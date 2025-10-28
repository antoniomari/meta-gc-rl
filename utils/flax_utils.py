import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence, List, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.
        It is the abstraction for everything needed to reconstruct the state of a model
        which is training.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()  # Optimizer (e.g Adam)
    opt_state: Any  # State of optimizer (E.g. Adam statistics)

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Perform optimization step to update model params (must provide gradients)."""

        # self.tx is the optimizer (Adam) -> update params using grads and opt_state
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """

        # Compute gradients
        # NOTE: jax.grad is a functional that represents "grad of loss (given as parameters)"
        # so it represents the gradient function that gets evaluated @ self.params
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        # grad_max, grad_min, grad_norm are PyTrees with the same structure as grads.
        # Each leaf in grad_max contains the maximum value of the corresponding gradient array,
        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)
        # Example output structure:
        # If grads is a dict like {'Dense_0': {'kernel': ...array..., 'bias': ...array...}, ...}
        # then grad_max will be {'Dense_0': {'kernel': <scalar>, 'bias': <scalar>}, ...}
        # and similarly for grad_min and grad_norm.

        # Flatten all leaves so we can aggregate statistics (e.g., global max/min/norm) across the entire parameter tree.
        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        # Sum of grad across all parameters
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


class MetaTrainState(flax.struct.PyTreeNode):
    """
    MetaTrainState for MAML-style meta-training.

    This class allows you to:
    1. Perform inner-loop adaptation (N steps of optimizer) for a task, producing updated parameters.
    2. Add those updated parameters to a list (one per task).
    3. Combine all updated parameters to perform a meta-update to the original parameters.

    To avoid recompilation, updated_params_list and test_loss_grads are fixed-size lists (PyTrees)
    with static shapes, initialized with dummy values. The number of tasks (meta_batch_size) is fixed.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    inner_opt: Any = nonpytree_field()
    inner_opt_state: Any
    meta_opt: Any = nonpytree_field()
    meta_opt_state: Any
    updated_params_list: Any  # PyTree: list of parameter PyTrees, fixed size [meta_batch_size]
    test_loss_grads: Any      # PyTree: list of gradient PyTrees, fixed size [meta_batch_size]
    meta_batch_size: int = nonpytree_field()
    max_training_steps: int = nonpytree_field()
    merging_eps: float = nonpytree_field()

    @classmethod
    def make_pytree_list(cls, example, n):
            return [jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, "copy") else jnp.array(x), example) for _ in range(n)]

    def init_updated_params_list(self, params):
        # Helper to create a list of deep-copied parameter PyTrees (not zerolike)
        updated_params_list = self.make_pytree_list(params, self.meta_batch_size)
        test_loss_grads = self.make_pytree_list(jax.tree_util.tree_map(jnp.zeros_like, params), self.meta_batch_size)
        return updated_params_list, test_loss_grads

    @classmethod
    def create(cls, model_def, params, inner_opt=None, meta_opt=None, meta_batch_size=1, max_training_steps=100000, merging_eps=1.0, **kwargs):
        """
        meta_batch_size: number of tasks per meta-update (fixed for the lifetime of the object)
        max_training_steps: total number of meta-training steps (for annealing merging_eps)
        """
        if inner_opt is not None:
            inner_opt_state = inner_opt.init(params)
        else:
            inner_opt_state = None

        if meta_opt is not None:
            meta_opt_state = meta_opt.init(params)
        else:
            meta_opt_state = None


        updated_params_list = cls.make_pytree_list(params, meta_batch_size)
        test_loss_grads = cls.make_pytree_list(jax.tree_util.tree_map(jnp.zeros_like, params), meta_batch_size)

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            inner_opt=inner_opt,
            inner_opt_state=inner_opt_state,
            meta_opt=meta_opt,
            meta_opt_state=meta_opt_state,
            updated_params_list=updated_params_list,
            test_loss_grads=test_loss_grads,
            meta_batch_size=meta_batch_size,
            max_training_steps=max_training_steps,
            merging_eps=merging_eps,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """

        # Compute gradients
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        # Use meta_opt for the meta-update step
        updates, new_inner_opt_state = self.inner_opt.update(grads, self.inner_opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            inner_opt_state=new_inner_opt_state,
        ), info

    def add_task_adaptation_result(self, updated_params, test_loss_grads, final_opt_state, i):
        self.updated_params_list[i] = updated_params
        if test_loss_grads is not None:
            self.test_loss_grads[i] = test_loss_grads

        return self.replace(inner_opt_state=final_opt_state)

    def inner_update(self,
        loss_fn,
        test_loss_fn=None,
        is_fomaml=False,
        params=None,
        reset_inner_opt=False,
        test_loss_pre_update=False,
        debug_print: Optional[str] = None,
    ):
        """
        Perform a single step of inner-loop optimization for meta-learning.

        Args:
            loss_fn (Callable): Loss function for training, takes parameters as input and returns (loss, info).
            test_loss_fn (Callable, optional): Loss function for testing. Used to compute test-time gradients.
            is_fomaml (bool, optional): If True, computes gradients for FOMAML by differentiating test_loss_fn w.r.t. updated params.
                                         If False (MAML), differentiates test_loss_fn(updated_params) w.r.t. original params. Default: False.
            params (PyTree, optional): Initial parameters to optimize. If None, uses self.params.
            reset_inner_opt (bool, optional): If True, re-initializes the optimizer state. Default: False.

        Returns:
            updated_params (PyTree): Parameters after inner update.
            test_grads (PyTree or None): Gradients of test_loss_fn w.r.t. parameters, or None if test_loss_fn not provided.
            new_inner_opt_state: Optimizer state after update.
            info (dict): Information dictionary from the loss function (may be from training or testing).
        """

        info = {}
        # 1. Test loss pre-update
        if test_loss_fn is not None:
            if debug_print == "pre":
                jax.debug.print("Computing test loss pre-update")
            if is_fomaml:
                # FOMAML: grad of test_loss_fn w.r.t. updated_params
                test_grads, test_info_pre = jax.grad(test_loss_fn, has_aux=True)(params)
                if debug_print == "pre":
                    jax.debug.print('Test info: {}', info['actor/actor_loss'])
            else:
                assert NotImplementedError("MAML implementation has to be fixed.")
                # MAML: grad of test_loss_fn(updated_params) w.r.t. original params
                def test_loss_on_orig_params(orig_params):
                    return test_loss_fn(updated_params)
                test_grads, test_info_pre = jax.grad(test_loss_on_orig_params, has_aux=True)(params)

            info.update({f"pre_test/{k}": v for k, v in test_info_pre.items()})


        # 1. Peform 1 steps of inner-loop optimization, train the model on the train_batch
        if reset_inner_opt:
            opt_state = self.inner_opt.init(params)
        else:
            opt_state = self.inner_opt_state

        if params is None:
            params = self.params

        grads, train_info = jax.grad(loss_fn, has_aux=True)(params)
        info.update({f"train/{k}": v for k, v in train_info.items()})
        updates, new_inner_opt_state = self.inner_opt.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)

        # 2. Compute test gradients using updated_params
        test_grads = None
        if test_loss_fn is not None:
            if debug_print == "post":
                jax.debug.print("Computing test loss post-update")
            if is_fomaml:
                # FOMAML: grad of test_loss_fn w.r.t. updated_params
                test_grads, test_info_post = jax.grad(test_loss_fn, has_aux=True)(updated_params)

            else:
                assert NotImplementedError("MAML implementation has to be fixed.")
                # MAML: grad of test_loss_fn(updated_params) w.r.t. original params
                def test_loss_on_orig_params(orig_params):
                    return test_loss_fn(updated_params)
                test_grads, test_info_post = jax.grad(test_loss_on_orig_params, has_aux=True)(params)

            info.update({f"test/{k}": v for k, v in test_info_post.items()})
            if debug_print == "post":
                jax.debug.print(f'Test info post-update: {info}')

        return updated_params, test_grads, new_inner_opt_state, info

    def meta_update(self, use_model_merging=False, eps=None, **kwargs):
        """
        Perform a meta-update using the list of updated parameters.
        If use_model_merging is True, perform a model merge update:
            new_params = self.params + eps * (self.updated_params_list[0] - self.params)
        Otherwise, the meta-gradient is computed as the difference between the mean of updated parameters and the current parameters.
        Returns a new MetaTrainState with updated parameters and an empty updated_params_list.

        eps: (optional) stepsize for model merging. If None, will use self.merging_eps (which is annealed).
        """
        if not self.updated_params_list:
            raise ValueError("No updated parameters to perform meta-update.")

        mean_updated_params = jax.tree_util.tree_map(
            lambda *ps: jnp.stack(ps).mean(axis=0), *self.updated_params_list
        )

        # Anneal merging_eps linearly from 1.0 to 0 over max_training_steps
        if use_model_merging:
            if eps is None:
                # Compute annealed merging_eps
                if self.max_training_steps > 1:
                    merging_eps = self.merging_eps - (self.step - 1) / (self.max_training_steps - 1)
                    merging_eps = jnp.clip(merging_eps, 0.0, 1.0)
                else:
                    merging_eps = self.merging_eps
            else:
                merging_eps = eps

            print(f'Merging eps: {merging_eps}')

            merged_params = jax.tree_util.tree_map(
                lambda p, up: p + merging_eps * (up - p), self.params, mean_updated_params
            )

            updated_params_list, test_loss_grads = self.init_updated_params_list(merged_params)

            return self.replace(
                step=self.step + 1,
                params=merged_params,
                updated_params_list=updated_params_list,
                test_loss_grads=test_loss_grads,
                **kwargs,
            )
        else:
            # Compute meta-gradient as the average of test_loss_grads
            meta_grads = jax.tree_util.tree_map(
                lambda *gs: jnp.stack(gs).mean(axis=0), *self.test_loss_grads
            )

            # Use meta_opt for the meta-update step
            updates, new_meta_opt_state = self.meta_opt.update(meta_grads, self.meta_opt_state, self.params)
            new_params = optax.apply_updates(self.params, updates)

            updated_params_list, test_loss_grads = self.init_updated_params_list(new_params)

            return self.replace(
                step=self.step + 1,
                params=new_params,
                meta_opt_state=new_meta_opt_state,
                updated_params_list=updated_params_list,
                test_loss_grads=test_loss_grads,
                **kwargs,
            )

    def __call__(self, *args, params=None, method=None, **kwargs):
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None
        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        return functools.partial(self, name=name)



def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)


    # If agent has MetaTrainState, only restore specific fields using replace
    if hasattr(agent, 'network') and agent.network is not None:
        # Get the fields we want to restore from the checkpoint

        if 'meta_train_state' in load_dict['agent']:
            # Old implementation naming
            network = load_dict['agent']['meta_train_state']
        else:
            # New implementation naming
            network = load_dict['agent']['network']

        # Only restore params, skip optimizer states to avoid serialization issues
        agent = agent.replace(
            network=agent.network.replace(
                params=network.get('params', agent.network.params)
                # Skip inner_opt_state and meta_opt_state to avoid serialization issues
            )
        )
        # Setup updated_params_list and test_loss_grads
        agent.network.init_updated_params_list(agent.network.params)
    else:
        # For non-meta agents, restore normally
        agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
