import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

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
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any
    updated_params_list: Any  # List of parameter PyTrees
    test_loss_grads: Any # List of gradients of test losses (only for MAML/FOMAML)

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
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
            updated_params_list=[],
            test_loss_grads=[],
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

        # self.tx is the optimizer (Adam) -> update params using grads and opt_state
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        ), info


    def inner_loop_update(self, loss_fn, num_steps=1, tx=None):
        """
        Perform N steps of inner-loop optimization starting from given params (or self.params).
        Returns the updated parameters after N steps.
        """
        # I should use statistics of the optimizer but don't update it now
        # opt_state = self.tx.init(self.params) # TODO: why this is not clear
        current_params = self.params

        for _ in range(num_steps):
            # current_params = self.apply_loss_fn(loss_fn)
            grads, info = jax.grad(loss_fn, has_aux=True)(current_params)
            updates, opt_state = self.tx.update(grads, self.opt_state, current_params)
            current_params = optax.apply_updates(current_params, updates)
        return current_params, info

    def add_task_adaptation(self, loss_fn, num_steps=1, test_loss_fn=None, is_fomaml=False):
        """
        Perform inner-loop adaptation for a task (N steps of optimizer), then add the resulting parameters to the list.
        If test_loss_fn is provided, compute the gradient of test_loss_fn evaluated at updated_params,
        using either FOMAML or Reptile/MAML style depending on is_fomaml.
        In both cases, add the test_loss_grads to self.test_loss_grads.
        Returns a new MetaTrainState with the updated lists.
        """
        updated_params, info = self.inner_loop_update(loss_fn, num_steps=num_steps)

        if test_loss_fn is not None:
            if is_fomaml:
                # FOMAML: grad of test_loss_fn w.r.t. updated_params
                grads, info = jax.grad(test_loss_fn, has_aux=True)(updated_params)
            else:
                # MAML: grad of test_loss_fn(updated_params) w.r.t. original params
                def test_loss_on_orig_params(orig_params):
                    return test_loss_fn(updated_params)
                grads, info = jax.grad(test_loss_on_orig_params, has_aux=True)(self.params)

            # Optimized: Use jax.tree_map for efficient list operations
            # This is more efficient than list concatenation for large lists
            return self.replace(
                updated_params_list=self.updated_params_list + [updated_params],
                test_loss_grads=self.test_loss_grads + [grads]
            ), info
        else:
            return self.replace(
                updated_params_list=self.updated_params_list + [updated_params]
            ), info

    def clear_updated_params(self):
        """
        Clear the list of updated parameters.
        Returns a new MetaTrainState with an empty list.
        """
        return self.replace(updated_params_list=[])

    def meta_update(self, use_model_merging=False, eps=0.1, **kwargs):
        """
        Perform a meta-update using the list of updated parameters.
        If use_model_merging is True, perform a model merge update:
            new_params = self.params + eps * (self.updated_params_list[0] - self.params)
        Otherwise, the meta-gradient is computed as the difference between the mean of updated parameters and the current parameters.
        Returns a new MetaTrainState with updated parameters and an empty updated_params_list.
        """
        if not self.updated_params_list:
            raise ValueError("No updated parameters to perform meta-update.")

        mean_updated_params = jax.tree_util.tree_map(
            lambda *ps: jnp.stack(ps).mean(axis=0), *self.updated_params_list
        )

        if use_model_merging:
            # Model merging: interpolate between self.params and the average of updated params
            merged_params = jax.tree_util.tree_map(
                lambda p, up: p + eps * (up - p), self.params, mean_updated_params
            )
            return self.replace(
                step=self.step + 1,
                params=merged_params,
                updated_params_list=[],
                **kwargs,
            )
        else:
            # Compute meta-gradient as the average of test_loss_grads
            meta_grads = jax.tree_util.tree_map(
                lambda *gs: jnp.stack(gs).mean(axis=0), *self.test_loss_grads
            )

            # Standard optimizer update
            updates, new_opt_state = self.tx.update(meta_grads, self.opt_state, self.params)
            new_params = optax.apply_updates(self.params, updates)

            return self.replace(
                step=self.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                updated_params_list=[],
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

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
