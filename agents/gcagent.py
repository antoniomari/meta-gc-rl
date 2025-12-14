import flax
import jax
import functools
import jax.numpy as jnp
from typing import Any, Callable, Optional
from utils.flax_utils import nonpytree_field, MetaTrainState, TrainState

class GCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature: float = 1.0):
        dist = self.network.select('actor')(observations, goals, temperature=temperature, params=self.network.params)
        actions = dist.sample(seed=seed)
        if not self.config.get('discrete'):
            actions = jnp.clip(actions, -1, 1)
        return actions

    def build_loss_fn(self, batch, rng) -> Callable[[Any], tuple[Any, dict]]:
        raise NotImplementedError

    @jax.jit
    def update(self, batch):
        new_rng, step_rng = jax.random.split(self.rng)
        loss_fn = self.build_loss_fn(batch, step_rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info


class MetaGCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: MetaTrainState
    config: Any = nonpytree_field()

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature: float = 1.0):
        dist = self.network.select('actor')(observations, goals, temperature=temperature, params=self.network.params)
        actions = dist.sample(seed=seed)
        if not self.config.get('discrete'):
            actions = jnp.clip(actions, -1, 1)
        return actions

    def total_loss(self, batch, grad_params, rng=None, fixed_params=None, actor_only=False):
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnames=("finetuning", "reset_inner_opt", "actor_only"))
    def update(self, batch, finetuning: bool = False, reset_inner_opt: bool = False, actor_only=False) -> tuple["MetaGCAgent", dict]:
        """Update the agent and return a new agent with information dictionary."""
        # NOTE: Finetuning argument is unused, kept for interface unification
        new_rng, rng = jax.random.split(self.rng) # rng used now, new_rng for next step
        def loss_fn(grad_params):
            return self.total_loss(
                batch,
                grad_params,
                rng=rng,
                fixed_params=self.network.params,
                actor_only=actor_only,
            )
        new_meta_train_state, info = self.network.apply_loss_fn(loss_fn=loss_fn, reset_opt=reset_inner_opt)
        # Return a new immutable agent with updated network and PRNG + metrics
        return self.replace(network=new_meta_train_state, rng=new_rng), info

    def meta_inner_update(
        self,
        i: int,
        train_batch: dict,
        test_batch: Optional[dict] = None,
        reset_inner_opt: bool = False,
        debug_print: Optional[str] = None,
        average_test_gradients: bool = False,
        inner_step: Optional[int] = None,
        actor_only: bool = False,
    ) -> tuple["MetaGCAgent", dict]:
        """
        Perform an inner-loop meta-update for a single task.

        Args:
            train_batch (dict): Training batch for inner adaptation.
            i (int): Index of the task_batch.
            test_batch (dict, optional): Test batch for meta-gradient computation. Defaults to None.
            num_steps (int, optional): Number of steps to take in the inner update. Defaults to 1.
        Returns:
            Tuple[MetaGCAgent, dict]:
                - A new MetaGCAgent with updated meta_train_state reflecting the task adaptation result.
                - An info dictionary with statistics from the inner update.
        """
        # Split RNG at the start for all operations
        rng = self.rng
        rng, pre_test_rng = jax.random.split(rng)
        rng, inner_update_rng = jax.random.split(rng)
        rng, post_test_rng = jax.random.split(rng)
        # rng now holds the key for the next operation (to be stored in agent)

        # Compute pre-test loss and gradients
        pre_test_grads = None
        if test_batch is not None:
            pre_test_grads, test_info_pre = self.compute_test_loss_and_grads(
                test_batch=test_batch,
                params=self.network.updated_params_list[i],
                actor_only=actor_only,
                rng=pre_test_rng
            )
        # Call get_inner_update_result and add the result to meta_train_state
        updated_params, final_opt_state, info, unscaled_updates, _ = self.get_inner_update_result(
            train_batch,
            params_idx=i,
            reset_inner_opt=reset_inner_opt,
            actor_only=actor_only,
            rng=inner_update_rng
        )
        # Compute test loss

        # Compute test loss and gradients
        test_grads = None
        if test_batch is not None:
            test_grads, test_info = self.compute_test_loss_and_grads(
                test_batch,
                updated_params,
                actor_only,
                rng=post_test_rng
            )
            # Update info with pre-test and test loss info
            info.update({f"pre_test/{k}": v for k, v in test_info_pre.items()})
            info.update({f"test/{k}": v for k, v in test_info.items()})

        new_network = self.network.add_task_adaptation_result(
            updated_params,
            pre_test_grads,
            test_grads,
            final_opt_state,
            i,
            inner_updates=unscaled_updates,
            average_test_gradients=average_test_gradients,
            inner_step=inner_step,
            info=info
        )
        return self.replace(network=new_network, rng=rng), info

    # TODO: consider generalization here
    @functools.partial(jax.jit, static_argnames=("actor_only"))
    def compute_test_loss_and_grads(self, test_batch, params, actor_only, rng=None):

        def test_loss_fn(grad_params):
            return self.total_loss(
                test_batch,
                grad_params=grad_params,
                rng=rng if rng is not None else self.rng,
                fixed_params=params,
                actor_only=actor_only,
            )

        test_grads, test_info = jax.grad(test_loss_fn, has_aux=True)(params)
        test_info.update(self.network.compute_grad_stats(test_grads))
        return test_grads, test_info

    @functools.partial(jax.jit, static_argnames=("reset_inner_opt", "params_idx", "actor_only"))
    def get_inner_update_result(self,
        train_batch,
        params_idx: int = 0,
        reset_inner_opt: bool = False,
        actor_only: bool = False,
        rng=None,
    ):
        """
        NOTE: This methods is overridden
        Get the result of the inner update for a single task.
        Args:
            train_batch (dict): Training batch for inner adaptation.
            test_batch (dict, optional): Test batch for meta-gradient computation. Defaults to None.
            num_steps (int, optional): Number of steps to take in the inner update. Defaults to 1.
            rng: RNG key to use for inner update. If None, uses self.rng and splits it.

        Returns:
            Tuple[Any, Any, dict, Any]:
                - Updated parameters after the inner update.
                - Final optimizer state.
                - Info dictionary with statistics from the inner update.
                - New RNG key (for next operation).
        """
        if rng is None:
            new_rng, step_rng = jax.random.split(self.rng)
        else:
            new_rng, step_rng = jax.random.split(rng)

        updated_params, final_opt_state, info, unscaled_updates = self.network.inner_update(
            self.total_loss,
            train_batch=train_batch,
            params=self.network.updated_params_list[params_idx],
            reset_inner_opt=reset_inner_opt,
            actor_only=actor_only,
            rng=step_rng,
        )
        return updated_params, final_opt_state, info, unscaled_updates, new_rng


    @functools.partial(jax.jit, static_argnames=("use_model_merging", "use_meta_optimizer", "annealing", "use_best_checkpoint"))
    def meta_update(self, use_model_merging=False, use_meta_optimizer=False, annealing=False, use_best_checkpoint=False):
        new_network = self.network.meta_update(use_model_merging=use_model_merging, use_meta_optimizer=use_meta_optimizer, annealing=annealing, use_best_checkpoint=use_best_checkpoint)
        return self.replace(network=new_network)


    @functools.partial(jax.jit, static_argnames=("use_meta_optimizer", "annealing"))
    def distillation_update(self, use_meta_optimizer=False, annealing=False):

        # Create loss function ||theta[i] - self.network.params||^2

        def loss_fn(grad_params):
            # grad_params and self.network.updated_params_list[0] are pytrees
            # Compute mean squared difference for each parameter tensor (scalar per tensor)
            mean_squared_diffs = jax.tree_util.tree_map(lambda a, b: jnp.mean(jnp.square(a - b)), grad_params, self.network.updated_params_list[0])
            # Average all the per-parameter scalars
            leaves = jax.tree_util.tree_leaves(mean_squared_diffs)
            loss = jnp.mean(jnp.stack(leaves))
            info = {"distillation_loss": loss}
            return loss, info

        new_network = self.network.distillation_update(loss_fn=loss_fn)
        return self.replace(network=new_network)
