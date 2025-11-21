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
        is_fomaml: bool = False,
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
            is_fomaml (bool, optional): Whether to use FOMAML-style gradient computation. Defaults to False.
            num_steps (int, optional): Number of steps to take in the inner update. Defaults to 1.
        Returns:
            Tuple[MetaGCAgent, dict]:
                - A new MetaGCAgent with updated meta_train_state reflecting the task adaptation result.
                - An info dictionary with statistics from the inner update.
        """
        # Call get_inner_update_result and add the result to meta_train_state
        updated_params, test_grads, final_opt_state, info, unscaled_updates = self.get_inner_update_result(train_batch, test_batch, is_fomaml, reset_inner_opt, params_idx=i, debug_print=debug_print, actor_only=actor_only)
        new_network = self.network.add_task_adaptation_result(updated_params, test_grads, final_opt_state, i, inner_updates=unscaled_updates, average_test_gradients=average_test_gradients, inner_step=inner_step)
        return self.replace(network=new_network), info

    @functools.partial(jax.jit, static_argnames=("is_fomaml", "reset_inner_opt", "params_idx", "debug_print", "actor_only"))
    def get_inner_update_result(self,
        train_batch,
        test_batch: Optional[dict] = None,
        is_fomaml: bool = True,
        reset_inner_opt: bool = False,
        params_idx: int = 0,
        debug_print: Optional[str] = None,
        actor_only: bool = False,
    ):
        """
        Get the result of the inner update for a single task.
        Args:
            train_batch (dict): Training batch for inner adaptation.
            test_batch (dict, optional): Test batch for meta-gradient computation. Defaults to None.
            is_fomaml (bool, optional): Whether to use FOMAML-style gradient computation. Defaults to False.
            num_steps (int, optional): Number of steps to take in the inner update. Defaults to 1.

        Returns:
            Tuple[Any, Any, dict]:
                - Updated parameters after the inner update.
                - Test gradients after the inner update.
                - Info dictionary with statistics from the inner update.
        """
        new_rng, step_rng = jax.random.split(self.rng)

        # loss_fn is partial of total loss giving step_rng
        def loss_fn(grad_params):
            return self.total_loss(
                train_batch,
                grad_params,
                rng=step_rng,
                fixed_params=self.network.updated_params_list[params_idx],
                actor_only=actor_only,
            )

        if test_batch is not None:
            new_rng, test_step_rng = jax.random.split(new_rng)
            def test_loss_fn(grad_params):
                return self.total_loss(
                    test_batch,
                    grad_params,
                    rng=test_step_rng,
                    fixed_params=self.network.updated_params_list[params_idx],
                    actor_only=actor_only,
                )
        else:
            test_loss_fn = None

        updated_params, test_grads, final_opt_state, info, unscaled_updates = self.network.inner_update(
            loss_fn=loss_fn,
            test_loss_fn=test_loss_fn,
            is_fomaml=is_fomaml,
            params=self.network.updated_params_list[params_idx],
            reset_inner_opt=reset_inner_opt,
        )
        return updated_params, test_grads, final_opt_state, info, unscaled_updates


    @functools.partial(jax.jit, static_argnames=("use_model_merging", "use_meta_optimizer", "annealing"))
    def meta_update(self, use_model_merging=False, use_meta_optimizer=False, annealing=False):
        new_network = self.network.meta_update(use_model_merging=use_model_merging, use_meta_optimizer=use_meta_optimizer, annealing=annealing)
        return self.replace(network=new_network)

