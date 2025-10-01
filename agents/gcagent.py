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
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
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
    meta_train_state: MetaTrainState
    config: Any = nonpytree_field()

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature: float = 1.0):
        dist = self.meta_train_state.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config.get('discrete'):
            actions = jnp.clip(actions, -1, 1)
        return actions

    def build_loss_fn(self, batch, rng) -> Callable[[Any], tuple[Any, dict]]:
        raise NotImplementedError

    @jax.jit
    def update(self, batch, finetuning: bool = False):
        """Update the agent and return a new agent with information dictionary."""
        # NOTE: Finetuning argument is unused, kept for interface unification
        new_rng, rng = jax.random.split(self.rng) # rng used now, new_rng for next step
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)
        new_meta_train_state, info = self.meta_train_state.apply_loss_fn(loss_fn=loss_fn)
        # Return a new immutable agent with updated network and PRNG + metrics
        return self.replace(meta_train_state=new_meta_train_state, rng=new_rng), info

    @functools.partial(jax.jit, static_argnames=("is_fomaml", "i"))
    def meta_inner_update(
        self,
        train_batch: dict,
        i: int,
        test_batch: Optional[dict] = None,
        is_fomaml: bool = False,

    ) -> tuple["MetaGCAgent", dict]:
        """
        Perform an inner-loop meta-update for a single task.

        Args:
            train_batch (dict): Training batch for inner adaptation.
            i (int): Index of the task_batch.
            test_batch (dict, optional): Test batch for meta-gradient computation. Defaults to None.
            is_fomaml (bool, optional): Whether to use FOMAML-style gradient computation. Defaults to False.

        Returns:
            Tuple[MetaGCAgent, dict]:
                - A new MetaGCAgent with updated meta_train_state reflecting the task adaptation result.
                - An info dictionary with statistics from the inner update.
        """
        # Call get_inner_update_result and add the result to meta_train_state
        updated_params, test_grads, info = self.get_inner_update_result(train_batch, test_batch, is_fomaml)
        new_meta_train_state = self.meta_train_state.add_task_adaptation_result(updated_params, test_grads, i)
        return self.replace(meta_train_state=new_meta_train_state), info


    def get_inner_update_result(self, train_batch, test_batch = None, is_fomaml = True):
        new_rng, step_rng = jax.random.split(self.rng)

        # loss_fn is partial of total loss giving step_rng
        def loss_fn(grad_params):
            return self.total_loss(train_batch, grad_params, rng=step_rng)

        if test_batch is not None:
            new_rng, test_step_rng = jax.random.split(new_rng)
            def test_loss_fn(grad_params):
                return self.total_loss(test_batch, grad_params, rng=test_step_rng)
        else:
            test_loss_fn = None

        updated_params, test_grads, info = self.meta_train_state.inner_update(loss_fn=loss_fn, num_steps=1, test_loss_fn=test_loss_fn, is_fomaml=is_fomaml)
        return updated_params, test_grads, info


    @functools.partial(jax.jit, static_argnames=("use_model_merging",))
    def meta_update(self, use_model_merging=False):

        # Type annotation for self.meta_train_state
        meta_train_state: MetaTrainState = self.meta_train_state
        new_meta_train_state = meta_train_state.meta_update(use_model_merging)
        return self.replace(meta_train_state=new_meta_train_state)



