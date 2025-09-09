import flax
import jax
from typing import Any, Callable
from utils.flax_utils import nonpytree_field

class GCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
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
