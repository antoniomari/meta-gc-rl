from typing import Any
import functools
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, MetaTrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor
from agents.gcagent import GCAgent, MetaGCAgent
from typing import cast

class GCBCAgent(MetaGCAgent):
    """Goal-conditioned behavioral cloning (GCBC) agent."""
    # NOTE: flax.struct.PyTreeNode is turned into a frozen dataclass like Flax struct
    # - attributes declared in the class body are instance fields
    # - nonpytree_field() is a Flax helper marks a field as non-pytree (excluded from JAX transformations/trees)

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the BC actor loss."""
        # Outputs means and variances of gaussian distributions over actions
        # Each distribution is conditioned on state and goal
        # mu(s, g) and sigma(s, g)
        # They are computed using grad_params, the gradient will be computed wrt to these
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)

        # log_prob is the log probability (density) of the actions under the actor distribution
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -log_prob.mean()

        actor_info = {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    # policy deterministic action vs dataset action
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    # mean of distribution diagonal standard deviation
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    @functools.partial(jax.jit, static_argnames=("actor_only"))  # mindful of arguments that must be static (e.g. shapes), recompiles if shapes change
    def total_loss(self, batch, grad_params, rng=None, fixed_params=None, actor_only=False):
        """Compute the total loss.

            Note: fixed_params and actor_only are not used in this implementation, kept for interface unification
            Args:
                batch: Batch of data.
                grad_params: Gradient parameters.
                rng: Random number generator.
                fixed_params: Fixed parameters (unused for GC-BC).
                actor_only: Whether to only compute the actor loss (unused for GC-BC).

        Returns:
            Tuple[Any, dict]: Loss and information dictionary.
        """
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss
        return loss, info

    @functools.partial(jax.jit, static_argnames=("finetuning", "reset_inner_opt", "actor_only"))
    def update(self, batch, finetuning: bool = False, reset_inner_opt: bool = False, actor_only: bool = False):
        return super().update(batch, finetuning, reset_inner_opt, actor_only)


    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        train_steps,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        # jax random splits keys deterministically, to have multiple reproducible random streams
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoder.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define actor network.
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        # Define two separate Adam optimizers: one for inner loop, one for meta-update
        inner_opt = optax.adam(learning_rate=config['inner_lr'])
        meta_opt = optax.adam(learning_rate=config['lr'])
        # init + dummy forward pass, returns PyTree?
        network_params = network_def.init(init_rng, **network_args)['params']
        # wrapper -> tracks params and optimizer state, to pass to update steps
        meta_train_state = MetaTrainState.create(
            network_def,
            network_params,
            inner_opt=inner_opt,
            meta_opt=meta_opt,
            meta_batch_size=config['meta_batch_size'],
            max_training_steps=train_steps, # TODO: adjust config next
            merging_eps=config['merging_eps'],
        )

        return cls(rng, network=meta_train_state, config=flax.core.FrozenDict(**config))



def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcbc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
