import copy
from typing import Any, Optional

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, MetaTrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue
from agents.gcagent import MetaGCAgent
import functools


class GCIQLAgent(MetaGCAgent):
    """Goal-conditioned implicit Q-learning (GCIQL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    """

    # Defined in parent class MetaGCAgent
    # rng: Any
    # network:  MetaTrainState
    # config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params, fixed_params=None):
        """Compute the IQL value loss."""

        # This one uses the fixed parameters, not the gradient parameters (as it is the target network)
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'], params=fixed_params)
        # q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)
        q = jnp.minimum(q1, q2)

        # This one uses the gradient parameters, as it is the value network
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params, fixed_params=None):
        """Compute the IQL critic loss."""
        # This one uses the fixed parameters, not the gradient parameters (as it is the value network)
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'], params=fixed_params)
        # next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'], params=grad_params)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        # This one uses the gradient parameters, as it is the critic network
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None, fixed_params=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(batch['observations'], batch['actor_goals'], params=fixed_params)
            # v = self.network.select('value')(batch['observations'], batch['actor_goals'], params=grad_params)
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'], params=fixed_params)
            # q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'], params=grad_params)
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

            # TODO: check if it is correct not to use gradient parameters here (ask marco)
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions, params=fixed_params)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            # jax.lax.stop_gradient prevents gradients from flowing through its argument during backprop.
            # Here, it normalizes q.mean() by the absolute mean detached from the computation graph, so
            # the denominator does not affect gradients.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        elif self.config['actor_loss'] == 'bc':
            # BC loss.
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            # NOTE: temporary code
            # Add L2 regularizer to the actor loss
            """
            l2_reg = 0.0
            for p in jax.tree_leaves(grad_params):
                l2_reg += jnp.sum(p ** 2)
            l2_weight =  0 # 1e-05
            actor_loss = -log_prob.mean() + l2_weight * l2_reg  # Original  actor_loss = -log_prob.mean()
            """
            actor_loss = -log_prob.mean()


            actor_info = {
                'actor_loss': actor_loss,
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @functools.partial(jax.jit, static_argnames=("actor_only"))
    def total_loss(self, batch, grad_params, rng=None, fixed_params=None, actor_only=False):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params, fixed_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, fixed_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, fixed_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        if actor_only:
            loss = actor_loss
        else:
            loss = value_loss + critic_loss + actor_loss
        info["total_loss"] = loss
        return loss, info

    def target_update(self,
        initial_params,
        updated_params,
        module_name: str = "critic"
    ) -> None:
        """Updates the target network, inplace.

        Args:
            initial_params: The initial parameters of the network.
            updated_params: The updated parameters of the network.
            module_name: The name of the module to update the target network for.
        """
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            initial_params[f'modules_{module_name}'],
            initial_params[f'modules_target_{module_name}'],
        )
        updated_params[f'modules_target_{module_name}'] = new_target_params

    @functools.partial(jax.jit, static_argnames=("finetuning", "reset_inner_opt", "actor_only"))
    def update(self, batch, finetuning=False, reset_inner_opt=False, actor_only=False):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(
                batch,
                grad_params,
                rng=rng,
                fixed_params=self.network.params,
                actor_only=actor_only,
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn, reset_opt=reset_inner_opt)
        self.target_update(self.network.params, new_network.params, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @functools.partial(jax.jit, static_argnames=("reset_inner_opt", "params_idx", "actor_only"))
    def get_inner_update_result(
        self,
        train_batch,
        params_idx: int = 0,
        reset_inner_opt: bool = False,
        actor_only: bool = False,
        rng=None,
    ):
        # This is an override, call the parent class's get_inner_update_result
        # The class should be exactly MetaGCAgent, so we can call the parent class's get_inner_update_result
        initial_params = self.network.updated_params_list[params_idx]
        updated_params, final_opt_state, info, unscaled_updates, new_rng = super().get_inner_update_result(train_batch, params_idx, reset_inner_opt, actor_only, rng)
        # Perform target update here
        # TODO: check for case of actor_only
        self.target_update(initial_params, updated_params, 'critic')
        return updated_params, final_opt_state, info, unscaled_updates, new_rng

    @functools.partial(jax.jit, static_argnames=("use_model_merging", "use_meta_optimizer", "annealing", "use_best_checkpoint"))
    def meta_update(self, use_model_merging=False, use_meta_optimizer=False, annealing=False, use_best_checkpoint=False):
        new_network = self.network.meta_update(use_model_merging=use_model_merging, use_meta_optimizer=use_meta_optimizer, annealing=annealing, use_best_checkpoint=use_best_checkpoint)
        # USE TARGET UPDATE HERE instead of in inner-update
        # TODO: check for case of actor_only
        self.target_update(self.network.params, new_network.params, 'critic')
        return self.replace(network=new_network)

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
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )

        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
            )

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
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        # Define two separate Adam optimizers: one for inner loop, one for meta-update
        """
        if "max_grad_norm" in config and config['max_grad_norm'] is not None:
            print(f"Using max grad norm: {config['max_grad_norm']}")
            inner_opt = optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                optax.adam(learning_rate=config['inner_lr']),
            )
            meta_opt = optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                optax.adam(learning_rate=config['lr']),
            )
        else:
        """
        inner_opt = optax.adam(learning_rate=config['inner_lr'])
        meta_opt = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network =  MetaTrainState.create(
            network_def,
            network_params,
            inner_opt=inner_opt,
            meta_opt=meta_opt,
            meta_batch_size=config['meta_batch_size'],
            max_training_steps=train_steps, # TODO: adjust config next
            merging_eps=config['merging_eps'],
        )

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gciql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.3,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
