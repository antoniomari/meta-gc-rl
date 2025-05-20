from collections import defaultdict
import copy

import jax
import numpy as np
from tqdm import trange
import optax


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    finetune_config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    train_dataset=None,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    exp_name=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        finetune_config: Configuration dictionary specific to finetuning.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    trajs = []
    stats = defaultdict(list)
    # check if agent.agent_name in config is hiql
    agent_name = config.get('agent_name', None)
    is_hiql = (agent_name == 'hiql')

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        # Simple script to plot critic and policy output in a 2D environment

        _batch = train_dataset.sample(10000)
        import numpy as np
        def make_plots(suffix):
            import matplotlib.pyplot as plt
            _batch = train_dataset.sample(10000)
            _obs = _batch['observations']
        #     print(values[:3])
            if is_hiql:
                # We need to sample from the high-level policy and then the low-level policy, below is sample code
                # Sample from the high-level policy
                (values_1, values_2) = agent.network.select('value')(_obs, goal.reshape(1, -1).repeat(10000, 0))
                # v = (v1 + v2) / 2
                values = (values_1 + values_2) / 2
                high_dist = agent.network.select('high_actor')(_obs, goal.reshape(1, -1).repeat(10000, 0))
                goal_reps = high_dist.sample(seed=jax.random.PRNGKey(np.random.randint(0, 2**32)))
                goal_reps = goal_reps / np.linalg.norm(goal_reps, axis=-1, keepdims=True) * np.sqrt(goal_reps.shape[-1])
                # Sample from the low-level policy
                actions = agent.network.select('low_actor')(_obs, goal_reps, goal_encoded=True).mean()
            else:
                values = agent.network.select('value')(_obs, goal.reshape(1, -1).repeat(10000, 0))
                actions = agent.network.select('actor')(_obs, goal.reshape(1, -1).repeat(10000, 0)).mean()
        #     print(actions[:3])
        #     print(values[8350] - values[8300])
            import io
            import wandb
            from PIL import Image

            try:
                buf = io.BytesIO()
                plt.scatter(_obs[:, 0], _obs[:, 1], c=values)
                #plt.savefig(f'Zvalues_{suffix}_{exp_name}.png')
                plt.savefig(buf, format='png', dpi=200)
                plt.close()
                buf.seek(0)
                wandb.log({f'Zvalues_{suffix}': wandb.Image(np.array(Image.open(buf)))})
                del buf
            except Exception as e:
                print(f"Error logging Zvalues_{suffix}: {e}")
                plt.close() # Ensure plot is closed even on error

            try:
                buf = io.BytesIO()
                plt.quiver(_obs[:, 0], _obs[:, 1], actions[:, 0], actions[:, 1], angles='xy', scale_units='xy', scale=2)
                #plt.savefig(f'Zactions_{suffix}_{exp_name}.png', dpi=900)
                plt.savefig(buf, format='png', dpi=200)
                plt.close()
                buf.seek(0)
                wandb.log({f'Zactions_{suffix}': wandb.Image(np.array(Image.open(buf)))})
                del buf
            except Exception as e:
                print(f"Error logging Zactions_{suffix}: {e}")
                plt.close()
            #wandb.log({f'Zactions_{suffix}': wandb.Image(buf)})
            
        make_plots('pre')

        finetune_stats = defaultdict(list)
        recursive_mdp = finetune_config.get('filter_by_recursive_mdp', False)
        aggregated_filters = []  
        if recursive_mdp:
            done = False
            step = 0
            render = []
            old_train_state, old_config = None, None
            actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

            # Define how many steps to execute between replanning phases.
            replan_horizon = finetune_config.get('replan_horizon', 100)

            # Replanning loop: repeatedly fine-tune and execute a short horizon.
            while not done:
                # Active test-time fine-tuning:
                if old_train_state is not None:
                    agent = agent.replace(network=old_train_state, config=old_config)
                
                old_config = agent.config
                new_config = agent.config.unfreeze()
                if finetune_config.actor_loss is not None and not is_hiql:
                    new_config['actor_loss'] = finetune_config.actor_loss
                if finetune_config.alpha is not None and not is_hiql:
                    new_config['alpha'] = finetune_config.alpha
                new_config = old_config.__class__(new_config)
                old_train_state = copy.deepcopy(agent.network)
                opt_state = agent.network.opt_state
                finetune_tx = optax.adam(learning_rate=finetune_config.lr)
                agent = agent.replace(network=agent.network.replace(tx=finetune_tx, opt_state=opt_state), config=new_config)

                finetune_stats = defaultdict(list)
                if hasattr(finetune_config, 'unfreeze'):
                    current_finetune_config = finetune_config.unfreeze()
                else:
                    # If it's a standard dictionary, create a copy to modify
                    current_finetune_config = dict(finetune_config)

                cube_env = current_finetune_config.get('cube_env', False)
                if cube_env: # A way to detect CubeEnv, or use env.spec.id
                    # env._num_cubes should be available if 'env' is an instance of your CubeEnv
                    num_cubes = env._num_cubes if hasattr(env, '_num_cubes') else 1 # Default to 1 if not found, adjust as needed
                    current_finetune_config['num_cubes'] = num_cubes
                    # The 9 elements per cube state: 3 (pos) + 4 (quat) + 2 (sin/cos yaw)
                    current_finetune_config['proprio_dim'] = env.observation_space.shape[0] - num_cubes * 9

                # If finetune_config was unfrozen, you might need to freeze it again or use the unfrozen version
                # For example, if the original was a frozen ConfigDict:
                # finetune_config_to_pass = config_dict.ConfigDict(current_finetune_config)
                # For simplicity, assuming current_finetune_config can be passed directly:
                finetune_config_to_pass = current_finetune_config
                _filter = train_dataset.prepare_active_sample(agent, observation, goal, finetune_config_to_pass, exp_name=exp_name,
                                                              log_filter=False, is_hiql=is_hiql)
                aggregated_filters.append(_filter)
                if _filter.sum() > 0:
                    for _ in range(finetune_config.num_steps):
                        batch = train_dataset.active_sample(
                            finetune_config.batch_size,
                            _filter,
                            goal,
                            finetune_config.ratio,
                            finetune_config.fix_actor_goal,
                            is_hiql = is_hiql
                        )
                        agent, update_info = agent.update(batch, finetuning=True)
                        add_to(finetune_stats, flatten(update_info))
                
                actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
                # Execute the policy for a fixed short horizon before replanning.
                for _ in range(replan_horizon):
                    if done:
                        break
                    action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
                    action = np.array(action)
                    if not config.get('discrete'):
                        if eval_gaussian is not None:
                            action = np.random.normal(action, eval_gaussian)
                        action = np.clip(action, -1, 1)

                    next_observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    step += 1
                    done = done or step >= 3000

                    if should_render and (step % video_frame_skip == 0 or done):
                        frame = env.render().copy()
                        if goal_frame is not None:
                            render.append(np.concatenate([goal_frame, frame], axis=0))
                        else:
                            render.append(frame)

                    transition = dict(
                        observation=observation,
                        next_observation=next_observation,
                        action=action,
                        reward=reward,
                        done=done,
                        info=info,
                    )
                    add_to(traj, transition)
                    observation = next_observation  # Update state for the next replan phase.

            # At the end of the recursive loop, aggregate all collected filters.
            if aggregated_filters:
                import numpy as np
                combined_filter = np.zeros_like(aggregated_filters[0])
                numberofallfiltered = 0
                for f in aggregated_filters:
                    # Combine filters via element-wise maximum (logical OR for binary masks)
                    combined_filter = np.maximum(combined_filter, f)
                    # check non zero elements in the f and add to numberofallfiltered
                    numberofallfiltered += np.count_nonzero(f)
                import wandb
                wandb.log({'Z_NumberOfFineTunePoints': numberofallfiltered})
                # Now log the aggregated filter.
                import matplotlib.pyplot as plt
                import io
                from PIL import Image
                _obs = train_dataset.dataset['observations']  # assuming dataset is available here
                filtered_pbs = _obs[combined_filter.astype(bool)]
                buf = io.BytesIO()
                plt.scatter(_obs[:5000, 0], _obs[:5000, 1])
                plt.scatter(filtered_pbs[:, 0], filtered_pbs[:, 1], alpha=0.5)
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                img = Image.open(buf)
                img_array = np.array(img)
                wandb.log({'ZFilter': wandb.Image(img_array)})
                del img, img_array, buf
            del aggregated_filters
            import gc
            gc.collect()

            if i < num_eval_episodes:
                add_to(stats, flatten(info))
                trajs.append(traj)
            else:
                renders.append(np.array(render))
            make_plots('post')

        else:
            # Prepare fine-tuning
            old_config = agent.config
            new_config = agent.config.unfreeze()
            # Override training parameters
            if finetune_config.actor_loss is not None and not is_hiql:
                new_config['actor_loss'] = finetune_config.actor_loss
            if finetune_config.alpha is not None and not is_hiql:
                new_config['alpha'] = finetune_config.alpha
            new_config = old_config.__class__(new_config)
            # Copy parameters and state
            old_train_state = copy.deepcopy(agent.network)
            opt_state = agent.network.opt_state
            # Define new optimizer
            finetune_tx = optax.adam(learning_rate=finetune_config.lr)
            agent = agent.replace(network=agent.network.replace(tx=finetune_tx, opt_state=opt_state), config=new_config)

            if finetune_config.num_steps:
                # Default fine tuning aspect

                # _filter is a binary mask over the entire dataset
                if hasattr(finetune_config, 'unfreeze'):
                    current_finetune_config = finetune_config.unfreeze()
                else:
                    # If it's a standard dictionary, create a copy to modify
                    current_finetune_config = dict(finetune_config)

                cube_env = current_finetune_config.get('cube_env', False)
                if cube_env: # A way to detect CubeEnv, or use env.spec.id
                    # env._num_cubes should be available if 'env' is an instance of your CubeEnv
                    num_cubes = env._num_cubes if hasattr(env, '_num_cubes') else 1 # Default to 1 if not found, adjust as needed
                    current_finetune_config['num_cubes'] = num_cubes
                    # The 9 elements per cube state: 3 (pos) + 4 (quat) + 2 (sin/cos yaw)
                    current_finetune_config['proprio_dim'] = env.observation_space.shape[0] - num_cubes * 9
                    try:
                        current_finetune_config['goal_is_oracle_rep'] = env._use_oracle_rep
                    except Exception as e:
                        #print(f"Error accessing _use_oracle_rep: {e}")
                        current_finetune_config['goal_is_oracle_rep'] = False
                
                
                # If finetune_config was unfrozen, you might need to freeze it again or use the unfrozen version
                # For example, if the original was a frozen ConfigDict:
                # finetune_config_to_pass = config_dict.ConfigDict(current_finetune_config)
                # For simplicity, assuming current_finetune_config can be passed directly:
                finetune_config_to_pass = current_finetune_config
                _filter = train_dataset.prepare_active_sample(agent, observation, goal, finetune_config_to_pass, exp_name=exp_name, is_hiql=is_hiql)
                # Skip fine-tuning if the filter would select nothing
                num_steps = finetune_config.num_steps if _filter.sum() else 0
                for _ in range(num_steps):
                    batch = train_dataset.active_sample(finetune_config.batch_size, _filter, goal, finetune_config.ratio, finetune_config.fix_actor_goal, is_hiql=is_hiql)
                    agent, info = agent.update(batch, finetuning=True)
                    #print('finetune', info)
                    add_to(finetune_stats, flatten(info))

            #print('finetune_stats', finetune_stats)
            make_plots('post')
            actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

            done = False
            step = 0
            render = []
            while not done:
                action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
                action = np.array(action)
                if not config.get('discrete'):
                    if eval_gaussian is not None:
                        action = np.random.normal(action, eval_gaussian)
                    action = np.clip(action, -1, 1)

                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1
                done = done or step >= 3000

                if should_render and (step % video_frame_skip == 0 or done):
                    frame = env.render().copy()
                    if goal_frame is not None:
                        render.append(np.concatenate([goal_frame, frame], axis=0))
                    else:
                        render.append(frame)

                transition = dict(
                    observation=observation,
                    next_observation=next_observation,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                )
                add_to(traj, transition)
                observation = next_observation
            if i < num_eval_episodes:
                add_to(stats, flatten(info))
                trajs.append(traj)
            else:
                renders.append(np.array(render))

        # Reset agent parameters and state after each episode
        agent = agent.replace(network=old_train_state, config=old_config)

    stats.update({'finetune/' + k: v for k, v in finetune_stats.items()})
    print('stats', stats)
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
