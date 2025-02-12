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

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')

        old_config = agent.config
        new_config = agent.config.unfreeze()
        if finetune_config.actor_loss is not None:
            new_config.actor_loss = finetune_config.actor_loss
        new_config = old_config.__class__(new_config)
        old_train_state = copy.deepcopy(agent.network)
        opt_state = agent.network.opt_state
        finetune_tx = optax.adam(learning_rate=finetune_config.lr)
        agent = agent.replace(network=agent.network.replace(tx=finetune_tx, opt_state=opt_state), config=new_config)

        finetune_stats = defaultdict(list)
        if finetune_config.num_steps:
            _filter = train_dataset.prepare_active_sample(agent, observation, goal, finetune_config)
            for _ in range(finetune_config.num_steps):
                batch = train_dataset.active_sample(finetune_config.batch_size, _filter, goal, finetune_config.ratio, finetune_config.fix_actor_goal)
                agent, info = agent.update(batch)
                add_to(finetune_stats, flatten(info))
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

        agent = agent.replace(network=old_train_state, config=old_config)

    stats.update({'finetune/' + k: v for k, v in finetune_stats.items()})
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
