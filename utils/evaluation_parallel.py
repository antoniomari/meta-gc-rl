import math
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import struct  # optional, for typed states
from utils.evaluation import _cfg_get, supply_rng, add_to, make_plots, flatten
import time
from collections import defaultdict
import copy
import jax
import jax.numpy as jnp
from tqdm import trange
from utils.config import FinetuneConfig
from typing import Optional, Tuple, Any
import optax
import numpy as np
import matplotlib.pyplot as plt
import io
import wandb
from PIL import Image
from utils.datasets import GCDataset, HGCDataset, Dataset
from agents.gcagent import GCAgent
from typing import Dict
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from utils.env_utils import make_env_and_datasets
from utils.config import GCTTTConfig
import importlib
import multiprocessing

# ---- your imports (env creation, datasets, agent utilities) ---- #
# from your_project import make_env_and_datasets, GCDataset, gc_ttt_critic_free, copy_current_agent

######### ------ Helpers to partition and pad ------ #########
import math
import numpy as np
import jax
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 1. Functional version of your agent.update
# ============================================================

def update_fn(agent, batch):
    """Functional update: one gradient step for one agent on one batch."""
    new_rng, rng = jax.random.split(agent.rng)

    def loss_fn(grad_params):
        return agent.total_loss(batch, grad_params, rng=rng)

    new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
    return agent.replace(network=new_network, rng=new_rng), info

# vmap over multiple models on one device
vmapped_update = jax.vmap(update_fn, in_axes=(0, 0))

# pmap over devices
pmapped_update = jax.pmap(vmapped_update, in_axes=(0, 0))


# ============================================================
# 2. Helpers for padding & partitioning
# ============================================================

def partition_into_devices(M, n_devices):
    models_per_device = math.ceil(M / n_devices)
    total_slots = models_per_device * n_devices
    return models_per_device, total_slots

def pad_list(items, total_slots, pad_item):
    return items + [pad_item] * (total_slots - len(items))

def to_sharded_array(pytrees, n_devices, models_per_device):
    """Convert list length total_slots into [n_devices, models_per_device] pytree and put on devices."""
    # Group into [n_devices, models_per_device]
    grouped = [
        pytrees[i*models_per_device:(i+1)*models_per_device]
        for i in range(n_devices)
    ]
    # Stack each group into a pytree with leading axis models_per_device
    per_device = [jax.tree_map(lambda *xs: jnp.stack(xs), *grp) for grp in grouped]
    # Shard across devices
    return jax.device_put_sharded(per_device, jax.local_devices())


# ============================================================
# 3. Parallel training driver
# ============================================================

def parallel_train(agents, batches, rng_seed=0):
    """
    Args:
      agents: list of M GCAgent objects (Flax pytrees).
      batches: list of M batches (dicts with same keys, arrays).
    Returns:
      new_agents: list of M updated agents
      infos: list of M metrics dicts
    """
    n_devices = jax.local_device_count()
    M = len(agents)
    models_per_device, total_slots = partition_into_devices(M, n_devices)

    # Pad agents and batches
    pad_agent = agents[-1]
    pad_batch = batches[-1]
    agents_padded = pad_list(agents, total_slots, pad_agent)
    batches_padded = pad_list(batches, total_slots, pad_batch)

    # Shard to devices
    agents_sharded = to_sharded_array(agents_padded, n_devices, models_per_device)
    batches_sharded = to_sharded_array(batches_padded, n_devices, models_per_device)

    # Run parallel update
    new_agents_sharded, infos_sharded = pmapped_update(agents_sharded, batches_sharded)

    # Bring back to host numpy
    new_agents_host = jax.tree_map(lambda x: np.array(x), new_agents_sharded)
    infos_host = jax.tree_map(lambda x: np.array(x), infos_sharded)

    # Unstack into list of length total_slots
    new_agents_list = [
        jax.tree_map(lambda arr: arr[d, m], new_agents_host)
        for d in range(n_devices)
        for m in range(models_per_device)
    ]
    infos_list = [
        jax.tree_map(lambda arr: arr[d, m], infos_host)
        for d in range(n_devices)
        for m in range(models_per_device)
    ]

    # Trim padding back to M
    return new_agents_list[:M], infos_list[:M]


# ============================================================
# 4. Example usage inside your evaluate
# ============================================================
def gc_ttt_critic_free_parallel(
    agents,                # list of M agents
    train_datasets,        # list of M GCDataset/HGCDataset objects
    envs,                  # list of M environments
    observations,          # list of M initial observations
    config: GCTTTConfig,
    goals,                 # list of M goals
    goal_frames,           # list of M goal frames
    should_render: bool,
):
    """
    Parallel critic-free fine-tuning for M agents across N GPUs.
    """

    trajs = [defaultdict(list) for _ in agents]
    finetune_stats = [defaultdict(list) for _ in agents]
    infos = [None for _ in agents]
    renders = [[] for _ in agents]

    num_steps = int(_cfg_get(finetune_config, "num_steps", 0))

    if num_steps > 0:
        # === Collect one batch per agent ===
        batches = []

        # NOTE: preparing active sample is not parallel

        print('Preparing samples', end="")
        t0 = time.time()
        for agent, dataset, goal in zip(agents, train_datasets, goals):
            _filter, _ = dataset.prepare_active_sample(agent, observations[0], goal, finetune_config)
            batch = dataset.active_sample(
                _cfg_get(finetune_config, "batch_size"),
                _filter,
                goal,
                _cfg_get(finetune_config, "ratio"),
                _cfg_get(finetune_config, "fix_actor_goal"),
                finetune_kwargs=finetune_config,
            )
            batches.append(batch)
        t1 = time.time()
        print(f"[{t1 - t0:.2f}] seconds")

        # === Fine-tune all agents in parallel ===
        print('Parallel training', end="")
        t0 = time.time()
        new_agents, infos_list = parallel_train(agents, batches)
        t1 = time.time()
        print(f"[{t1 - t0:.2f}] seconds")

        # Collect stats
        for stats_dict, info in zip(finetune_stats, infos_list):
            add_to(stats_dict, flatten(info))

        agents = new_agents  # replace with updated agents

    # === Rollout episodes independently on CPU threads ===
    def rollout_one(idx):
        agent = agents[idx]
        env = envs[idx]
        observation = observations[idx]
        goal = goals[idx]
        goal_frame = goal_frames[idx]

        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
        traj = defaultdict(list)
        render = []
        done, step = False, 0
        while not done:
            next_obs, action, reward, terminated, truncated, info = actor_step(
                actor_fn, observation, env, config, goal, eval_gaussian, eval_temperature
            )
            step += 1
            done = terminated or truncated or step >= 3000

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_obs

        return traj, info, render

    # Run rollouts in parallel threads (CPU only)
    print('Rollout after finetuning', end="")
    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        results = list(ex.map(rollout_one, range(len(agents))))

    for i, (traj, info, render) in enumerate(results):
        trajs[i] = traj
        infos[i] = info
        renders[i] = render

    return agents, trajs, infos, finetune_stats, renders



def eval_episodes(ep_idx: List[int]):
    is_video = list(map(lambda i: i >= num_eval_episodes))

    envs = []
    datasets = []
    agents = []
    observations = []
    goals = []
    goal_frames = []

    for i in ep_idx:
        # Create fresh environment for this thread
        ep_env, ep_train_dataset, _ = make_env_and_datasets(
            config.env_name, config.data_ratio,
            frame_stack=config.agent["frame_stack"]
        )
        ep_env.reset(seed=config.seed + ep_idx)
        observation, info = ep_env.reset(options=dict(task_id=task_id, render_goal=is_video))
        ep_env.action_space.seed(config.seed + ep_idx)

        # Create dataset for this thread
        dataset_class = {
            "GCDataset": GCDataset,
            "HGCDataset": HGCDataset,
        }[config.agent["dataset_class"]]
        ep_train_dataset = dataset_class(Dataset.create(**ep_train_dataset), config.agent)

        envs.append(ep_env)
        datasets.append(ep_train_dataset)
        agents.append(copy_current_agent(agent, config.finetune))
        observations.append(observation)
        goals.append(info.get("goal"))
        goal_frames.append(info.get("goal_rendered"))



    # Run evaluation
    recursive_mdp = bool(_cfg_get(config.finetune, "filter_by_recursive_mdp", False))

        if recursive_mdp:
            raise NotImplementedError
            # Default GC-TTT (with critic)
            gc_ttt_critic(
                ep_train_dataset, agent_cpu, ep_env, config.agent, config.finetune,
                goal, config.eval_gaussian, config.eval_temperature
            )
            return {"ep": ep_idx, "info": {}, "traj": None, "render": None, "stats": {}}
        else:
            # Critic-free branch
            agent_copy, old_train_state, old_config = copy_current_agent(agent_cpu, config.finetune)

            traj, info_out, finetune_stats, render = gc_ttt_critic_free_parallel(
                agents=agents,
                train_datasets=datasets,
                envs=envs,
                observations=observations,
                config=config,
                goals=goals,
                goal_frames=goal_frames,
                should_render=False,
            )

            print(f"Episode completed: {ep_idx}")

            return {
                "ep": ep_idx,
                "info": info_out,
                "traj": traj,
                "render": render,
                "stats": finetune_stats
            }

def evaluate(
    agent: GCAgent,
    env,
    task_id: int,
    config: GCTTTConfig,
    train_dataset,
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
    num_eval_episodes = config.eval_episodes # Default: 50?
    num_video_episodes = config.video_episodes # Default: 0?

    trajs = []
    stats = defaultdict(list)
    renders = []


    # Run episodes in parallel using threads, in batches of size N
    N = 5
    episode_indices = list(range(num_eval_episodes + num_video_episodes))
    results = []
    for i in range(0, num_eval_episodes + num_video_episodes, N):
        steps = list(range(i, min(i+N, num_eval_episodes + num_video_episodes)))
        # I want to perform N episodes at a time
        results.append(eval_episodes())


        results = list(executor.map(eval_episode_threaded, episode_indices))

    # Process results
    for result in results:
        ep_idx = result["ep"]
        if ep_idx < num_eval_episodes:
            add_to(stats, flatten(result["info"]))
            trajs.append(result["traj"])
            stats.update({f"finetune/{k}": v for k, v in result["stats"].items()})
        else:
            renders.append(np.array(result["render"]))
    # print('stats', stats)
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
