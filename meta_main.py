import sys
import yaml
import os
import random
import time
import psutil
import gc
import argparse
from collections import defaultdict, OrderedDict


os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_async_all_gather=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO off, 2=WARNING off, 3=ERROR only
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# Enable JAX compilation logging
# os.environ["JAX_LOG_COMPILES"] = "1"

import gymnasium as gym
import wandb
import jax
from dataclasses import asdict
import numpy as np
import tqdm
from agents import agents
from ml_collections import FrozenConfigDict
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate, _cfg_get
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_wandb_video, setup_wandb
from utils.config import GCTTTConfig, load_config
from agents.gcagent import GCAgent, MetaGCAgent
import matplotlib.pyplot as plt
import io
from PIL import Image
import importlib

def _to_np(x):
    # Convert to numpy array for distance computation
    if isinstance(x, np.ndarray):
        return x
    elif hasattr(x, "cpu"):
        return np.asarray(x.cpu())
    else:
        return np.asarray(x)


def override_config_value(cfg: GCTTTConfig, key_path: str, value: str):
    """Override a config value using dot notation (e.g., 'finetune.num_steps').

    Args:
        cfg: The configuration object to modify
        key_path: Dot-separated path to the config value (e.g., 'finetune.num_steps')
        value: String value to set (will be converted to appropriate type)
    """
    keys = key_path.split('.')
    current = cfg

    # Navigate to the parent of the target attribute
    for key in keys[:-1]:
        if hasattr(current, key):
            current = getattr(current, key)
        else:
            raise ValueError(f"Config path '{key_path}' not found: '{key}' does not exist")

    # Get the final key and set the value
    final_key = keys[-1]
    if not hasattr(current, final_key):
        raise ValueError(f"Config path '{key_path}' not found: '{final_key}' does not exist")

    # Get the current value to determine the type for conversion
    current_value = getattr(current, final_key)

    # Convert the string value to the appropriate type
    if isinstance(current_value, bool):
        converted_value = value.lower() in ('true', '1', 'yes', 'on')
    elif isinstance(current_value, int):
        converted_value = int(value)
    elif isinstance(current_value, float):
        converted_value = float(value)
    elif current_value is None:
        # Try to infer type from the string value
        try:
            converted_value = int(value)
        except ValueError:
            try:
                converted_value = float(value)
            except ValueError:
                if value.lower() in ('true', 'false'):
                    converted_value = value.lower() == 'true'
                else:
                    converted_value = value
    else:
        converted_value = value

    setattr(current, final_key, converted_value)
    print(f"Override: {key_path} = {converted_value} (type: {type(converted_value).__name__})")


def parse_args():
    """Parse command line arguments for config overrides."""
    parser = argparse.ArgumentParser(
        description="Train meta agent with optional config overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python meta_main.py config.yaml --finetune.num_steps 100 --train_steps 50000
  python meta_main.py config.yaml --finetune.num_steps 200 --train_steps 100000 --meta_algorithm reptile
  python meta_main.py config.yaml --seed 42 --meta_algorithm maml
  python meta_main.py config.yaml --agent.inner_loop_steps 10 --meta_algorithm maml
  python meta_main.py config.yaml --agent.meta_batch_size 16 --agent.inner_loop_steps 5 --meta_algorithm maml
  python meta_main.py config.yaml --eval_interval 5000 --meta_algorithm maml
  python meta_main.py config.yaml --train_on_test_goal --use_random_batch
        """
    )

    parser.add_argument('config_file', help='Path to the YAML configuration file')
    parser.add_argument('--finetune.num_steps', type=str, dest='finetune_num_steps',
                        help='Override finetune.num_steps value')
    parser.add_argument('--train_steps', type=str, dest='train_steps',
                        help='Override train_steps value')
    parser.add_argument('--meta_algorithm', type=str, dest='meta_algorithm',
                        choices=['maml', 'fomaml', 'reptile'],
                        help='Override meta_algorithm value (choices: maml, fomaml, reptile)')
    parser.add_argument('--seed', type=int, dest='seed',
                        help='Override seed value')
    parser.add_argument('--agent.inner_loop_steps', type=int, dest='inner_loop_steps',
                        help='Override agent.inner_loop_steps value')
    parser.add_argument('--agent.meta_batch_size', type=int, dest='meta_batch_size',
                        help='Override agent.meta_batch_size value')
    parser.add_argument('--eval_interval', type=int, dest='eval_interval',
                        help='Override eval_interval value')
    parser.add_argument('--use_random_batch', action='store_true', dest='use_random_batch',
                        help='Use random batch sampling instead of goal-conditioned sampling')
    parser.add_argument('--train_on_test_goal', action='store_true', dest='train_on_test_goal',
                        help='Override train_on_test_goal to True (use test goal for training batch fetching)')
    parser.add_argument('--actor_uniform_sample', action='store_true', dest='actor_uniform_sample',
                        help='Set agent.actor_geom_sample to False (use uniform sampling instead of geometry sampling)')
    parser.add_argument('--finetune.mc_quantile', type=float, dest='finetune_mc_quantile',
                        help='Override finetune.mc_quantile value (float)')

    return parser.parse_args()


# META Learning methods to run: standard maml, FOMAML, Reptile

class DataSelectionCache:
    def __init__(self, cache_max_size=100):
        self.cache_max_size = cache_max_size
        self.starts = None
        self.goals = None
        self.caches = [None] * cache_max_size
        self.size = 0
        self.write_idx = 0
        self.start_dim = None
        self.goal_dim = None

    def maybe_reset(self, start_state_np, goal_np, cache_max_size):
        if self.cache_max_size != cache_max_size:
            self.__init__(cache_max_size)
        if self.starts is None or self.goals is None:
            self.start_dim = start_state_np.shape[0]
            self.goal_dim = goal_np.shape[0]
            self.starts = np.zeros((self.cache_max_size, self.start_dim), dtype=start_state_np.dtype)
            self.goals = np.zeros((self.cache_max_size, self.goal_dim), dtype=goal_np.dtype)
            self.caches = [None] * self.cache_max_size
            self.size = 0
            self.write_idx = 0

    def lookup(self, start_state_np, goal_np, threshold_dist):
        if self.size == 0:
            return None
        valid_range = slice(0, self.size)
        cached_starts = self.starts[valid_range]
        cached_goals = self.goals[valid_range]
        start_dists = np.linalg.norm(cached_starts - start_state_np, axis=1)
        goal_dists = np.linalg.norm(cached_goals - goal_np, axis=1)
        mask = (start_dists < threshold_dist) & (goal_dists < threshold_dist)
        if np.any(mask):
            valid_indices = np.where(mask)[0]
            total_dists = start_dists[valid_indices] + goal_dists[valid_indices]
            idx = valid_indices[np.argmin(total_dists)]
            return self.caches[idx]
        return None

    def insert(self, start_state_np, goal_np, _filter, max_len):
        idx = self.write_idx
        self.starts[idx] = np.copy(start_state_np)
        self.goals[idx] = np.copy(goal_np)
        self.caches[idx] = (np.copy(_filter), max_len)
        self.write_idx = (idx + 1) % self.cache_max_size
        if self.size < self.cache_max_size:
            self.size += 1




def fetch_goal_conditioned_batch(
    agent: GCAgent,
    train_dataset: GCDataset,
    start_states,
    goals,
    finetune_config,
    threshold_dist: float = 1.5,
    cache_max_size: int = 100
):
    """
    Parallel version: start_states and goals are batches (np.ndarray or list) of shape [B, ...].
    Returns lists of train_batch and test_batch, one per (start_state, goal) pair.
    This version is parallelized over the batch using numpy vectorization where possible (not multithreaded, no for over batch).
    """
    # Use a function attribute for the cache instead of a singleton/global
    if not hasattr(fetch_goal_conditioned_batch, "_cache"):
        fetch_goal_conditioned_batch._cache = DataSelectionCache(cache_max_size)
    cache = fetch_goal_conditioned_batch._cache

    # Convert to numpy arrays if not already
    start_states_np = np.asarray([_to_np(s).flatten() for s in start_states])
    goals_np = np.asarray([_to_np(g).flatten() for g in goals])

    batch_size = start_states_np.shape[0]
    # For cache, still use numpy arrays (cache is CPU)
    cache.maybe_reset(
        start_states_np[0],
        goals_np[0],
        cache_max_size
    )

    # Prepare arrays for cache lookup
    if cache.size > 0:
        valid_range = slice(0, cache.size)
        cached_starts = cache.starts[valid_range]  # [N, D]
        cached_goals = cache.goals[valid_range]    # [N, D]
    else:
        cached_starts = np.zeros((0, start_states_np.shape[1]), dtype=start_states_np.dtype)
        cached_goals = np.zeros((0, goals_np.shape[1]), dtype=goals_np.dtype)

    # Timer for getting the filter
    t_filter_start = time.time()
    # Vectorized cache lookup for all batch elements (on CPU)
    if cache.size > 0:
        # [B, N, D]
        start_diffs = start_states_np[:, None, :] - cached_starts[None, :, :]
        goal_diffs = goals_np[:, None, :] - cached_goals[None, :, :]
        # [B, N]
        start_dists = np.linalg.norm(start_diffs, axis=2)
        goal_dists = np.linalg.norm(goal_diffs, axis=2)
        # [B, N]
        mask = (start_dists < threshold_dist) & (goal_dists < threshold_dist)
        # For each batch element, find the best cache index (or -1 if none)
        any_mask = np.any(mask, axis=1)
        # [B] - index of best cache or -1
        best_indices = -np.ones(batch_size, dtype=int)
        for i in np.where(any_mask)[0]:
            valid_indices = np.where(mask[i])[0]
            if len(valid_indices) > 0:
                total_dists = start_dists[i, valid_indices] + goal_dists[i, valid_indices]
                idx = valid_indices[np.argmin(total_dists)]
                best_indices[i] = idx
        # Now, for each batch element, get cache hit or miss
        cache_hit_mask = best_indices != -1
        cache_miss_mask = ~cache_hit_mask
        cached_results = [
            cache.caches[int(idx)] if idx != -1 else None
            for idx in best_indices
        ]
    else:
        cache_hit_mask = np.zeros(batch_size, dtype=bool)
        cache_miss_mask = ~cache_hit_mask
        cached_results = [None] * batch_size

    # Prepare _filter and max_len for all batch elements
    filters = [None] * batch_size
    max_lens = [None] * batch_size

    # Fill in cache hits
    if np.any(cache_hit_mask):
        hit_indices = np.where(cache_hit_mask)[0]
        for i in hit_indices:
            _filter, max_len = cached_results[i]
            filters[i] = _filter
            max_lens[i] = max_len

    # For cache misses, call prepare_active_sample in a vectorized way (as much as possible)
    miss_indices = np.where(cache_miss_mask)[0]
    if len(miss_indices) > 0:
        # Prepare all args for misses
        miss_start_states = [start_states[i] for i in miss_indices]
        miss_goals = [goals[i] for i in miss_indices]
        # No for loop: use numpy vectorization is not possible for arbitrary python calls,
        # but we can use list comprehensions (which is as parallel as possible in python)
        results = [
            train_dataset.prepare_active_sample(agent, s, g, finetune_config, log_filter=False)
            for s, g in zip(miss_start_states, miss_goals)
        ]
        for idx, (filt, max_len) in zip(miss_indices, results):
            filters[idx] = filt
            max_lens[idx] = max_len
            if cache.size < cache.cache_max_size:
                # Insert using numpy arrays (cache is CPU)
                cache.insert(
                    start_states_np[idx],
                    goals_np[idx],
                    filt,
                    max_len
                )
    t_filter_end = time.time()
    # print(f"Time for getting the filter: {t_filter_end - t_filter_start:.4f} seconds")

    # Timer for fetching the batch
    t_batch_start = time.time()

    # Now, for each batch element, build train/test batches
    # Vectorized: build all batches at once if possible, else use list comprehension
    batch_size_val = _cfg_get(finetune_config, "batch_size")
    total_batch_size = 2 * batch_size_val

    # Prepare all batch args
    t_all_batches_start = time.time()
    all_batches = []
    for i in range(batch_size):
        _filter = filters[i]
        goal = goals[i]
        if _filter is None or _filter.sum() == 0:
            all_batches.append(None)
        else:
            batch = train_dataset.active_sample(
                total_batch_size,
                _filter,
                goal,
                _cfg_get(finetune_config, "ratio"),
                _cfg_get(finetune_config, "fix_actor_goal"),
                finetune_kwargs=finetune_config,
            )
            all_batches.append(batch)
    t_all_batches_end = time.time()
    # print(f"Time for getting all batches: {t_all_batches_end - t_all_batches_start:.4f} seconds")

    # Now split into train/test batches
    batches = []

    for batch in all_batches:
        if batch is not None:
            batches.append(
                (
                    {k: v[:batch_size_val] for k, v in batch.items()},
                    {k: v[batch_size_val:] for k, v in batch.items()}
                )
            )

    return batches


def fetch_random_batch(
    train_dataset: GCDataset,
    finetune_config,
    batch_size: int = None
):
    """
    Fetch a random batch from the dataset using dataset.sample function.

    Args:
        train_dataset: The training dataset
        finetune_config: Fine-tuning configuration
        batch_size: Batch size for sampling. If None, uses finetune_config.batch_size

    Returns:
        A tuple of (train_batch, test_batch) where both are random samples from the dataset
    """
    if batch_size is None:
        batch_size = _cfg_get(finetune_config, "batch_size")

    # Sample random batch from dataset
    random_batch = train_dataset.sample(batch_size)

    # Split into train and test batches (half each)
    train_size = batch_size // 2
    test_size = batch_size - train_size

    train_batch = {k: v[:train_size] for k, v in random_batch.items()}
    test_batch = {k: v[train_size:train_size + test_size] for k, v in random_batch.items()}

    return (train_batch, test_batch)


def sample_start_goal_pairs(
    train_dataset: GCDataset,
    env: gym.Env,
    meta_batch_size: int,
    train_on_test_goal: bool,
):
    """
    Sample start and goal pairs from the dataset.
    """
    # Sample start states and goals for each task
    start_states = []
    goals = []

    if train_on_test_goal:
        # 1. Sample task METABATCH_SIZE times
        task_ids = np.random.randint(1, 6, meta_batch_size)

        for task_id in task_ids:
            obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
            start_states.append(obs)
            goals.append(info.get("goal"))
    else:
        start_batch, goal_batch = fetch_random_batch(train_dataset, cfg.finetune, batch_size=2 * meta_batch_size)
        start_states.extend(start_batch['observations'])
        goals.extend(goal_batch['next_observations'])

    return start_states, goals


def evaluation_loop(
    agent: GCAgent,
    env: gym.Env,
    cfg: GCTTTConfig,
    train_dataset: GCDataset,
    eval_logger: CsvLogger,
    step: int,
    num_ttt_steps: int):
    print("Evaluating...")

    if cfg.eval_on_cpu:
        eval_agent = jax.device_put(agent, device=jax.devices("cpu")[0])
    else:
        eval_agent = agent
    renders = []
    eval_metrics = {}
    overall_metrics = defaultdict(list)
    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )

    num_tasks = (
        len(cfg.eval_tasks) if cfg.eval_tasks is not None else len(task_infos)
    )
    for task_id in tqdm.trange(1, num_tasks + 1):
        task_name = task_infos[task_id - 1]["task_name"]
        # Test-time fine-tuning happens in here
        # Test-time fine-tuning happens in here
        eval_start_time = time.time()
        eval_info, trajs, cur_renders = evaluate(
            agent=eval_agent,
            env=env,
            task_id=task_id,
            config=cfg,
            train_dataset=train_dataset,
            num_ttt_steps=num_ttt_steps,
        )
        eval_duration = time.time() - eval_start_time
        print(f"Evaluation for task {task_id} took {eval_duration:.2f} seconds")

        # Simple script to plot rollouts, assuming that the first 2 dimensions
        # of the data represent XY CoM coordinates.
        # TODO: remove
        plotit = True
        if plotit:
            buf = io.BytesIO()
            _obs = np.stack(trajs[0]["observation"])
            _background = train_dataset.sample(1000)["observations"]
            plt.scatter(_background[:, 0], _background[:, 1])
            plt.scatter(_obs[:, 0], _obs[:, 1])
            # plt.savefig(f'Zfig_{exp_name}.png', dpi=900)
            plt.savefig(buf, format="png", dpi=900)
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            wandb.log({"Zfig": wandb.Image(img_array)}, step=step)
            del img_array, img, buf

        # --- MINIMAL MODIFICATION START ---

        finetune_actor_loss_key = "finetune/actor/actor_loss"
        # Check for the specific key and add its list to eval_metrics
        if finetune_actor_loss_key in eval_info:
            loss_list_raw = eval_info[finetune_actor_loss_key]
            if isinstance(loss_list_raw, list):  # Make sure it's a list
                try:
                    # Convert JAX arrays/other numerics to standard Python floats
                    loss_values_float = [
                        (
                            float(val.item())
                            if hasattr(val, "item")
                            else float(val)
                        )
                        for val in loss_list_raw
                    ]

                    # Add the list directly to eval_metrics.
                    # Use a key that indicates it's the raw trend/list.
                    # Replace '/' in the metric name segment with '_' for cleaner W&B key
                    log_key_segment = finetune_actor_loss_key.replace("/", "_")
                    eval_metrics[
                        f"finetune/{task_name}_{log_key_segment}_trend"
                    ] = loss_values_float
                except Exception as e:
                    # Log a warning if conversion fails, but don't crash
                    print(
                        f"Warning: Could not process {finetune_actor_loss_key} list for task {task_name}: {e}"
                    )
        # --- MINIMAL MODIFICATION END ---

        renders.extend(cur_renders)
        metric_names = ["success"]
        eval_metrics.update(
            {
                f"evaluation/{task_name}_{k}/{num_ttt_steps}_TTT": v
                for k, v in eval_info.items()
                if k in metric_names
            }
        )
        # wandb.log({f'evaluation_logged/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names})
        for k, v in eval_info.items():
            if k in metric_names:
                overall_metrics[k].append(v)

    # TODO: check are we averaging over task?
    for k, v in overall_metrics.items():
        eval_metrics[f"evaluation/overall_{k}/{num_ttt_steps}_TTT"] = np.mean(v)

    if cfg.video_episodes > 0:
        video = get_wandb_video(renders=renders, n_cols=num_tasks)
        eval_metrics[f"video/{num_ttt_steps}_TTT"] = video

    try:
        # Use the same step as the training loop to maintain monotonic ordering
        wandb.log(eval_metrics, step=step)
    except Exception as e:
        print(f"Error during wandb.log: {e}")

    # Log to the separate eval_logger if it exists
    try:
        eval_logger.log(eval_metrics, step=step)
    except Exception as e:
        print(f"Error logging to eval_logger: {e}")

    # Clear memory after evaluation
    gc.collect()
    jax.clear_caches()
    print("[Memory] Cleared memory after evaluation")


def main(cfg: GCTTTConfig):

    if cfg.agent['agent_name'].startswith("meta_"):
        agent_file_name = cfg.agent['agent_name'][len("meta_"):]
    else:
        agent_file_name = cfg.agent['agent_name']
    agent_cfg = importlib.import_module(f"agents.{agent_file_name}").get_config()
    for k, v in agent_cfg.items():
        if k not in cfg.agent:
            cfg.agent[k] = v

    # Set up logger.
    # split env_name by '-'
    exp_name = get_exp_name(cfg)

    env_name_short = cfg.env_name.split("-")[0]

    # Create group name based on meta learning algorithm and parameters
    if cfg.train_steps == 0:
        group_name = f"{env_name_short}-RANDOM-INIT"
    elif cfg.meta_algorithm is None:
        group_name = f"{env_name_short}-PT"  # Pretraining
    else:
        group_name = f"{env_name_short}-{cfg.meta_algorithm.upper()}"

    # Add goal type prefix to group name (only ALL-GOALS case)
    if not cfg.train_on_test_goal:
        group_name = "ALL-GOALS-" + group_name

    # Add finetune steps to group name
    if cfg.finetune.num_steps > 0:
        group_name += f"-{cfg.finetune.num_steps}-TTT"
    else:
        group_name += "-0-TTT"

    # Add meta learning parameters if meta algorithm is present
    if cfg.meta_algorithm is not None:
        inner_steps = cfg.agent.get("inner_loop_steps", 1)
        meta_batch_size = cfg.agent.get("meta_batch_size", 32)
        group_name += f"-{inner_steps}inner-{meta_batch_size}meta"

    # Add batch sampling configuration to group name
    if cfg.use_random_batch:
        group_name += "-RANDOMBATCH"

    # Add mc_quantile to group name
    mc_quantile = cfg.finetune.get("mc_quantile")
    if mc_quantile is not None:
        group_name += f"-mc{mc_quantile}"

    # Build a serializable config for logging only
    wandb_config = {
        "run_group": cfg.run_group,
        "seed": cfg.seed,
        "env_name": cfg.env_name,
        "data_ratio": cfg.data_ratio,
        "working_dir": cfg.working_dir,
        "restore_path": cfg.restore_path,
        "restore_epoch": cfg.restore_epoch,
        "agent": cfg.agent,
        "finetune": asdict(cfg.finetune),
        "train_steps": cfg.train_steps,
        "log_interval": cfg.log_interval,
        "eval_interval": cfg.eval_interval,
        "save_interval": cfg.save_interval,
        "eval_start": cfg.eval_start,
        "eval_tasks": cfg.eval_tasks,
        "eval_episodes": cfg.eval_episodes,
        "eval_temperature": cfg.eval_temperature,
        "eval_gaussian": cfg.eval_gaussian,
        "video_episodes": cfg.video_episodes,
        "video_frame_skip": cfg.video_frame_skip,
        "eval_on_cpu": cfg.eval_on_cpu,
        "train_on_test_goal": cfg.train_on_test_goal,
        "use_random_batch": cfg.use_random_batch,
    }
    setup_wandb(
        project="TTT_AllFinalRuns", group=group_name, name=exp_name, config=wandb_config
    )

    # Save current expanded config in the experiment dir
    os.makedirs(cfg.working_dir, exist_ok=True)
    with open(os.path.join(cfg.working_dir, "config.yaml"), "w") as f:
        yaml.dump(wandb_config, f)

    # Set up environment and dataset.
    config_agent = cfg.agent
    env: gym.Env
    env, train_dataset, val_dataset = make_env_and_datasets(
        cfg.env_name, cfg.data_ratio, frame_stack=config_agent["frame_stack"]
    )
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    dataset_class = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
    }[config_agent["dataset_class"]]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config_agent)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config_agent)

    # Initialize agent.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    example_batch = train_dataset.sample(1)
    if config_agent["discrete"]:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch["actions"] = np.full_like(
            example_batch["actions"], env.action_space.n - 1
        )

    agent_class = agents[config_agent["agent_name"]]
    agent: MetaGCAgent = agent_class.create(
        cfg.seed,
        example_batch["observations"],
        example_batch["actions"],
        config_agent,
        cfg.train_steps,
    )

    # Restore agent.
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(cfg.working_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(cfg.working_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()

    # add warmup
    # GC-BC antmaze (0 meta learning step) no TTT
    # Without TTT we can inspect "how much" params are updated l2-distance param space, l2-distance output space,
    # TTT-with-critic: precomputation to be updated periodically

    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def check_compilation_cache():
        """Check JAX compilation cache status."""
        try:
            # Get compilation cache info
            cache_info = jax._src.api_util._compilation_cache_info()
            print(f"[JAX] Compilation cache entries: {len(cache_info) if cache_info else 0}")

            # Check if we can get more detailed cache info
            if hasattr(jax, 'get_compilation_cache'):
                cache = jax.get_compilation_cache()
                print(f"[JAX] Cache size: {len(cache) if cache else 0}")
        except Exception as e:
            print(f"[JAX] Could not get compilation cache info: {e}")

    def monitor_compilation():
        """Monitor JAX compilation events."""
        print(f"[JAX] Available devices: {jax.devices()}")
        print(f"[JAX] Default backend: {jax.default_backend()}")
        check_compilation_cache()

    # No / Full meta-learning here
    META_BATCH_SIZE = config_agent["meta_batch_size"]   # Jonas: much 10-100 or more?
    # Start by cheating case with Meta-learning
    # GC-BC-TTT-No critic VS GC-BC-TTT-No critic with meta-learning in this environment
    # MAML, FOMAML, Reptile
    META_LEARNING_ALGORITHM = cfg.meta_algorithm
    # META_LEARNING_START_STEP is 0 if meta algo is not none otherwise 1000000
    META_LEARNING_START_STEP = 0 if META_LEARNING_ALGORITHM is not None else 1000000


    # test-time gradient steps x-axis (during TTT)
    # X-axis: pre-training steps (or actual pre-training time) -> y-axis success rate after TTT
    # Use 3 seeds for intervals

    # two lines normal TTT
    # meta TTT

    if cfg.train_steps == 0:
        for num_ttt_steps in cfg.finetune.num_steps_list:
            evaluation_loop(
                agent=agent,
                env=env,
                cfg=cfg,
                train_dataset=train_dataset,
                eval_logger=eval_logger,
                step=0,
                num_ttt_steps=num_ttt_steps)
        # Evaluate only

    for i in tqdm.tqdm(
        range(1, cfg.train_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):

        if i < META_LEARNING_START_STEP:
            # 1. Sample 1 batch for one task

            fetched = False

            # Measure memory and time before sampling
            memory_before_fetch = get_memory_usage()
            t_fetch_start = time.time()

            while not fetched:
                task_id = np.random.randint(1, 6)
                if cfg.use_random_batch:
                    # Use normal dataset sampling
                    train_batch = train_dataset.sample(cfg.finetune.batch_size)
                    fetched = True
                else:
                    if cfg.train_on_test_goal:
                        # Sample start states and goals for each task
                        obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
                        start_states = [obs]
                        goals = [info.get("goal")]
                    else:
                        # Sample a random goal
                        start_batch, goal_batch = fetch_random_batch(train_dataset, cfg.finetune, batch_size=2)
                        start_states = start_batch['observations']
                        goals = goal_batch['next_observations']

                    # Get goal-conditioned batch
                    batches = fetch_goal_conditioned_batch(
                        agent, train_dataset, start_states, goals, cfg.finetune,
                        threshold_dist=1.5,
                        cache_max_size=5000
                    )

                    if len(batches) > 0:
                        fetched = True
                        train_batch = batches[0][0]  # batches = [(train_batch, test_batch)]

            t_fetch_end = time.time()
            memory_after_fetch = get_memory_usage()
            print(f"[Timer] Sampling 1 batch for 1 task took {t_fetch_end - t_fetch_start:.4f} seconds.")
            print(f"[Memory] Before fetch: {memory_before_fetch:.2f} MB, After fetch: {memory_after_fetch:.2f} MB (+{memory_after_fetch - memory_before_fetch:.2f} MB)")

            agent, update_info = agent.update(train_batch)
        else:
            # Measure memory and time before sampling
            memory_before_fetch = get_memory_usage()
            t_fetch_start = time.time()

            # Sample start states and goals for each task
            start_states, goals = sample_start_goal_pairs(train_dataset, env, META_BATCH_SIZE, cfg.train_on_test_goal)

            t_fetch_end = time.time()
            memory_after_fetch = get_memory_usage()

            print(f"[Timer] Sampling batches took {t_fetch_end - t_fetch_start:.4f} seconds.")
            print(f"[Memory] Before fetch: {memory_before_fetch:.2f} MB, After fetch: {memory_after_fetch:.2f} MB (+{memory_after_fetch - memory_before_fetch:.2f} MB)")

            # --- Inner Update (Meta-Learning) ---
            t_inner_start = time.time()
            memory_before_inner = get_memory_usage()


            update_info = []
            is_fomaml = META_LEARNING_ALGORITHM == "fomaml"

            for inner_step in range(config_agent.get("inner_loop_steps", 1)):

                # Fetch task batches
                task_batches = fetch_goal_conditioned_batch(
                    agent, train_dataset, start_states, goals, cfg.finetune,
                    threshold_dist=1.5,
                    cache_max_size=5000
                )

                for num_task, (train_batch, test_batch) in enumerate(task_batches):

                    test_batch_arg = None if META_LEARNING_ALGORITHM == "reptile" else test_batch

                    agent, batch_info = agent.meta_inner_update(
                        num_task, train_batch, test_batch_arg, is_fomaml=is_fomaml, num_steps=1
                    )

                    if inner_step == config_agent.get("inner_loop_steps", 1) - 1:
                        update_info.append(batch_info)

            t_inner_end = time.time()
            memory_after_inner = get_memory_usage()
            inner_update_time = t_inner_end - t_inner_start

            print(f"[Timer] Inner update took {inner_update_time:.4f} seconds.")
            print(
                f"[Memory] Before inner: {memory_before_inner:.2f} MB, "
                f"After inner: {memory_after_inner:.2f} MB "
                f"(+{memory_after_inner - memory_before_inner:.2f} MB)"
            )
            if len(update_info) == 0:
                continue

            # Average batch info
            update_info = {k: np.mean([info[k] for info in update_info]) for k in update_info[0].keys()}

            # --- Meta Update (Meta-Learning) ---
            t_meta_start = time.time()
            memory_before_meta = get_memory_usage()
            agent = agent.meta_update(use_model_merging=META_LEARNING_ALGORITHM == "reptile")
            memory_after_meta = get_memory_usage()
            t_meta_end = time.time()
            meta_update_time = t_meta_end - t_meta_start
            print(f"[Timer] Meta update took {meta_update_time:.4f} seconds.")
            print(f"[Memory] Before meta: {memory_before_meta:.2f} MB, After meta: {memory_after_meta:.2f} MB (+{memory_after_meta - memory_before_meta:.2f} MB)\n")

            # Clear JAX caches periodically to prevent memory growth
            if i % 100_000 == 0:  # Every 100k iterations
                print(f"[JAX] Before clearing caches at iteration {i}")
                monitor_compilation()
                jax.clear_caches()
                gc.collect()
                memory_mb = get_memory_usage()
                print(f"[Memory] Cleared JAX caches. Current memory: {memory_mb:.2f} MB")
                print(f"[JAX] After clearing caches:")
                monitor_compilation()


        # Log metrics.
        if i % cfg.log_interval == 0:

            print(update_info)
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                # So... we sample only 1 batch for validation
                val_batch = val_dataset.sample(config_agent["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update(
                    {f"validation/{k}": v for k, v in val_info.items()}
                )
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / cfg.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i % cfg.eval_interval == 0 and i >= cfg.eval_start:

            for num_ttt_steps in cfg.finetune.num_steps_list:
                evaluation_loop(
                    agent=agent,
                    env=env,
                    cfg=cfg,
                    train_dataset=train_dataset,
                    eval_logger=eval_logger,
                    step=i,
                    num_ttt_steps=num_ttt_steps)

        # Save agent.
        if i % cfg.save_interval == 0:
            save_agent(agent, cfg.working_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load base configuration
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file '{args.config_file}' not found.")
    cfg = load_config(args.config_file)

    # Apply command line overrides
    if args.finetune_num_steps is not None:
        override_config_value(cfg, 'finetune.num_steps', args.finetune_num_steps)

    if args.train_steps is not None:
        override_config_value(cfg, 'train_steps', args.train_steps)

    if args.meta_algorithm is not None:
        override_config_value(cfg, 'meta_algorithm', args.meta_algorithm)

    if args.seed is not None:
        override_config_value(cfg, 'seed', args.seed)

    if args.inner_loop_steps is not None:
        cfg.agent['inner_loop_steps'] = args.inner_loop_steps

    if args.meta_batch_size is not None:
        cfg.agent['meta_batch_size'] = args.meta_batch_size

    if args.eval_interval is not None:
        override_config_value(cfg, 'eval_interval', args.eval_interval)

    if args.use_random_batch:
        cfg.use_random_batch = True

    if args.train_on_test_goal:
        cfg.train_on_test_goal = True

    if args.actor_uniform_sample:
        cfg.agent['actor_geom_sample'] = False

    if args.finetune_mc_quantile is not None:
        override_config_value(cfg, 'finetune.mc_quantile', str(args.finetune_mc_quantile))

    print(f"Number of steps list: {cfg.finetune.num_steps_list}")

    main(cfg)
