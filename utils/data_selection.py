import sys
import yaml
import os
import random
import time
import psutil
import gc
import argparse
from collections import defaultdict, OrderedDict
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
from typing import List, Tuple, Optional
from agents.gcagent import GCAgent, MetaGCAgent
import matplotlib.pyplot as plt
import io


def _to_np(x):
    # Convert to numpy array for distance computation
    if isinstance(x, np.ndarray):
        return x
    elif hasattr(x, "cpu"):
        return np.asarray(x.cpu())
    else:
        return np.asarray(x)

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


def get_batch_filters(train_dataset: GCDataset, agent: GCAgent, start_states, goals, finetune_config, mc_quantile: Optional[float] = None) -> List[Tuple[np.ndarray, int]]:
    """
    Get batch filters for each task.
    Returns a list of (filter, max_len) tuples for each (start_state, goal) pair.
    """
    # [(filt, max_len), ...]
    return [
        train_dataset.prepare_active_sample(agent, s, g, finetune_config, log_filter=False, mc_quantile=mc_quantile)
        for s, g in zip(start_states, goals)
    ]


def fetch_meta_batches(
    train_dataset: GCDataset,
    goals,
    finetune_config,
    filters_and_max_lens,
    meta_batch_size: int = None,
):
    """
    Build meta batches from a dataset for meta-learning fine-tuning.

    For each pair of (filter, goal) in filters_and_max_lens and goals, construct a batch of data by selecting samples
    that satisfy the filter and are conditioned on the given goal. Only includes batches if the filter is not None
    and selects at least one sample.

    Args:
        train_dataset: The dataset from which to sample.
        goals: A list or array of goal states, each corresponding to a meta-batch task.
        finetune_config: Dictionary/config object holding fine-tuning hyperparameters (must have "batch_size", "ratio", etc).
        filters_and_max_lens: List of (filter, max_len) tuples, strongly aligned with goals.
        meta_batch_size: Optional override for batch size per meta task.

    Returns:
        all_batches: List of batches (dicts of arrays), one for each task/filter with non-empty samples.
    """

    # Now, for each batch element, build train/test batches
    # Vectorized: build all batches at once if possible, else use list comprehension
    if meta_batch_size is not None:
        batch_size = meta_batch_size
    else:
        batch_size = int(finetune_config.get("batch_size"))

    # Prepare all batch args
    all_batches = []
    for i in range(len(filters_and_max_lens)):
        filter, _ = filters_and_max_lens[i]
        goal = goals[i]

        # Add batch if filter contains samples
        if filter is not None and filter.sum() > 0:
            batch = train_dataset.active_sample(
                batch_size,
                filter,
                goal,
                _cfg_get(finetune_config, "ratio"),
                _cfg_get(finetune_config, "fix_actor_goal"),
                finetune_kwargs=finetune_config
            )
            all_batches.append(batch)

    return all_batches


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
