import gymnasium as gym
import numpy as np
from utils.datasets import GCDataset
from utils.evaluation import _cfg_get
from typing import List, Tuple, Optional, Any, TypedDict
from agents.gcagent import GCAgent
from utils.config import FinetuneConfig

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


def get_batch_filters(train_dataset: GCDataset, agent: GCAgent, start_states, goals, config: GCTTTConfig) -> List[Tuple[np.ndarray, int]]:
    """
    Get batch filters for each task.
    Returns a list of (filter, max_len) tuples for each (start_state, goal) pair.
    """
    # [(filt, max_len), ...]

    assert len(goals) == 1, "Only one goal is supported for now"
    state_to_goal_dists = train_dataset.prepare_values(agent, goals[0], finetune_config)

    return [
        train_dataset.prepare_active_sample(agent, s, g, env_name=config.env_name, finetune_kwargs=config.finetune, state_to_goal_dist=state_to_goal_dists)
        for s, g in zip(start_states, goals)
    ]


def fetch_meta_batch(
    train_dataset: 'GCDataset',
    goal: Any,
    finetune_config: dict,
    filter_and_max_len: tuple,
    meta_batch_size: Optional[int] = None,
    exclude: Optional[List[int]] = None,
    fix_actor_goal: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[dict, np.ndarray]:
    """
    Build a meta batch from a dataset for meta-learning fine-tuning,
    for a single (goal, filter_and_max_len) pair.

    Optionally, you can provide an 'exclude' argument to mask out indices from the filter.
    'exclude' should be an iterable of indices to exclude from the batch.

    Returns a list of MetaBatchDict(s), each containing:
      - 'batch': Dict of arrays for the active sample.
      - 'batch_idx': Indices (within original dataset) of selected samples.

    Args:
        train_dataset: The dataset from which to sample.
        goal: Goal state for the batch/task.
        finetune_config: Dictionary/config holding fine-tuning hyperparameters ("batch_size", "ratio", etc).
        filter_and_max_len: Tuple of (filter, max_len).
        meta_batch_size: Optional override for batch size.
        exclude: Optional iterable of indices to mask out (set False in filter).

    Returns:
        all_batches: List of MetaBatchDict(s), for the given input (empty list if filter is invalid or empty).
    """

    if meta_batch_size is not None:
        batch_size = meta_batch_size
    else:
        batch_size = int(finetune_config.get("batch_size"))

    filt, _ = filter_and_max_len

    if filt is not None and filt.sum() > 0:
        # Exclude indices if provided
        if exclude is not None:
            filt_used = np.copy(filt)
            filt_used[exclude] = 0
        else:
            filt_used = filt

        if verbose:
            print(f"Fetching meta batch: filter has {filt_used.sum()} samples")

        # Only add batch if there are remaining samples after exclusion
        if filt_used is not None and filt_used.sum() > 0:
            batch, idxs = train_dataset.active_sample(
                batch_size,
                filt_used,
                goal,
                _cfg_get(finetune_config, "ratio"),
                fix_actor_goal if fix_actor_goal is not None else _cfg_get(finetune_config, "fix_actor_goal"),
                hierarchical=(agent.config['agent_name'] == 'saw'),
                return_indices=True,
            )

            return batch, idxs

    return (None, None)


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
    is_stitch_dataset: bool = False,
    finetune_config: FinetuneConfig = None,
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
            goals.append(info.get("goal"))
            if not is_stitch_dataset:
                start_states.append(obs)

        if is_stitch_dataset:
            start_batch, _ = fetch_random_batch(train_dataset, finetune_config, batch_size=2*meta_batch_size)
            start_states.extend(start_batch['observations'])

    else:
        start_batch, goal_batch = fetch_random_batch(train_dataset, finetune_config, batch_size=2 * meta_batch_size)
        start_states.extend(start_batch['observations'])
        goals.extend(goal_batch['next_observations'])

    return start_states, goals
