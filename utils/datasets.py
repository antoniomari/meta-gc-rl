import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
import rustworkx as rx
from heapq import heappop, heappush


def maximum_edge_length(G, source, target):
    """
    Finds the highest threshold such that, if all edges longer
    than the threshold were removed, source and target would
    be disconnected.
    """
    dist = {}  # maximum distances
    seen = {source: 0}
    fringe = [(0, source)]
    while fringe:
        (d, v) = heappop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        for u, cost in G.adj(v).items():
            vu_dist = max(dist[v], cost)
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError("Contradictory paths found:", "negative weights?")
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                heappush(fringe, (vu_dist, u))
    return dist[target]


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)

import numpy as np

def extract_cube_positions_from_full_obs(observation_array, proprio_dim, num_cubes, state_dim_per_cube=9, pos_dim_per_cube=3):
    """
    Extracts concatenated cube positions from a full observation array.
    Handles both single (1D) and batched (nD) observation arrays.
    The positions are the first `pos_dim_per_cube` elements of each cube's state block.
    """
    positions_list = []
    for i in range(num_cubes):
        start_idx = proprio_dim + i * state_dim_per_cube
        
        if observation_array.ndim == 1: # Single observation (e.g., current `obs`)
            positions_list.append(observation_array[start_idx : start_idx + pos_dim_per_cube])
        else: # Batched observations (e.g., dataset `_obs`)
            positions_list.append(observation_array[..., start_idx : start_idx + pos_dim_per_cube])

    # Concatenate along the last dimension for batched, or axis 0 for single
    if observation_array.ndim == 1:
        return np.concatenate(positions_list, axis=0)
    else:
        return np.concatenate(positions_list, axis=-1)


def filter_by_recursive_mdp(dataset, agent, obs, goal, finetune_kwargs, state_to_goal_dist=None, start_to_state_dist=None,
                            _start_values=None, _values=None):
    """
    GC-TTT with critic: find "optimal trajectories" and stitch them together
    - trajectories with high Monte-Carlo returns
    - trajectory passes close to current state (in terms of reward)
    - trajectory passes close to current goal (in terms of reward)
    Args:
        dataset: Dataset to filter.
        obs: Current observation (1D array).
        goal: Goal observation (1D array).
        finetune_kwargs: Additional fine-tuning parameters, e.g., proprio_dim, num_cubes, etc.
        state_to_goal_dist: Precomputed distance from current state to all states in the dataset.
        start_to_state_dist: Precomputed distance from current state to all states in the dataset.
    """

    _obs = dataset['observations']
    ep_id = dataset['terminals'].cumsum() // 2
    ep_id[1:] = ep_id[:-1]
    ep_lens = np.unique(ep_id, return_counts= True)[1]
    assert len(set(ep_lens)) == 1, "All episodes need to have the same length."
    ep_len = ep_lens[0].item()

    subtraj_min_steps = finetune_kwargs.get('min_steps', 10) # subgoals that are at least this many steps away from the start
    start_threshold   = finetune_kwargs.get('mc_similarity_threshold', 1.0) # distance threshold for start matches
    num_selected_points  = finetune_kwargs.get('recursive_selected_num_points', 10) # number of subgoals to select
    non_optimality = finetune_kwargs.get('no_optimality', False) # if true we only sample transitions close to the state regardless of whether they are any good
    non_relevance = finetune_kwargs.get('no_relevance', False) # if true we sample transitions from the buffer that are good under the optimality criterion but may be from anywhere over the state space (not necessarily close to our agents state)
    cube_env = finetune_kwargs.get('cube_env', False) 
    visual_env = finetune_kwargs.get('visual_env', False) # if true we use visual env logic for selecting subgoals

    # --- Start NaN Handling ---
    if state_to_goal_dist is not None:
        state_to_goal_dist_np = np.array(state_to_goal_dist) # Convert to NumPy for handling
        nan_mask = np.isnan(state_to_goal_dist_np)
        num_nans = nan_mask.sum()
        if num_nans > 0:
            # Choose a large value (effectively infinity for practical purposes)
            large_value = np.finfo(state_to_goal_dist_np.dtype).max / 2
            state_to_goal_dist_np[nan_mask] = large_value
            state_to_goal_dist = jnp.array(state_to_goal_dist_np) # Convert back to JAX array
    # --- End NaN Handling ---
    
    mask = np.zeros_like(ep_id)
    _obs = _obs.reshape(-1, ep_len, _obs.shape[-1])

    # Select subtrajectories that start close to the current state
    # Relevance criterion
    if cube_env:
        proprio_dim = finetune_kwargs['proprio_dim']
        num_cubes = finetune_kwargs['num_cubes']
        
        # Extract cube positions from current observation (obs)
        # obs is 1D array (current environment observation)
        current_obs_cube_positions = extract_cube_positions_from_full_obs(obs, proprio_dim, num_cubes)
        
        # Extract cube positions from dataset observations (_obs_reshaped)
        # _obs_reshaped has shape (num_episodes, ep_len, feature_dim)
        dataset_cube_positions = extract_cube_positions_from_full_obs(_obs, proprio_dim, num_cubes)
        
        # Calculate distance for start_matches
        # Resulting shape for start_matches_dist: (num_episodes, ep_len)
        start_matches_dist = jnp.sqrt(jnp.sum((dataset_cube_positions - current_obs_cube_positions)**2, axis=-1))
        start_matches = start_matches_dist < start_threshold

        if non_relevance:
            # If non-relevance is enabled, we sample transitions from the buffer that are good under the optimality criterion but may be from anywhere over the state space (not necessarily close to our agent's state)
            start_matches = jnp.sqrt(jnp.sum((_obs[..., :2] - obs[:2])**2, axis=-1)) < 10000.0

    if visual_env:
        goal_value_threshold = np.quantile(_values, 0.9) # e.g., top 10% of values
        start_value_threshold = np.quantile(_start_values, 0.9) # e.g., top 10% of values
        # start_matches are the ones that are close to the current state and have high value
        start_matches = (_start_values >= start_value_threshold)

    else: 
        # Original maze logic
        # obs is 1D, _obs_reshaped is (num_episodes, ep_len, feature_dim)
        start_matches = jnp.sqrt(jnp.sum((_obs[..., :2] - obs[:2])**2, axis=-1)) < start_threshold
        if finetune_kwargs['relevance_by_value']:
            assert start_to_state_dist is not None, 'Distance from current obs to all states is needed.'
            start_matches = (start_to_state_dist < start_threshold).reshape(start_matches.shape)
        if non_relevance:
            # If non-relevance is enabled, we sample transitions from the buffer that are good under the optimality criterion but may be from anywhere over the state space (not necessarily close to our agent's state)
            start_matches = jnp.sqrt(jnp.sum((_obs[..., :2] - obs[:2])**2, axis=-1)) < 10000.0

    # Now based on the return estimates, we select the subtrajectories that are optimal
    # Optimality criterion
    if state_to_goal_dist is not None:
        # Shift start_matches to align with the subtrajectory minimum steps
        shift_start_matches = np.zeros_like(start_matches)
        shift_start_matches[:, subtraj_min_steps:] = start_matches[:, :-subtraj_min_steps]
        
        scores = ((shift_start_matches.cumsum(-1) > 0) * state_to_goal_dist.reshape(start_matches.shape))
        scores = np.where(scores==0, scores.max(), scores)
        ep_idxs = np.argsort(scores.min(-1))[:num_selected_points]

        if non_optimality:
            # randomly select from  np.argsort(scores.min(-1)) instead of taking the best ones
            ep_idxs = np.random.choice(np.arange(len(scores)), num_selected_points, replace=False)

        mask = mask.reshape(-1, ep_len)
        mask[ep_idxs] = 1.
        mask *= (start_matches.cumsum(-1) > 0)  # only keep from matches
        col_indices = np.arange(ep_len)
        mask *= col_indices[None] < scores.argmin(-1)[..., None]  # discard after best point
        #return mask.flatten()
        max_len = mask.sum(-1).max()
        return mask.flatten(), max_len


def filter_from_state_goal(dataset, obs, goal, quantile, slack, sim_threshold, finetune_kwargs=None):
    """
    GC-TTT without critic: only find "optimal trajectories", no stitching
    - trajectories with high Monte-Carlo returns
    - trajectory passes close to current state (in terms of reward)
    Args:
        dataset: Dataset to filter.
        obs: Current observation (1D array).
        goal: Goal observation (1D array).
        quantile: Quantile for filtering: 0.5 means that we keep the best half of the trajectories.
        slack: Slack for filtering: how many steps we allow to deviate from the goal.
        sim_threshold: Similarity threshold for filtering: how close the trajectory should be to the goal.
        finetune_kwargs: Additional fine-tuning parameters, e.g., proprio_dim, num_cubes, etc.
    """

    cube_env = finetune_kwargs.get('cube_env', False)
    if cube_env:
        proprio_dim = finetune_kwargs['proprio_dim']
        num_cubes = finetune_kwargs['num_cubes']
    visual_env = finetune_kwargs.get('visual_env', False) # if true we use visual env logic for selecting subgoals

    _obs = dataset['observations']
    ep_id = dataset['terminals'].cumsum() // 2
    ep_id[1:] = ep_id[:-1]
    ep_lens = np.unique(ep_id, return_counts=True)[1]
    # This is needed for parallel computation, but can be worked around
    assert len(set(ep_lens)) == 1, "All episodes need to have the same length."
    ep_len = ep_lens[0].item()

    mask = np.zeros_like(ep_id)
    _obs = _obs.reshape(-1, ep_len, _obs.shape[-1])

    # Find all trajectories that pass close to the current state and goal
    # If cube_env is True, we first need to extract cube positions from the observations
    if cube_env:
        current_obs_cube_positions = extract_cube_positions_from_full_obs(obs, proprio_dim, num_cubes)
        dataset_cube_positions = extract_cube_positions_from_full_obs(_obs, proprio_dim, num_cubes)
        goal_cube_positions = goal # goal is already [pos1, pos2, ...] for CubeEnv
        goal_is_oracle_representation = finetune_kwargs.get('goal_is_oracle_rep', True)

        if goal_is_oracle_representation:
            # Goal is already in the format [pos1_scaled, pos2_scaled, ...] (shape: num_cubes * 3)
            goal_cube_positions = goal
            # Sanity check for expected shape if it's an oracle goal
            expected_shape = (num_cubes * 3,)
            if goal.shape != expected_shape:
                print(f"Warning: 'goal_is_oracle_rep' is True, but goal shape is {goal.shape}, expected {expected_shape}. This might lead to errors.")
                # This implies a mismatch in configuration or how goal_is_oracle_rep was set.
                # For the error (1000,1001,3) vs (28,), if num_cubes=1, then goal.shape here *must* be (3,).
                # If it's (28,), then goal_is_oracle_representation should have been False.
        else:
            # Goal is a full observation of the goal state. Extract cube positions from it.
            # Input `goal` shape e.g. (28,) for num_cubes=1, proprio_dim=19
            # Output `goal_cube_positions` shape (num_cubes * 3,), e.g. (3,) for num_cubes=1
            goal_cube_positions = extract_cube_positions_from_full_obs(goal, proprio_dim, num_cubes)

        start_matches_dist = jnp.sqrt(jnp.sum((dataset_cube_positions - current_obs_cube_positions)**2, axis=-1))
        start_matches = start_matches_dist < sim_threshold

        # This line should now work if goal_cube_positions is correctly shaped to (num_cubes * 3,)
        goal_matches_dist = jnp.sqrt(jnp.sum((dataset_cube_positions - goal_cube_positions)**2, axis=-1))
        goal_matches = goal_matches_dist < sim_threshold
    else: 
        # Original maze logic
        start_matches = jnp.sqrt(jnp.sum((_obs[..., :2] - obs[:2]) ** 2, axis=-1)) < sim_threshold
        goal_matches = jnp.sqrt(jnp.sum((_obs[..., :2] - goal[:2]) ** 2, axis=-1)) < sim_threshold

    # Only proceed if there are trajectories matching both start and goal
    filtered_eps = (start_matches.sum(-1) * goal_matches.sum(-1)) > 0 
    if filtered_eps.sum():
        # Here, we are trying to select the best subtrajectory in each matching trajectory
        goal_matches_id = np.arange(ep_len).reshape(1, -1) * goal_matches
        goal_matches_id = np.where(goal_matches_id == 0, ep_len, goal_matches_id)
        acc_min = np.minimum.accumulate(goal_matches_id[..., ::-1], -1)[..., ::-1]
        steps_to_goal = acc_min - np.arange(ep_len).reshape(1, -1)
        # candidates contains one value for all possible starting states
        # this value roughly indicates how distant is the closest goal match
        candidates = steps_to_goal * start_matches
        candidates = np.where(candidates == 0, ep_len, candidates)
        candidates = np.where(acc_min == ep_len, ep_len, candidates)
        # for each trajectory, we select the most promising candidate (highest MC return)
        solutions = np.argmin(candidates, -1)
        goal_offset = np.min(candidates, -1)
        threshold = ep_len - 1   
        if (goal_offset < ep_len).sum():
            threshold = np.quantile(goal_offset[goal_offset < ep_len], quantile)
        goal_offset = np.where(goal_offset > threshold, 0, goal_offset)
        goal_offset = np.where(goal_offset > 0, np.minimum(goal_offset + slack, ep_len - solutions), 0)
        col_indices = np.arange(ep_len)
        mask = (col_indices >= solutions[:, np.newaxis]) & (col_indices < (solutions + goal_offset)[:, np.newaxis])

    return mask.flatten()


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        goal_idxs = np.where(
            np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6), middle_goal_idxs, random_goal_idxs
        )

        # Goals at the current state.
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    # This function implements the main fine-tuning logic for data selection
    def active_sample(self, batch_size: int, _filter, goal, ratio, fix_actor_goal, finetune_kwargs):

        finetune_bs = int(batch_size * ratio)
        # First, sample a batch normally
        uniform_batch = self.sample(batch_size - finetune_bs)
        idxs = np.random.choice(np.where(_filter)[0], finetune_bs)
        # Then, sample a batch actively
        active_batch = self.sample(finetune_bs, idxs)

        # We set the actor goals to the same goal for fix_actor_goal percentage of transitions in the active batch.
        idxs = np.random.uniform(size=(finetune_bs,)) < fix_actor_goal
        if finetune_kwargs.get('saw', False):
            active_batch['high_actor_goals'][idxs] = goal
            #active_batch['low_actor_goals'][idxs] = goal
        else:
            active_batch['actor_goals'][idxs] = goal

        return {k: np.concatenate([uniform_batch[k], active_batch[k]]) for k in uniform_batch}

    def prepare_active_sample(self, agent, obs, goal, finetune_kwargs, batch_size=2048, exp_name = None,
                              log_filter=True):
        
        _obs = self.dataset['observations']
        _filter = jnp.ones_like(self.dataset['terminals'])
        mc_quantile = finetune_kwargs['mc_quantile']
        mc_slack = finetune_kwargs['mc_slack']
        mc_similarity_threshold = finetune_kwargs['mc_similarity_threshold']
        max_len = np.inf
        visual_env = finetune_kwargs.get('visual_env', False) # if true we use visual env logic for selecting subgoals

        # GC-TTT without critic: only find "optimal trajectories", no stitching
        # - trajectories with high Monte-Carlo returns
        # - trajectory passes close to current state (in terms of reward)
        if finetune_kwargs['filter_by_mc']:
            mc_filter = filter_from_state_goal(self.dataset, obs, goal, mc_quantile, mc_slack, mc_similarity_threshold, finetune_kwargs)
            _filter = _filter * mc_filter

        # Randomly select 10k transitions for fine-tuning.
        elif finetune_kwargs.get('random_selection', False):
            _filter = np.zeros_like(self.dataset['terminals'])
            _filter[np.random.choice(len(_obs), 10000)] = 1.

        # GC-TTT with critic: find "optimal trajectories" and stitch them together
        # - trajectories with high Monte-Carlo returns
        # - trajectory passes close to current state (in terms of reward)
        # - trajectory passes close to current goal (in terms of reward)
        elif finetune_kwargs.get('filter_by_recursive_mdp', False):
            _values = []
            _start_values = []
            batch_size=10000
            for i in range((len(_obs) // batch_size) + 1):
                _sli, _ce = i*batch_size, min((i+1)*batch_size, len(_obs))
                if finetune_kwargs.get('saw', False):
                    v1, v2 = agent.network.select('value')(_obs[_sli:_ce], goal.reshape(1, -1).repeat(_ce - _sli, 0))
                    v = (v1 + v2) / 2
                    _values.append(v)
                    del v1, v2, v
                    v1, v2 = agent.network.select('value')(_obs[_sli:_ce], obs.reshape(1, -1).repeat(_ce - _sli, 0))
                    v = (v1 + v2) / 2
                    _start_values.append(v)
                    del v1, v2, v
                else:
                    _values.append(agent.network.select('value')(_obs[_sli:_ce], goal.reshape(1, -1).repeat(_ce - _sli, 0)))
                    _start_values.append(agent.network.select('value')(_obs[_sli:_ce], obs.reshape(1, -1).repeat(_ce - _sli, 0)))
            _values = jnp.concatenate(_values, 0)
            state_to_goal_dist = (jnp.log((_values/(1/(1 - 0.99)) + 1)) / jnp.log(0.99))
            _start_values = jnp.concatenate(_start_values, 0)
            start_to_state_dist = (jnp.log((_start_values/(1/(1 - 0.99)) + 1)) / jnp.log(0.99))
            
            #td_filter = filter_by_recursive_mdp(self.dataset, agent, obs, goal, finetune_kwargs, state_to_goal_dist, start_to_state_dist)
            td_filter, max_len = filter_by_recursive_mdp(self.dataset, agent, obs, goal, finetune_kwargs, state_to_goal_dist, start_to_state_dist,
                                                         _start_values, _values)
            _filter = _filter * td_filter 

        
        # Simple visualization of the filter for 2D environments
        if log_filter and not visual_env:
            import matplotlib.pyplot as plt
            import io
            import wandb
            from PIL import Image
            filtered_pbs = _obs[_filter.astype(bool)]

            try:
                buf = io.BytesIO()
                plt.scatter(_obs[:5000, 0], _obs[:5000, 1])
                plt.scatter(filtered_pbs[:, 0], filtered_pbs[:, 1], alpha=0.5)
                #plt.savefig(f'Zfilter_{exp_name}.png')
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                wandb.log({'ZFilter': wandb.Image(np.array(Image.open(buf)))})
                del buf
            except Exception as e:
                print(f"Error in logging filter image: {e}")
                plt.close()
            # wandb.log({'ZFilter': wandb.Image(f'Zfilter_{exp_name}.png')})

        #return _filter
        return _filter, max_len

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                )

        return batch
