import dataclasses
from functools import partial
from typing import Any, Optional

import time
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
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


def compute_reward(obs, goal, env_name, is_goal=True):
    if 'maze' in env_name:
        # Shape results: (num_episodes, ep_len)
        return jnp.sqrt(jnp.sum((obs[..., :2] - goal[:2]) ** 2, axis=-1)) < 1.0
    elif 'cube-single' in env_name:
        # revert obs transformation
        _obs = obs[..., -9:-6] * 0.1 + jnp.array([[[0.425, 0., 0.]]])
        _goal = goal[-9:-6] * 0.1 + jnp.array([0.425, 0., 0.])
        _obs_grip = obs[..., -16:-13] * 0.1 + jnp.array([[[0.425, 0., 0.]]])
        _goal_grip = goal[-16:-13] * 0.1 + jnp.array([0.425, 0., 0.])
        cube_matches = jnp.sqrt(jnp.sum((_obs - _goal) ** 2, axis=-1)) < 0.04
        goal_matches = jnp.sqrt(jnp.sum((_obs_grip - _goal_grip) ** 2, axis=-1)) < 0.04
        return cube_matches if is_goal else cube_matches * goal_matches
    else:
        raise NotImplementedError("Environment not supported")

#def filter_by_recursive_mdp(dataset, agent, obs, goal, finetune_kwargs, state_to_goal_dist=None, start_to_state_dist=None,
#                            _start_values=None, _values=None):
def filter_by_td(dataset, obs, env_name, state_to_goal_dist, agent, finetune_kwargs):


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
        state_to_goal_dist: Precomputed distance from goal state to all states in the dataset.
        start_to_state_dist: Precomputed distance from start state to all states in the dataset.
    """

    _obs = dataset['observations'] # shape: (total_data_size, obs_dim)
    # There are two terminals for the last two states of an episode
    # So the sequence of terminal is like [0 0 1 1 0 0 0 1 1 0 0 0 0 0 1]
    # (Cumsum terminal)//2  will be       [0 0 0 1 1 1 1 1 2 2 2 2 2 2 2]
    # Shifted by one so the next episode starts after the last terminal
    # of previous episode                 [0 0 0 0 1 1 1 1 1 2 2 2 2 2 2]
    ep_id = dataset['terminals'].cumsum() // 2 # shape: (total_data_size,)
    ep_id[1:] = ep_id[:-1]
    ep_lens = np.unique(ep_id, return_counts= True)[1] # shape: (num_episodes,)
    assert len(set(ep_lens)) == 1, "All episodes need to have the same length."
    ep_len = ep_lens[0].item() # length of each episode

    subtraj_min_steps = finetune_kwargs.get('min_steps', 10) # subgoals that are at least this many steps away from the start
    sim_threshold = finetune_kwargs.get('mc_similarity_threshold', 1.0) # distance threshold for start matches
    num_selected_points  = finetune_kwargs.get('recursive_selected_num_points', 10) # number of subgoals to select
    non_optimality = finetune_kwargs.get('no_optimality', False) # if true we only sample transitions close to the state regardless of whether they are any good
    non_relevance = finetune_kwargs.get('no_relevance', False) # if true we sample transitions from the buffer that are good under the optimality criterion but may be from anywhere over the state space (not necessarily close to our agents state)
    cube_env = finetune_kwargs.get('cube_env', False)
    visual_env = finetune_kwargs.get('visual_env', False) # if true we use visual env logic for selecting subgoals
    fixed_horizon = finetune_kwargs.get('fixed_horizon', False)

    # Check nan, if there are raise Error
    if np.isnan(state_to_goal_dist).sum() > 0:
        raise ValueError("state_to_goal_dist contains nans")

    mask = np.zeros_like(ep_id)  # shape: (total_data_size,)

    # Select subtrajectories that start close to the current state
    if finetune_kwargs['relevance_by_value']:
        # Compute starting values now
        _start_values = []
        batch_size = 400_000 # TODO: Ask marco # old: 10000
        for i in range((len(_obs) // batch_size) + 1):
            _sli, _ce = i*batch_size, min((i+1)*batch_size, len(_obs))
            _start_value = agent.network.select('value')(_obs[_sli:_ce], obs[None].repeat(_ce-_sli, 0), params=agent.network.params)
            # handle twin critics
            _start_value = ((_start_value[0] + _start_value[1]) / 2) if len(_start_value) == 2 else _start_value
            _start_values.append(_start_value)
        _start_values = jnp.concatenate(_start_values, 0)
        # Convert values to distances
        start_to_state_dist = (jnp.log((_start_values/(1/(1 - 0.99)) + 1)) / jnp.log(0.99))
        start_matches = (start_to_state_dist < sim_threshold).reshape(-1, ep_len)
    else:
        _obs = _obs.reshape(-1, ep_len, _obs.shape[-1]) # Shape: (num_episodes, ep_len, obs_dim)
        start_matches = compute_reward(_obs, obs, env_name)
    if non_relevance:
        # shuffle start matches, i.e. match states randomly
        np.random.shuffle(start_matches)

    # Now based on the return estimates, we select the subtrajectories that are optimal
    # Shift start_matches to align with the subtrajectory minimum steps
    shift_start_matches = np.zeros_like(start_matches)

    # For each episode, the first subtraj_min_steps steps are set to 0
    # and the rest are set to the start_matches of the previous steps
    # NOTE: what if is there a match of start state within the last subtraj_min_steps steps?
    # Answer: we want only to select subtrajectories whose length is at least subtraj_min_steps
    # So for the steps 1, 2, ..., subtraj_min_steps-1, we set the start_matches to 0
    # While for the rest we set the start_matches to the start_matches the step which is subtraj_min_steps steps ago

    # Logic: if state at position t matches start, then subtraj starts at t
    # and t+subtraj_min_steps is a valid end position (because the start matches)
    # so shift_start_matches semantic is "is this end position such that the start matches?"
    subtraj_min_steps = finetune_kwargs["min_steps"]

    # The first 10 will not match, the 11-th match if the 1st is relevant and so on
    shift_start_matches[:, subtraj_min_steps:] = start_matches[:, :-subtraj_min_steps]

    if fixed_horizon:
        scores = ((shift_start_matches.cumsum(-1) == 1) * state_to_goal_dist.reshape(start_matches.shape))
    else:
        scores = ((shift_start_matches.cumsum(-1) > 0) * state_to_goal_dist.reshape(start_matches.shape))
    scores = np.where(scores==0, scores.max(), scores)
    num_selected_points  = finetune_kwargs.get('recursive_selected_num_points', 10) # number of subgoals to select

    # For each episode, scores is a 2D array (episodes x trajectory steps).
    # scores.min(-1) finds the minimum score along the time axis for each episode,
    # so scores.min(-1) is a 1D array where each element corresponds to the "best" (lowest) score found within that episode.
    # np.argsort(...): sorts all episodes based on these minimum scores (from lowest to highest).
    # The result is that ep_idxs contains the indices of the `num_selected_points` episodes that have the best (lowest) minimal score along their trajectory.
    ep_idxs = np.argsort(scores.min(-1))[:num_selected_points]

    if non_optimality:
        # randomly select from  np.argsort(scores.min(-1)) instead of taking the best ones
        ep_idxs = np.random.choice(np.where(scores.min(-1) < scores.max())[0], num_selected_points, replace=False)


    mask = mask.reshape(-1, ep_len)
    mask[ep_idxs] = 1.

    if finetune_kwargs.get('latest_starting_state', False):
        # Take last state that matches the start
        # Take cumsum and then set to 0 those not equal to the max
        # NOTE: I actually want to get the closest one, so start from latest - offset
        cumsum = start_matches.cumsum(-1)
        max_cumsum = cumsum.max(-1)
        mask_last_start = (cumsum == max_cumsum[:, None])
        mask_start = mask_last_start
    elif finetune_kwargs.get('start_from_first', False):
        mask_start = np.ones_like(mask)
    else:
        mask_start = start_matches.cumsum(-1) > 0

    mask *= mask_start  # only keep from matches until the end
    col_indices = np.arange(ep_len) # [0, 1, 2, ..., ep_len-1]
    # discard best point and all points after it TODO: ask marco why this??
    mask *= col_indices[None] <= scores.argmin(-1)[..., None]  # discard after best point
    # for each ep, count selected steps and take maximum
    max_len = mask.sum(-1).max()
    return mask.flatten(), max_len


#def filter_from_state_goal(dataset, obs, goal, quantile, slack, sim_threshold, finetune_kwargs=None):
def filter_by_mc(dataset, obs, goal, env_name, finetune_kwargs):
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
    quantile = finetune_kwargs['mc_quantile']
    slack = finetune_kwargs['mc_slack']

    _obs = dataset['observations']
    ep_id = dataset['terminals'].cumsum() // 2
    ep_id[1:] = ep_id[:-1]
    ep_lens = np.unique(ep_id, return_counts=True)[1]
    # This is needed for parallel computation, but can be worked around
    assert len(set(ep_lens)) == 1, "All episodes need to have the same length."
    ep_len = ep_lens[0].item()

    mask = np.zeros_like(ep_id)
    _obs = _obs.reshape(-1, ep_len, _obs.shape[-1])

    start_matches = compute_reward(_obs, obs, env_name)
    goal_matches = compute_reward(_obs, goal, env_name)

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
        threshold = ep_len - 1  # in case no solutions are found
        # we further filter these promising candidates, and only take the top quantile%
        if (goal_offset < ep_len).sum():
            threshold = np.quantile(goal_offset[goal_offset < ep_len], quantile)
        goal_offset = np.where(goal_offset > threshold, 0, goal_offset)
        goal_offset = np.where(goal_offset > 0, np.minimum(goal_offset + slack, ep_len - solutions), 0)
        col_indices = np.arange(ep_len)
        mask = (col_indices >= solutions[:, np.newaxis]) & (col_indices < (solutions + goal_offset)[:, np.newaxis])

    max_len = mask.sum(-1).max()
    return mask.flatten(), max_len


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
    values = None
    start_values = None

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

    # def active_sample(self, batch_size: int, _filter, goal, ratio, fix_actor_goal, finetune_kwargs, return_indices=False):
    def active_sample(self, batch_size: int, _filter, goal, ratio, fix_actor_goal, hierarchical=False, return_indices=False):
        """
        TODO: set hierarchical to True if using SAW
        This function samples a batch of data for fine-tuning, combining both uniform and actively selected samples.
        A portion of the batch is sampled uniformly, while the rest is sampled from transitions that satisfy a given filter.
        For a specified percentage of the actively sampled transitions, the actor goals are set to a fixed goal.

        Args:
            batch_size (int): Total number of samples to draw.
            _filter (array-like): Boolean mask or filter to select eligible transitions for active sampling.
            goal (np.ndarray): The goal to assign to a subset of actor goals in the active batch.
            ratio (float): Fraction of the batch to sample actively (between 0 and 1).
            fix_actor_goal (float): Fraction of the active batch to set the actor goal to the provided goal.
            finetune_kwargs (dict): Additional keyword arguments for fine-tuning logic.

        Returns:
            dict: A batch dictionary with concatenated uniform and active samples.
            np.ndarray: Indices of the selected samples.
        """
        if return_indices:
            assert ratio == 1.0, "To get indices, ratio of active samples must be 1.0"
        finetune_bs = int(batch_size * ratio)
        # First, sample a batch normally
        uniform_batch = self.sample(batch_size - finetune_bs)
        idxs = np.random.choice(np.where(_filter)[0], finetune_bs)
        # Then, sample a batch actively
        active_batch = self.sample(finetune_bs, idxs)

        # We set the actor goals to the same goal for fix_actor_goal percentage of transitions in the active batch.
        idxs_fix_actor_goal = np.random.uniform(size=(finetune_bs,)) < fix_actor_goal

        active_batch['high_actor_goals' if hierarchical else 'actor_goals'][idxs_fix_actor_goal] = goal

        if return_indices:
            return {k: np.concatenate([uniform_batch[k], active_batch[k]]) for k in uniform_batch}, idxs
        else:
            return {k: np.concatenate([uniform_batch[k], active_batch[k]]) for k in uniform_batch}


    def prepare_values(self, agent, goal, finetune_kwargs, return_values=False):

        if not finetune_kwargs['filter_by_recursive_mdp']: # Note: marco changed into `filter_by_td`
            return None
        _obs = self.dataset['observations']
        _values = []
        batch_size = 400_000 # TODO: Ask marco # old: 10000
        for i in range((len(_obs) // batch_size) + 1):
            _sli, _ce = i*batch_size, min((i+1)*batch_size, len(_obs))
            _value = agent.network.select('value')(_obs[_sli:_ce], goal[None].repeat(_ce-_sli, 0), params=agent.network.params)
            # handle twin critics
            _value = ((_value[0] + _value[1]) / 2) if len(_value) == 2 else _value
            _values.append(_value)
        _values = jnp.concatenate(_values, 0)
        # Convert values to distances
        discount = agent.config['discount']
        state_to_goal_dist = (jnp.log((_values/(1/(1 - discount)) + 1)) / jnp.log(discount))

        if return_values:
            return state_to_goal_dist, _values
        else:
            return state_to_goal_dist

    # NOTE: batch size here is useless
    #def prepare_active_sample(self, agent, obs, goal, finetune_kwargs, batch_size=2048, exp_name = None,
    #                          log_filter=True, mc_quantile: Optional[float] = None):
    def prepare_active_sample(self, agent, obs, goal, env_name, finetune_kwargs, state_to_goal_dist=None):

        _obs = self.dataset['observations']
        _filter = jnp.ones_like(self.dataset['terminals'])

        mc_slack = finetune_kwargs['mc_slack']
        mc_similarity_threshold = finetune_kwargs['mc_similarity_threshold']
        max_len = np.inf # TODO: maybe adjust
        # TODO: ask marco about visual_env, if we still need it
        visual_env = finetune_kwargs.get('visual_env', False) # if true we use visual env logic for selecting subgoals

        # GC-TTT without critic: only find "optimal trajectories", no stitching
        # - trajectories with high Monte-Carlo returns
        # - trajectory passes close to current state (in terms of reward)
        if finetune_kwargs['filter_by_mc']:
            # OLD: mc_filter = filter_from_state_goal(self.dataset, obs, goal, mc_quantile, mc_slack, mc_similarity_threshold, finetune_kwargs)
            # NOTE: consider passing mc_quantile for adjusting optimality
            mc_filter, max_len = filter_by_mc(self.dataset, obs, goal, env_name, finetune_kwargs)
            _filter = _filter * mc_filter

        # Randomly select 10k transitions for fine-tuning.
        elif finetune_kwargs.get('random_selection', False):
            _filter = np.zeros_like(self.dataset['terminals'])
            _filter[np.random.choice(len(_obs), 10000)] = 1.

            # TODO: ask marco here, why he uses the following
            # _filter[np.random.choice(len(_obs), len(_obs) * 0.99)] = 0.

        # GC-TTT with critic: find "optimal trajectories" and stitch them together
        # - trajectories with high Monte-Carlo returns
        # - trajectory passes close to current state (in terms of reward)
        # - trajectory passes close to current goal (in terms of reward)
        # In another version, we use 'filter_by_td' instead of 'filter_by_recursive_mdp'
        elif finetune_kwargs.get('filter_by_recursive_mdp', False):
            assert state_to_goal_dist is not None, "state_to_goal_dist must be provided for recursive Value-based optimality scoring"
            td_filter, max_len = filter_by_td(self.dataset, obs, env_name, state_to_goal_dist, agent, finetune_kwargs)
            _filter = _filter * td_filter

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
