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


def filter_from_state_goal(dataset, obs, goal, quantile, slack):

    _obs = dataset['observations']
    ep_id = dataset['terminals'].cumsum() // 2
    ep_id[1:] = ep_id[:-1]
    ep_lens = np.unique(ep_id, return_counts= True)[1]
    assert len(set(ep_lens)) == 1, "All episodes need to have the same length."
    ep_len = ep_lens[0].item()

    mask = np.zeros_like(ep_id)
    _obs = _obs.reshape(-1, ep_len, _obs.shape[-1])
    start_matches = jnp.sqrt(((_obs[...] - obs)**2).sum(-1)) < 0.5
    goal_matches = jnp.sqrt(((_obs[...] - goal)**2).sum(-1)) < 0.5
    filtered_eps = (start_matches.sum(-1) * goal_matches.sum(-1)) > 0
    if filtered_eps.sum():
        goal_matches_id = np.arange(ep_len).reshape(1, -1) * goal_matches
        goal_matches_id = np.where(goal_matches_id == 0, ep_len, goal_matches_id)
        acc_min = np.minimum.accumulate(goal_matches_id[..., ::-1], -1)[..., ::-1]
        steps_to_goal = acc_min - np.arange(ep_len).reshape(1, -1)
        candidates = steps_to_goal * start_matches
        candidates = np.where(candidates == 0, ep_len, candidates)
        candidates = np.where(acc_min == ep_len, ep_len, candidates)
        solutions = np.argmin(candidates, -1)
        goal_offset = np.min(candidates, -1)
        threshold = np.quantile(goal_offset[goal_offset < ep_len], quantile)
        threshold = min(threshold, ep_len - 1)  # in case no solutions are found
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

    def active_sample(self, batch_size: int, _filter, goal, ratio, fix_actor_goal):
        finetune_bs = int(batch_size * ratio)
        uniform_batch = self.sample(batch_size - finetune_bs)
        idxs = np.random.choice(np.where(_filter)[0], finetune_bs)
        active_batch = self.sample(finetune_bs, idxs)

        # active_batch['value_goals'] = active_batch['value_goals'] * 0 + goal
        # successes = (jnp.sqrt(((active_batch['observations'] - active_batch['value_goals'])**2).sum(-1)) < 1).astype(float)
        # active_batch['masks'] = 1.0 - successes
        # active_batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)
        # fix actor goal to fine-tuning goal
        idxs = np.random.uniform(size=(finetune_bs,)) < fix_actor_goal
        active_batch['actor_goals'][idxs] = goal

        return {k: np.concatenate([uniform_batch[k], active_batch[k]]) for k in uniform_batch}

    def prepare_active_sample(self, agent, obs, goal, finetune_kwargs, batch_size=2048):

        _obs = self.dataset['observations']
        _filter = jnp.ones_like(self.dataset['terminals'])
        mc_quantile = finetune_kwargs['mc_quantile']
        mc_slack = finetune_kwargs['mc_slack']

        # On-policy filtering: only find "optimal trajectories", no stitching
        # - trajectories with high Monte-Carlo returns
        # - trajectory passes close to current state (in terms of reward)
        if finetune_kwargs['filter_by_mc']:
            mc_filter = filter_from_state_goal(self.dataset, obs, goal, mc_quantile, mc_slack)
            _filter = _filter * mc_filter

        # Off-policy filtering (needs a value function, can stitch)
        # - (sub)trajectories with low TD-error mean_t (V(s_t+1) - \gammaV(s_t))^2
        # - state respects triangle inequality
        if finetune_kwargs['filter_by_td']:

            batch_size = 1000
            get_steps_between = lambda x, y: jnp.log((agent.network.select('value')(x, y) / (1/(1 - 0.99)) + 1)) / jnp.log(0.99)
            batch = _obs[np.random.choice(len(_obs), batch_size)]
            batch = np.concatenate([obs.reshape(1, -1), goal.reshape(1, -1), batch], 0)
            distances = [get_steps_between(batch, o.reshape(1, -1).repeat(len(batch), 0)) for o in batch]
            distances = np.stack(distances, 0).astype(np.float64)
            distances = np.maximum(0., distances)
            graph = rx.PyDiGraph.from_adjacency_matrix(distances)
            threshold = maximum_edge_length(graph, 0, 1)
            graph = rx.PyDiGraph.from_adjacency_matrix(np.where(distances > threshold, 10000, distances))
            path = [idx for idx in rx.dijkstra_shortest_paths(graph, 0, 1, weight_fn=lambda x: x)[1]]
            path = path[::max(1, len(path) // finetune_kwargs['sorb_len'])]

            _td_filter = jnp.zeros_like(self.dataset['terminals'])
            for start_id, end_id in zip(path[:-1], path[1:]):
                _segment = filter_from_state_goal(self.dataset, batch[start_id], batch[end_id], mc_quantile, mc_slack)
                _td_filter = np.maximum(_td_filter, _segment)
            _filter = _filter * _td_filter

            # TODO: finish writing this
            # check that trajectories are well selected
            # try to BC them
            # or try to AWR with the fixed value

            # this computes values along the graph :)
            # distances_in_path = np.array([0] + [distances[a, b] for a, b in zip(path[:-1], path[1:])])
            # distances_in_path = np.cumsum(distances_in_path)
            # min_steps = np.ones(len(_obs)) * 10000
            # for idx, dip in zip(path, distances_in_path):
            #     steps = []
            #     for i in range((len(_obs) // batch_size) + 1):
            #         _sli, _ce = i*batch_size, min((i+1)*batch_size, len(_obs))
            #         steps.append(get_steps_between(_obs[_sli:_ce], _obs[[idx]].repeat(_ce - _sli, 0)))
            #     steps = np.concatenate(steps, 0)
            #     steps = np.where(steps > threshold, 10000, steps) + dip
            #     min_steps = np.minimum(min_steps, steps)
            # self._values = - (1 - 0.99 ** min_steps) / (1 - 0.99)

            # recursive selection
            # batch = _obs[np.random.choice(len(_obs), batch_size)]
            # batch = np.concatenate([obs.reshape(1, -1), batch, goal.reshape(1, -1)])
            # subgoal_ids = [0, len(batch) - 1]
            # distances_from = {i: get_steps_between(batch[[i]].repeat(len(batch), 0), batch) for i in subgoal_ids}
            # distances_to = {i: get_steps_between(batch, batch[[i]].repeat(len(batch), 0)) for i in subgoal_ids}
            # for _ in range(iters):
            #     new_subgoals = []
            #     # new subgoal is al least 0.5 distance away from either
            #     # it must be closer to neighbors than to second neighbors
            #     for start_id, end_id in zip(subgoal_ids[:-1], subgoal_ids[1:]):
            #         half_distance = distances_from[start_id][end_id] / 1.5
            #         distance_from = np.maximum(distances_from[start_id], half_distance)
            #         distance_to = np.maximum(distances_to[end_id], half_distance)
            #         mid_id = np.argmin(np.maximum(distance_from, distance_to)).item()
            #         distances_from[mid_id] = get_steps_between(batch[[start_id]].repeat(len(batch), 0), batch)
            #         distances_to[mid_id] = get_steps_between(batch, batch[[end_id]].repeat(len(batch), 0))
            #         new_subgoals.extend([start_id, mid_id])
            #     subgoal_ids = new_subgoals + [subgoal_ids[-1]]
            # print(subgoal_ids)
            # import matplotlib.pyplot as plt
            # plt.scatter(batch[:, 0], batch[:, 1], c='red', alpha=0.01)
            # plt.scatter(batch[subgoal_ids][:, 0], batch[subgoal_ids][:, 1], c=np.arange(len(subgoal_ids)))
            # plt.savefig('test.png')
            # plt.close()

            # 0/1 function on BC
            # _values = []
            # for i in range((len(_obs) // batch_size) + 1):
            #     _sli, _ce = i*batch_size, min((i+1)*batch_size, len(_obs))
            #     _values.append(agent.network.select('value')(_obs[_sli:_ce], goal.reshape(1, -1).repeat(_ce - _sli, 0)))
            # _values = jnp.concatenate(_values, 0)
            # _state_to_goal = (jnp.log((_values / (1/(1 - 0.99)) + 1)) / jnp.log(0.99))
            # td_filter = jnp.zeros_like(ep_id)
            # same_ep = (ep_id[:-horizon] == ep_id[horizon:])
            # td = ((0.99**horizon) * _values[horizon:] - _values[:-horizon]) * same_ep
            # min_td = jnp.quantile(td, mc_quantile)
            # for h in range(horizon):
            #     td_filter = td_filter.at[h:-horizon+h].set(jnp.logical_or((td > min_td) * same_ep, td_filter[h:-horizon+h]))
            # _filter = _filter * td_filter

            # equality
            # _start_values = []
            # for i in range((len(_obs) // batch_size) + 1):
            #     _sli, _ce = i*batch_size, min((i+1)*batch_size, len(_obs))
            #     _start_values.append(agent.network.select('value')(obs.reshape(1, -1).repeat(_ce - _sli, 0), _obs[_sli:_ce]))
            # _start_values = jnp.concatenate(_start_values, 0)
            # _start_to_state = (jnp.log((_start_values / (1/(1 - 0.99)) + 1)) / jnp.log(0.99))
            # eq_score = _start_to_state + _state_to_goal  # picking bottom 50% of this score is insufficient
            # eq_filter = eq_score < jnp.quantile(eq_score, eq_quantile)
            # _filter = _filter * eq_filter

            # how many steps closer?
            # td_filter = jnp.zeros_like(ep_id)
            # same_ep = (ep_id[:-horizon] == ep_id[horizon:])
            # progress = (_state_to_goal[:-horizon] - _state_to_goal[horizon:]) * same_ep * _filter[:-horizon]  # how many steps closer?
            # min_progress = jnp.quantile(progress, 0.9)
            # for h in range(horizon):
            #     td_filter = td_filter.at[h:-horizon+h].set(jnp.logical_or((progress > min_progress) * same_ep, td_filter[h:-horizon+h]))
            # _filter = _filter * td_filter

        # import matplotlib.pyplot as plt
        # __obs = _obs[_filter.astype(bool)]
        # plt.scatter(_obs[:5000, 0], _obs[:5000, 1])
        # plt.scatter(__obs[:, 0], __obs[:, 1], alpha=0.1)
        # plt.savefig('zfilter.png')
        # plt.close()

        return _filter

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
