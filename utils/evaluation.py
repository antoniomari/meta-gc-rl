from collections import defaultdict
import copy
import jax
from tqdm import trange
import flax
from utils.config import FinetuneConfig
from typing import Optional, Any
import optax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import io
import wandb
from PIL import Image
from utils.datasets import GCDataset
from agents.gcagent import GCAgent
from typing import Dict
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from utils.config import GCTTTConfig
import gymnasium as gym
import time
import gc


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


# Helper to read values from finetune_config whether it's a dict or an object
def _cfg_get(cfg: Optional[FinetuneConfig], key: str, default=None):
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if hasattr(cfg, "get"):
            # ml_collections.ConfigDict supports get
            return cfg.get(key, default)
        return getattr(cfg, key, default)
    except Exception:
        return default


def make_plots(train_dataset, agent, goal, suffix, is_saw):

    _batch = train_dataset.sample(10000)
    _obs = _batch["observations"]
    if is_saw:
        v1, v2 = agent.network.select("value")(
            _obs, goal.reshape(1, -1).repeat(10000, 0)
        )
        values = (v1 + v2) / 2
        del v1, v2
    else:
        values = agent.network.select("value")(
            _obs, goal.reshape(1, -1).repeat(10000, 0)
        )
    actions = agent.network.select("actor")(
        _obs, goal.reshape(1, -1).repeat(10000, 0)
    ).mean()

    try:
        buf = io.BytesIO()
        plt.scatter(_obs[:, 0], _obs[:, 1], c=values)
        plt.savefig(buf, format="png", dpi=200)
        plt.close()
        buf.seek(0)
        wandb.log({f"Zvalues_{suffix}": wandb.Image(np.array(Image.open(buf)))})
        del buf
    except Exception as e:
        print(f"Error logging Zvalues_{suffix}: {e}")
        plt.close()  # Ensure plot is closed even on error

    try:
        buf = io.BytesIO()
        plt.quiver(
            _obs[:, 0],
            _obs[:, 1],
            actions[:, 0],
            actions[:, 1],
            angles="xy",
            scale_units="xy",
            scale=2,
        )
        plt.savefig(buf, format="png", dpi=200)
        plt.close()
        buf.seek(0)
        wandb.log({f"Zactions_{suffix}": wandb.Image(np.array(Image.open(buf)))})
        del buf
    except Exception as e:
        print(f"Error logging Zactions_{suffix}: {e}")
        plt.close()


def copy_current_agent(agent: GCAgent, finetune_config: FinetuneConfig) -> GCAgent:

    old_config = agent.config

    # Create a copy of the original config to preserve its type
    if hasattr(agent.config, "unfreeze"):
        new_config = agent.config.unfreeze()
    else:
        new_config = copy.deepcopy(agent.config)

    # Update specific fields from finetune_config
    if actor_loss_val := _cfg_get(finetune_config, "actor_loss", None) is not None:
        new_config["actor_loss"] = actor_loss_val
    if alpha_val := _cfg_get(finetune_config, "alpha", None) is not None:
        new_config["alpha"] = alpha_val
    old_train_state = copy.deepcopy(agent.network)
    opt_state = copy.deepcopy(agent.network.opt_state)
    finetune_tx = optax.adam(learning_rate=_cfg_get(finetune_config, "lr"))

    copy_agent = agent.replace(
        network=agent.network.replace(tx=finetune_tx, opt_state=opt_state),
        config=new_config,
    )

    return copy_agent, agent, old_config


def clone_agent(agent: GCAgent) -> GCAgent:
    # Deep, side-effect-free clone
    state = flax.serialization.to_state_dict(agent)
    return flax.serialization.from_state_dict(agent, state)


def snapshot_agent(agent: GCAgent) -> Dict[str, Any]:
    # Keep only the serialized snapshot; cheaper to store/restore
    return flax.serialization.to_state_dict(agent)

def restore_agent_from_snapshot(agent: GCAgent, snapshot: Dict[str, Any]) -> GCAgent:
    return flax.serialization.from_state_dict(agent, snapshot)



def make_current_config(finetune_config: FinetuneConfig, env: gym.Env) -> FinetuneConfig:
    # _filter is a binary mask over the entire dataset
    if hasattr(finetune_config, "unfreeze"):
        current_finetune_config = finetune_config.unfreeze()
    else:
        current_finetune_config = copy.deepcopy(finetune_config)

    cube_env = current_finetune_config.get("cube_env", False)
    if cube_env:  # A way to detect CubeEnv, or use env.spec.id
        # env._num_cubes should be available if 'env' is an instance of your CubeEnv
        num_cubes = (
            env._num_cubes if hasattr(env, "_num_cubes") else 1
        )  # Default to 1 if not found, adjust as needed
        current_finetune_config["num_cubes"] = num_cubes
        # The 9 elements per cube state: 3 (pos) + 4 (quat) + 2 (sin/cos yaw)
        current_finetune_config["proprio_dim"] = (
            env.observation_space.shape[0] - num_cubes * 9
        )
        try:
            current_finetune_config["goal_is_oracle_rep"] = env._use_oracle_rep
        except Exception as e:
            # print(f"Error accessing _use_oracle_rep: {e}")
            current_finetune_config["goal_is_oracle_rep"] = False

    return current_finetune_config


def actor_step(
    actor_fn,
    observation,
    env,
    config: GCTTTConfig,
    goal,
):
    action = actor_fn(
        observations=observation, goals=goal, temperature=config.eval_temperature
    )
    action = np.array(action)
    if not config.agent.get("discrete"):
        if config.eval_gaussian is not None:
            action = np.random.normal(action, config.eval_gaussian)
        action = np.clip(action, -1, 1)

    next_observation, reward, terminated, truncated, info = env.step(action)
    return next_observation, action, reward, terminated, truncated, info


def gc_ttt_critic(
    train_dataset: GCDataset,
    agent: GCAgent,
    env,
    observation,
    config: GCTTTConfig,
    goal,
    goal_frame,
    should_render: bool,
    num_ttt_steps: Optional[int] = None
):

    finetune_config = config.finetune
    # GC-TTT without critic
    traj = defaultdict(list)
    finetune_stats = defaultdict(list)
    info = None

    done = False
    step = 0
    aggregated_filters = []
    render = []
    old_train_state, old_config = None, None
    actor_fn = supply_rng(
        agent.sample_actions,
        rng=jax.random.PRNGKey(np.random.randint(0, 2**32)),
    )

    # Define how many steps to execute between replanning phases. K in the paper.
    replan_horizon: int = finetune_config.get("replan_horizon", 100)

    # Define how many steps to finetune
    num_steps: int = num_ttt_steps if num_ttt_steps is not None else finetune_config.get("num_steps", 0)

    # Replanning loop: repeatedly fine-tune and execute a short horizon.
    agent_ft = None
    while not done:
        # working copy for finetuning
        # if agent_ft is not None:
            # check if parameters are equal
            # using jax.tree_util.tree_all
            #print("Before cloning: ",
            #    jax.tree_util.tree_all(jax.tree_map(lambda x, y: jnp.array_equal(x, y), agent.network.params, agent_ft.network.params))
            #)
        agent_ft = clone_agent(agent)
        #print("After cloning: ",
        #    jax.tree_util.tree_all(jax.tree_map(lambda x, y: jnp.array_equal(x, y), agent.network.params, agent_ft.network.params))
        #)

        # New finetuning config?
        finetune_stats = defaultdict(list)


        if finetune_config.get("cube_env", False):
            # A way to detect CubeEnv, or use env.spec.id
            # env._num_cubes should be available if 'env' is an instance of your CubeEnv
            num_cubes = (
                env._num_cubes if hasattr(env, "_num_cubes") else 1
            )  # Default to 1 if not found, adjust as needed
            finetune_config["num_cubes"] = num_cubes
            # The 9 elements per cube state: 3 (pos) + 4 (quat) + 2 (sin/cos yaw)
            finetune_config["proprio_dim"] = (
                env.observation_space.shape[0] - num_cubes * 9
            )

        # Filtering the dataset for active test-time fine-tuning.

        filter_start_time = time.time()
        _filter, max_len = train_dataset.prepare_active_sample(
            agent,
            observation,
            goal,
            finetune_config,
            log_filter=False,
        )
        filter_end_time = time.time()
        # print(f"[Timing] Filtering dataset for active test-time fine-tuning took {filter_end_time - filter_start_time:.4f} seconds")

        aggregated_filters.append(_filter)
        if _filter.sum() > 0:
            # Finetune for N steps
            for i in range(num_steps):
                # Sample a batch from the dataset using the filter.
                # The batch will contain only the samples that match the filter.
                batch = train_dataset.active_sample(
                    finetune_config.get("batch_size", 1024),
                    _filter,
                    goal,
                    finetune_config.get("ratio", 1.0),
                    finetune_config.get("fix_actor_goal", 1.0),
                    finetune_kwargs=finetune_config,
                )

                # Update the agent with the sampled batch.
                # Note: in the original code (as here) the optimizer is never reset, consider using reset_inner_opt=i==0
                agent_ft, update_info = agent_ft.update(batch, finetuning=True, reset_inner_opt=False)
                add_to(finetune_stats, flatten(update_info))

            # Log the filter after fine-tuning, also show the cuurent state of the agent as red
            if not _cfg_get(finetune_config, "visual_env", False):

                pass # TODO: restore this code accordingly
                """
                    _obs = train_dataset.dataset[
                        "observations"
                    ]  # assuming dataset is available here
                    filtered_pbs = _obs[_filter.astype(bool)]
                    buf = io.BytesIO()
                    plt.scatter(_obs[:5000, 0], _obs[:5000, 1])
                    plt.scatter(filtered_pbs[:, 0], filtered_pbs[:, 1], alpha=0.5)
                    plt.scatter(observation[0], observation[1], color="red", s=50)
                    plt.savefig(buf, format="png")
                    plt.close()
                    buf.seek(0)
                    img = Image.open(buf)
                    img_array = np.array(img)
                    wandb.log({"ZFilter_Partial": wandb.Image(img_array)})
                    del img, img_array, buf
                """
        else:
            print(f"Empty filter")
        actor_fn = supply_rng(
            agent_ft.sample_actions,
            rng=jax.random.PRNGKey(np.random.randint(0, 2**32)),
        )

        # Execute the policy for a fixed short horizon before replanning.
        for _step in range(replan_horizon):
            if done:
                break
            # check if finetune.reset_after_horizon is set to True and enter the loop
            if _step > max_len and finetune_config.get("reset_after_horizon", False):

                agent_ft = clone_agent(agent)
                actor_fn = supply_rng(
                    agent_ft.sample_actions,
                    rng=jax.random.PRNGKey(np.random.randint(0, 2**32)),
                )

            # Perform action
            action = actor_fn(
                observations=observation,
                goals=goal,
                temperature=config.eval_temperature,
            )
            action = np.array(action)
            if not config.agent.get("discrete"):
                if config.eval_gaussian is not None:
                    action = np.random.normal(action, config.eval_gaussian)
                action = np.clip(action, -1, 1)

            # info contains "success" and is overwritten every time
            # So we get the success from the last step
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            done = done or step >= 3000

            # TODO: restore this code accordingly
            #if should_render and (step % video_frame_skip == 0 or done):
            #    frame = env.render().copy()
            #    if goal_frame is not None:
            #        render.append(np.concatenate([goal_frame, frame], axis=0))
            #    else:
            #        render.append(frame)

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

        combined_filter = np.zeros_like(aggregated_filters[0])
        numberofallfiltered = 0
        for f in aggregated_filters:
            # Combine filters via element-wise maximum (logical OR for binary masks)
            combined_filter = np.maximum(combined_filter, f)
            # check non zero elements in the f and add to numberofallfiltered
            numberofallfiltered += np.count_nonzero(f)

        # TODO: restore wandb.log({"Z_NumberOfFineTunePoints": numberofallfiltered})

        visual_env = _cfg_get(finetune_config, "visual_env", False)
        if not visual_env:
            pass # TODO: restore this code accordingly
            """

            _obs = train_dataset.dataset[
                "observations"
            ]  # assuming dataset is available here
            filtered_pbs = _obs[combined_filter.astype(bool)]
            buf = io.BytesIO()
            plt.scatter(_obs[:5000, 0], _obs[:5000, 1])
            plt.scatter(filtered_pbs[:, 0], filtered_pbs[:, 1], alpha=0.5)
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            wandb.log({"ZFilter": wandb.Image(img_array)})
            del img, img_array, buf
            """
    del aggregated_filters

    gc.collect()


    visual_env = finetune_config.get("visual_env", False)
    if not visual_env:
        # TODO: restore?
        # make_plots(train_dataset, agent, goal, "post", _cfg_get(finetune_config, "saw", False))
        pass

    return traj, info, finetune_stats, render


def gc_ttt_critic_free(
    train_dataset: GCDataset,
    agent: GCAgent,
    env,
    observation,
    config: GCTTTConfig,
    goal,
    goal_frame,
    should_render: bool,
    _filter,
    num_ttt_steps: Optional[int] = None
):
    # GC-TTT without critic
    traj = defaultdict(list)
    finetune_stats = defaultdict(list)
    info = None

    finetune_config = config.finetune

    if _cfg_get(finetune_config, "num_steps", 0):

        current_finetune_config = make_current_config(finetune_config, env)
        if num_ttt_steps is not None:
            num_steps = int(num_ttt_steps) if _filter.sum() else 0
        else:
            num_steps = int(_cfg_get(finetune_config, "num_steps", 0)) if _filter.sum() else 0

        t_finetune_start = time.time()
        for i in range(num_steps):
            # Sample a batch from the dataset using the filter.
            # The batch will contain only the samples that match the filter.
            batch = train_dataset.active_sample(
                _cfg_get(finetune_config, "batch_size"),
                _filter,
                goal,
                _cfg_get(finetune_config, "ratio"),
                _cfg_get(finetune_config, "fix_actor_goal"),
                finetune_kwargs=current_finetune_config,
            )
            # Update the agent with the sampled batch.
            # Note: in the original code (as here) the optimizer is never reset, consider using reset_inner_opt=i==0
            agent, info = agent.update(batch, finetuning=True, reset_inner_opt=False)
            add_to(finetune_stats, flatten(info))
        print(f"time for finetuning {num_steps} steps", time.time() - t_finetune_start)

    # Plotting values and actions after fine-tuning
    if not _cfg_get(finetune_config, "visual_env", False):
        # TODO: restore?
        # make_plots(train_dataset, agent, goal, "post", _cfg_get(finetune_config, "saw", False))
        pass

    actor_fn = supply_rng(
        agent.sample_actions,
        rng=jax.random.PRNGKey(np.random.randint(0, 2**32)),
    )

    # Rollout the episode with the updated agent
    done = False
    step = 0
    render = []
    rollout_start_time = time.time()
    while not done:
        next_observation, action, reward, terminated, truncated, info = actor_step(
            actor_fn, observation, env, config, goal
        )
        step += 1
        done = terminated or truncated or step >= 3000

        if should_render and (step % config.video_frame_skip == 0 or done):
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
    rollout_end_time = time.time()
    print(f"Rollout took {rollout_end_time - rollout_start_time:.4f} seconds")

    return traj, info, finetune_stats, render


def evaluate(
    agent: GCAgent,
    env,
    config: GCTTTConfig,
    task_id: int,
    train_dataset: GCDataset,
    num_ttt_steps: Optional[int] = None
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

    agent_snapshot = snapshot_agent(agent)

    trajs = []
    stats = defaultdict(list)

    renders = []

    # Create progress bar
    pbar = trange(config.eval_episodes + config.video_episodes, desc=f"Task {task_id}")
    success_count = 0

    for i in pbar:
        agent_ft = clone_agent(agent)             # working copy for finetuning

        # Render only video episodes
        should_render = i >= config.eval_episodes

        start_state, info = env.reset(
            options=dict(task_id=task_id, render_goal=should_render),
            seed=i
        )
        # For each task (determined by task_id) there is a different goal
        goal = info.get("goal")
        goal_frame = info.get("goal_rendered")

        # Debug print: uncomment if needed
        # print(f"Start state: {start_state} and goal: {goal}")

        # Prepare filter
        _filter = train_dataset.prepare_active_sample(agent, start_state, goal, config.finetune, log_filter=False)[0]

        # Simple script to plot critic and policy output in a 2D environment
        # We sample a batch from the training dataset, then calculate both values and actions on sampled batch
        # Plotting values and actions before fine-tuning
        visual_env = _cfg_get(config.finetune, "visual_env", False)
        if not visual_env:
            # TODO: restore?
            # make_plots(train_dataset, agent, goal, "pre", _cfg_get(finetune_config, "saw", False))
            pass

        recursive_mdp = config.finetune.get("filter_by_recursive_mdp", False)

        if recursive_mdp:
            # Default GC-TTT (with critic)
            # - recursive_mdp = True
            traj, info, finetune_stats, render = gc_ttt_critic(
                train_dataset,
                agent_ft,
                env,
                observation=start_state,
                goal=goal,
                config=config,
                goal_frame=goal_frame,
                should_render=should_render,
                num_ttt_steps=num_ttt_steps
            )
        else:
            traj, info, finetune_stats, render = gc_ttt_critic_free(
                train_dataset,
                agent_ft,
                env,
                start_state,
                config,
                goal,
                goal_frame,
                should_render,
                _filter=_filter,
                num_ttt_steps=num_ttt_steps
            )

        if i < config.eval_episodes:
            # Extract success from info
            success = info.get("success", False)
            if success:
                success_count += 1

            # Update progress bar with success info
            success_rate = success_count / (i + 1)
            success_str = "✓" if success else "✗"
            pbar.set_postfix({
                'success': f"{success_str} {success_rate:.1%} ({success_count}/{i+1})"
            })

            # print(info)
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

        # Reset agent state after each episode
        agent = restore_agent_from_snapshot(agent, agent_snapshot)

    finetune_stats = {"finetune/" + k: v for k, v in finetune_stats.items()}
    add_to(stats, finetune_stats)

    # Aggregate statistics over eval_episodes
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
