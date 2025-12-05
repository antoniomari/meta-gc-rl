#!/usr/bin/env python3
"""
Interactive visualization script for GC-TTT maze environments.

This script allows you to:
1. Load a trained agent
2. Run episodes and visualize the agent's trajectory in the maze
3. See the agent's path, goal, and optionally value function
4. Save videos or show interactive plots

Usage:
    python visualize_episode.py --config config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000
    python visualize_episode.py --config config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --task_id 1 --show_value
    python visualize_episode.py --config config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --save_video output.mp4
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import jax
import jax.numpy as jnp
from collections import defaultdict

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_async_all_gather=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import gymnasium as gym
from agents import agents
from utils.datasets import Dataset, GCDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent
from utils.config import GCTTTConfig, load_config
from agents.gcagent import GCAgent, MetaGCAgent
from utils.evaluation import supply_rng


def get_agent_position(observation, env_name):
    """Extract agent position from observation based on environment type."""
    if 'pointmaze' in env_name.lower():
        # For pointmaze, first 2 dimensions are x, y position
        return observation[:2]
    elif 'antmaze' in env_name.lower():
        # For antmaze, first 2 dimensions are x, y position
        return observation[:2]
    elif 'humanoidmaze' in env_name.lower():
        # For humanoidmaze, first 2 dimensions are x, y position
        return observation[:2]
    else:
        # Default: assume first 2 dimensions
        return observation[:2]


def visualize_trajectory_2d(env, agent, config, task_id=1, num_ttt_steps=0,
                            show_value=False, save_video=None, interactive=True, train_dataset=None):
    """
    Visualize an episode trajectory in 2D for maze environments.

    Args:
        env: Gymnasium environment
        agent: Trained agent
        config: Configuration object
        task_id: Task ID to evaluate
        num_ttt_steps: Number of test-time fine-tuning steps
        show_value: Whether to show value function heatmap
        save_video: Path to save video (None for no save)
        interactive: Whether to show interactive plot
    """
    # Reset environment with task
    obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
    goal = info.get('goal', obs[:2])  # Fallback to obs[:2] if goal not in info

    # Get maze map if available
    maze_map = None
    if hasattr(env.unwrapped, 'maze_map'):
        maze_map = env.unwrapped.maze_map
        maze_unit = getattr(env.unwrapped, '_maze_unit', 4.0)
        offset_x = getattr(env.unwrapped, '_offset_x', 4)
        offset_y = getattr(env.unwrapped, '_offset_y', 4)

    # Run episode
    print(f"Running episode with task_id={task_id}, num_ttt_steps={num_ttt_steps}")

    # Fine-tune if needed
    if num_ttt_steps > 0 and train_dataset is not None:
        from utils.evaluation import gc_ttt_critic, gc_ttt_critic_free

        try:
            if hasattr(config.finetune, 'filter_by_recursive_mdp') and config.finetune.filter_by_recursive_mdp:
                traj, info, finetune_stats, render = gc_ttt_critic(
                    train_dataset, agent, env, obs, config, goal, None, True, num_ttt_steps
                )
            else:
                traj, info, finetune_stats, render = gc_ttt_critic_free(
                    train_dataset, agent, env, obs, config, goal, None, True, num_ttt_steps
                )

            # Extract trajectory
            positions = []
            for i, obs_t in enumerate(traj.get('observation', [])):
                pos = get_agent_position(obs_t, config.env_name)
                positions.append(pos)

            if len(positions) == 0:
                print("Warning: No trajectory data collected. Running simple rollout...")
                positions = run_simple_rollout(env, agent, config, goal, obs)
        except Exception as e:
            print(f"Error during fine-tuning visualization: {e}")
            print("Falling back to simple rollout...")
            positions = run_simple_rollout(env, agent, config, goal, obs)

        try:
            if hasattr(config.finetune, 'filter_by_recursive_mdp') and config.finetune.filter_by_recursive_mdp:
                traj, info, finetune_stats, render = gc_ttt_critic(
                    train_dataset, agent, env, obs, config, goal, None, True, num_ttt_steps
                )
            else:
                traj, info, finetune_stats, render = gc_ttt_critic_free(
                    train_dataset, agent, env, obs, config, goal, None, True, num_ttt_steps
                )

            # Extract trajectory
            positions = []
            for i, obs_t in enumerate(traj.get('observation', [])):
                pos = get_agent_position(obs_t, config.env_name)
                positions.append(pos)

            if len(positions) == 0:
                print("Warning: No trajectory data collected. Running simple rollout...")
                positions = run_simple_rollout(env, agent, config, goal, obs)
        except Exception as e:
            print(f"Error during fine-tuning visualization: {e}")
            print("Falling back to simple rollout...")
            positions = run_simple_rollout(env, agent, config, goal, obs)
    else:
        # Simple rollout without fine-tuning
        positions = run_simple_rollout(env, agent, config, goal, obs)

    if len(positions) == 0:
        print("Error: No positions collected. Cannot visualize.")
        return

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw maze if available
    if maze_map is not None:
        draw_maze(ax, maze_map, maze_unit, offset_x, offset_y)

    # Show value function if requested
    if show_value and hasattr(agent, 'network'):
        show_value_function(ax, agent, env, goal, config, maze_map, maze_unit, offset_x, offset_y)

    # Plot trajectory
    positions = np.array(positions)
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7, label='Agent Path')
    ax.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='o',
               edgecolors='black', linewidths=2, label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='s',
               edgecolors='black', linewidths=2, label='End', zorder=5)

    # Plot goal
    goal_pos = get_agent_position(goal, config.env_name)
    ax.scatter(goal_pos[0], goal_pos[1], c='orange', s=300, marker='*',
               edgecolors='black', linewidths=2, label='Goal', zorder=5)

    # Add arrows to show direction
    if len(positions) > 1:
        for i in range(0, len(positions)-1, max(1, len(positions)//20)):
            dx = positions[i+1, 0] - positions[i, 0]
            dy = positions[i+1, 1] - positions[i, 1]
            ax.arrow(positions[i, 0], positions[i, 1], dx*0.8, dy*0.8,
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Episode Visualization - Task {task_id} (TTT Steps: {num_ttt_steps})', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Calculate success
    final_pos = positions[-1]
    distance_to_goal = np.linalg.norm(final_pos - goal_pos)
    success = distance_to_goal < 1.0  # Adjust threshold as needed

    ax.text(0.02, 0.98, f'Distance to goal: {distance_to_goal:.2f}\nSuccess: {success}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_video:
        # Save as image
        plt.savefig(save_video.replace('.mp4', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_video.replace('.mp4', '.png')}")

    if interactive:
        plt.show()
    else:
        plt.close()


def draw_maze(ax, maze_map, maze_unit, offset_x, offset_y):
    """Draw the maze walls on the plot."""
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 1:  # Wall
                x = (j - offset_x) * maze_unit
                y = (maze_map.shape[0] - 1 - i - offset_y) * maze_unit
                rect = Rectangle((x, y), maze_unit, maze_unit,
                               facecolor='gray', edgecolor='black', linewidth=1)
                ax.add_patch(rect)


def show_value_function(ax, agent, env, goal, config, maze_map, maze_unit, offset_x, offset_y):
    """Show value function as a heatmap."""
    try:
        # Create a grid of positions
        if maze_map is not None:
            x_min = (0 - offset_x) * maze_unit
            x_max = (maze_map.shape[1] - offset_x) * maze_unit
            y_min = (0 - offset_y) * maze_unit
            y_max = (maze_map.shape[0] - offset_y) * maze_unit
        else:
            x_min, x_max = -10, 10
            y_min, y_max = -10, 10

        x = np.linspace(x_min, x_max, 50)
        y = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x, y)

        # Get value for each position
        values = []
        goal_pos = get_agent_position(goal, config.env_name)

        for y_val in y:
            row_values = []
            for x_val in x:
                # Create observation (assuming first 2 dims are position)
                obs = np.zeros(env.observation_space.shape[0])
                obs[0] = x_val
                obs[1] = y_val

                # Get value
                try:
                    value = agent.network.select('value')(
                        obs.reshape(1, -1),
                        goal_pos.reshape(1, -1)
                    )
                    if isinstance(value, (list, tuple)):
                        value = value[0]
                    row_values.append(float(value))
                except:
                    row_values.append(0.0)
            values.append(row_values)

        values = np.array(values)

        # Plot heatmap
        im = ax.contourf(X, Y, values, levels=20, alpha=0.3, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Value')
    except Exception as e:
        print(f"Could not show value function: {e}")


def run_simple_rollout(env, agent, config, goal, initial_obs):
    """Run a simple rollout without fine-tuning."""
    positions = []
    obs = initial_obs
    done = False
    step = 0
    max_steps = 1000

    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(0))

    while not done and step < max_steps:
        # Get action
        action = actor_fn(obs, goal)
        if isinstance(action, (list, tuple)):
            action = action[0]
        action = np.array(action)
        action = np.clip(action, -1, 1)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Record position
        pos = get_agent_position(obs, config.env_name)
        positions.append(pos)

        obs = next_obs
        done = terminated or truncated
        step += 1

    # Record final position
    if not done:
        pos = get_agent_position(obs, config.env_name)
        positions.append(pos)

    return positions


def visualize_mujoco_render(env, agent, config, task_id=1, num_ttt_steps=0,
                            save_video=None, fps=10):
    """
    Visualize episode using MuJoCo renderer (for 3D visualization).

    Args:
        env: Gymnasium environment with MuJoCo renderer
        agent: Trained agent
        config: Configuration object
        task_id: Task ID to evaluate
        num_ttt_steps: Number of test-time fine-tuning steps
        save_video: Path to save video
        fps: Frames per second for video
    """
    # Set render mode
    if hasattr(env, 'render_mode'):
        env.render_mode = 'rgb_array'

    # Reset environment
    obs, info = env.reset(options=dict(task_id=task_id, render_goal=True))
    goal = info.get('goal', obs[:2])
    goal_frame = info.get('goal_rendered')

    frames = []
    if goal_frame is not None:
        frames.append(goal_frame)

    # Run episode (similar to evaluation code)
    done = False
    step = 0
    max_steps = 1000

    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(0))

    while not done and step < max_steps:
        # Get action
        action = actor_fn(obs, goal)
        if isinstance(action, (list, tuple)):
            action = action[0]
        action = np.array(action)
        action = np.clip(action, -1, 1)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Render
        if step % config.video_frame_skip == 0 or done:
            frame = env.render()
            if frame is not None:
                if goal_frame is not None:
                    frames.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    frames.append(frame)

        obs = next_obs
        done = terminated or truncated
        step += 1

    # Save video if requested
    if save_video and frames:
        try:
            import imageio
            imageio.mimsave(save_video, frames, fps=fps)
            print(f"Saved video to {save_video}")
        except ImportError:
            print("imageio not installed. Install with: pip install imageio")
        except Exception as e:
            print(f"Error saving video: {e}")

    return frames


def main():
    parser = argparse.ArgumentParser(
        description='Visualize episodes in GC-TTT maze environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 2D visualization
  python visualize_episode.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000

  # With value function heatmap
  python visualize_episode.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --show_value

  # With test-time fine-tuning
  python visualize_episode.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --num_ttt_steps 50

  # Save as image
  python visualize_episode.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --save_image output.png

  # MuJoCo 3D rendering
  python visualize_episode.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --mujoco_render --save_video output.mp4
        """
    )
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--restore_path', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--restore_epoch', type=int, required=True, help='Epoch to restore')
    parser.add_argument('--task_id', type=int, default=1, help='Task ID to visualize')
    parser.add_argument('--num_ttt_steps', type=int, default=0, help='Number of TTT steps')
    parser.add_argument('--show_value', action='store_true', help='Show value function heatmap')
    parser.add_argument('--save_video', type=str, default=None, help='Path to save video')
    parser.add_argument('--save_image', type=str, default=None, help='Path to save image')
    parser.add_argument('--mujoco_render', action='store_true', help='Use MuJoCo renderer (3D)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--interactive', action='store_true', default=True, help='Show interactive plot')

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    # Create environment
    env, train_dataset, val_dataset = make_env_and_datasets(
        cfg.env_name, cfg.data_ratio, frame_stack=cfg.agent.get('frame_stack')
    )
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    # Create agent
    example_batch = train_dataset.sample(1)
    if cfg.agent.get('discrete', False):
        example_batch['actions'] = np.full_like(
            example_batch['actions'], env.action_space.n - 1
        )

    agent_class = agents[cfg.agent['agent_name']]
    agent = agent_class.create(
        cfg.seed,
        example_batch['observations'],
        example_batch['actions'],
        cfg.agent,
        cfg.train_steps,
    )

    # Restore agent
    print(f"Loading agent from {args.restore_path} at epoch {args.restore_epoch}")
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)

    # Visualize
    if args.mujoco_render:
        frames = visualize_mujoco_render(
            env, agent, cfg, args.task_id, args.num_ttt_steps, args.save_video
        )
        if args.save_video:
            print(f"Video saved to {args.save_video}")
    else:
        # Create dataset for fine-tuning if needed
        train_dataset_for_viz = None
        if args.num_ttt_steps > 0:
            dataset_class_name = cfg.agent.get('dataset_class', 'GCDataset')
            if dataset_class_name == 'GCDataset':
                train_dataset_for_viz = GCDataset(train_dataset, cfg.agent)
            else:
                train_dataset_for_viz = train_dataset

        visualize_trajectory_2d(
            env, agent, cfg, args.task_id, args.num_ttt_steps,
            args.show_value, args.save_image or args.save_video, args.interactive, train_dataset_for_viz
        )


if __name__ == '__main__':
    main()

