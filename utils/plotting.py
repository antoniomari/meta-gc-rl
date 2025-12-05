import numpy as np
import matplotlib.pyplot as plt

# Helper function to extract position from observation
def get_agent_position(observation, env_name):
    """Extract agent position from observation based on environment type."""
    if 'pointmaze' in env_name.lower() or 'antmaze' in env_name.lower() or 'humanoidmaze' in env_name.lower():
        # For maze environments, first 2 dimensions are x, y position
        return observation[:2]
    else:
        # Default: assume first 2 dimensions
        raise AssertionError("Environment is not a maze environment")

# Helper function to draw maze walls
def draw_maze(ax, maze_map, maze_unit, offset_x, offset_y):
    """Draw the maze walls on the plot."""
    from matplotlib.patches import Rectangle
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 1:  # Wall
                # Standard mapping: j is x (horizontal), i is y (vertical)
                x = j * maze_unit
                y = (i) * maze_unit
                rect = Rectangle((x, y), maze_unit, maze_unit,
                                 facecolor='gray', edgecolor='black', linewidth=1)
                ax.add_patch(rect)


def plot_maze_and_samples(
    train_dataset,
    task_filter,
    env,
    start_state,
    goal,
    cfg
):
    maze_map = env.unwrapped.maze_map
    maze_unit = getattr(env.unwrapped, '_maze_unit')
    offset_x = getattr(env.unwrapped, '_offset_x')
    offset_y = getattr(env.unwrapped, '_offset_y')
    print(f"Maze map shape: {maze_map.shape}")
    print(f"Maze unit: {maze_unit}, Offset: ({offset_x}, {offset_y})")

    filtered_obs = train_dataset.dataset['observations'][task_filter.astype(bool)]
    all_obs_sample = train_dataset.dataset['observations'][np.random.choice(len(train_dataset.dataset['observations']), 5000)]

    # Extract x, y positions
    all_obs_xy = all_obs_sample[:, :2]
    filtered_obs_xy = filtered_obs[:, :2] if len(filtered_obs) > 0 else np.zeros((0, 2))

    # Infer maze_unit using max/min x and y from all_obs_sample
    x_max, x_min = np.max(all_obs_xy[:, 0]), np.min(all_obs_xy[:, 0])
    y_max, y_min = np.max(all_obs_xy[:, 1]), np.min(all_obs_xy[:, 1])

    # Get start and goal positions
    start_xy = get_agent_position(start_state, cfg.env_name)
    goal_xy = get_agent_position(goal, cfg.env_name)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw maze walls using actual maze parameters
    draw_maze(ax, maze_map, maze_unit, offset_x, offset_y)

    # Set axis limits based on inferred maze_unit and center observed region
    # For 8x8 maze, offset can be recomputed for plotting, but if you want to stay close to the original logic,
    # try using the inferred maze_unit with existing offset_x and offset_y, but warn if those aren't defined.

    # Min in all_obs_sample
    offset_x = (0 - x_min/maze_unit) + 1
    offset_y = (0 - y_min/maze_unit) + 1

    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)

    # Plot all observations
    ax.scatter(all_obs_xy[:, 0]+offset_x * maze_unit, all_obs_xy[:, 1]+offset_y * maze_unit,
            c='lightgray', alpha=0.4, s=5, label='All observations (sample)')

    # Plot filtered samples
    if len(filtered_obs_xy) > 0:
        ax.scatter(filtered_obs_xy[:, 0]+offset_x * maze_unit, filtered_obs_xy[:, 1]+offset_y * maze_unit,
                c='blue', alpha=0.5, s=10, label=f'Filtered observations ({len(filtered_obs_xy)})')
    else:
        print("Warning: No observations passed the filter")

    # Plot start and goal positions
    ax.scatter(start_xy[0]+offset_x * maze_unit, start_xy[1]+offset_y * maze_unit,
            c='green', marker='s', s=200, edgecolors='black', linewidth=2,
            label='Start', zorder=10)
    ax.scatter(goal_xy[0]+offset_x * maze_unit, goal_xy[1]+offset_y * maze_unit,
            c='red', marker='*', s=300, edgecolors='black', linewidth=2,
            label='Goal', zorder=10)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Maze Environment and Filtered Samples\n'
                f'Filter: {len(filtered_obs_xy)}/{len(train_dataset.dataset["observations"])} samples '
                f'({len(filtered_obs_xy)/len(train_dataset.dataset["observations"])*100:.1f}%)',
                fontsize=14)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
