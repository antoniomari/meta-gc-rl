import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import pandas as pd
from typing import Dict, Tuple, Any
from collections import defaultdict

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
    cfg,
    test_batch=None
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

    # Plot test batch points if provided
    if test_batch is not None and 'observations' in test_batch:
        test_obs = test_batch['observations']
        test_obs_xy = test_obs[:, :2] if len(test_obs) > 0 else np.zeros((0, 2))
        if len(test_obs_xy) > 0:
            ax.scatter(test_obs_xy[:, 0]+offset_x * maze_unit, test_obs_xy[:, 1]+offset_y * maze_unit,
                    c='red', alpha=0.6, s=15, label=f'Test batch ({len(test_obs_xy)})', zorder=5)

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


# Pattern for parsing IQL experiment group names
IQL_NAME_PATTERN = (
    r"^"
    r"(?:(?P<is_finetuned>FT)-)?"
    # Match optional ALL-GOALS, ALLGOALS, ALL, TEST-GOALS, TESTGOALS, or TEST (with or without -GOALS)
    r"(?:(?P<goals>ALL|TEST)(?:-GOALS)?-)?"
    r"(?:(?P<env>antmaze)-)?"
    r"(?:(?P<policy_extraction>ddpgbc)-)?"
    r"(?P<pretraining>PT|FX|RX|PTA|RXA|FXA|RXAFIX|RXAMOFIX)"
    r"(?:-(?P<inner_steps>\d+)-(?P<meta_batch_size>\d+))?"
    r"(?:-mc(?P<mc_quantile>[\d.]+))?"
    r"(?:-m_(?P<merging_eps>[\d.]+))?"
    r"(?:-lr(?P<lr>(?:[\d.]+|[\d.]+e[+\-]?\d+)))?"
    r"(?:-ilr(?P<ilr>(?:[\d.]+|[\d.]+e[+\-]?\d+)))?"
    r"$"
)


def parse_group_name(group_name: str):
    """Parse a group name string and extract experiment parameters."""
    match = re.match(IQL_NAME_PATTERN, group_name)

    if match:
        params_dict = match.groupdict()

        if 'goals' not in params_dict:
            params_dict['goals'] = "TEST"
        if 'env' not in params_dict or params_dict['env'] is None:
            params_dict['env'] = "pointmaze"
        if 'randbatch' in params_dict and params_dict['randbatch'] is not None:
            params_dict['goals'] = "RANDOM"
            del params_dict['randbatch']

        # Default mc_quantile is 0.2 (optimality)
        if 'mc_quantile' not in params_dict or params_dict['mc_quantile'] is None:
            params_dict['mc_quantile'] = 0.2
        else:
            params_dict['mc_quantile'] = float(params_dict['mc_quantile'])

        if 'lr' not in params_dict or params_dict['lr'] is None:
            params_dict['lr'] = 3e-04
        else:
            params_dict['lr'] = float(params_dict['lr'])
        if 'ilr' not in params_dict or params_dict['ilr'] is None:
            params_dict['ilr'] = 3e-04
        else:
            params_dict['ilr'] = float(params_dict['ilr'])

        if params_dict['pretraining'] == "RX":
            params_dict['pretraining'] = "REPTILE"
        elif params_dict['pretraining'] == "FX":
            params_dict['pretraining'] = "FOMAML"
        elif params_dict['pretraining'] == "RXA":
            params_dict['pretraining'] = "REPTILE-actor"
        elif params_dict['pretraining'] == "FXA":
            params_dict['pretraining'] = "FOMAML-actor"

        if 'merging_eps' not in params_dict or params_dict['merging_eps'] is None:
            if params_dict['pretraining'] == "REPTILE":
                params_dict['merging_eps'] = 1.0
        else:
            params_dict['merging_eps'] = float(params_dict['merging_eps'])

        # Convert numeric fields to int if present
        for k in ['inner_steps', 'meta_batch_size']:
            if k in params_dict and params_dict[k] is not None:
                params_dict[k] = int(params_dict[k])

        print(params_dict)
        return params_dict
    # If we get here, we failed to parse the group name
    return {}


def fetch_group_runs(
    group_name: str,
    download_data: bool = False,
    reset: bool = False,
    root_dir: str = "results_eval"
) -> Tuple[Dict[str, Any], Dict[int, pd.DataFrame]]:
    """
    Fetch all runs for a given group name and return settings and results.

    Results are collected from a directory structured as:
    notebooks/results_eval/{group_name}/results_seed{N}.csv

    Args:
        group_name: Name of the experiment group
        download_data: Unused parameter (kept for compatibility)
        reset: If True, look for results_reset_seed*.csv files instead of results_seed*.csv
        root_dir: Root directory for results. If None, defaults to "notebooks/results_eval"
                 (relative to current working directory)

    Returns:
        tuple[dict, dict]: A tuple containing two dictionaries.
            - The first dictionary contains the settings extracted from the group name.
            - The second dictionary contains the results for each run, keyed by seed.
    """
    # Parse settings from group_name
    settings = parse_group_name(group_name)
    results_dir = os.path.join(root_dir, group_name)
    results = {}

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if reset:
        result_files = glob.glob(os.path.join(results_dir, "results_reset_seed[0-9]*.csv"))
    else:
        result_files = glob.glob(os.path.join(results_dir, "results_seed[0-9]*.csv"))
    if not result_files:
        raise FileNotFoundError(f"No result files found in: {results_dir}")

    for file in result_files:
        # Expect filename contains ...results_seedN.csv
        basename = os.path.basename(file)
        try:
            seed_str = basename.split(".csv")[0][-1]
            seed = int(seed_str)
        except Exception as e:
            # Skip files with unexpected naming
            continue
        try:
            df = pd.read_csv(file)
            # Try to convert numeric columns to float
            for col in df.columns:
                if col.endswith("_success") or col in ["overall_success"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                if col in ["step", "TTT_steps"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")
            results[seed] = df
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            results[seed] = pd.DataFrame()  # put empty so later code will work

    return settings, results


def get_mean_std_for_all_steps(settings: dict, results: dict[int, pd.DataFrame], result_col="overall_success"):
    """
    Instead of taking group_name and results, takes settings and results as from fetch_group_runs.
    """
    # Collect all unique steps across all seeds
    all_steps = sorted(set(np.concatenate([df['step'].values for df in results.values() if not df.empty])))
    if not all_steps:
        raise ValueError(f"No data for group: {settings}")

    # Build a matrix: rows=seeds, columns=steps, values=overall_success (NaN if missing)
    seed_list = sorted(results.keys())
    overall_success_matrix = np.full((len(seed_list), len(all_steps)), np.nan)

    for i, seed in enumerate(seed_list):
        df = results[seed]
        if df.empty:
            continue
        step_to_success = dict(zip(df['step'], df[result_col]))
        for j, step in enumerate(all_steps):
            if step in step_to_success:
                overall_success_matrix[i, j] = step_to_success[step]

    # Compute mean and std across seeds (ignore NaNs)
    mean_success = np.nanmean(overall_success_matrix, axis=0)
    std_success = np.nanstd(overall_success_matrix, axis=0)

    return all_steps, mean_success, std_success


def plot_overall_success_for_groups(
    groups_dict,
    baselines: dict[int, float] = None,
    result_col: str = "overall_success",
    reset: bool = False,
    root_dir: str = "results_eval"
):
    """
    Plot the mean and standard deviation of a result metric (e.g., overall_success)
    as a function of training steps for several experiment groups.

    For each unique TTT_steps value found in the results of a group, plot a separate line (mean ± std across seeds).
    Shading denotes ± one standard deviation. Optionally, plot horizontal lines for baselines.

    Args:
        groups_dict: dictionary where each key is a label and each value is a dictionary with a "groups" key
            containing a list of group names.
        baselines (dict[int, float], optional): horizontal lines for given TTT_steps values.
        result_col (str, optional): result column to plot.

    Behavior:
        - For each TTT_steps value present among results, plot a separate line per group.
    """
    plt.figure(figsize=(8, 5))
    pretraining = set()

    for idx, (group_label, group_info) in enumerate(groups_dict.items()):
        group_name = group_info["groups"][0]
        settings, results = fetch_group_runs(group_name, reset=reset, root_dir=root_dir)
        pretraining.add(settings.get("pretraining", None))
        color_base = f"C{idx % 10}"

        # Determine unique TTT_steps present across all seeds' DataFrames
        ttt_to_indices = {}
        for seed, df in results.items():
            if df.empty or 'TTT_steps' not in df.columns:
                continue
            for ttt in df['TTT_steps'].unique():
                ttt_to_indices.setdefault(ttt, set()).add(seed)

        # Plot one line per TTT_steps value found in results
        for j, (ttt_steps_value, seed_set) in enumerate(sorted(ttt_to_indices.items())):
            # Collect all seeds with this TTT_steps value; build results subset
            seeds_with_ttt = list(seed_set)
            if not seeds_with_ttt:
                continue
            # Aggregate df for only these seeds and only rows where TTT_steps == ttt_steps_value
            subgroup_results = {
                seed: df[df['TTT_steps'] == ttt_steps_value] if not df.empty else pd.DataFrame()
                for seed, df in results.items() if seed in seeds_with_ttt
            }
            # Possibly different color for each line of this group
            color = color_base if len(ttt_to_indices) == 1 else f"C{(idx*4 + j) % 10}"

            # Optionally plot baseline for this TTT_steps value
            if baselines is not None and ttt_steps_value in baselines:
                plt.axhline(
                    y=baselines[ttt_steps_value],
                    color=color,
                    linestyle='--'
                )

            # Aggregate results across chosen seeds and steps
            all_steps, mean_success, std_success = get_mean_std_for_all_steps(
                settings, subgroup_results, result_col=result_col)
            label = f"{group_label} (TTT={ttt_steps_value})" if ttt_steps_value is not None else group_label
            plt.plot(
                all_steps, mean_success, label=label, color=color
            )
            plt.fill_between(
                all_steps, mean_success - std_success, mean_success + std_success,
                color=color, alpha=0.2)

    plt.xlabel('Pretraining Steps')
    plt.ylabel('Overall Success')
    plt.title(f"Algo: {pretraining}")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_best_mean_success_and_std(settings: dict, results: dict[int, pd.DataFrame], result_col="overall_success") -> Tuple[float, float]:
    """
    Compute the maximum of the mean overall_success across steps.

    Args:
        settings: dict: settings of the group (from parse_group_name)
        results: dict[int, pd.DataFrame]: dict of seed to dataframe of results
    Returns:
        tuple[float, float]: (max_mean_success, ci95_at_max)
    """
    all_steps = sorted(set(np.concatenate([df['step'].values for df in results.values() if not df.empty])))
    if not all_steps:
        raise AssertionError(f"No data for group: {settings}")

    seed_list = sorted(results.keys())
    overall_success_matrix = np.full((len(seed_list), len(all_steps)), np.nan)

    for i, seed in enumerate(seed_list):
        df = results[seed]
        if df.empty:
            continue
        step_to_success = dict(zip(df['step'], df[result_col]))
        for j, step in enumerate(all_steps):
            if step in step_to_success:
                overall_success_matrix[i, j] = step_to_success[step]

    mean_success = np.nanmean(overall_success_matrix, axis=0)
    n_valid = np.sum(~np.isnan(overall_success_matrix), axis=0)
    stderr_success = np.nanstd(overall_success_matrix, axis=0) / np.sqrt(np.maximum(n_valid, 1))
    ci95_success = 1.96 * stderr_success

    if np.all(np.isnan(mean_success)):
        raise AssertionError(f"All mean_success are NaN for group: {settings}")
    max_idx = np.nanargmax(mean_success)
    max_mean_success = mean_success[max_idx]
    ci95_at_max = ci95_success[max_idx]

    return max_mean_success, ci95_at_max


def get_algo_color(group_key):
    """Get color for algorithm based on group key."""
    # If geom is False, override color to black
    if group_key[4] is False or group_key[5] == 1.0:
        return "black"
    if group_key[1] == "FOMAML":
        return "green"
    elif group_key[1] == "MAML":
        return "blue"
    elif group_key[1] == "PT":
        return "purple"
    elif group_key[1] == "REPTILE":
        return "red"
    return None


def get_linestyle_and_dot_type(group_key):
    """Get linestyle and dot type based on group key."""
    if group_key[0] == "ALL":
        return "solid", "o"
    elif group_key[0] == "TEST":
        return "dashed", "s"
    elif group_key[0] == "RANDOM":
        return "dotted", "D"
    return "solid", "o"


def plot_max_overall_success_vs_ttt_steps(
    group_results: dict,
    result_col="overall_success",
    verbose: bool = False,
    title: str = None,
    reset: bool = False,
    xlim: tuple[int, int] = None,
    ylim: tuple[int, int] = None,
    error_bars: bool = False,
    root_dir: str = "results_eval",
    ax=None,
    figsize=(6, 4),
    show=True,
    print_table: bool = False,
    figure_path: str = None,
    use_last_checkpoint: bool = False  # TODO: implement this version
):
    """
    For all (settings,results) pairs in group_results, plot the maximum of the mean overall_success across steps.
    Groups are defined by the tuple (goals, pretraining, inner_steps, meta_batch_size).
    For each group, plot a line: X-axis is TTT_steps, Y-axis is max mean overall_success.
    The label for each line is the group fields.
    Error bars (std across seeds at the max mean step) are shown using fill_between by default,
    or using simple error bars if error_bars=True.

    If group_results is a dict, you can provide value as either a list of (settings, results) tuples
    or a dict with keys:
        - 'groups': list of (settings, results) tuples (required)
        - 'color', 'linestyle', 'dot_type': optional (for plot line customization)

    Args:
        error_bars: If True, show simple error bars instead of fill_between shaded area.
        ax: matplotlib Axes object. If provided, plot on this axes instead of creating a new figure.
        figsize: Figure size tuple (width, height) when creating a new figure.
        show: If True, call plt.show() at the end. Set to False when creating subplots.
        print_table: If True, print a formatted table with results for TTT_steps (0, 10, 20, 50, 100, 200, max).
    """
    group_points = defaultdict(list)
    group_plotopts = {}

    def append_group_points(settings: dict, results: dict[int, pd.DataFrame], label: str = None):
        # Extract grouping fields

        # "ttt_steps" is a column only in the latest implementation format
        first_df = next(iter(results.values()))
        for ttt_steps in [0, 5, 10, 20, 50, 100, 200]:
            # Overwrite TTT_steps in settings
            settings["TTT_steps"] = ttt_steps
            # Copy all DFs filtered by ttt_steps
            filtered_results = {k: v[v['TTT_steps'] == ttt_steps] for k, v in results.items()}
            if len(filtered_results) == 0 or all(df.empty for df in filtered_results.values()):
                if verbose:
                    print(f"[Skipping] No filtered results for TTT_steps={ttt_steps}, group={label}")
                continue
            if verbose:
                print(f"Processing group={label} TTT_steps={ttt_steps}")
            max_mean_success, std_at_max = get_best_mean_success_and_std(settings, filtered_results, result_col=result_col)
            n_seeds = len([k for k, v in filtered_results.items() if not v.empty])
            group_points[label].append((ttt_steps, max_mean_success, std_at_max, label, n_seeds))

    # If group_results is a dict, extract the values and custom plotting options if present
    for idx, (group_label, group_info) in enumerate(group_results.items()):
        group_name = group_info["groups"][0]
        settings, results = fetch_group_runs(group_name, reset=reset, root_dir=root_dir)
        append_group_points(settings, results, label=group_label)
        plotopts = {}
        for k in ("color", "linestyle", "dot_type"):
            if k in group_info:
                plotopts[k] = group_info[k]
        if plotopts:
            group_plotopts[group_label] = plotopts

    if verbose:
        print(group_points)

    # Create figure/axes if not provided
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # Print table if requested
    if print_table:
        # Define TTT_steps columns (excluding 5, as user specified 0, 10, 20, 50, 100, 200, max)
        ttt_columns = [0, 10, 20, 50, 100, 200]

        # Build table data
        table_data = []
        for label in sorted(group_points.keys()):
            points = group_points[label]
            row_data = {"Label": label}

            # Collect data for each TTT_steps value
            ttt_data = {}
            for point in points:
                ttt_steps, max_mean_success, std_at_max, _, n_seeds = point
                if ttt_steps in ttt_columns:
                    # Compute standard error
                    std_error = std_at_max / np.sqrt(n_seeds) if n_seeds > 0 else 0.0
                    ttt_data[ttt_steps] = (max_mean_success, std_error)

            # Fill in values for each column
            for ttt in ttt_columns:
                if ttt in ttt_data:
                    mean_val, std_err = ttt_data[ttt]
                    row_data[str(ttt)] = f"{mean_val:.4f} ± {std_err:.4f}"
                else:
                    row_data[str(ttt)] = "N/A"

            # Compute max across all TTT_steps for this label
            relevant_points = [p for p in points if p[0] in ttt_columns]
            if relevant_points:
                # Find the point with maximum mean
                max_point = max(relevant_points, key=lambda x: x[1])
                max_mean = max_point[1]
                _, _, std_at_max, _, n_seeds = max_point
                max_std_err = std_at_max / np.sqrt(n_seeds) if n_seeds > 0 else 0.0
                row_data["max"] = f"{max_mean:.4f} ± {max_std_err:.4f}"
            else:
                row_data["max"] = "N/A"

            table_data.append(row_data)

        # Print formatted table
        print("\n" + "="*120)
        print("Results Table: Max Overall Success vs TTT_steps")
        print("="*120)

        # Header
        header = f"{'Label':<50}"
        for ttt in ttt_columns:
            header += f"  {ttt:>15}"
        header += f"  {'max':>15}"
        print(header)
        print("-"*120)

        # Rows
        for row in table_data:
            row_str = f"{row['Label']:<50}"
            for ttt in ttt_columns:
                val = row.get(str(ttt), "N/A")
                row_str += f"  {val:>15}"
            row_str += f"  {row.get('max', 'N/A'):>15}"
            print(row_str)

        print("="*120 + "\n")

    color_map = {}
    for idx, (label, points) in enumerate(sorted(group_points.items())):
        points_sorted = sorted(points, key=lambda x: x[0])
        ttt_steps_sorted = [p[0] for p in points_sorted]
        max_mean_success_sorted = [p[1] for p in points_sorted]
        if verbose:
            print(list(zip(ttt_steps_sorted, max_mean_success_sorted)))
        std_at_max_sorted = [p[2] for p in points_sorted]

        plotopts = group_plotopts.get(label, {})

        color = plotopts.get("color", None)
        if color is None:
            color = get_algo_color(label)
            if color is None:
                color = f"C{idx % 10}"
        color_map[label] = color

        linestyle = plotopts.get("linestyle", None)
        dot_type = plotopts.get("dot_type", None)
        if linestyle is None or dot_type is None:
            _auto_ls, _auto_dot = get_linestyle_and_dot_type(label)
            linestyle = linestyle if linestyle is not None else _auto_ls
            dot_type = dot_type if dot_type is not None else _auto_dot

        ax.plot(
            ttt_steps_sorted,
            max_mean_success_sorted,
            marker=dot_type,
            linestyle=linestyle,
            color=color,
            label=label,
            alpha=0.9,
            markeredgecolor="black",
        )
        lower = np.array(max_mean_success_sorted) - np.array(std_at_max_sorted)
        upper = np.array(max_mean_success_sorted) + np.array(std_at_max_sorted)
        if error_bars:
            ax.errorbar(
                ttt_steps_sorted,
                max_mean_success_sorted,
                yerr=std_at_max_sorted,
                color=color,
                alpha=0.7,
                capsize=3,
                capthick=1,
                linestyle='none',
            )
        else:
            ax.fill_between(
                ttt_steps_sorted,
                lower,
                upper,
                color=color,
                alpha=0.17,
            )

    ax.set_xlabel('TTT steps')
    ax.set_ylabel('Mean Success Rate')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'Max {result_col} vs TTT_steps')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=14)
    ax.grid(True)
    # Set fontsize for axis labels
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    # Set fontsize for tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)
    # Set fontsize for title if present
    if ax.get_title():
        ax.title.set_size(16)


    # Y axis in log scale
    if figure_path is not None:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        print(f"Saving figure to {figure_path}")
        plt.savefig(figure_path, bbox_inches='tight')
    if show:
        plt.show()

    return ax
