import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import pandas as pd
from typing import Dict, Tuple, Any, Optional, TypedDict
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
    root_dir: str = "results_eval"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
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

    # Concatenate results in a single dataframe with a "seed" column
    dfs = []
    for seed, df in results.items():
        if not df.empty:
            df = df.copy()
            df['seed'] = seed
            dfs.append(df)
    results_df = pd.concat(dfs, ignore_index=True)

    return settings, results_df



def plot_mean_success_over_checkpoints(
    groups_dict,
    baselines: dict[int, float] = None,
    result_col: str = "overall_success",
    root_dir: str = "results_eval",
    new_style: bool = False,
    figsize=(6, 4),
    fontsize: int = 18,
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
    assert len(groups_dict) == 1, "Only one configuration can be plotted at a time"
    plt.figure(figsize=figsize)
    ax = plt.gca()

    group_label = list(groups_dict.keys())[0]
    group_info = groups_dict[group_label]

    print(f"Group label: {group_label}")
    print(f"Group info: {group_info['groups']}")
    group_name = group_info["groups"][0]
    settings, results = fetch_group_runs(group_name, root_dir=root_dir)

    metrics_df = compute_mean_success_and_std_error(
        results,
        result_col=result_col,
        group_label=group_name,
    )

    # Print for each meta_step the area under the success curve
    for meta_steps, group in metrics_df.groupby("meta_steps"):
        print(f"Meta steps: {meta_steps}, Area under success curve: {compute_area_under_success_curve(group)}")


    if new_style:

        size_map = {
            0: 10,
            10: 30,
            20: 50,
            50: 70,
            100: 120,
            200: 170,
        }

        # Plot one line per TTT_steps value found in results
        for j, (ttt_steps, group) in enumerate(metrics_df.groupby("TTT_steps")):
            # Possibly different color for each line of this group
            color = group_info["color"]

            # Sort group by meta_steps
            group = group.sort_values("meta_steps")
            all_steps = group["meta_steps"].values
            mean_success = group["mean_success"].values
            std_error = group["std_error"].values
            # Scatter plot where marker size depends on TTT_steps
            # Example: size scales with TTT_steps (feel free to tune the scaling factor)
            plt.scatter(
                all_steps, mean_success,
                label=f"TTT={ttt_steps}",
                color=color,
                s=size_map[ttt_steps],
                alpha=0.7,
                edgecolors='k'
            )
            plt.fill_between(
                all_steps, mean_success - std_error, mean_success + std_error,
                color=color, alpha=0.15)

    else:
        # Plot one line per TTT_steps value found in results
        for j, (ttt_steps, group) in enumerate(metrics_df.groupby("TTT_steps")):
            # Possibly different color for each line of this group
            color = f"C{(idx*4 + j) % 10}"

            # Optionally plot baseline for this TTT_steps value
            if baselines is not None and ttt_steps in baselines:
                plt.axhline(
                    y=baselines[ttt_steps],
                    color=color,
                    linestyle='--'
                )

            # Sort group by meta_steps
            group = group.sort_values("meta_steps")
            all_steps = group["meta_steps"].values
            mean_success = group["mean_success"].values
            std_error = group["std_error"].values
            plt.plot(
                all_steps, mean_success, label=f"{group_label} (TTT={ttt_steps})", color=color
            )
            plt.fill_between(
                all_steps, mean_success - std_error, mean_success + std_error,
                color=color, alpha=0.15)

    plt.xlabel('Post-training Steps (Millions)')
    plt.ylabel('Mean Success Rate')
    plt.title(f"Algo: {group_label}")
    plt.legend()

    # Place the grid below the plot elements (so dots are on top)
    ax.set_axisbelow(True)
    plt.grid(True)

    # Divide xtick labels by 1e6 and format as decimal
    xticks = ax.get_xticks()
    xticklabels = [f"{x/1e6:g}" for x in xticks]
    ax.set_xticklabels(xticklabels)

    # Set fontsize for axis labels and ticks
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # Set fontsize for title if present
    if ax.get_title():
        ax.title.set_fontsize(fontsize)

    plt.show()


def compute_mean_success_and_std_error(
    results_df: pd.DataFrame,
    result_col="overall_success",
    allow_nan: bool = False,
    group_label: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Compute the maximum of the mean overall_success across steps.

    Args:
        group_label: str: label of the group
        results: dict[int, pd.DataFrame]: dict of seed to dataframe of results
    Returns:
        tuple[float, float]: (max_mean_success, ci95_at_max)
    """
    ttt_steps_list = results_df['TTT_steps'].unique()
    all_steps = sorted(set(results_df['step'].values))
    seed_list = sorted(results_df['seed'].unique())

    print(f"num_seeds: {len(results_df['seed'].unique())}, num_meta_steps: {len(results_df['step'].unique())}")
    # Overwrite TTT_steps in settings
    # Copy all DFs filtered by ttt_steps


    # I want a dataframe with columns: TTT_steps, meta_steps, mean_success, std_error, n_seeds
    results = []
    for meta_steps in all_steps:
        for ttt_steps in ttt_steps_list:

            # filtered_results has len(seed_list) rows
            filtered_results = results_df[(results_df['TTT_steps'] == ttt_steps) & (results_df['step'] == meta_steps)]
            overall_success_vector = np.full((len(seed_list),), np.nan)

            # Rows are seeds, columns are steps
            for i, seed in enumerate(seed_list):
                df = filtered_results[filtered_results['seed'] == seed]
                assert len(df) == 1, f"Expected 1 row for seed {seed} and meta_steps {meta_steps} and ttt_steps {ttt_steps} - {group_label}"
                overall_success_vector[i] = df[result_col].values[0]

            # TODO: print(overall_success_vector)
            if not allow_nan:
                assert np.all(np.isfinite(overall_success_vector)), "Overall success vector contains NaN or Inf"

            # Mean_success, n_valid and stderr_success have shape (len(all_steps),)
            mean_success = np.nanmean(overall_success_vector, axis=0)
            n_valid = np.sum(~np.isnan(overall_success_vector), axis=0)
            if not allow_nan:
                assert n_valid == len(seed_list), f"Expected {len(seed_list)} valid results for meta_steps {meta_steps} and ttt_steps {ttt_steps}"
            stderr_success = np.nanstd(overall_success_vector, axis=0) / np.sqrt(np.maximum(n_valid, 1))
            ci95_success = 1.96 * stderr_success

            results.append({
                "TTT_steps": ttt_steps,
                "meta_steps": meta_steps,
                "mean_success": mean_success,
                "std_error": stderr_success,
                "n_seeds": n_valid
            })

    return pd.DataFrame.from_records(results, columns=["TTT_steps", "meta_steps", "mean_success", "std_error", "n_seeds"])


def compute_area_under_success_curve(results_df: pd.DataFrame, up_to: Optional[int] = None) -> float:
    """
    Given a df like this
        TTT_steps  mean_success  std_error
    0          0      0.366667   0.037444
    1         10      0.690667   0.021026
    2         20      0.713333   0.007139
    3         50      0.717333   0.008911
    4        100      0.726667   0.016257
    5        200      0.718667   0.014153

    return the area under the success curve using the trapezoidal rule.
    """

    df = results_df.copy()
    df = df[df["TTT_steps"] <= up_to]

    df = results_df.sort_values("TTT_steps").reset_index(drop=True)
    return float(np.trapezoid(df["mean_success"].values, df["TTT_steps"].values)) / df["TTT_steps"].max()



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


class ExperimentInfo(TypedDict, total=False):
    groups: list[str]       # required: list of group/folder names to plot
    color: str              # required: line color (e.g. "black", "blue", "#FF0000")
    linestyle: str          # required: line style (e.g. "-", "--", "-.", ":")
    dot_type: str           # optional: marker type (e.g. "o", "x", "s", "D")


def plot_max_overall_success_vs_ttt_steps(
    group_results: dict[str, ExperimentInfo],
    result_col="overall_success",
    ttt_steps_list: list[int] = [0, 10, 20, 50, 100, 200],
    verbose: bool = False,
    title: str = None,
    xlim: tuple[int, int] = None,
    ylim: tuple[int, int] = None,
    error_bars: bool = False,
    root_dir: str = "results_eval",
    ax=None,
    figsize=(6, 4),
    show=True,
    print_table: bool = False,
    figure_path: str = None,
    use_best_auc_up_to: Optional[int] = None, # TODO: implement this version
    fontsize: int = 20,
    show_legend: bool = True,
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

    # Process each group as a separate line to plot
    for idx, (group_label, group_info) in enumerate[tuple[str, ExperimentInfo]](group_results.items()):
        group_name = group_info["groups"][0]
        settings, results_df = fetch_group_runs(group_name, root_dir=root_dir)
        # Compute mean success and std error for all TTT_steps

        # Columns: "TTT_steps", "meta_steps", "mean_success", "std_error", "n_seeds"
        metrics_df = compute_mean_success_and_std_error(
            results_df,
            result_col=result_col,
            group_label=group_label,
        )


        # For all meta_steps, compute area under the success curve and select meta_steps with max area
        if use_best_auc_up_to is not None:
            max_area = -np.inf
            best_meta_steps = None
            for meta_steps, group in metrics_df.groupby("meta_steps"):
                area = compute_area_under_success_curve(group, up_to=use_best_auc_up_to)
                if area > max_area:
                    max_area = area
                    best_meta_steps = meta_steps
            # Now, best_meta_steps contains the meta_steps with the maximal area under the success curve
            metrics_df = metrics_df[metrics_df["meta_steps"] == best_meta_steps]
            print(f"Area under success curve for best checkpoint {group_label}: {max_area} at meta_steps {best_meta_steps}")
        else:
            # For each TTT_steps, select the meta_steps row with the highest mean_success
            best_rows = []
            for ttt_steps, subdf in metrics_df.groupby("TTT_steps"):
                max_idx = subdf["mean_success"].idxmax()
                best_rows.append({
                    "TTT_steps": ttt_steps,
                    "mean_success": subdf.at[max_idx, "mean_success"],
                    "std_error": subdf.at[max_idx, "std_error"],
                })
            metrics_df = pd.DataFrame(best_rows)
            print(f"Area under success curve for {group_label}: {compute_area_under_success_curve(metrics_df)}")

        plotopts = {k: group_info[k] for k in ("color", "linestyle", "dot_type") if k in group_info}
        group_points[group_label] = metrics_df
        group_plotopts[group_label] = plotopts

    if verbose:
        print(group_points)
    # Create figure/axes if not provided
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ########### Print table if requested ###########
    if print_table:
        table_data = []
        for label in sorted(group_points.keys()):
            points = group_points[label]
            row_data = {"Label": label}

            # Fill in values for each column
            for ttt in ttt_steps_list:
                mean_val_arr = points.loc[points['TTT_steps'] == ttt, 'mean_success'].values
                std_err_arr = points.loc[points['TTT_steps'] == ttt, 'std_error'].values
                assert len(mean_val_arr) == 1 and len(std_err_arr) == 1, f"Expected 1 value for TTT_steps {ttt}"
                mean_val = mean_val_arr[0]
                std_err = std_err_arr[0]
                if np.isnan(mean_val) or np.isnan(std_err):
                    row_data[str(ttt)] = "N/A"
                else:
                    row_data[str(ttt)] = f"{mean_val:.4f} ± {std_err:.4f}"

            # TODO: restore and adjust code or delete
            # Compute max across all TTT_steps for this label
            #if relevant_points:
                # Find the point with maximum mean
                #max_point = max(relevant_points, key=lambda x: x[1])
                #max_mean = max_point[1]
                #_, _, std_at_max, _, n_seeds = max_point
                #max_std_err = std_at_max / np.sqrt(n_seeds) if n_seeds > 0 else 0.0
                #row_data["max"] = f"{max_mean:.4f} ± {max_std_err:.4f}"
            #else:
            #    row_data["max"] = "N/A"
            table_data.append(row_data)

        # Print formatted table
        print("\n" + "="*120)
        print("Results Table: Max Overall Success vs TTT_steps")
        print("="*120)

        # Header
        header = f"{'Label':<50}"
        for ttt in ttt_steps_list:
            header += f"  {ttt:>15}"
        header += f"  {'max':>15}"
        print(header)
        print("-"*120)

        # Rows
        for row in table_data:
            row_str = f"{row['Label']:<50}"
            for ttt in ttt_steps_list:
                val = row.get(str(ttt), "N/A")
                row_str += f"  {val:>15}"
            row_str += f"  {row.get('max', 'N/A'):>15}"
            print(row_str)

        print("="*120 + "\n")


    ########### Plot lines ###########
    for idx, (label, points) in enumerate(sorted(group_points.items())):
        points_sorted = points.sort_values(by='TTT_steps')
        ttt_steps_sorted = points_sorted['TTT_steps'].values
        mean_success_sorted = points_sorted['mean_success'].values
        if verbose:
            print(list(zip(ttt_steps_sorted, mean_success_sorted)))
        std_error_sorted = points_sorted['std_error'].values

        plotopts = group_plotopts.get(label, {})
        color = plotopts.get("color", None)
        linestyle = plotopts.get("linestyle", None)
        dot_type = plotopts.get("dot_type", None)
        if color is None:
            color = f"C{idx % 10}"
        if linestyle is None or dot_type is None:
            _auto_ls, _auto_dot = get_linestyle_and_dot_type(label)
            linestyle = linestyle if linestyle is not None else _auto_ls
            dot_type = dot_type if dot_type is not None else _auto_dot

        ax.plot(
            ttt_steps_sorted,
            mean_success_sorted,
            marker=dot_type,
            linestyle=linestyle,
            color=color,
            label=label,
            alpha=0.9,
            markeredgecolor="black",
        )
        lower = np.array(mean_success_sorted) - np.array(std_error_sorted)
        upper = np.array(mean_success_sorted) + np.array(std_error_sorted)
        if error_bars:
            ax.errorbar(
                ttt_steps_sorted,
                mean_success_sorted,
                yerr=std_error_sorted,
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

    ########### Set axes labels and title ###########
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

    if show_legend:
        legend_fontsize = fontsize - 2
        ax.legend(fontsize=legend_fontsize)

    ax.grid(True)
    # Set fontsize for axis lbels
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    # Set fontsize for tick labels
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # Set fontsize for title if present
    if ax.get_title():
        ax.title.set_size(fontsize)

    # Y axis in log scale
    if figure_path is not None:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        print(f"Saving figure to {figure_path}")
        plt.savefig(figure_path, bbox_inches='tight')
    if show:
        plt.show()

    return ax
