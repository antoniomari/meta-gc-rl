# analyze_sweeps.py (Query-Based, Parallelized, and Optimized)

import wandb.apis.public as wandb_api
import pandas as pd
import os
import concurrent.futures
from tqdm import tqdm
import itertools

# --- User Configuration ---
# 1. Set your wandb entity and project name
WANDB_ENTITY = 'rlforimitation'
WANDB_PROJECT = "TTT_AllFinalRuns"

# 2. Define the metric you want to analyze
METRIC_TO_ANALYZE = "evaluation/overall_success"

# 3. Define the output filename for the text report
OUTPUT_FILENAME = "analysis_report_query_based.txt"

# 4. Set the number of parallel workers to fetch data.
MAX_WORKERS = 512

# 5. --- NEW: Define the Hyperparameter Grid ---
# This is the core of the new, faster approach. Instead of fetching all runs,
# we generate all possible hyperparameter combinations and query for them directly.
#
# IMPORTANT: You must manually list all the values you swept over for each key.
# This should match the 'grid' section of your original YAML experiment file.
PARAM_GRID = {
    'env_name': ['pointmaze-medium-stitch-v0', 'pointmaze-medium-navigate-v0', 'antmaze-medium-stitch-v0', 'antmaze-medium-navigate-v0', 'humanoidmaze-medium-stitch-v0', 'humanoidmaze-medium-navigate-v0', 'cube-single-play-v0'],
    'agent.agent_name': ['gciql'],
    'finetune.actor_loss': ['ddpgbc', 'awr'],
    'finetune.lr': [3.e-4, 3.e-5],
    'finetune.num_steps': [50, 100, 200],
    'finetune.filter_by_mc': [True, False],
    'finetune.filter_by_recursive_mdp': [True, False],
    'finetune.replan_horizon': [100, 200],
    # NOTE: Do NOT include 'seed' here. We query for all seeds implicitly.
}
# --- End of Configuration ---

# Add a timeout to the API client to prevent hanging requests.
api = wandb_api.Api(timeout=30)


def fetch_and_process_group(config_combination):
    """
    Fetches runs for a specific hyperparameter combination, analyzes them,
    and returns a formatted string for the report.
    """
    # Build the filter for the wandb API query.
    # This creates a dictionary like: {"config.env_name": "...", "config.finetune.lr": ...}
    filters = {f"config.{key}": value for key, value in config_combination.items()}

    # Add a filter to only get runs that have finished successfully.
    #filters["state"] = "finished"

    # Fetch the specific runs (e.g., the 3 seeds) matching this exact configuration.
    try:
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters)
    except Exception as e:
        # If a specific query fails (e.g., due to a timeout), print a warning and skip.
        print(f"\n[Warning] API call failed for config {config_combination}. Error: {e}")
        return None

    if not runs:
        return None  # No finished runs found for this combination.

    mean_successes = []
    for run in runs:
        # Using run.history() is fine for this use case.
        df = run.history(
            samples=50000, keys=['evaluation/overall_success'], pandas=(True), stream="default"
        )
        if not df.empty and METRIC_TO_ANALYZE in df.columns:
            df[METRIC_TO_ANALYZE] = pd.to_numeric(df[METRIC_TO_ANALYZE], errors='coerce')
            df = df.dropna(subset=[METRIC_TO_ANALYZE])
            if not df.empty:
                mean_successes.append(df[METRIC_TO_ANALYZE].mean())

    if not mean_successes:
        return None # No valid data found in any run for this group

    mean_over_seeds = sum(mean_successes) / len(mean_successes)
    std_over_seeds = pd.Series(mean_successes).std() if len(mean_successes) > 1 else 0
    std_error_over_seeds = std_over_seeds / (len(mean_successes) ** 0.5) if len(mean_successes) > 0 else 0
    
    # --- FIX: The error occurs here. ---
    # When creating a DataFrame from scalars, you must provide an index.
    results_df = pd.DataFrame({
        f"mean": mean_over_seeds,
        f"std": std_over_seeds,
        f"std_error": std_error_over_seeds
    }, index=[0]) # By adding index=[0], we create a one-row DataFrame.

    # Build the report string for this group
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"Configuration ({len(runs)} seeds):")
    for key, value in config_combination.items():
        report_lines.append(f"  - {key}: {value}")
    report_lines.append("-" * 20)
    report_lines.append("Results:")
    report_lines.append(results_df.to_string())
    report_lines.append("\n")

    return "\n".join(report_lines)


# --- Main Script ---
print("Starting analysis...")

# Step 1: Generate all hyperparameter combinations from the grid
keys = PARAM_GRID.keys()
values = PARAM_GRID.values()
all_combinations = list(itertools.product(*values))
tasks = [dict(zip(keys, combo)) for combo in all_combinations]

print(f"Generated {len(tasks)} unique hyperparameter configurations to query from wandb.")

# Step 2: Process each combination in parallel
print(f"Fetching and analyzing groups using up to {MAX_WORKERS} parallel workers...")
processed_results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_task = {executor.submit(fetch_and_process_group, task): task for task in tasks}

    for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks)):
        try:
            result = future.result()
            if result:
                processed_results.append(result)
        except Exception as exc:
            # This will catch errors happening inside the thread and print them
            print(f'\n[Error] A task generated an exception: {exc}')


# Step 3: Write all collected results to the final report file
print(f"Writing final report to {OUTPUT_FILENAME}...")
with open(OUTPUT_FILENAME, 'w') as f:
    f.write(f"Analysis Report for Project: {WANDB_ENTITY}/{WANDB_PROJECT}\n")
    f.write(f"Metric: {METRIC_TO_ANALYZE}\n\n")

    for report_string in processed_results:
        f.write(report_string)

print(f"\nAnalysis complete. Report saved to '{os.path.abspath(OUTPUT_FILENAME)}'")
