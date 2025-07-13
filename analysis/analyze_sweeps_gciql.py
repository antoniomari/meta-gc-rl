# analyze_sweeps.py (Query-Based, Parallelized, with Categorization and Sorting)

import wandb.apis.public as wandb_api
import pandas as pd
import os
import concurrent.futures
from tqdm import tqdm
import itertools
from collections import defaultdict

# --- User Configuration ---
# 1. Set your wandb entity and project name
WANDB_ENTITY = 'rlforimitation'
WANDB_PROJECT = "TTT_AllFinalRuns"

# 2. Define the metric you want to analyze
METRIC_TO_ANALYZE = "evaluation/overall_success"

# 3. Define the output filename for the text report
OUTPUT_FILENAME = "analysis_report_GCIQL.txt"

# 4. Set the number of parallel workers to fetch data.
MAX_WORKERS = 256 # Increased for potentially faster network IO

# 5. Define the Hyperparameter Grid
PARAM_GRID = {
    'env_name': ['pointmaze-medium-stitch-v0', 'pointmaze-medium-navigate-v0', 'antmaze-medium-stitch-v0', 'antmaze-medium-navigate-v0', 'humanoidmaze-medium-stitch-v0', 'humanoidmaze-medium-navigate-v0', 'cube-single-play-v0'],
    'agent.agent_name': ['gciql'],
    'finetune.actor_loss': ['ddpgbc', 'awr'],
    'finetune.lr': [0, 3.e-4, 3.e-5],
    'finetune.num_steps': [0, 50, 100, 200],
    'finetune.filter_by_mc': [True, False],
    'finetune.filter_by_recursive_mdp': [True, False],
    'finetune.replan_horizon': [100, 200],
}
# --- End of Configuration ---

# Add a timeout to the API client to prevent hanging requests.
api = wandb_api.Api(timeout=60)

def fetch_and_process_group(config_combination):
    """
    Fetches runs for a specific hyperparameter combination and returns a structured result dictionary.
    """
    filters = {f"config.{key}": value for key, value in config_combination.items()}
    
    try:
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters)
    except Exception as e:
        print(f"\n[Warning] API call failed for config {config_combination}. Error: {e}")
        return None

    if not runs:
        return None

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
    std_over_seeds = pd.Series(mean_successes).std() if len(mean_successes) > 1 else 0.0
    std_error_over_seeds = std_over_seeds / (len(mean_successes) ** 0.5) if len(mean_successes) > 0 else 0.0
    
    # Return a structured dictionary instead of a formatted string
    return {
        'config': config_combination,
        'mean_score': mean_over_seeds,
        'std_dev': std_over_seeds,
        'std_error': std_error_over_seeds,
        'num_seeds': len(runs)
    }

# --- Main Script ---
print("Starting analysis...")

# Step 1: Generate all hyperparameter combinations from the grid
keys = PARAM_GRID.keys()
values = PARAM_GRID.values()
all_combinations = list(itertools.product(*values))
tasks = [dict(zip(keys, combo)) for combo in all_combinations]

print(f"Generated {len(tasks)} unique hyperparameter configurations to query from wandb.")

# Step 2: Process each combination in parallel and collect structured results
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
            print(f'\n[Error] A task generated an exception: {exc}')

# --- NEW: Step 3: Categorize and Sort the Results ---
print("Categorizing and sorting results...")
categorized_results = defaultdict(lambda: defaultdict(list))

for res in processed_results:
    env = res['config']['env_name']
    actor_loss = res['config']['finetune.actor_loss']
    
    # Create category names like "GCIQL - DDPGBC" and "GCIQL - AWR"
    category_name = f"GCIQL - {actor_loss.upper()}"
    
    categorized_results[env][category_name].append(res)

# Sort the results within each category by mean_score (descending)
for env, categories in categorized_results.items():
    for category, results_list in categories.items():
        results_list.sort(key=lambda x: x['mean_score'], reverse=True)


# --- NEW: Step 4: Write the Final Sorted and Categorized Report ---
print(f"Writing final report to {OUTPUT_FILENAME}...")
with open(OUTPUT_FILENAME, 'w') as f:
    f.write(f"Analysis Report for Project: {WANDB_ENTITY}/{WANDB_PROJECT}\n")
    f.write(f"Metric: {METRIC_TO_ANALYZE}\n")

    # Iterate through environments, sorted alphabetically for consistent order
    for env_name in sorted(categorized_results.keys()):
        categories = categorized_results[env_name]
        
        f.write("\n\n" + "#" * 80 + "\n")
        f.write(f"### Environment: {env_name}\n")
        f.write("#" * 80 + "\n")

        # Iterate through categories (e.g., AWR, DDPGBC), sorted alphabetically
        for category_name in sorted(categories.keys()):
            results_list = categories[category_name]
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"===== Category: {category_name} =====\n")
            f.write("=" * 80 + "\n")

            if not results_list:
                f.write("No results found for this category.\n")
                continue

            # Write out each sorted result
            for i, result in enumerate(results_list):
                f.write(f"\n--- Rank #{i+1} in Category ---\n")
                f.write(f"Score (Mean ± Std Dev): {result['mean_score']:.4f} ± {result['std_dev']:.4f}\n")
                f.write(f"Standard Error: {result['std_error']:.4f}\n")
                f.write(f"Configuration ({result['num_seeds']} seeds):\n")
                
                # Print config details, skipping keys used for grouping
                for key, value in result['config'].items():
                    if key not in ['env_name', 'finetune.actor_loss', 'agent.agent_name']:
                        f.write(f"  - {key}: {value}\n")

print(f"\nAnalysis complete. Report saved to '{os.path.abspath(OUTPUT_FILENAME)}'")
