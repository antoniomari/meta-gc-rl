import os
import random
import time
import argparse
import warnings
import csv
import fcntl
from collections import defaultdict

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_async_all_gather=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import gymnasium as gym
import jax
import numpy as np
import tqdm
from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.config import GCTTTConfig, load_config
from agents.gcagent import GCAgent, MetaGCAgent
import gc
import importlib


def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_main.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000
  python eval_main.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --output_file results.csv
  python eval_main.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --num_ttt_steps 50

  # Concurrent execution: Run multiple processes with different TTT steps, all writing to the same file
  python eval_main.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --num_ttt_steps 10 --output_file results.csv &
  python eval_main.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --num_ttt_steps 50 --output_file results.csv &
  python eval_main.py config.yaml --restore_path /path/to/checkpoint --restore_epoch 1000 --num_ttt_steps 100 --output_file results.csv &
        """
    )

    parser.add_argument('--restore_path', type=str, required=True,
                        help='Path to the checkpoint directory')
    parser.add_argument('--restore_epoch', type=int, required=True,
                        help='Epoch number to restore from')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation results (default: auto-generated)')
    parser.add_argument('--num_ttt_steps', type=int, default=None,
                        help='Override number of TTT steps (if None, uses config.finetune.num_steps_list)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override seed value')
    parser.add_argument('--eval_tasks', type=int, nargs='+', default=None,
                        help='Override eval_tasks (list of task IDs to evaluate)')
    parser.add_argument('--eval_episodes', type=int, default=None,
                        help='Override number of evaluation episodes per task')
    parser.add_argument('--reset_after_horizon', action='store_true',
                        help='Override finetune.reset_after_horizon to True')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


def load_existing_results(output_file):
    """Load existing results from CSV file if it exists.

    Returns a set of (step, TTT_steps) tuples to check for duplicates.
    """
    existing_keys = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                try:
                    reader = csv.DictReader(f)
                    for row in reader:
                        step = row.get('step', '')
                        ttt_steps = row.get('TTT_steps', '')
                        if step:
                            existing_keys.add((step, ttt_steps))
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (csv.Error, IOError) as e:
            print(f"Warning: Could not read existing results file: {e}")
    return existing_keys


def save_results(results, output_file, max_retries=10, retry_delay=0.1):
    """Save evaluation results to a CSV file with file locking for concurrent access.

    This function supports concurrent writes by:
    1. Reading existing results to check for duplicates
    2. Appending new rows (avoiding duplicates)
    3. Using file locking to prevent race conditions
    4. Retrying if lock acquisition fails

    CSV format: each row represents one evaluation step (epoch).
    Columns: step, overall_success, task1_success, task2_success, ...
    """
    # Handle case where output_file is in current directory (no dirname)
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create directory if there's a directory component
        os.makedirs(output_dir, exist_ok=True)

    # Extract epoch from results
    epoch = str(results.get('epoch') or results.get('restore_epoch', ''))

    # Prepare rows to write - one row per TTT step evaluation
    rows_to_write = []
    for ttt_key, eval_data in results.get('evaluations', {}).items():
        # Extract TTT steps from key (e.g., "10_TTT" -> 10)
        ttt_steps = ttt_key.replace('_TTT', '')

        # Create step identifier: use epoch as step (or epoch_ttt_steps if multiple TTT steps)
        step = epoch

        # Extract overall metrics
        overall_metrics = eval_data.get('overall', {})
        overall_success = overall_metrics.get('success', 0.0)

        # Convert overall_success to float if needed
        if isinstance(overall_success, (list, np.ndarray)):
            overall_success = float(np.mean(overall_success))
        elif hasattr(overall_success, 'item'):
            overall_success = float(overall_success.item())
        else:
            try:
                overall_success = float(overall_success)
            except (TypeError, ValueError):
                overall_success = 0.0

        # Create row with step, TTT_steps, and overall_success
        row = {
            'step': step,
            'TTT_steps': str(ttt_steps),
            'overall_success': str(overall_success),
        }

        # Add task-specific success columns
        # Sort tasks by task_id or task_name for consistent column ordering
        tasks = sorted(
            eval_data.get('tasks', {}).items(),
            key=lambda x: x[1].get('task_id', x[0])
        )

        for task_name, task_data in tasks:
            task_metrics = task_data.get('metrics', {})
            task_success = task_metrics.get('success', 0.0)

            # Convert task_success to float if needed
            if isinstance(task_success, (list, np.ndarray)):
                task_success = float(np.mean(task_success))
            elif hasattr(task_success, 'item'):
                task_success = float(task_success.item())
            else:
                try:
                    task_success = float(task_success)
                except (TypeError, ValueError):
                    task_success = 0.0

            # Use task name as column name (e.g., "task1_success")
            row[f'{task_name}_success'] = str(task_success)

        rows_to_write.append(row)

    if not rows_to_write:
        print("Warning: No rows to write")
        return

    # Define CSV columns: step, overall_success, then task columns sorted by name
    # Collect all task columns from all rows
    all_task_columns = set()
    for row in rows_to_write:
        all_task_columns.update([k for k in row.keys() if k.endswith('_success') and k != 'overall_success'])

    # Sort task columns for consistent ordering
    all_task_columns = sorted(all_task_columns)
    all_columns = ['step', 'TTT_steps', 'overall_success'] + all_task_columns

    # Retry loop for file locking
    for attempt in range(max_retries):
        try:
            # Check if file exists and read existing keys
            file_exists = os.path.exists(output_file)
            existing_keys = load_existing_results(output_file) if file_exists else set()

            # Check file size to determine if it's empty
            file_is_empty = not file_exists or os.path.getsize(output_file) == 0

            # Read existing columns first if file exists and has content
            if file_exists and not file_is_empty:
                try:
                    with open(output_file, 'r', newline='') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                        try:
                            reader = csv.DictReader(f)
                            existing_columns = reader.fieldnames
                            if existing_columns:
                                # Merge columns: existing first, then any new ones
                                all_columns = list(existing_columns) + [c for c in all_columns if c not in existing_columns]
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except (csv.Error, IOError) as e:
                    print(f"Warning: Could not read existing columns: {e}, using new columns")

            # Open file and acquire exclusive lock for writing
            with open(output_file, 'a' if file_exists else 'w', newline='') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                try:

                    writer = csv.DictWriter(f, fieldnames=all_columns)

                    # Write header if file is new or empty
                    if file_is_empty:
                        writer.writeheader()

                    # Write rows (skip duplicates based on step and TTT_steps)
                    rows_written = 0
                    for row in rows_to_write:
                        step = row.get('step', '')
                        ttt_steps = row.get('TTT_steps', '')
                        key = (step, ttt_steps)
                        if key not in existing_keys:
                            # Ensure row has all columns (fill missing with empty string)
                            complete_row = {col: row.get(col, '') for col in all_columns}
                            writer.writerow(complete_row)
                            rows_written += 1
                            existing_keys.add(key)  # Update set to avoid duplicates in same batch

                    f.flush()
                    os.fsync(f.fileno())  # Ensure data is written to disk

                    if rows_written > 0:
                        print(f"Results saved to {output_file} ({rows_written} rows written)")
                    else:
                        print(f"Results already exist in {output_file}, skipping duplicate entries")

                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            return

        except IOError as e:
            if attempt < max_retries - 1:
                print(f"Warning: Could not acquire lock (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                raise IOError(f"Failed to save results after {max_retries} attempts: {e}")


def evaluation_loop(
    agent: GCAgent,
    env: gym.Env,
    cfg: GCTTTConfig,
    train_dataset: GCDataset,
    num_ttt_steps: int,
    eval_tasks: list = None,
):
    """Run evaluation loop and return results."""

    print("Evaluating...")

    if cfg.eval_on_cpu:
        warnings.warn("eval_on_cpu is True, but it is not supported for evaluation. Setting it to False.")
    cfg.eval_on_cpu = False

    eval_agent = agent

    results = {
        'config': {
            'env_name': cfg.env_name,
            'restore_path': cfg.restore_path,
            'restore_epoch': cfg.restore_epoch,
            'num_ttt_steps': num_ttt_steps,
            'eval_episodes': cfg.eval_episodes,
        },
        'tasks': {},
        'overall': {}
    }

    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )

    if eval_tasks is None:
        eval_tasks = cfg.eval_tasks if cfg.eval_tasks is not None else list(range(1, len(task_infos) + 1))

    num_tasks = len(eval_tasks)

    overall_metrics = defaultdict(list)

    # Create progress bar with dynamic postfix
    pbar = tqdm.tqdm(eval_tasks, desc="Evaluating tasks")

    for task_id in pbar:
        task_name = task_infos[task_id - 1]["task_name"]

        eval_start_time = time.time()
        eval_info, trajs, cur_renders = evaluate(
            agent=eval_agent,
            env=env,
            task_id=task_id,
            config=cfg,
            train_dataset=train_dataset,
            num_ttt_steps=num_ttt_steps,
        )
        eval_duration = time.time() - eval_start_time

        # Extract success metric
        success_value = None
        metric_names = ["success"]
        for k, v in eval_info.items():
            if k in metric_names:
                # Convert to float, handling numpy arrays, lists, and scalars
                if isinstance(v, (list, np.ndarray)):
                    success_value = float(np.mean(v))
                elif hasattr(v, 'item'):
                    success_value = float(v.item())
                else:
                    try:
                        success_value = float(v)
                    except (TypeError, ValueError):
                        success_value = None
                break

        # Update progress bar with success info
        if success_value is not None:
            success_str = "✓" if success_value > 0.5 else "✗"
            success_pct = f"{success_value:.1%}"
            pbar.set_postfix({
                'task': task_name[:15],  # Truncate long task names
                'success': f"{success_str} {success_pct}"
            })
        else:
            pbar.set_postfix({'task': task_name[:15]})

        print(f"Task {task_id} ({task_name}): {eval_info}")
        print(f"Evaluation for task {task_id} took {eval_duration:.2f} seconds")

        # Store task-specific results
        task_results = {
            'task_id': task_id,
            'task_name': task_name,
            'eval_duration': eval_duration,
            'metrics': {}
        }

        # Extract metrics
        for k, v in eval_info.items():
            if k in metric_names:
                task_results['metrics'][k] = float(v) if hasattr(v, 'item') else v
                overall_metrics[k].append(v)

        # Store finetune stats if available
        finetune_stats = {k.replace('finetune/', ''): v for k, v in eval_info.items() if k.startswith('finetune/')}
        if finetune_stats:
            task_results['finetune_stats'] = finetune_stats

        results['tasks'][task_name] = task_results

    # Compute overall metrics
    for k, v in overall_metrics.items():
        results['overall'][k] = float(np.mean(v))

    # Clear memory after evaluation
    gc.collect()
    jax.clear_caches()
    print("[Memory] Cleared memory after evaluation")

    return results


def main(cfg: GCTTTConfig, args):
    """Main evaluation function."""

    # Load agent config
    if cfg.agent['agent_name'].startswith("meta_"):
        agent_file_name = cfg.agent['agent_name'][len("meta_"):]
    else:
        agent_file_name = cfg.agent['agent_name']
    agent_cfg = importlib.import_module(f"agents.{agent_file_name}").get_config()
    for k, v in agent_cfg.items():
        if k not in cfg.agent:
            cfg.agent[k] = v

    # Set up environment and dataset
    config_agent = cfg.agent
    env: gym.Env
    env, train_dataset, val_dataset = make_env_and_datasets(
        cfg.env_name, cfg.data_ratio, frame_stack=config_agent["frame_stack"]
    )
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    dataset_class = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
    }[config_agent["dataset_class"]]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config_agent)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config_agent)

    # Initialize agent
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    example_batch = train_dataset.sample(1)
    if config_agent["discrete"]:
        example_batch["actions"] = np.full_like(
            example_batch["actions"], env.action_space.n - 1
        )

    agent_class = agents[config_agent["agent_name"]]
    agent: MetaGCAgent
    agent = agent_class.create(
        cfg.seed,
        example_batch["observations"],
        example_batch["actions"],
        config_agent,
        cfg.train_steps,
    )

    # Restore agent
    print(f"[Restore] Restoring agent from {cfg.restore_path} at epoch {cfg.restore_epoch}")
    agent: MetaGCAgent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Determine TTT steps to evaluate
    if args.num_ttt_steps is not None:
        num_ttt_steps_list = [args.num_ttt_steps]
    else:
        num_ttt_steps_list = cfg.finetune.num_steps_list

    # Determine output file path
    if args.output_file is None:
        # Auto-generate output file name
        output_dir = os.path.join(cfg.working_dir, "eval_results")
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_name = os.path.basename(cfg.restore_path.rstrip('/'))
        output_file = os.path.join(
            output_dir,
            f"eval_{checkpoint_name}_epoch{cfg.restore_epoch}.csv"
        )
    else:
        output_file = args.output_file
        # Ensure output directory exists (handle case where file is in current directory)
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if there's a directory component
            os.makedirs(output_dir, exist_ok=True)

    # Run evaluation for each TTT step count
    # Note: When running with --num_ttt_steps, only one TTT step will be evaluated
    # Multiple processes can run concurrently with different --num_ttt_steps values
    # and they will all append to the same output file

    eval_tasks = args.eval_tasks if args.eval_tasks is not None else cfg.eval_tasks

    for num_ttt_steps in num_ttt_steps_list:
        print(f"\n{'='*60}")
        print(f"Evaluating with {num_ttt_steps} TTT steps")
        print(f"{'='*60}\n")

        results = evaluation_loop(
            agent=agent,
            env=env,
            cfg=cfg,
            train_dataset=train_dataset,
            num_ttt_steps=num_ttt_steps,
            eval_tasks=eval_tasks,
        )

        # Prepare results for this TTT step
        ttt_key = f'{num_ttt_steps}_TTT'
        all_results = {
            'restore_path': cfg.restore_path,
            'restore_epoch': cfg.restore_epoch,
            'epoch': cfg.restore_epoch,  # Also include as 'epoch' for clarity
            'config': {
                'env_name': cfg.env_name,
                'seed': cfg.seed,
                'eval_episodes': cfg.eval_episodes,
                'restore_epoch': cfg.restore_epoch,
                'epoch': cfg.restore_epoch,
            },
            'evaluations': {
                ttt_key: results
            }
        }

        # Save results (will merge with existing file if it exists)
        save_results(all_results, output_file)

    # Print summary for this run
    print("\n" + "="*60)
    print("EVALUATION SUMMARY (This Run)")
    print("="*60)
    for num_ttt_steps in num_ttt_steps_list:
        ttt_key = f'{num_ttt_steps}_TTT'
        if ttt_key in all_results.get('evaluations', {}):
            eval_results = all_results['evaluations'][ttt_key]
            print(f"\n{ttt_key}:")
            print(f"  Overall success rate: {eval_results['overall'].get('success', 'N/A'):.4f}")
            for task_name, task_results in eval_results['tasks'].items():
                success = task_results['metrics'].get('success', 'N/A')
                print(f"    {task_name}: {success:.4f}")

    # Load and print full summary from CSV file (includes all concurrent runs)
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if rows:
                    print("\n" + "="*60)
                    print("FULL EVALUATION SUMMARY (All Steps)")
                    print("="*60)
                    for row in rows:
                        step = row.get('step', 'N/A')
                        overall_success = row.get('overall_success', 'N/A')
                        print(f"\nStep {step}:")
                        print(f"  Overall success rate: {overall_success}")

                        # Print task-specific successes
                        task_columns = [k for k in row.keys() if k.endswith('_success') and k != 'overall_success']
                        for task_col in sorted(task_columns):
                            task_name = task_col.replace('_success', '')
                            success = row.get(task_col, 'N/A')
                            print(f"    {task_name}: {success}")
                    print("="*60)
        except (csv.Error, IOError) as e:
            print(f"Warning: Could not read CSV file for summary: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load base configuration
    cfg = load_config(os.path.join(os.path.dirname(args.restore_path), "config.yaml"))

    # Apply command line overrides
    cfg.restore_path = args.restore_path
    cfg.restore_epoch = args.restore_epoch

    if args.seed is not None:
        cfg.seed = args.seed

    if args.eval_tasks is not None:
        cfg.eval_tasks = args.eval_tasks

    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes

    if args.reset_after_horizon:
        cfg.finetune.reset_after_horizon = True

    print(f"Configuration:")
    print(f"  Restore path: {cfg.restore_path}")
    print(f"  Restore epoch: {cfg.restore_epoch}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Eval episodes: {cfg.eval_episodes}")
    print(f"  TTT steps list: {cfg.finetune.num_steps_list}")
    print(f"  Reset after horizon: {cfg.finetune.reset_after_horizon}")
    if args.num_ttt_steps is not None:
        print(f"  Overriding TTT steps: {args.num_ttt_steps}")

    main(cfg, args)

