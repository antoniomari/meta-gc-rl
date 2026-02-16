import sys
import yaml
import os
import random
import time
import psutil
import gc
import argparse
import warnings
import csv
from collections import defaultdict, OrderedDict

# Test job command
# srun --time=4:0:0 --mem-per-cpu=32G --gpus=1 --pty bash -l
# ssh -L 8888:localhost:8888 eu-lo-g3-022


os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_async_all_gather=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO off, 2=WARNING off, 3=ERROR only
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# Enable JAX compilation logging
# os.environ["JAX_LOG_COMPILES"] = "1"

import gymnasium as gym
import wandb
import jax
from dataclasses import asdict
import numpy as np
import tqdm
from agents import agents
from ml_collections import FrozenConfigDict
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate, _cfg_get
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_wandb_video, setup_wandb, get_exp_name
from datetime import datetime
from utils.data_selection import get_task_filter, fetch_meta_batch, sample_task
from utils.config import GCTTTConfig, load_config
from agents.gcagent import GCAgent, MetaGCAgent
import matplotlib.pyplot as plt
import io
from PIL import Image
import importlib

def _to_np(x):
    # Convert to numpy array for distance computation
    if isinstance(x, np.ndarray):
        return x
    elif hasattr(x, "cpu"):
        return np.asarray(x.cpu())
    else:
        return np.asarray(x)


def override_config_value(cfg: GCTTTConfig, key_path: str, value: str):
    """Override a config value using dot notation (e.g., 'finetune.num_steps').

    Args:
        cfg: The configuration object to modify
        key_path: Dot-separated path to the config value (e.g., 'finetune.num_steps')
        value: String value to set (will be converted to appropriate type)
    """
    keys = key_path.split('.')
    current = cfg

    # Navigate to the parent of the target attribute
    for key in keys[:-1]:
        if hasattr(current, key):
            current = getattr(current, key)
        else:
            raise ValueError(f"Config path '{key_path}' not found: '{key}' does not exist")

    # Get the final key and set the value
    final_key = keys[-1]
    if not hasattr(current, final_key):
        raise ValueError(f"Config path '{key_path}' not found: '{final_key}' does not exist")

    # Get the current value to determine the type for conversion
    current_value = getattr(current, final_key)

    # Convert the string value to the appropriate type
    if isinstance(current_value, bool):
        converted_value = value.lower() in ('true', '1', 'yes', 'on')
    elif isinstance(current_value, int):
        converted_value = int(value)
    elif isinstance(current_value, float):
        converted_value = float(value)
    elif current_value is None:
        # Try to infer type from the string value
        try:
            converted_value = int(value)
        except ValueError:
            try:
                converted_value = float(value)
            except ValueError:
                if value.lower() in ('true', 'false'):
                    converted_value = value.lower() == 'true'
                else:
                    converted_value = value
    else:
        converted_value = value

    setattr(current, final_key, converted_value)
    # Note: This print is always shown as it's important for debugging config overrides
    print(f"Override: {key_path} = {converted_value} (type: {type(converted_value).__name__})")


def parse_args():
    """Parse command line arguments for config overrides."""
    parser = argparse.ArgumentParser(
        description="Train meta agent with optional config overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python meta_main.py config.yaml --finetune.num_steps 100 --train_steps 50000
  python meta_main.py config.yaml --finetune.num_steps 200 --train_steps 100000 --meta_algorithm reptile
  python meta_main.py config.yaml --seed 42 --meta_algorithm maml
  python meta_main.py config.yaml --agent.inner_loop_steps 10 --meta_algorithm maml
  python meta_main.py config.yaml --agent.meta_batch_size 16 --agent.inner_loop_steps 5 --meta_algorithm maml
  python meta_main.py config.yaml --eval_interval 5000 --meta_algorithm maml
  python meta_main.py config.yaml --train_on_test_goal --use_random_batch
        """
    )

    parser.add_argument('config_file', help='Path to the YAML configuration file')
    parser.add_argument('--finetune.num_steps', type=str, dest='finetune_num_steps',
                        help='Override finetune.num_steps value')
    parser.add_argument('--train_steps', type=str, dest='train_steps',
                        help='Override train_steps value')
    parser.add_argument('--meta_algorithm', type=str, dest='meta_algorithm',
                        choices=['maml', 'fomaml', 'reptile'],
                        help='Override meta_algorithm value (choices: maml, fomaml, reptile)')
    parser.add_argument('--seed', type=int, dest='seed',
                        help='Override seed value')
    parser.add_argument('--agent.inner_loop_steps', type=int, dest='inner_loop_steps',
                        help='Override agent.inner_loop_steps value')
    parser.add_argument('--agent.meta_batch_size', type=int, dest='meta_batch_size',
                        help='Override agent.meta_batch_size value')
    parser.add_argument('--eval_interval', type=int, dest='eval_interval',
                        help='Override eval_interval value')
    parser.add_argument('--use_random_batch', action='store_true', dest='use_random_batch',
                        help='Use random batch sampling instead of goal-conditioned sampling')
    parser.add_argument('--train_on_test_goal', action='store_true', dest='train_on_test_goal',
                        help='Override train_on_test_goal to True (use test goal for training batch fetching)')
    parser.add_argument('--actor_uniform_sample', action='store_true', dest='actor_uniform_sample',
                        help='Set agent.actor_geom_sample to False (use uniform sampling instead of geometry sampling)')
    parser.add_argument('--finetune.mc_quantile', type=float, dest='finetune_mc_quantile',
                        help='Override finetune.mc_quantile value (float)')
    parser.add_argument('--finetune.mc_quantile_train', type=float, dest='finetune_mc_quantile_train',
                        help='Override finetune.mc_quantile_train value (float) - used for training only')
    parser.add_argument('--save_interval', type=int, dest='save_interval',
                        help='Override save_interval value')
    parser.add_argument('--restore_path', type=str, dest='restore_path',
                        help='Override restore_path value')
    parser.add_argument('--restore_epoch', type=int, dest='restore_epoch',
                        help='Override restore_epoch value')
    parser.add_argument('--log_interval', type=int, dest='log_interval',
                        help='Override log_interval value')
    parser.add_argument('--finetune.lr', type=float, dest='finetune_lr',
                        help='Override finetune.lr value (float)')
    parser.add_argument('--agent.merging_eps', type=float, dest='agent_merging_eps',
                        help='Override agent.merging_eps value (float)')
    parser.add_argument('--agent.max_grad_norm', type=float, dest='agent_max_grad_norm',
                        help='Override agent.max_grad_norm value (float)')
    parser.add_argument('--finetune.inner_lr', type=float, dest='finetune_inner_lr',
                        help='Override finetune.inner_lr value (float) - used for inner optimizer')
    parser.add_argument('--average_test_gradients', action='store_true', dest='average_test_gradients',
                        help='Override average_test_gradients to True (average test gradients across tasks)')
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        help='Enable verbose output (prints timing, memory, and debug information)')
    parser.add_argument('--wandb_group', type=str, dest='wandb_group',
                        help='Override automatic wandb group name (for testing purposes)')
    parser.add_argument('--training_fix_actor_goal', type=float, dest='training_fix_actor_goal',
                        help='Override training_fix_actor_goal value (float, default: 1.0)')
    parser.add_argument('--plot_interval', type=int, dest='plot_interval',
                        help='Override plot_interval value (int, interval for creating plots, default: 100)')
    parser.add_argument('--finetune.actor_only', action='store_true', dest='finetune_actor_only',
                        help='Override finetune.actor_only to True (only update actor during training)')
    parser.add_argument('--use_meta_optimizer', action='store_true', dest='use_meta_optimizer',
                        help='Use meta optimizer (only for reptile algorithm)')
    parser.add_argument('--annealing', action='store_true', dest='annealing',
                        help='Enable annealing (default: False, only for reptile algorithm)')
    parser.add_argument('--no_optimality', action='store_true', dest='no_optimality',
                        help='Set finetune.no_optimality to True (disable optimality filtering)')
    parser.add_argument('--use_best_checkpoint', action='store_true', dest='use_best_checkpoint',
                        help='Override use_best_checkpoint to True (use best checkpoint)')
    parser.add_argument('--actor_loss', type=str, dest='actor_loss',
                        help='Override both cfg.finetune.actor_loss and cfg.agent["actor_loss"]')

    return parser.parse_args()


def evaluation_loop(
    agent: GCAgent,
    env: gym.Env,
    cfg: GCTTTConfig,
    train_dataset: GCDataset,
    eval_logger: CsvLogger,
    step: int,
    num_ttt_steps: int,
):

    print("Evaluating...")

    if cfg.eval_on_cpu:
        warnings.warn("eval_on_cpu is True, but it is not supported for evaluation. Setting it to False.")
    cfg.eval_on_cpu = False
    if cfg.eval_on_cpu:
        eval_agent = jax.device_put(agent, device=jax.devices("cpu")[0])
    else:
        eval_agent = agent
    renders = []
    eval_metrics = {}
    overall_metrics = defaultdict(list)
    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )

    num_tasks = (
        len(cfg.eval_tasks) if cfg.eval_tasks is not None else len(task_infos)
    )
    for task_id in tqdm.trange(1, num_tasks + 1):
        task_name = task_infos[task_id - 1]["task_name"]
        # Test-time fine-tuning happens in here
        # Test-time fine-tuning happens in here
        eval_start_time = time.time()
        eval_info, trajs, cur_renders = evaluate(
            agent=eval_agent,
            env=env,
            task_id=task_id,
            config=cfg,
            train_dataset=train_dataset,
            num_ttt_steps=num_ttt_steps,
        )
        print(eval_info)
        eval_duration = time.time() - eval_start_time
        print(f"Evaluation for task {task_id} took {eval_duration:.2f} seconds")

        # Simple script to plot rollouts, assuming that the first 2 dimensions
        # of the data represent XY CoM coordinates.
        # TODO: remove
        plotit = True
        if plotit:
            buf = io.BytesIO()
            _obs = np.stack(trajs[0]["observation"])
            _background = train_dataset.sample(1000)["observations"]
            plt.scatter(_background[:, 0], _background[:, 1])
            plt.scatter(_obs[:, 0], _obs[:, 1])
            # plt.savefig(f'Zfig_{exp_name}.png', dpi=900)
            plt.savefig(buf, format="png", dpi=900)
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            wandb.log({"Zfig": wandb.Image(img_array)}, step=step)
            del img_array, img, buf

        # --- MINIMAL MODIFICATION START ---

        finetune_actor_loss_key = "finetune/actor/actor_loss"
        # Check for the specific key and add its list to eval_metrics
        if finetune_actor_loss_key in eval_info:
            loss_list_raw = eval_info[finetune_actor_loss_key]
            if isinstance(loss_list_raw, list):  # Make sure it's a list
                try:
                    # Convert JAX arrays/other numerics to standard Python floats
                    loss_values_float = [
                        (
                            float(val.item())
                            if hasattr(val, "item")
                            else float(val)
                        )
                        for val in loss_list_raw
                    ]

                    # Add the list directly to eval_metrics.
                    # Use a key that indicates it's the raw trend/list.
                    # Replace '/' in the metric name segment with '_' for cleaner W&B key
                    log_key_segment = finetune_actor_loss_key.replace("/", "_")
                    eval_metrics[
                        f"finetune/{task_name}_{log_key_segment}_trend"
                    ] = loss_values_float
                except Exception as e:
                    # Log a warning if conversion fails, but don't crash
                    print(
                        f"Warning: Could not process {finetune_actor_loss_key} list for task {task_name}: {e}"
                    )
        # --- MINIMAL MODIFICATION END ---

        renders.extend(cur_renders)
        metric_names = ["success"]
        eval_metrics.update(
            {
                f"evaluation/{task_name}_{k}/{num_ttt_steps}_TTT": v
                for k, v in eval_info.items()
                if k in metric_names
            }
        )

        wandb.log({f'evaluation_logged/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}, step=step)
        for k, v in eval_info.items():
            if k in metric_names:
                overall_metrics[k].append(v)

    # TODO: check are we averaging over task?
    for k, v in overall_metrics.items():
        eval_metrics[f"evaluation/overall_{k}/{num_ttt_steps}_TTT"] = np.mean(v)

    if cfg.video_episodes > 0:
        video = get_wandb_video(renders=renders, n_cols=num_tasks)
        eval_metrics[f"video/{num_ttt_steps}_TTT"] = video

    try:
        # Use the same step as the training loop to maintain monotonic ordering
        wandb.log(eval_metrics, step=step)
    except Exception as e:
        print(f"Error during wandb.log: {e}")

    # Log to the separate eval_logger if it exists
    try:
        eval_logger.log(eval_metrics, step=step)
    except Exception as e:
        print(f"Error logging to eval_logger: {e}")

    # Clear memory after evaluation
    gc.collect()
    jax.clear_caches()
    print("[Memory] Cleared memory after evaluation")



# TODO: adjust implementation
def get_exp_name_new(cfg):
    """Return the experiment name."""
    # experiment name consists:
    # goal type prefix (TEST-GOALS or ALL-GOALS)
    # agent_name (gciql or hiql)
    # environment name parts: env_name_split[0] + '-' + env_name_split[2]
    # agent.actor_loss (bc or awr)
    # finetune.actor_loss (bc or awr)
    # finetune.filter_by_mc
    # finetune.mc_quantile
    # finetune.mc_slack
    # finetune.mc_similarity_threshold
    # finetune.filter_by_recursive_mdp
    # finetune.min_steps
    # finetune.replan_horizon
    # meta_algorithm (if not None)
    # timestamp
    # seed

    exp_name = env_name
    exp_name += f'_{cfg.agent["agent_name"]}'
    exp_name += '_' + cfg.finetune.actor_loss
    exp_name += '_' + str(cfg.finetune.filter_by_mc)
    exp_name += '_' + str(cfg.finetune.mc_quantile)
    exp_name += '_' + str(cfg.finetune.mc_slack)
    exp_name += '_' + str(cfg.finetune.mc_similarity_threshold)
    exp_name += '_' + str(cfg.finetune.filter_by_recursive_mdp)
    exp_name += '_' + str(cfg.finetune.min_steps)
    exp_name += '_' + str(cfg.finetune.replan_horizon)
    # Add inner_loop_steps if present
    inner_loop_steps = cfg.agent.get("inner_loop_steps", 1)
    exp_name += '_' + str(inner_loop_steps) + 'inner'
    # Add meta_batch_size if present
    meta_batch_size = cfg.agent.get("meta_batch_size", 32)
    exp_name += '_' + str(meta_batch_size) + 'meta'
    exp_name += '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name += f'_s{cfg.seed:03d}'
    return exp_name


# Create group name and experiment name using a helper function
def get_exp_and_group_names(cfg, env_name_short, exp_name):
    # Create group name based on meta learning algorithm and parameters

    if cfg.train_steps == 0:
        group_name = f"INIT-{env_name_short}-{cfg.finetune.actor_loss}"
    else:
        if cfg.meta_algorithm is None:
            # Old implemnetation: PT
            algorithm_suffix = "JT"  # Joint training
        else:
            if cfg.meta_algorithm == "reptile":
                # Old implemnetation: RX
                algorithm_suffix = "RR"  # Stands for Reptile-fixed
            elif cfg.meta_algorithm == "fomaml":
                # Old implemnetation: FX
                algorithm_suffix = "FF"  # Stands for FOMAML-fixed
            else:
                algorithm_suffix = cfg.meta_algorithm.upper()

        # Append "A" to algorithm suffix if actor_only is True
        if cfg.finetune.actor_only:
            algorithm_suffix += "A"

        if cfg.use_best_checkpoint:
            algorithm_suffix += "B"

        # Append "MO" to algorithm suffix if use_meta_optimizer is True (only for reptile)
        if cfg.meta_algorithm == "reptile" and getattr(cfg, 'use_meta_optimizer', False):
            algorithm_suffix += "MO"

        # Append "fix" to algorithm suffix if annealing is False (only for reptile)
        if cfg.meta_algorithm == "reptile" and getattr(cfg, 'annealing', False):
            algorithm_suffix += "ANN"

        group_name = f"{env_name_short}-{cfg.finetune.actor_loss}-{algorithm_suffix}"

        # Add type of task
        if cfg.train_on_test_goal:
            assert not cfg.use_random_batch, "Cannot use random batch and train on test goal at the same time"
            group_name = "TEST-" + group_name
        elif cfg.use_random_batch:
            # Case of one task per sample in the batch
            group_name = "RANDOM-" + group_name
        else:
            # Default case: all samples in the batch are from the same task
            group_name = "ALL-" + group_name

        # Add meta learning parameters if meta algorithm is present
        inner_steps = cfg.agent.get("inner_loop_steps", 1)
        meta_batch_size = cfg.agent.get("meta_batch_size", 32)
        group_name += f"-{inner_steps}-{meta_batch_size}"

    # Add mc_quantile_train to group name if it was overwritten
    mc_quantile_train = cfg.finetune.get("mc_quantile_train")
    mc_quantile = cfg.finetune.get("mc_quantile")
    if mc_quantile_train is not None:
    #    group_name += f"-mc{mc_quantile_train}"
        exp_name += f"-mc{mc_quantile_train}"
    elif mc_quantile is not None:
    #    group_name += f"-mc{mc_quantile}"
        exp_name += f"-mc{mc_quantile}"

    # Add merging_eps to group name if different from 1.0
    merging_eps = cfg.agent.get("merging_eps", 1.0)
    if merging_eps != 1.0 and cfg.meta_algorithm == "reptile":
        if not cfg.use_meta_optimizer:
            group_name += f"-m_{merging_eps}"
            exp_name += f"-m_{merging_eps}"

    # Add LR to group name if it was overwritten
    if 'lr' in cfg.agent:
        lr_value = cfg.agent['lr']
        group_name += f"-lr{lr_value}"
        exp_name += f"-lr{lr_value}"

    # Add inner_lr to group name if it was overwritten
    if cfg.finetune.inner_lr is not None:
        inner_lr_value = cfg.finetune.inner_lr
        group_name += f"-ilr{inner_lr_value}"
        exp_name += f"-ilr{inner_lr_value}"

    # Add training_fix_actor_goal to group name if different from 1.0
    if cfg.training_fix_actor_goal != 1.0:
        group_name += f"-rg{cfg.training_fix_actor_goal}"
        exp_name += f"-rg{cfg.training_fix_actor_goal}"

    # Add FT prefix if restoring from checkpoint
    if cfg.restore_path is not None:
        group_name = "FT-" + group_name
        exp_name = "FT-" + exp_name

    return group_name, exp_name


def log_test_loss(test_info, cfg: GCTTTConfig, save_path: str, inner_step: int, meta_step: int):
    # Note: save_path will be a csv
    # I want to save these columns: meta_step, inner_step, test_loss
    # Create save_path directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_exists = os.path.isfile(save_path)
    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['meta_step', 'inner_step', 'test_loss'])
        writer.writerow([meta_step, inner_step, test_info])


def checkpoint_save_path(cfg: GCTTTConfig, wandb_group: str):
    # folder structure:
    # $working_dir/   # Contains env and agent info
    #   - {wandb_group}/
    #       - seed/
    #         -
    #           - checkpoint.pt
    save_path = os.path.join(cfg.working_dir, wandb_group, f"seed{cfg.seed}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def plot_test_loss_inner_steps(plot_dict: dict, step: int, cfg: GCTTTConfig):
    """
    Create plots of train and test losses vs inner steps and log them to wandb.

    Args:
        plot_dict: Dictionary where keys are inner_step (int) and values are batch_info dicts
        step: Current training step
        cfg: Configuration object

    Returns:
        None (logs plots directly to wandb)
    """
    if not plot_dict:
        raise ValueError(f"[Plot] plot_dict is empty at step {step}, skipping plot creation")

    # Extract inner steps and sort them
    inner_steps = sorted([k for k in plot_dict.keys() if isinstance(k, int)])
    if not inner_steps:
        raise ValueError(f"[Plot] No valid inner steps found in plot_dict at step {step}, skipping plot creation")

    print(f"[Plot] Creating plots at step {step} with inner_steps: {inner_steps}")

    # Extract the metrics for each inner step (both train and test)
    metrics_to_plot = {
        "total_loss": {
            "train": "train/total_loss",
            "test": "test/total_loss"
        },
        "value/value_loss": {
            "train": "train/value/value_loss",
            "test": "test/value/value_loss"
        },
        "critic/critic_loss": {
            "train": "train/critic/critic_loss",
            "test": "test/critic/critic_loss"
        },
        "actor/actor_loss": {
            "train": "train/actor/actor_loss",
            "test": "test/actor/actor_loss"
        },
    }

    # Try alternative key names if the primary ones don't exist
    alt_keys = {
        "train/total_loss": ["train/total_loss", "total_loss"],
        "test/total_loss": ["test/total_loss"],
        "train/value/value_loss": ["train/value/value_loss", "value/value_loss"],
        "test/value/value_loss": ["test/value/value_loss"],
        "train/critic/critic_loss": ["train/critic/critic_loss", "critic/critic_loss"],
        "test/critic/critic_loss": ["test/critic/critic_loss"],
        "train/actor/actor_loss": ["train/actor/actor_loss", "actor/actor_loss"],
        "test/actor/actor_loss": ["test/actor/actor_loss"],
    }

    # Collect data for each metric (both train and test)
    plot_data = {}
    for metric_name, keys in metrics_to_plot.items():
        train_values = []
        test_values = []
        valid_steps = []

        for inner_step in inner_steps:
            batch_info = plot_dict[inner_step]
            train_value = None
            test_value = None

            # Try to get train value
            train_key = keys["train"]
            if train_key in batch_info:
                train_value = batch_info[train_key]
            else:
                # Try alternative keys
                for alt_key in alt_keys.get(train_key, []):
                    if alt_key in batch_info:
                        train_value = batch_info[alt_key]
                        break

            # Try to get test value
            test_key = keys["test"]
            if test_key in batch_info:
                test_value = batch_info[test_key]
            else:
                # Try alternative keys
                for alt_key in alt_keys.get(test_key, []):
                    if alt_key in batch_info:
                        test_value = batch_info[alt_key]
                        break

            # Only process if at least one value is available
            if train_value is not None or test_value is not None:
                valid_steps.append(inner_step)

                # Convert to float if needed and store (or None if missing)
                if train_value is not None:
                    if hasattr(train_value, 'item'):
                        train_value = float(train_value.item())
                    else:
                        train_value = float(train_value)
                    train_values.append(train_value)
                else:
                    train_values.append(None)

                if test_value is not None:
                    if hasattr(test_value, 'item'):
                        test_value = float(test_value.item())
                    else:
                        test_value = float(test_value)
                    test_values.append(test_value)
                else:
                    test_values.append(None)

        if train_values or test_values:
            plot_data[metric_name] = {
                "inner_steps": valid_steps,
                "train_values": train_values,
                "test_values": test_values
            }

    if not plot_data:
        raise ValueError(f"[Plot] No plot data collected at step {step}. Available keys in batch_info: {list(plot_dict[inner_steps[0]].keys()) if inner_steps else 'N/A'}")

    # Create plots
    for metric_name, data in plot_data.items():
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out None values for plotting
        train_steps = [s for s, v in zip(data["inner_steps"], data["train_values"]) if v is not None]
        train_vals = [v for v in data["train_values"] if v is not None]
        test_steps = [s for s, v in zip(data["inner_steps"], data["test_values"]) if v is not None]
        test_vals = [v for v in data["test_values"] if v is not None]

        # Plot train losses in blue
        if train_vals:
            ax.plot(train_steps, train_vals, marker='o', linewidth=2, markersize=6, color='blue', label='Train')

        # Plot test losses in red
        if test_vals:
            ax.plot(test_steps, test_vals, marker='s', linewidth=2, markersize=6, color='red', label='Test')

        ax.set_xlabel("Inner Step", fontsize=12)
        ax.set_ylabel(metric_name.split("/")[-1].replace("_", " ").title(), fontsize=12)
        ax.set_title(f"{metric_name} vs Inner Step (Training Step {step})", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        # Use plot/ prefix to organize plots in wandb
        plot_name = "plot/" + metric_name.replace("/", "_")
        wandb.log({plot_name: wandb.Image(img_array)}, step=step)
        print(f"[Plot] Logged {plot_name} to wandb at step {step}")
        del img_array, img, buf


def main(cfg: GCTTTConfig, verbose: bool = False, wandb_group: str = None):

    if cfg.agent['agent_name'].startswith("meta_"):
        agent_file_name = cfg.agent['agent_name'][len("meta_"):]
    else:
        agent_file_name = cfg.agent['agent_name']
    agent_cfg = importlib.import_module(f"agents.{agent_file_name}").get_config()
    for k, v in agent_cfg.items():
        if k not in cfg.agent:
            cfg.agent[k] = v

    # Set up logger.
    # split env_name by '-'
    exp_name = get_exp_name(cfg)
    env_name_short = cfg.env_name.split("-")[0]

    if "navigate" in cfg.env_name:
        assert "maze" in env_name_short, "Expert environment must contain 'maze' in the name"
        env_name_short = env_name_short.replace("maze", "_exp")

    group_name, exp_name = get_exp_and_group_names(cfg, env_name_short, exp_name)

    # Override group name if wandb_group is provided (for testing purposes)
    if wandb_group is not None:
        group_name = wandb_group

    # Build a serializable config for logging only
    wandb_config = {
        "run_group": cfg.run_group,
        "seed": cfg.seed,
        "env_name": cfg.env_name,
        "data_ratio": cfg.data_ratio,
        "working_dir": cfg.working_dir,
        "restore_path": cfg.restore_path,
        "restore_epoch": cfg.restore_epoch,
        "agent": cfg.agent,
        "finetune": asdict(cfg.finetune),
        "train_steps": cfg.train_steps,
        "log_interval": cfg.log_interval,
        "eval_interval": cfg.eval_interval,
        "save_interval": cfg.save_interval,
        "eval_start": cfg.eval_start,
        "eval_tasks": cfg.eval_tasks,
        "eval_episodes": cfg.eval_episodes,
        "eval_temperature": cfg.eval_temperature,
        "eval_gaussian": cfg.eval_gaussian,
        "video_episodes": cfg.video_episodes,
        "video_frame_skip": cfg.video_frame_skip,
        "eval_on_cpu": cfg.eval_on_cpu,
        "train_on_test_goal": cfg.train_on_test_goal,
        "use_random_batch": cfg.use_random_batch,
        "training_fix_actor_goal": cfg.training_fix_actor_goal,
        "plot_interval": getattr(cfg, "plot_interval", 1000),
        "use_best_checkpoint": cfg.use_best_checkpoint,
    }
    setup_wandb(
        project="TTT_AllFinalRuns", group=group_name, name=exp_name, config=wandb_config
    )

    # Save current expanded config in the experiment dir
    os.makedirs(cfg.working_dir, exist_ok=True)
    exp_path = os.path.dirname(checkpoint_save_path(cfg, group_name))
    with open(os.path.join(exp_path, "config.yaml"), "w") as f:
        yaml.dump(wandb_config, f)

    # Set up environment and dataset.
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

    # Initialize agent.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    example_batch = train_dataset.sample(1)
    if config_agent["discrete"]:
        # Fill with the maximum action to let the agent know the action space size.
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

    # Restore agent.
    if cfg.restore_path is not None:
        print(f"[Restore] Restoring agent from {cfg.restore_path} at epoch {cfg.restore_epoch}")
        agent: MetaGCAgent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(cfg.working_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(cfg.working_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()


    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def check_compilation_cache():
        """Check JAX compilation cache status."""
        try:
            # Get compilation cache info
            cache_info = jax._src.api_util._compilation_cache_info()
            print(f"[JAX] Compilation cache entries: {len(cache_info) if cache_info else 0}")

            # Check if we can get more detailed cache info
            if hasattr(jax, 'get_compilation_cache'):
                cache = jax.get_compilation_cache()
                print(f"[JAX] Cache size: {len(cache) if cache else 0}")
        except Exception as e:
            print(f"[JAX] Could not get compilation cache info: {e}")

    def monitor_compilation():
        """Monitor JAX compilation events."""
        print(f"[JAX] Available devices: {jax.devices()}")
        print(f"[JAX] Default backend: {jax.default_backend()}")
        check_compilation_cache()

    # No / Full meta-learning here
    META_BATCH_SIZE = config_agent["meta_batch_size"]   # Jonas: much 10-100 or more?
    # Start by cheating case with Meta-learning
    # GC-BC-TTT-No critic VS GC-BC-TTT-No critic with meta-learning in this environment
    # MAML, FOMAML, Reptile
    META_LEARNING_ALGORITHM = cfg.meta_algorithm

    if cfg.train_steps == 0:
        for num_ttt_steps in cfg.finetune.num_steps_list:
            evaluation_loop(
                agent=agent,
                env=env,
                cfg=cfg,
                train_dataset=train_dataset,
                eval_logger=eval_logger,
                step=0,
                num_ttt_steps=num_ttt_steps)
        # Evaluate only

    plot_interval = getattr(cfg, "plot_interval", 100)

    # If no meta learning, than meta_batch_size is 1
    if META_LEARNING_ALGORITHM is None:
       assert META_BATCH_SIZE == 1, "If no meta learning, then meta_batch_size must be 1"

    for i in tqdm.tqdm(
        range(1, cfg.train_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        plot_dict = {}  # Initialize plot_dict for each iteration

        # Save agent at the beginning of the iteration to avoid skipping saving when
        # The loop enters a "continue" branch.
        if i % cfg.save_interval == 0:
            save_agent(agent, checkpoint_save_path(cfg, group_name), i)

        # Pretraining
        if cfg.use_random_batch:

            assert META_LEARNING_ALGORITHM is None, "If use_random_batch is True, then meta_learning_algorithm must be None"

            train_batch = train_dataset.sample(cfg.finetune.batch_size)
            agent, update_info = agent.update(train_batch, actor_only=cfg.finetune.actor_only)

        # Meta-learning aligned
        else:
            update_info = []
            # --- Inner Update (Meta-Learning) ---
            t_inner_start = time.time()
            memory_before_inner = get_memory_usage()

            for num_task in range(META_BATCH_SIZE):
                info_task = {}
                info_task_lowest_test_loss = {}

                fetched = False
                while not fetched:
                    start_state, goal = sample_task(train_dataset, env, test_task=cfg.train_on_test_goal, is_stitch_dataset="stitch" in cfg.env_name)
                    task_filter, max_len = get_task_filter(train_dataset, agent, start_state, goal, cfg)
                    if task_filter.sum() > 0:
                        fetched = True

                # Sample test batch for the task
                test_batch, test_batch_idx = fetch_meta_batch(
                    train_dataset,
                    goal,
                    agent,
                    cfg.finetune,
                    (task_filter, max_len),
                    meta_batch_size=128, # NOTE: I fixed this to avoid recompilation
                    fix_actor_goal=cfg.training_fix_actor_goal,
                    verbose=i%100 == 0
                )

                for inner_step in range(config_agent.get("inner_loop_steps", 1)):

                    train_batch, _ = fetch_meta_batch(
                        train_dataset,
                        goal,
                        agent,
                        cfg.finetune,
                        (task_filter, max_len),
                        exclude=test_batch_idx,
                        fix_actor_goal=cfg.training_fix_actor_goal
                    )

                    if train_batch is None:
                        break

                    if META_LEARNING_ALGORITHM is None:
                        # Inner optimizer is not reset for Joint training
                        reset_inner_opt = False
                    else:
                        # For meta-learning, the inner optimizer is reset at the first inner step
                        reset_inner_opt = inner_step == 0
                    agent, batch_info = agent.meta_inner_update(
                        num_task,
                        train_batch,
                        test_batch,
                        reset_inner_opt=reset_inner_opt,
                        average_test_gradients=cfg.average_test_gradients,
                        inner_step=inner_step,
                        actor_only=cfg.finetune.actor_only
                    )

                    # Save batch_info for plotting (for all meta-learning algorithms when it's time to plot)
                    if (i - 1) % plot_interval == 0:
                        # Save batch_info for current inner_step to a dictionary
                        plot_dict[inner_step] = batch_info


                    ### [Update information to log] ###

                    # Update pre-test info only at the first inner step
                    if inner_step == 0:
                        # take all keys that starts with "training/pre_test" and insert them into update_info_to_append
                        info_task.update({k: batch_info[k] for k in batch_info.keys() if k.startswith("pre_test")})
                        # Set info_task_lowest_test_loss
                        info_task_lowest_test_loss = batch_info

                    # Update lowest test loss info
                    if batch_info['test/actor/actor_loss'] < info_task_lowest_test_loss['test/actor/actor_loss']:
                        info_task_lowest_test_loss = batch_info
                    # If last inner step, prepare full info to log
                    if inner_step == config_agent.get("inner_loop_steps", 1) - 1:
                        # If cfg.use_best_checkpoint then use such info
                        if cfg.use_best_checkpoint:
                            batch_info = info_task_lowest_test_loss

                        # Include all other keys coming to batch_info
                        info_task.update({k: batch_info[k] for k in batch_info.keys() if not k.startswith("pre_test")})

                        # For all metrics with "training/pre_test/{metric_name}", compute "diff/{metric_name}" = "training/pre_test/{metric_name}" - "training/test/{metric_name}"
                        diff_stats = {}
                        for k in info_task.keys():
                            if k.startswith("pre_test/"):
                                metric_name = k[len("pre_test/"):]
                                diff_stats[f"diff/{metric_name}"] = info_task[f"pre_test/{metric_name}"] - info_task[f"test/{metric_name}"]

                        info_task.update(diff_stats)
                        # Append info batch
                        update_info.append(info_task)

            t_inner_end = time.time()
            memory_after_inner = get_memory_usage()
            inner_update_time = t_inner_end - t_inner_start

            if verbose:
                print(f"[Timer] Inner update took {inner_update_time:.4f} seconds.")
                print(
                    f"[Memory] Before inner: {memory_before_inner:.2f} MB, "
                    f"After inner: {memory_after_inner:.2f} MB "
                    f"(+{memory_after_inner - memory_before_inner:.2f} MB)"
                )
            if len(update_info) == 0:
                print(f"[Warning] No update info found for meta-step {i}")
                continue
            # Average batch info
            update_info = {k: np.mean([info[k] for info in update_info]) for k in update_info[0].keys()}

            # --- Meta Update (Meta-Learning) ---
            t_meta_start = time.time()
            memory_before_meta = get_memory_usage()

            # Joint training does model merging with eps=1.0
            if META_LEARNING_ALGORITHM is None:
                assert cfg.agent["merging_eps"] == 1.0

            agent: MetaGCAgent = agent.meta_update(
                use_model_merging=META_LEARNING_ALGORITHM == "reptile" or META_LEARNING_ALGORITHM is None,
                use_meta_optimizer=cfg.use_meta_optimizer,
                annealing=cfg.annealing,
                use_best_checkpoint=cfg.use_best_checkpoint,
            )
            memory_after_meta = get_memory_usage()
            t_meta_end = time.time()
            meta_update_time = t_meta_end - t_meta_start
            if verbose:
                print(f"[Timer] Meta update took {meta_update_time:.4f} seconds.")
                print(f"[Memory] Before meta: {memory_before_meta:.2f} MB, After meta: {memory_after_meta:.2f} MB (+{memory_after_meta - memory_before_meta:.2f} MB)\n")

            # Clear JAX caches periodically to prevent memory growth
            if i % 100_000 == 0:  # Every 100k iterations
                if verbose:
                    print(f"[JAX] Before clearing caches at iteration {i}")
                    monitor_compilation()
                jax.clear_caches()
                gc.collect()
                if verbose:
                    memory_mb = get_memory_usage()
                    print(f"[Memory] Cleared JAX caches. Current memory: {memory_mb:.2f} MB")
                    print(f"[JAX] After clearing caches:")
                    monitor_compilation()

        # Log metrics.
        if i % cfg.log_interval == 0:
            if verbose:
                print(update_info)
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                # So... we sample only 1 batch for validation
                val_batch = val_dataset.sample(config_agent["batch_size"])
                _, val_info = agent.total_loss(
                    val_batch,
                    grad_params=agent.network.params,
                    fixed_params=agent.network.params,
                )
                train_metrics.update(
                    {f"validation/{k}": v for k, v in val_info.items()}
                )
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / cfg.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i % cfg.eval_interval == 0 and i >= cfg.eval_start:

            for num_ttt_steps in cfg.finetune.num_steps_list:
                evaluation_loop(
                    agent=agent,
                    env=env,
                    cfg=cfg,
                    train_dataset=train_dataset,
                    eval_logger=eval_logger,
                    step=i,
                    num_ttt_steps=num_ttt_steps)

        # Create plot.
        if (i - 1) % plot_interval == 0 and not cfg.use_random_batch:
            plot_test_loss_inner_steps(plot_dict, i, cfg)

    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load base configuration
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file '{args.config_file}' not found.")
    cfg = load_config(args.config_file)

    # Apply command line overrides
    if args.finetune_num_steps is not None:
        override_config_value(cfg, 'finetune.num_steps_list', [int(args.finetune_num_steps)])

    if args.train_steps is not None:
        override_config_value(cfg, 'train_steps', args.train_steps)

    if args.meta_algorithm is not None:
        override_config_value(cfg, 'meta_algorithm', args.meta_algorithm)

    if args.seed is not None:
        override_config_value(cfg, 'seed', args.seed)

    if args.inner_loop_steps is not None:
        cfg.agent['inner_loop_steps'] = args.inner_loop_steps

    if args.meta_batch_size is not None:
        cfg.agent['meta_batch_size'] = args.meta_batch_size

    if args.eval_interval is not None:
        override_config_value(cfg, 'eval_interval', args.eval_interval)

    if args.use_random_batch:
        cfg.use_random_batch = True

    if args.train_on_test_goal:
        cfg.train_on_test_goal = True

    if args.actor_uniform_sample:
        cfg.agent['actor_geom_sample'] = False

    if args.finetune_mc_quantile is not None:
        override_config_value(cfg, 'finetune.mc_quantile', str(args.finetune_mc_quantile))

    if args.finetune_mc_quantile_train is not None:
        override_config_value(cfg, 'finetune.mc_quantile_train', str(args.finetune_mc_quantile_train))

    if args.save_interval is not None:
        override_config_value(cfg, 'save_interval', str(args.save_interval))

    if args.restore_path is not None:
        override_config_value(cfg, 'restore_path', args.restore_path)

    if args.restore_epoch is not None:
        override_config_value(cfg, 'restore_epoch', str(args.restore_epoch))

    if args.log_interval is not None:
        override_config_value(cfg, 'log_interval', str(args.log_interval))

    if args.finetune_lr is not None:
        override_config_value(cfg, 'finetune.lr', str(args.finetune_lr))
        cfg.agent['lr'] = float(args.finetune_lr)
        print(f"Finetune learning rate: {cfg.agent['lr']}")

    if args.agent_merging_eps is not None:
        cfg.agent['merging_eps'] = args.agent_merging_eps
    else:
        cfg.agent['merging_eps'] = 1.0

    if args.agent_max_grad_norm is not None:
        cfg.agent['max_grad_norm'] = args.agent_max_grad_norm

    if args.finetune_inner_lr is not None:
        override_config_value(cfg, 'finetune.inner_lr', str(args.finetune_inner_lr))
    # Use the finetune.inner_lr value (maybe after overwriting it)
    cfg.agent['inner_lr'] = cfg.finetune.inner_lr

    if args.average_test_gradients:
        cfg.average_test_gradients = True

    if args.training_fix_actor_goal is not None:
        cfg.training_fix_actor_goal = args.training_fix_actor_goal

    if args.plot_interval is not None:
        cfg.plot_interval = args.plot_interval

    if args.finetune_actor_only:
        cfg.finetune.actor_only = True

    if args.no_optimality:
        cfg.finetune.no_optimality = True

    if args.use_meta_optimizer:
        assert cfg.meta_algorithm == "reptile", "use_meta_optimizer is only supported for reptile algorithm"
        assert cfg.agent['merging_eps'] == cfg.agent['lr']
        cfg.use_meta_optimizer = True
    else:
        cfg.use_meta_optimizer = False
    if args.annealing:
        cfg.annealing = True
    else:
        cfg.annealing = False

    if args.use_best_checkpoint:
        cfg.use_best_checkpoint = True

    if args.actor_loss is not None:
        cfg.finetune.actor_loss = args.actor_loss
        cfg.agent['actor_loss'] = args.actor_loss
        print(f"Override: finetune.actor_loss = {args.actor_loss}")
        print(f"Override: agent.actor_loss = {args.actor_loss}")

    print(f"Number of steps list: {cfg.finetune.num_steps_list}")

    if cfg.finetune.filter_by_recursive_mdp:
        print("TTT with critique")
    else:
        print("TTT no critique")

    main(cfg, verbose=args.verbose if args.verbose is not None else False, wandb_group=args.wandb_group)

