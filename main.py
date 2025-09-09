import sys
import yaml
import os
import random
import time
from collections import defaultdict

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import wandb
import jax
from dataclasses import asdict
import numpy as np
import tqdm
from agents import agents
from ml_collections import FrozenConfigDict
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_wandb_video, setup_wandb
from utils.config import GCTTTConfig, load_config
import matplotlib.pyplot as plt
import io
from PIL import Image


def main(cfg: GCTTTConfig):

    # Load agent defaults
    import importlib

    cfg_dict = asdict(cfg)
    agent_cfg = importlib.import_module(f"agents.{cfg.agent.agent_name}").get_config()
    for k, v in agent_cfg.items():
        if k not in cfg_dict["agent"]:
            cfg_dict["agent"][k] = v

    # Set up logger.
    # split env_name by '-'
    env_name_split = cfg.env_name.split("-")
    # set wandb_env_name as the first part and second part of env_name_split
    wandb_env_name = env_name_split[0] + "-" + env_name_split[2]
    exp_name = get_exp_name(cfg)
    setup_wandb(
        project="TTT_AllFinalRuns", group=cfg.run_group, name=exp_name, config=cfg_dict
    )

    # Save current expanded config in the experiment dir
    os.makedirs(cfg.working_dir, exist_ok=True)
    with open(os.path.join(cfg.working_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg_dict, f)

    # Set up environment and dataset.
    config_agent = cfg_dict["agent"]
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
    agent = agent_class.create(
        cfg.seed,
        example_batch["observations"],
        example_batch["actions"],
        config_agent,
    )

    # Restore agent.
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(cfg.working_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(cfg.working_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(
        range(1, cfg.train_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        # Update agent. TODO: determine batch shape
        batch = train_dataset.sample(config_agent["batch_size"])
        agent, update_info = agent.update(
            batch
        )  # I assume training logic is in the agent
        # TODO: check update_info structure

        # Log metrics.
        if i % cfg.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                # So... we sample only 1 batch for validation
                val_batch = val_dataset.sample(config_agent["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
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
            print("Evaluating...")
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

            # TODO: check this task_infos
            num_tasks = (
                cfg.eval_tasks if cfg.eval_tasks is not None else len(task_infos)
            )
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]["task_name"]
                # Test-time fine-tuning happens in here
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config_agent,
                    finetune_config=cfg.finetune,
                    num_eval_episodes=cfg.eval_episodes,
                    num_video_episodes=cfg.video_episodes,
                    train_dataset=train_dataset,
                    video_frame_skip=cfg.video_frame_skip,
                    eval_temperature=cfg.eval_temperature,
                    eval_gaussian=cfg.eval_gaussian,
                    exp_name=exp_name,
                )

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
                    wandb.log({"Zfig": wandb.Image(img_array)})
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
                        f"evaluation/{task_name}_{k}": v
                        for k, v in eval_info.items()
                        if k in metric_names
                    }
                )
                # wandb.log({f'evaluation_logged/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names})
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)

            # TODO: check are we averaging over task?
            for k, v in overall_metrics.items():
                eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

            if cfg.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics["video"] = video

            try:
                # Assuming 'i' is your global training step counter
                wandb.log(eval_metrics)
            except Exception as e:
                print(f"Error during wandb.log: {e}")

            # Log to the separate eval_logger if it exists
            try:
                eval_logger.log(eval_metrics, step=i)
            except Exception as e:
                print(f"Error logging to eval_logger: {e}")

            time.sleep(10)  # Sleep for a minute to avoid too many logs in a short time.

        # Save agent.
        if i % cfg.save_interval == 0:
            save_agent(agent, cfg.working_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("No .yaml configs found.")

    if not os.path.exists(sys.argv[1]):
        raise FileNotFoundError(f"Config file '{sys.argv[1]}' not found.")
    cfg = load_config(sys.argv[1])
    main(cfg)
