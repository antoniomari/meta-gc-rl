import sys
import yaml
import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from agents import agents
from ml_collections import FrozenConfigDict
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_wandb_video, setup_wandb


def main(cfg):

    # load agent defaults
    import importlib
    agent_cfg = importlib.import_module(f"agents.{cfg['agent']['agent_name']}").get_config()
    for k, v in agent_cfg.items():
        if k not in cfg['agent']:
            cfg['agent'][k] = v
    cfg = FrozenConfigDict(cfg)

    # Set up logger.
    exp_name = get_exp_name(cfg.seed)
    setup_wandb(project='OGBench', group=cfg.run_group, name=exp_name, config=cfg.to_dict())

    os.makedirs(cfg.working_dir, exist_ok=True)
    with open(os.path.join(cfg.working_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg.to_dict(), f)

    # Set up environment and dataset.
    config = cfg.agent
    env, train_dataset, val_dataset = make_env_and_datasets(cfg.env_name, frame_stack=config['frame_stack'])

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        cfg.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(cfg.working_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(cfg.working_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, cfg.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % cfg.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i % cfg.eval_interval == 0:
            if cfg.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = cfg.eval_tasks if cfg.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=cfg.eval_episodes,
                    num_video_episodes=cfg.video_episodes,
                    num_finetune_steps=cfg.finetune_steps,
                    finetune_lr=cfg.finetune_lr,
                    train_dataset=train_dataset,
                    video_frame_skip=cfg.video_frame_skip,
                    eval_temperature=cfg.eval_temperature,
                    eval_gaussian=cfg.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if cfg.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % cfg.save_interval == 0:
            save_agent(agent, cfg.working_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('No .yaml configs found.')

    if not os.path.exists(sys.argv[1]):
        raise FileNotFoundError(f"Config file '{sys.argv[1]}' not found.")
    with open(sys.argv[1], 'r') as file:
        cfg = yaml.safe_load(file)

    main(cfg)
