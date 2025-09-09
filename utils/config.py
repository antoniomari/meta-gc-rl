from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class AgentConfig:
    """Configuration for the agent selection.

    Attributes:
        agent_name: Identifier of the agent to run (e.g., 'gciql', 'gcbc').
    """

    agent_name: str = "gciql"
    actor_loss: str = "bc"


@dataclass
class FinetuneConfig:
    """Hyperparameters for test-time fine-tuning.

    Attributes:
        ratio: Proportion of on-the-fly data to use for fine-tuning.
        num_steps: Gradient steps per episode for fine-tuning.
        lr: Fine-tuning learning rate.
        actor_loss: Loss type used during fine-tuning (e.g., 'ddpgbc').
        alpha: Optional temperature/BC coefficient override.
        batch_size: Batch size for fine-tuning updates.
        fix_actor_goal: Probability to fix actor goal to evaluation goal.
        mc_quantile: Quantile threshold for Monte Carlo filtering of demos.
        mc_slack: Extra padding steps kept around selected segments.
        sorb_len: Number of subgoals for off-policy GCFT.
        filter_by_mc: Enable on-policy GCFT filtering by MC returns.
        filter_by_td: Enable off-policy GCFT filtering by TD error.
        relevance_by_value: Use value-based relevance criterion instead of rewards.
        saw: Enable SAW behavior at fine-tuning time.
        reset_after_horizon: Reset agent after reaching replan horizon.
        mc_similarity_threshold: Threshold for MC similarity when filtering.
        filter_by_recursive_mdp: Enable recursive-MDP filtering.
        min_steps: Minimum number of fine-tuning steps to perform.
        replan_horizon: Horizon for replanning during fine-tuning.
    """

    ratio: float = 0.5
    num_steps: int = 0
    lr: float = 3e-5
    actor_loss: str = "ddpgbc"
    alpha: Optional[float] = None
    batch_size: int = 1024
    fix_actor_goal: float = 0.0
    mc_quantile: float = 0.2
    mc_slack: int = 5
    sorb_len: int = 10
    filter_by_mc: bool = False
    filter_by_td: bool = False
    relevance_by_value: bool = False
    saw: bool = False
    reset_after_horizon: bool = False
    mc_similarity_threshold: float = 1.0
    filter_by_recursive_mdp: bool = False
    min_steps: int = 10
    replan_horizon: int = 100

    def __getitem__(self, key):
        """Make the config subscriptable like a dictionary."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Allow setting attributes like a dictionary."""
        setattr(self, key, value)

    def get(self, key, default=None):
        """Provide dictionary-like get method."""
        return getattr(self, key, default)


@dataclass
class GCTTTConfig:
    """Top-level configuration reflecting the fields in default.yaml.

    Attributes cover experiment identity, environment and dataset selection,
    optional checkpoints, agent and fine-tuning sub-configs, and evaluation
    controls (frequency, tasks, and device hints).
    """

    run_group: str = "debug"
    seed: int = 0
    env_name: str = "pointmaze-medium-navigate-v0"
    data_ratio: float = 1.0
    working_dir: str = "exp"
    restore_path: Optional[str] = None
    restore_epoch: Optional[int] = None

    agent: AgentConfig = field(default_factory=AgentConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    train_steps: int = 1_000_000
    log_interval: int = 5000
    eval_interval: int = 100_000
    save_interval: int = 100_000
    eval_start: int = 800_000

    eval_tasks: Optional[List[int]] = None
    eval_episodes: int = 50
    eval_temperature: float = 0.0
    eval_gaussian: Optional[float] = None
    video_episodes: int = 0
    video_frame_skip: int = 3
    eval_on_cpu: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GCTTTConfig":
        return cls(
            run_group=data.get("run_group", cls.run_group),
            seed=int(data.get("seed", cls.seed)),
            env_name=str(data.get("env_name", cls.env_name)),
            data_ratio=float(data.get("data_ratio", cls.data_ratio)),
            working_dir=str(data.get("working_dir", cls.working_dir)),
            restore_path=data.get("restore_path", None),
            restore_epoch=data.get("restore_epoch", None),
            agent=AgentConfig(**data.get("agent", {})),
            finetune=FinetuneConfig(**data.get("finetune", {})),
            train_steps=int(data.get("train_steps", cls.train_steps)),
            log_interval=int(data.get("log_interval", cls.log_interval)),
            eval_interval=int(data.get("eval_interval", cls.eval_interval)),
            save_interval=int(data.get("save_interval", cls.save_interval)),
            eval_start=int(data.get("eval_start", cls.eval_start)),
            eval_tasks=data.get("eval_tasks", None),
            eval_episodes=int(data.get("eval_episodes", cls.eval_episodes)),
            eval_temperature=float(data.get("eval_temperature", cls.eval_temperature)),
            eval_gaussian=data.get("eval_gaussian", None),
            video_episodes=int(data.get("video_episodes", cls.video_episodes)),
            video_frame_skip=int(data.get("video_frame_skip", cls.video_frame_skip)),
            eval_on_cpu=int(data.get("eval_on_cpu", cls.eval_on_cpu)),
        )


def load_config(config_path: Union[str, Path]) -> GCTTTConfig:
    """Load configuration from a YAML file into a GCTTTConfig object.

    Args:
        config_path: Path to a YAML file with fields like default.yaml.

    Returns:
        Parsed configuration as a GCTTTConfig instance.
    """
    path = Path(config_path)
    with path.open("r") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}
    return GCTTTConfig.from_dict(data)


# Controls what gets imported when you do `from utils.config import *`
__all__ = [
    "AgentConfig",
    "FinetuneConfig",
    "GCTTTConfig",
    "load_config",
]
