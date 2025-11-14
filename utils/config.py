from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypedDict, Literal, cast

import yaml


class AgentConfig(TypedDict, total=False):
    """Dictionary-style configuration for agent selection with flexible keys.

    Known keys are documented here; additional keys may be added at runtime.
    """
    agent_name: str  # e.g., 'gciql', 'gcbc'
    actor_loss: str  # e.g., 'bc', 'ddpgbc'


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
        actor_only: If True, only update actor during training (not critic/value).
    """

    ratio: float = 0.5
    num_steps: int = 0
    num_steps_list: List[int] = field(default_factory=lambda: [0, 5, 10, 20, 50])
    lr: float = 3e-5
    inner_lr: Optional[float] = None
    actor_loss: str = "ddpgbc"
    alpha: Optional[float] = None
    batch_size: int = 1024
    fix_actor_goal: float = 0.0
    mc_quantile: float = 0.2
    mc_quantile_train: Optional[float] = None
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
    actor_only: bool = False

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

    Attributes:
        plot_interval: Interval (in training steps) for creating plots during training.
    """

    run_group: str = "debug"
    seed: int = 0
    env_name: str = "pointmaze-medium-navigate-v0"
    data_ratio: float = 1.0
    working_dir: str = "exp"
    restore_path: Optional[str] = None
    restore_epoch: Optional[int] = None

    agent: Dict[str, Any] = field(default_factory=lambda: {"agent_name": "gciql", "actor_loss": "bc"})
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    meta_algorithm: Optional[Literal["maml", "fomaml", "reptile"]] = None  # Options: "maml", "fomaml", "reptile"
    train_on_test_goal: bool = False  # Whether to use test goal for training batch fetching
    use_random_batch: bool = False  # Whether to use random batch sampling instead of goal-conditioned
    average_test_gradients: bool = False  # Whether to average test gradients across tasks

    train_steps: int = 1_000_000
    log_interval: int = 5000
    eval_interval: int = 100_000
    save_interval: int = 100_000
    eval_start: int = 800_000
    test_batch_fraction: float = 0.2

    eval_tasks: Optional[List[int]] = None
    eval_episodes: int = 50
    eval_temperature: float = 0.0
    eval_gaussian: Optional[float] = None
    video_episodes: int = 0
    video_frame_skip: int = 3
    eval_on_cpu: int = 1
    training_fix_actor_goal: float = 1.0
    plot_interval: int = 100  # Interval for creating plots during training

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
            agent=data.get("agent", {"agent_name": "gciql", "actor_loss": "bc"}),
            finetune=FinetuneConfig(**data.get("finetune", {})),
            meta_algorithm=cast(Literal["maml", "fomaml", "reptile"], data.get("meta_algorithm", cls.meta_algorithm)),
            train_on_test_goal=bool(data.get("train_on_test_goal", cls.train_on_test_goal)),
            use_random_batch=bool(data.get("use_random_batch", cls.use_random_batch)),
            average_test_gradients=bool(data.get("average_test_gradients", cls.average_test_gradients)),
            train_steps=int(data.get("train_steps", cls.train_steps)),
            log_interval=int(data.get("log_interval", cls.log_interval)),
            eval_interval=int(data.get("eval_interval", cls.eval_interval)),
            save_interval=int(data.get("save_interval", cls.save_interval)),
            eval_start=int(data.get("eval_start", cls.eval_start)),
            test_batch_fraction=float(data.get("test_batch_fraction", cls.test_batch_fraction)),
            eval_tasks=data.get("eval_tasks", None),
            eval_episodes=int(data.get("eval_episodes", cls.eval_episodes)),
            eval_temperature=float(data.get("eval_temperature", cls.eval_temperature)),
            eval_gaussian=data.get("eval_gaussian", None),
            video_episodes=int(data.get("video_episodes", cls.video_episodes)),
            video_frame_skip=int(data.get("video_frame_skip", cls.video_frame_skip)),
            eval_on_cpu=int(data.get("eval_on_cpu", cls.eval_on_cpu)),
            training_fix_actor_goal=float(data.get("training_fix_actor_goal", cls.training_fix_actor_goal)),
            plot_interval=int(data.get("plot_interval", cls.plot_interval)),
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
        try:
            # Try safe_load first (preferred for security)
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        except yaml.constructor.ConstructorError as e:
            # If safe_load fails due to Python-specific tags (like !!python/tuple),
            # fall back to unsafe_load. This is safe here since we're loading
            # config files saved by our own codebase.
            if "python" in str(e) or "tuple" in str(e):
                f.seek(0)  # Reset file pointer
                data: Dict[str, Any] = yaml.unsafe_load(f) or {}
            else:
                raise
    return GCTTTConfig.from_dict(data)


# Controls what gets imported when you do `from utils.config import *`
__all__ = [
    "AgentConfig",
    "FinetuneConfig",
    "GCTTTConfig",
    "load_config",
]
