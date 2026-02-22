# How to Run Experiments

All commands should be run from the project root (`gc_ttt/`).

## Step 1: Pretraining

Run `send_jobs_pretraining.sh` for each of the 9 config files in `config/pretraining/`.

The script uses the following default hyperparameters:

| Hyperparameter              | Default Value |
|-----------------------------|---------------|
| `train_settings`            | RANDOM        |
| `average_test_gradients`    | False         |
| `use_pretrained_model`      | False         |
| `actor_only`                | False         |
| `use_meta_optimizer`        | False         |
| `annealing`                 | False         |
| `use_best_checkpoint`       | False         |
| `seeds`                     | 0             |
| `inner_loop_steps`          | 1             |
| `meta_batch_sizes`          | 1             |
| `inner_lrs`                 | 3e-4          |
| `outer_lr`                  | 3e-4          |
| `train_steps`               | 500000        |
| `eval_interval`             | 2000000       |
| `save_interval`             | 50000         |
| `training_fix_actor_goal`   | 1.0           |
| `merging_eps`               | 1             |

```bash
# Expert BC (3 envs)
bash scripts/send_jobs_pretraining.sh config/pretraining/pointmaze_expert_bc.yaml
bash scripts/send_jobs_pretraining.sh config/pretraining/antmaze_expert_bc.yaml
bash scripts/send_jobs_pretraining.sh config/pretraining/humanoidmaze_expert_bc.yaml

# Expert IQL-BC (3 envs)
bash scripts/send_jobs_pretraining.sh config/pretraining/pointmaze_iql_bc.yaml
bash scripts/send_jobs_pretraining.sh config/pretraining/antmaze_iql_bc.yaml
bash scripts/send_jobs_pretraining.sh config/pretraining/humanoidmaze_iql_bc.yaml

# Stitch IQL DDPG+BC (3 envs)
bash scripts/send_jobs_pretraining.sh config/pretraining/pointmaze_iql_ddpgbc.yaml
bash scripts/send_jobs_pretraining.sh config/pretraining/antmaze_iql_ddpgbc.yaml
bash scripts/send_jobs_pretraining.sh config/pretraining/humanoidmaze_iql_ddpgbc.yaml

```

Each invocation submits an `sbatch` job via `scripts/finetune_iql.sh` with the configured hyperparameters.

## Step 2: Evaluate Baselines

Once pretraining is complete, evaluate the baseline models using `send_eval_jobs.sh`.

```bash
bash scripts/send_eval_jobs.sh <env> <model>
```

| Argument | Values                                                                                                           |
|----------|------------------------------------------------------------------------------------------------------------------|
| `env`    | `pointmaze_expert`, `pointmaze_stitch`, `antmaze_expert`, `antmaze_stitch`, `humanoidmaze_expert`, `humanoidmaze_stitch` |
| `model`  | `bc`, `iql_bc`, `iql_ddpgbc`                                                                                    |

The checkpoints are looked up under `exp/<env>/<model>/`. For example:

```bash
bash scripts/send_eval_jobs.sh pointmaze_expert bc          # -> exp/pointmaze_expert/bc/
bash scripts/send_eval_jobs.sh antmaze_expert iql_bc        # -> exp/antmaze_expert/iql_bc/
bash scripts/send_eval_jobs.sh humanoidmaze_stitch iql_ddpgbc  # -> exp/humanoidmaze_stitch/iql_ddpgbc/
```

Each invocation submits one SLURM job per combination of seed, TTT steps, and restore epoch.
Default evaluation settings (editable inside the script):

| Setting                | Default   |
|------------------------|-----------|
| `restore_epochs`       | 300k-500k (step 50k) |
| `seeds`                | 0, 1, 2   |
| `num_ttt_steps_list`   | 0, 10, 20, 50, 100, 200 |
| `eval_episodes`        | 50        |
| `finetune_lr`          | 3e-4      |
| `finetune_actor_loss`  | bc        |

## Step 3: Post-Training Meta-learning

After pretraining, post-train with meta-learning with different algorithms using `send_jobs.sh`.

```bash
bash scripts/send_jobs.sh <config_file> [restore_epoch] [use_best_checkpoint] [inner_lr] [outer_lr] [algo]
```

| Argument               | Default   | Description                                |
|------------------------|-----------|--------------------------------------------|
| `config_file`          | (required)| Config from `config/pretraining/`          |
| `restore_epoch`        | 400000    | Epoch of the pretrained checkpoint         |
| `use_best_checkpoint`  | True      | If True, uses `inner_loop_steps=(100)`     |
| `inner_lr`             | 3e-4      | Inner learning rate                        |
| `outer_lr`             | 3e-4      | Outer learning rate (ignored for ttt_reptile) |
| `algo`                 | ttt_reptile | `continual_ttt`, `tt_fomaml`, or `tt_reptile` |

Algo name mapping to internal `meta_algorithm`:

| CLI name        | Internal value |
|-----------------|----------------|
| `continual_ttt` | PT             |
| `tt_fomaml`    | fomaml         |
| `tt_reptile`   | reptile        |

The script internally loops over `seeds=(0 1 2)`. For `tt_reptile`, it also sweeps
`merging_eps=(0.01 0.1 0.2 0.5)` with `lr=merging_eps`. For others, `merging_eps=1`.

### Hyperparameter grid

Run for each `<config>` file and each `inner_lr` in `{3e-4, 3e-5}`:

```bash
# TT-Fomaml
bash scripts/send_jobs.sh <config> 400000 True 3e-4 3e-4 tt_fomaml
bash scripts/send_jobs.sh <config> 400000 True 3e-5 3e-4 tt_fomaml

# Continual-TTT
bash scripts/send_jobs.sh <config> 400000 True 3e-4 3e-4 continual_ttt
bash scripts/send_jobs.sh <config> 400000 True 3e-5 3e-4 continual_ttt

# RRAB (sweeps merging_eps internally: 0.01, 0.1, 0.2, 0.5)
bash scripts/send_jobs.sh <config> 400000 True 3e-4 3e-4 tt_reptile
bash scripts/send_jobs.sh <config> 400000 True 3e-5 3e-4 tt_reptile
```

For example, to run the full grid for `antmaze_expert_bc`:

```bash
for inner_lr in 3e-4 3e-5; do
    for algo in continual_ttt tt_fomaml tt_reptile; do
        bash scripts/send_jobs.sh config/pretraining/antmaze_expert_bc.yaml 400000 True $inner_lr 3e-4 $algo
    done
done
```

To run the sweep over num of inner steps (1 10 20 50 100 200):
```bash
for inner_lr in 3e-4 3e-5; do
    for algo in continual_ttt tt_fomaml; do
        bash scripts/send_jobs.sh config/pretraining/antmaze_expert_bc.yaml 400000 False $inner_lr 3e-4 $algo
    done
done
```

### Summary of jobs per config

| Algo            | inner_lr values | merging_eps values       | seeds | Jobs per config |
|-----------------|-----------------|--------------------------|-------|-----------------|
| `continual_ttt` | 3e-4, 3e-5      | 1                        | 0,1,2 | 6               |
| `tt_fomaml`    | 3e-4, 3e-5      | 1                        | 0,1,2 | 6               |
| `tt_reptile`   | 3e-4, 3e-5      | 0.01, 0.1, 0.2, 0.5     | 0,1,2 | 24              |
| **Total**       |                 |                          |       | **36**          |

## Step 4: Evaluate Finetuned Models

After meta-finetuning completes, evaluate the finetuned models using the same
`send_eval_jobs.sh` from Step 2, passing the finetuned model path as the optional
third argument:

```bash
bash scripts/send_eval_jobs.sh <env> <model> <restore_path_base>
```

| Argument             | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `env`                | Environment (same as Step 2)                                       |
| `model`              | Model type (same as Step 2)                                        |
| `restore_path_base`  | Full base path to the finetuned experiment directory               |

For example, to evaluate a FOMAML-finetuned antmaze model:

```bash
bash scripts/send_eval_jobs.sh antmaze_stitch iql_ddpgbc \
    /cluster/home/anmari/gc_ttt/exp/antmaze_stitch/iql_ddpgbc/FT-ALL-ant_sti-ddpgbc-JTAB-100-1-lr0.0003-ilr0.0003
```

This will look for checkpoints under `<restore_path_base>/seed{0,1,2}/` and evaluate
across the same grid of epochs, seeds, and TTT steps as in Step 2.

When the third argument is omitted, the script falls back to the default baseline path

To evaluate `antmaze_expert/bc` models





## 1. antmaze_expert/bc

### a. TT-Fomaml (with various inner steps and learning rates as well as early-stop version)
```bash
# List all relevant model directories to evaluate
model_paths=(
    exp/antmaze_stitch/iql_ddpgbc/FT-ALL-ant_sti-ddpgbc-FFAB-100-1-lr0.0003-ilr0.0003
    exp/antmaze_stitch/iql_ddpgbc/FT-ALL-ant_sti-ddpgbc-JTA-1-1-lr0.0003-ilr0.0003
    exp/antmaze_stitch/iql_ddpgbc/FT-ALL-ant_sti-ddpgbc-JTA-10-1-lr0.0003-ilr0.0003
    exp/antmaze_stitch/iql_ddpgbc/FT-ALL-ant_sti-ddpgbc-JTA-20-1-lr0.0003-ilr0.0003
    exp/antmaze_stitch/iql_ddpgbc/FT-ALL-ant_sti-ddpgbc-JTA-50-1-lr0.0003-ilr0.0003
    exp/antmaze_stitch/iql_ddpgbc/FT-ALL-ant_sti-ddpgbc-JTA-100-1-lr0.0003-ilr0.0003
    exp/antmaze_stitch/iql_ddpgbc/FT-ALL-ant_sti-ddpgbc-JTA-200-1-lr0.0003-ilr0.0003
    exp/antmaze_stitch/iql_ddpgbc/FT-ALL-ant_sti-ddpgbc-JTAB-100-1-lr0.0003-ilr0.0003
)

for path in "${model_paths[@]}"; do
    bash scripts/send_eval_jobs.sh antmaze_stitch iql_ddpgbc default False "$path"
done
```

### b. Continual-TTT





### To run eval TTT (Actor+Critic+Value) vs TTT Actor only
Example: Humanoidmaze stitch

```bash
for lr in 3e-05 1e-04 3e-04 1e-03 3e-03; do
    for actor_only in True False; do
        bash scripts/send_eval_jobs.sh  humanoidmaze_stitch iql_ddpgbc $lr $actor_only
    done
done
```
