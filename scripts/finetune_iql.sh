#!/bin/bash
#SBATCH --job-name=ft_iql
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --output=/cluster/home/anmari/gc_ttt/logs/pointmaze/ft_%j.out
#SBATCH --error=/cluster/home/anmari/gc_ttt/logs/pointmaze/ft_%j.err

cd /cluster/home/anmari/gc_ttt
source venv/bin/activate

# Default values
config_file="meta_antmaze_iql.yaml"
seed=0
train_settings="TEST"  # "ALL", "TEST", or "RANDOM"
average_test_gradients="False"
use_pretrained_model="False"
meta_algorithm="fomaml"
inner_loop_steps=1
meta_batch_size=1
inner_lr=3e-5
lr=3e-4
train_steps=10000
eval_interval=2000
save_interval="None"
training_fix_actor_goal="None"
merging_eps=1.0
actor_only="False"
use_meta_optimizer="False"
annealing="False"
use_best_checkpoint="False"
wandb_group=""
restore_path="/cluster/home/anmari/gc_ttt/exp/antmaze_stitch/iql"
restore_epoch=100000
mc=0.2
actor_loss=""

# Parse named arguments (supports both --key=value and --key value formats)
while [[ $# -gt 0 ]]; do
    case $1 in
        --config_file=*)
            config_file="${1#*=}"
            shift
            ;;
        --config_file)
            config_file="$2"
            shift 2
            ;;
        --seed=*)
            seed="${1#*=}"
            shift
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --train_settings=*)
            train_settings="${1#*=}"
            shift
            ;;
        --train_settings)
            train_settings="$2"
            shift 2
            ;;
        --average_test_gradients=*)
            average_test_gradients="${1#*=}"
            shift
            ;;
        --average_test_gradients)
            average_test_gradients="$2"
            shift 2
            ;;
        --use_pretrained_model=*)
            use_pretrained_model="${1#*=}"
            shift
            ;;
        --use_pretrained_model)
            use_pretrained_model="$2"
            shift 2
            ;;
        --meta_algorithm=*)
            meta_algorithm="${1#*=}"
            shift
            ;;
        --meta_algorithm)
            meta_algorithm="$2"
            shift 2
            ;;
        --inner_loop_steps=*)
            inner_loop_steps="${1#*=}"
            shift
            ;;
        --inner_loop_steps)
            inner_loop_steps="$2"
            shift 2
            ;;
        --meta_batch_size=*)
            meta_batch_size="${1#*=}"
            shift
            ;;
        --meta_batch_size)
            meta_batch_size="$2"
            shift 2
            ;;
        --inner_lr=*)
            inner_lr="${1#*=}"
            shift
            ;;
        --inner_lr)
            inner_lr="$2"
            shift 2
            ;;
        --lr=*)
            lr="${1#*=}"
            shift
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --train_steps=*)
            train_steps="${1#*=}"
            shift
            ;;
        --train_steps)
            train_steps="$2"
            shift 2
            ;;
        --eval_interval=*)
            eval_interval="${1#*=}"
            shift
            ;;
        --eval_interval)
            eval_interval="$2"
            shift 2
            ;;
        --save_interval=*)
            save_interval="${1#*=}"
            shift
            ;;
        --save_interval)
            save_interval="$2"
            shift 2
            ;;
        --training_fix_actor_goal=*)
            training_fix_actor_goal="${1#*=}"
            shift
            ;;
        --training_fix_actor_goal)
            training_fix_actor_goal="$2"
            shift 2
            ;;
        --merging_eps=*)
            merging_eps="${1#*=}"
            shift
            ;;
        --merging_eps)
            merging_eps="$2"
            shift 2
            ;;
        --actor_only=*)
            actor_only="${1#*=}"
            shift
            ;;
        --actor_only)
            actor_only="$2"
            shift 2
            ;;
        --use_meta_optimizer=*)
            use_meta_optimizer="${1#*=}"
            shift
            ;;
        --use_meta_optimizer)
            use_meta_optimizer="$2"
            shift 2
            ;;
        --annealing=*)
            annealing="${1#*=}"
            shift
            ;;
        --annealing)
            annealing="$2"
            shift 2
            ;;
        --use_best_checkpoint=*)
            use_best_checkpoint="${1#*=}"
            shift
            ;;
        --use_best_checkpoint)
            use_best_checkpoint="$2"
            shift 2
            ;;
        --wandb_group=*)
            wandb_group="${1#*=}"
            shift
            ;;
        --wandb_group)
            wandb_group="$2"
            shift 2
            ;;
        --restore_path=*)
            restore_path="${1#*=}"
            shift
            ;;
        --restore_path)
            restore_path="$2"
            shift 2
            ;;
        --restore_epoch=*)
            restore_epoch="${1#*=}"
            shift
            ;;
        --restore_epoch)
            restore_epoch="$2"
            shift 2
            ;;
        --actor_loss=*)
            actor_loss="${1#*=}"
            shift
            ;;
        --actor_loss)
            actor_loss="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$train_settings" = "TEST" ]; then
    train_settings_flag="--train_on_test_goal"
elif [ "$train_settings" = "RANDOM" ]; then
    train_settings_flag="--use_random_batch"
else
    train_settings_flag=""
fi


if [ "$average_test_gradients" = "True" ]; then
    average_test_gradients_flag="--average_test_gradients"
else
    average_test_gradients_flag=""
fi

# Only pass --meta_algorithm if it's not "PT"
meta_algorithm_flag=""
if [ "$meta_algorithm" != "PT" ]; then
    meta_algorithm_flag="--meta_algorithm $meta_algorithm"
fi

if [ "$use_pretrained_model" = "True" ]; then
    use_pretrained_model_flag="--restore_path $restore_path --restore_epoch $restore_epoch"
else
    use_pretrained_model_flag=""
fi

if [ "$actor_only" = "True" ]; then
    actor_only_flag="--finetune.actor_only"
else
    actor_only_flag=""
fi

if [ "$use_meta_optimizer" = "True" ]; then
    use_meta_optimizer_flag="--use_meta_optimizer"
else
    use_meta_optimizer_flag=""
fi

if [ "$annealing" = "True" ]; then
    annealing_flag="--annealing"
else
    annealing_flag=""
fi

if [ "$use_best_checkpoint" = "True" ]; then
    use_best_checkpoint_flag="--use_best_checkpoint"
else
    use_best_checkpoint_flag=""
fi

if [ -n "$wandb_group" ]; then
    wandb_group_flag="--wandb_group $wandb_group"
else
    wandb_group_flag=""
fi

if [ -n "$actor_loss" ]; then
    actor_loss_flag="--actor_loss $actor_loss"
else
    actor_loss_flag=""
fi

#      --agent.max_grad_norm 1000 \
#      --wandb_group "fomaml_max_grad_1000" \
#       --wandb_group "original_params_${meta_algorithm}" \
# --no_optimality

cmd="python meta_main.py \"$config_file\" \
      $meta_algorithm_flag \
      --seed $seed \
      --agent.inner_loop_steps $inner_loop_steps \
      --agent.meta_batch_size $meta_batch_size \
      --log_interval 1 \
      --eval_interval $eval_interval \
      --train_steps $train_steps \
      --save_interval $save_interval \
      --finetune.lr $lr \
      --finetune.inner_lr $inner_lr \
      --training_fix_actor_goal $training_fix_actor_goal \
      --agent.merging_eps $merging_eps \
      $train_settings_flag \
      $average_test_gradients_flag \
      $use_pretrained_model_flag \
      $actor_only_flag \
      $use_meta_optimizer_flag \
      $annealing_flag \
      $use_best_checkpoint_flag \
      $wandb_group_flag \
      $actor_loss_flag"

echo "Running command:"
echo $cmd

eval $cmd

# This script accepts named arguments in the format --key=value or --key value
# Arguments can be provided in any order.
#
# Available arguments:
#   --config_file (default: meta_antmaze_iql.yaml)
#   --seed (default: 0)
#   --train_settings (default: TEST) - "ALL", "TEST", or "RANDOM"
#   --average_test_gradients (default: False)
#   --use_pretrained_model (default: False)
#   --meta_algorithm (default: fomaml)
#   --inner_loop_steps (default: 1)
#   --meta_batch_size (default: 1)
#   --inner_lr (default: 3e-5)
#   --lr (default: 3e-4)
#   --train_steps (default: 10000)
#   --eval_interval (default: 2000)
#   --save_interval (default: None)
#   --training_fix_actor_goal (default: None)
#   --merging_eps (default: 1.0)
#   --actor_only (default: False)
#   --use_meta_optimizer (default: False)
#   --annealing (default: False)
#   --use_best_checkpoint (default: False)
#   --wandb_group (optional: if empty, --wandb_group flag won't be passed)
#   --restore_path (default: /cluster/home/anmari/gc_ttt/exp/antmaze_stitch/iql)
#   --restore_epoch (default: 100000)
#   --actor_loss (optional: if set, overrides both cfg.finetune.actor_loss and cfg.agent["actor_loss"])
#
# Examples:
#   # Using --key=value format
#   bash scripts/antmaze/finetune_iql.sh --config_file=config/meta_antmaze_no_opt.yaml --seed=0 --train_settings=TEST
#
#   # Using --key value format
#   bash scripts/antmaze/finetune_iql.sh --config_file config/meta_antmaze_no_opt.yaml --seed 0 --meta_algorithm PT
#
#   # With wandb_group and restore path
#   bash scripts/antmaze/finetune_iql.sh --config_file=meta_antmaze_iql.yaml --seed=0 --wandb_group="my-group" --restore_path="/path/to/checkpoint" --restore_epoch=50000
#
#   # Without wandb_group (it will be omitted)
#   bash scripts/antmaze/finetune_iql.sh --config_file=meta_antmaze_iql.yaml --seed=0 --restore_path="/path/to/checkpoint" --restore_epoch=50000


