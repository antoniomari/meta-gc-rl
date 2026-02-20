#!/bin/bash
# Script to submit meta-finetuning jobs for all environments.
# Inlines the worker logic from finetune_iql.sh into heredoc sbatch submissions.
#
# Usage:
#   bash scripts/send_jobs.sh <config_file> [restore_epoch] [use_best_checkpoint] [inner_lr] [outer_lr] [algo]
#   algo: continual_ttt, tttreptile, tt_fomaml
#
# Example:
#   bash scripts/send_jobs.sh config/pretraining/antmaze_expert_bc.yaml 400000 True 3e-04 3e-04 tt_reptile

config_file=${1:-""}
restore_epoch=${2:-400000}
use_best_checkpoint=${3:-True}
inner_lr=${4:-3e-04}
outer_lr=${5:-3e-04}
algo_input=${6:-"tt_reptile"}

# Remap algo names to internal meta_algorithm values
case "$algo_input" in
    continual_ttt) algo="PT" ;;
    tt_reptile)   algo="reptile" ;;
    tt_fomaml)    algo="fomaml" ;;
    *)
        echo "Error: algo must be one of: continual_ttt, tt_reptile, tt_fomaml"
        exit 1
        ;;
esac

# ============== MODEL PATH MAPPING (9 configs) ==============
case "$config_file" in
    # Expert BC
    config/pretraining/pointmaze_expert_bc.yaml)
        model_path="/cluster/home/anmari/gc_ttt/exp/pointmaze_expert/bc/RANDOM-point_exp-bc-JT-1-1-lr0.0003-ilr0.0003" ;;
    config/pretraining/antmaze_expert_bc.yaml)
        model_path="/cluster/home/anmari/gc_ttt/exp/antmaze_expert/bc/RANDOM-ant_exp-bc-JT-1-1-lr0.0003-ilr0.0003" ;;
    config/pretraining/humanoidmaze_expert_bc.yaml)
        model_path="/cluster/home/anmari/gc_ttt/exp/humanoidmaze_expert/bc/RANDOM-humanoid_exp-bc-JT-1-1-lr0.0003-ilr0.0003" ;;
    # Expert IQL-BC
    config/pretraining/pointmaze_iql_bc.yaml)
        model_path="/cluster/home/anmari/gc_ttt/exp/pointmaze_expert/iql_bc/RANDOM-point_exp-bc-JT-1-1-lr0.0003-ilr0.0003" ;;
    config/pretraining/antmaze_iql_bc.yaml)
        model_path="/cluster/home/anmari/gc_ttt/exp/antmaze_expert/iql_bc/RANDOM-ant_exp-bc-JT-1-1-lr0.0003-ilr0.0003" ;;
    config/pretraining/humanoidmaze_iql_bc.yaml)
        model_path="/cluster/home/anmari/gc_ttt/exp/humanoidmaze_expert/iql_bc/RANDOM-humanoid_exp-bc-JT-1-1-lr0.0003-ilr0.0003" ;;
    # Stitch IQL DDPG+BC
    config/pretraining/pointmaze_iql_ddpgbc.yaml)
        model_path="/cluster/home/anmari/gc_ttt/exp/pointmaze_stitch/iql_ddpgbc/RANDOM-point_sti-ddpgbc-JT-1-1-lr0.0003-ilr0.0003" ;;
    config/pretraining/antmaze_iql_ddpgbc.yaml)
        model_path="/cluster/home/anmari/gc_ttt/exp/antmaze_stitch/iql_ddpgbc/RANDOM-ant_sti-ddpgbc-JT-1-1-lr0.0003-ilr0.0003" ;;
    config/pretraining/humanoidmaze_iql_ddpgbc.yaml)
        model_path="/cluster/home/anmari/gc_ttt/exp/humanoidmaze_stitch/iql_ddpgbc/RANDOM-humanoid_sti-ddpgbc-JT-1-1-lr0.0003-ilr0.0003" ;;
    *)
        echo "Invalid config file: $config_file"
        exit 1 ;;
esac

# ============== HYPERPARAMETER GRID ==============
if [ "$use_best_checkpoint" = "True" ]; then
    inner_loop_steps=(100)
else
    inner_loop_steps=(1 10 20 50 100 200)
fi

if [ "$algo" = "reptile" ]; then
    merging_eps=(0.01 0.1 0.2 0.5)
else
    merging_eps=(1)
fi

seeds=(0 1 2)
meta_batch_sizes=(1)
train_steps=50000
eval_interval=200000 # don't eval
save_interval=5000
wandb_group=""  # This?
actor_loss=""  # This?

# Fixed settings
training_fix_actor_goal=1.0  # Do not edit
train_settings=ALL  # Do not edit
average_test_gradients=False  # Do not edit
use_pretrained_model=True  # Do not edit
actor_only=True # Whether to finetune only the actor
use_meta_optimizer=False  # Do not edit
annealing=False  # Do not edit

# ============== DERIVED FLAGS (computed once) ==============
# train_settings flag
if [ "$train_settings" = "TEST" ]; then
    train_settings_flag="--train_on_test_goal"
elif [ "$train_settings" = "RANDOM" ]; then
    train_settings_flag="--use_random_batch"
else
    train_settings_flag=""
fi

# average_test_gradients flag
average_test_gradients_flag=""
[ "$average_test_gradients" = "True" ] && average_test_gradients_flag="--average_test_gradients"

# meta_algorithm flag (only pass if not PT)
meta_algorithm_flag=""
[ "$algo" != "PT" ] && meta_algorithm_flag="--meta_algorithm $algo"

# actor_only flag
actor_only_flag=""
[ "$actor_only" = "True" ] && actor_only_flag="--finetune.actor_only"

# use_meta_optimizer flag
use_meta_optimizer_flag=""
[ "$use_meta_optimizer" = "True" ] && use_meta_optimizer_flag="--use_meta_optimizer"

# annealing flag
annealing_flag=""
[ "$annealing" = "True" ] && annealing_flag="--annealing"

# use_best_checkpoint flag
use_best_checkpoint_flag=""
[ "$use_best_checkpoint" = "True" ] && use_best_checkpoint_flag="--use_best_checkpoint"

# wandb_group flag
wandb_group_flag=""
[ -n "$wandb_group" ] && wandb_group_flag="--wandb_group $wandb_group"

# actor_loss flag
actor_loss_flag=""
[ -n "$actor_loss" ] && actor_loss_flag="--actor_loss $actor_loss"

# ============== SLURM SETTINGS ==============
TIME="48:00:00"
MEM="32G"
GPU="1"
LOG_DIR="/cluster/home/anmari/gc_ttt/logs/finetuning"

# ============== JOB SUBMISSION ==============
mkdir -p "$LOG_DIR"

echo "Submitting finetuning jobs for config=${config_file}, algo=${algo}..."
echo "=================================="

job_count=0

for inner_step in "${inner_loop_steps[@]}"; do
    for meta_batch_size in "${meta_batch_sizes[@]}"; do
        for eps in "${merging_eps[@]}"; do
            # If algo is reptile, the lr should be the same as merging_eps
            if [ "$algo" = "reptile" ]; then
                lr=$eps
            else
                lr=$outer_lr
            fi

            # use_pretrained_model restore flag
            use_pretrained_model_flag=""
            [ "$use_pretrained_model" = "True" ] && use_pretrained_model_flag="--restore_epoch $restore_epoch"

            for seed in "${seeds[@]}"; do
                restore_path="${model_path}/seed${seed}"

                JOB_NAME="ft_s${seed}_is${inner_step}_eps${eps}"

                echo "Submitting: seed=$seed, inner_steps=$inner_step, eps=$eps, lr=$lr"

                sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --gpus=${GPU}
#SBATCH --output=${LOG_DIR}/ft_%j.out
#SBATCH --error=${LOG_DIR}/ft_%j.err

cd /cluster/home/anmari/gc_ttt
source venv/bin/activate

python meta_main.py "${config_file}" \
    ${meta_algorithm_flag} \
    --seed ${seed} \
    --agent.inner_loop_steps ${inner_step} \
    --agent.meta_batch_size ${meta_batch_size} \
    --log_interval 1 \
    --eval_interval ${eval_interval} \
    --train_steps ${train_steps} \
    --save_interval ${save_interval} \
    --finetune.lr ${lr} \
    --finetune.inner_lr ${inner_lr} \
    --training_fix_actor_goal ${training_fix_actor_goal} \
    --agent.merging_eps ${eps} \
    --restore_path ${restore_path} \
    ${use_pretrained_model_flag} \
    ${train_settings_flag} \
    ${average_test_gradients_flag} \
    ${actor_only_flag} \
    ${use_meta_optimizer_flag} \
    ${annealing_flag} \
    ${use_best_checkpoint_flag} \
    ${wandb_group_flag} \
    ${actor_loss_flag}
EOF

                job_count=$((job_count + 1))
                sleep 0.1

            done
        done
    done
done

echo "=================================="
echo "Submitted ${job_count} jobs total"
echo "Check status with: squeue -u \$USER"
