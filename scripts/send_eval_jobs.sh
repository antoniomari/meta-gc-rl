#!/bin/bash
# Script to submit evaluation jobs for all environments.
# Structure mirrors meta_vlas/meta_libero/jobs/send_jobs.sh
#
# Usage:
#   bash scripts/send_eval_jobs.sh <env> <model> [finetune_lr] [finetune_actor_only] [restore_path_base]
#   env:   pointmaze_expert, pointmaze_stitch, antmaze_expert, antmaze_stitch,
#          humanoidmaze_expert, humanoidmaze_stitch
#   model: bc, iql_bc, iql_ddpgbc
#   finetune_lr: (optional, default "default")
#   finetune_actor_only: (optional, default "False") - set to "True" to enable finetune_actor_only flag
#   restore_path_base: (optional) full base path to exp dirs, e.g. /my/custom/path/foo

env=${1:?"Usage: $0 <env> <model> [finetune_lr] [finetune_actor_only] [restore_path_base] (e.g. bash $0 pointmaze_expert bc)"}
model=${2:?"Usage: $0 <env> <model> [finetune_lr] [finetune_actor_only] [restore_path_base] (e.g. bash $0 pointmaze_expert bc)"}

# Optional finetune_lr, default = "default"
finetune_lr="${3:-default}"

# Optional finetune_actor_only, default = "False"
finetune_actor_only="${4:-False}"

# Optional restore_path_base (now last)
restore_path_base_arg="${5:-}"

# ============== HYPERPARAMETER GRID ==============
if [[ -n "$restore_path_base_arg" ]]; then
    # A post-trained model is saved every 5k steps
    restore_epochs=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000)
else
    # Pretrained models are saved every 50k steps
    restore_epochs=(50000 100000 150000 200000 250000 300000 350000 400000 450000 500000)  # (400000)
fi

# 3 seeds -> fixed
seeds=(0 1 2)
# List of TTT steps,
num_ttt_steps_list=(0 10 20 50 100 200)

# Validate env
maze_part="${env%_*}"          # e.g. pointmaze
dataset_part="${env##*_}"      # e.g. expert

if [[ "$maze_part" != "pointmaze" && "$maze_part" != "antmaze" && "$maze_part" != "humanoidmaze" ]]; then
    echo "Error: env must start with pointmaze, antmaze, or humanoidmaze"
    exit 1
fi
if [[ "$dataset_part" != "expert" && "$dataset_part" != "stitch" ]]; then
    echo "Error: env must end with _expert or _stitch"
    exit 1
fi

# Validate model
if [[ "$model" != "bc" && "$model" != "iql_bc" && "$model" != "iql_ddpgbc" ]]; then
    echo "Error: model must be one of: bc, iql_bc, iql_ddpgbc"
    exit 1
fi

# Derive short names for restore path naming
# e.g. pointmaze_expert -> point_exp, antmaze_stitch -> ant_sti
env_short="${maze_part/maze/}_${dataset_part:0:3}"

# model_short: bc -> bc, iql_bc -> bc, iql_ddpgbc -> ddpgbc
case "$model" in
    bc)         model_short="bc" ;;
    iql_bc)     model_short="bc" ;;
    iql_ddpgbc) model_short="ddpgbc" ;;
esac

# exp folder is directly env/model: e.g. exp/pointmaze_expert/bc/
exp_dir="${env}"
algo_dir="${model}"

# ============== EXPERIMENT PATHS ==============
if [[ -n "$restore_path_base_arg" ]]; then
    # If user specified the base path, just use it
    restore_paths_base=("$restore_path_base_arg")
else
    # Construct the default base path
    restore_paths_base=(
        "/cluster/home/anmari/gc_ttt/exp/${exp_dir}/${algo_dir}/RANDOM-${env_short}-${model_short}-JT-1-1-lr0.0003-ilr0.0003"
    )
fi

if [ ${#restore_paths_base[@]} -eq 0 ]; then
    echo "Error: restore_paths_base is empty. Edit this script to add experiment paths."
    exit 1
fi

# Fixed parameters
eval_episodes=50  # don't change this
num_trajectories=10  # don't change this
finetune_actor_loss=""

# Output directory (auto-computed from env and model_type)
base_output_dir="results/${exp_dir}/${algo_dir}"

# ============== SLURM SETTINGS ==============
TIME="24:00:00"
MEM="32G"
GPU="1"
LOG_DIR="/cluster/home/anmari/gc_ttt/logs/eval_${exp_dir}"

# ============== JOB SUBMISSION ==============
mkdir -p "$LOG_DIR"

echo "Submitting evaluation jobs for env=${env}, model_type=${model} (${exp_dir}/${algo_dir})..."
echo "=================================="

job_count=0

for seed in "${seeds[@]}"; do
    for num_ttt_steps in "${num_ttt_steps_list[@]}"; do
        for restore_epoch in "${restore_epochs[@]}"; do
            for base_path in "${restore_paths_base[@]}"; do
                [ -z "$base_path" ] && continue

                restore_path="${base_path}/seed${seed}"
                exp_name=$(basename "$base_path")

                # Build exp_name with suffixes
                [ "$finetune_lr" != "default" ] && exp_name="${exp_name}_finetune_lr${finetune_lr}"
                [ -n "$finetune_actor_loss" ] && exp_name="${exp_name}_actor_loss${finetune_actor_loss}"
                [ "$num_trajectories" != "10" ] && exp_name="${exp_name}_traj_${num_trajectories}"
                [ "$finetune_actor_only" == "True" ] && exp_name="${exp_name}_TTT_actor"

                output_file="${base_output_dir}/${exp_name}/results_seed${seed}.csv"

                JOB_NAME="eval_${env}_s${seed}_ttt${num_ttt_steps}_e${restore_epoch}"

                echo "Submitting: path=$(basename "$base_path"), seed=$seed, ttt=$num_ttt_steps, epoch=$restore_epoch"

                # Build optional flags
                finetune_lr_flag=""
                [ "$finetune_lr" != "default" ] && finetune_lr_flag="--finetune_lr ${finetune_lr}"

                finetune_actor_only_flag=""
                [ "$finetune_actor_only" == "True" ] && finetune_actor_only_flag="--finetune_actor_only"

                finetune_actor_loss_flag=""
                [ -n "$finetune_actor_loss" ] && finetune_actor_loss_flag="--finetune_actor_loss ${finetune_actor_loss}"

                sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --gpus=${GPU}
#SBATCH --output=${LOG_DIR}/eval_%j.out
#SBATCH --error=${LOG_DIR}/eval_%j.err

cd /cluster/home/anmari/gc_ttt
source venv/bin/activate

mkdir -p "$(dirname "${output_file}")"

python eval_main.py \
    --restore_path "${restore_path}" \
    --restore_epoch ${restore_epoch} \
    --num_ttt_steps ${num_ttt_steps} \
    --seed ${seed} \
    --eval_episodes ${eval_episodes} \
    --output_file "${output_file}" \
    --num_trajectories ${num_trajectories} ${finetune_lr_flag} ${finetune_actor_only_flag} ${finetune_actor_loss_flag}
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
