#!/bin/bash
#SBATCH --job-name=eval_iql
#SBATCH --time=24:00:00  # Increased time for multiple paths
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --output=/cluster/home/anmari/gc_ttt/logs/eval_%j.out
#SBATCH --error=/cluster/home/anmari/gc_ttt/logs/eval_%j.err

cd /cluster/home/anmari/gc_ttt
source venv/bin/activate

# Parse arguments
# Format: restore_paths_base_string restore_epochs_string num_ttt_steps seed eval_episodes output_file_base finetune_lr finetune_actor_only num_trajectories finetune_actor_loss
restore_paths_base_string="${1}"
restore_epochs_string="${2:-100000}"
num_ttt_steps=${3:-20}
seed=${4:-0}
eval_episodes=${5:-50}
output_file_base=${6:-"notebooks/results_eval"}
finetune_lr=${7:-"default"}
finetune_actor_only=${8:-"False"}
num_trajectories=${9:-10}
finetune_actor_loss=${10:-""}

# Split restore_paths_base_string into array (space-separated)
IFS=' ' read -ra restore_paths_base <<< "$restore_paths_base_string"

# Split restore_epochs_string into array (space-separated)
IFS=' ' read -ra restore_epochs <<< "$restore_epochs_string"

# If no paths provided, exit
if [ ${#restore_paths_base[@]} -eq 0 ] || [ -z "$restore_paths_base_string" ]; then
    echo "No restore paths provided. Exiting."
    exit 0
fi

# If no epochs provided, exit
if [ ${#restore_epochs[@]} -eq 0 ] || [ -z "$restore_epochs_string" ]; then
    echo "No restore epochs provided. Exiting."
    exit 0
fi

echo "=========================================="
echo "Seed: $seed"
echo "TTT steps: $num_ttt_steps"
echo "Epochs: ${restore_epochs[*]}"
echo "Paths: ${restore_paths_base[*]}"
if [ "$finetune_lr" != "default" ]; then
    echo "Finetune LR: $finetune_lr"
fi
if [ "$finetune_actor_only" == "True" ]; then
    echo "Finetune actor only: enabled"
fi
if [ -n "$finetune_actor_loss" ]; then
    echo "Finetune actor loss: $finetune_actor_loss"
fi
echo "Num trajectories: $num_trajectories"
echo "=========================================="

# Loop through each restore path base
for base_path in "${restore_paths_base[@]}"; do
    # Skip empty paths
    [ -z "$base_path" ] && continue

    # Build restore path with seed appended
    restore_path="${base_path}/seed${seed}"

    # Build output file path
    # Extract experiment identifier from restore_path
    # Example: /cluster/home/anmari/gc_ttt/exp/antmaze_stitch/iql_ddpgbc/TEST-antmaze-ddpgbc-PT-mc0.2-lr0.003-ilr0.003/seed0
    # -> Extract: TEST-antmaze-ddpgbc-PT-mc0.2-lr0.003-ilr0.003
    exp_name=$(basename $(dirname "$restore_path"))

    # Append finetune_lr to exp_name if different from "default"
    if [ "$finetune_lr" != "default" ]; then
        exp_name="${exp_name}_finetune_lr${finetune_lr}"
    fi


    # Append actor_loss to exp_name if finetune_actor_loss is set
    if [ -n "$finetune_actor_loss" ]; then
        exp_name="${exp_name}_actor_loss${finetune_actor_loss}"
    fi

    # Append _traj_{num_trajectories} to exp_name if num_trajectories is different from 10
    if [ "$num_trajectories" != "10" ]; then
        exp_name="${exp_name}_traj_${num_trajectories}"
    fi

    # Append _TTT_actor to exp_name if finetune_actor_only is "True"
    if [ "$finetune_actor_only" == "True" ]; then
        exp_name="${exp_name}_TTT_actor"
    fi

    # Loop through each restore epoch
    for restore_epoch in "${restore_epochs[@]}"; do
        # Skip empty epochs
        [ -z "$restore_epoch" ] && continue

        # Build output file path with epoch included
        if [ -z "$output_file_base" ]; then
            output_file="notebooks/results_eval/${exp_name}/results_seed${seed}.csv"
        else
            output_file="${output_file_base}/${exp_name}/results_seed${seed}.csv"
        fi

        # Create output directory if it doesn't exist
        output_dir=$(dirname "$output_file")
        mkdir -p "$output_dir"

        echo "----------------------------------------"
        echo "Evaluating path: $restore_path"
        echo "Epoch: $restore_epoch"
        echo "Output: $output_file"
        if [ "$finetune_lr" != "default" ]; then
            echo "Finetune LR: $finetune_lr"
        fi
        if [ "$finetune_actor_only" == "True" ]; then
            echo "Finetune actor only: enabled"
        fi
        if [ -n "$finetune_actor_loss" ]; then
            echo "Finetune actor loss: $finetune_actor_loss"
        fi
        echo "Num trajectories: $num_trajectories"
        echo "----------------------------------------"

        # Build python command
        python_cmd="python eval_main.py \
            --restore_path \"$restore_path\" \
            --restore_epoch $restore_epoch \
            --num_ttt_steps $num_ttt_steps \
            --seed $seed \
            --eval_episodes $eval_episodes \
            --output_file \"$output_file\""

        # Add --finetune_lr if different from "default"
        if [ "$finetune_lr" != "default" ]; then
            python_cmd="$python_cmd --finetune_lr $finetune_lr"
        fi

        # Add --finetune_actor_only if "True"
        if [ "$finetune_actor_only" == "True" ]; then
            python_cmd="$python_cmd --finetune_actor_only"
        fi

        # Add --finetune_actor_loss if set
        if [ -n "$finetune_actor_loss" ]; then
            python_cmd="$python_cmd --finetune_actor_loss $finetune_actor_loss"
        fi

        # Add --num_trajectories (always pass it, defaults to 10)
        python_cmd="$python_cmd --num_trajectories $num_trajectories"

        # Execute the command
        eval $python_cmd

        echo "Completed evaluation for path: $restore_path (epoch: $restore_epoch)"
        echo ""
    done
done

echo "All evaluations completed!"
