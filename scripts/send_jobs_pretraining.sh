#!/bin/bash

# ---- Pretraining settings ----
# Note: train_settings can be "ALL", "TEST", or "RANDOM"
train_settings=RANDOM
average_test_gradients=False
use_pretrained_model=False
actor_only=False # Whether to finetune only the actor
use_meta_optimizer=False
annealing=False
use_best_checkpoint=False

config_file=${1:-"config/pretraining/pointmaze_expert_bc.yaml"} # or specify path like "config/meta_pointmaze_iql.yaml"
seeds=(0 1 2)
inner_loop_steps=(1) # (1 10 50 100 200) # (1 10 50 100 200)
meta_batch_sizes=(1)
inner_lrs=(3e-4)
outer_lr=3e-4
train_steps=500000
eval_interval=2000000 # don't eval
save_interval=50000
training_fix_actor_goal=1.0
merging_eps=(1) # (0.5 0.1 0.01)
wandb_group=""
actor_loss=""

for algo in "PT"; do
    for inner_lr in ${inner_lrs[@]}; do
        # If algo is PT, the lr should be the same as the inner lr
        if [ "$algo" = "PT" ]; then
            lr=$inner_lr
        else
            lr=$outer_lr
        fi
        for inner_step in ${inner_loop_steps[@]}; do
            for meta_batch_size in ${meta_batch_sizes[@]}; do
                for eps in ${merging_eps[@]}; do
                    for seed in ${seeds[@]}; do
                        args=(
                            "--config_file=$config_file"
                            "--seed=$seed"
                            "--train_settings=$train_settings"
                            "--average_test_gradients=$average_test_gradients"
                            "--use_pretrained_model=$use_pretrained_model"
                            "--meta_algorithm=$algo"
                            "--inner_loop_steps=$inner_step"
                            "--meta_batch_size=$meta_batch_size"
                            "--inner_lr=$inner_lr"
                            "--lr=$lr"
                            "--train_steps=$train_steps"
                            "--eval_interval=$eval_interval"
                            "--save_interval=$save_interval"
                            "--training_fix_actor_goal=$training_fix_actor_goal"
                            "--merging_eps=$eps"
                            "--actor_only=$actor_only"
                            "--use_meta_optimizer=$use_meta_optimizer"
                            "--annealing=$annealing"
                            "--use_best_checkpoint=$use_best_checkpoint"
                            "--wandb_group=$wandb_group"
                            "--actor_loss=$actor_loss"
                        )

                        echo "sbatch scripts/finetune_iql.sh ${args[*]}"
                        sbatch scripts/finetune_iql.sh "${args[@]}"
                    done
                done
            done
        done
    done
done
