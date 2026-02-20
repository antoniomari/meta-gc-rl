#!/bin/bash
#SBATCH --job-name=test_lr
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --output=/cluster/home/anmari/gc_ttt/logs/antmaze/test_lr_%j.out
#SBATCH --error=/cluster/home/anmari/gc_ttt/logs/antmaze/test_lr_%j.err

cd /cluster/home/anmari/gc_ttt
source venv/bin/activate

MODE="$1"

if [[ "$MODE" == "all" ]]; then
    for lr in 3e-4 3e-5; do
        python meta_main.py meta_antmaze_iql.yaml  \
            --meta_algorithm fomaml \
            --seed 0 \
            --restore_path /cluster/home/anmari/gc_ttt/exp/antmaze_stitch/iql \
            --restore_epoch 100000 \
            --agent.inner_loop_steps 200 \
            --agent.meta_batch_size 1 \
            --log_interval 20000 \
            --eval_interval 10000 \
            --train_steps 10000 \
            --finetune.lr $lr \
            --finetune.inner_lr 3e-6
    done
    for inner_lr in 3e-6 3e-5 3e-4 3e-3; do
        python meta_main.py meta_antmaze_iql.yaml  \
            --meta_algorithm fomaml \
            --seed 0 \
            --restore_path /cluster/home/anmari/gc_ttt/exp/antmaze_stitch/iql \
            --restore_epoch 100000 \
            --agent.inner_loop_steps 200 \
            --agent.meta_batch_size 1 \
            --log_interval 20000 \
            --eval_interval 10000 \
            --train_steps 10000 \
            --finetune.lr 3e-4 \
            --finetune.inner_lr $inner_lr
    done


elif [[ "$MODE" == "test" ]]; then

    for lr in 3e-4 3e-5; do
        for inner_loop_steps in 10 50; do
            python meta_main.py meta_antmaze_iql.yaml  \
                --meta_algorithm fomaml \
                --seed 0 \
                --restore_path /cluster/home/anmari/gc_ttt/exp/antmaze_stitch/iql \
                --restore_epoch 100000 \
                --agent.inner_loop_steps $inner_loop_steps \
                --agent.meta_batch_size 1 \
                --log_interval 20000 \
                --eval_interval 10000 \
                --train_steps 10000 \
                --finetune.lr $lr \
                --finetune.inner_lr 3e-6 \
                --train_on_test_goal \
                --average_test_gradients
        done
    done
    for inner_lr in 3e-5 3e-4 3e-3; do
        python meta_main.py meta_antmaze_iql.yaml  \
            --meta_algorithm fomaml \
            --seed 0 \
            --restore_path /cluster/home/anmari/gc_ttt/exp/antmaze_stitch/iql \
            --restore_epoch 100000 \
            --agent.inner_loop_steps 200 \
            --agent.meta_batch_size 1 \
            --log_interval 20000 \
            --eval_interval 10000 \
            --train_steps 10000 \
            --finetune.lr 3e-4 \
            --finetune.inner_lr $inner_lr \
            --train_on_test_goal \
            --average_test_gradients
    done

else
    echo "Usage: $0 [all|test]"
    exit 1
fi

