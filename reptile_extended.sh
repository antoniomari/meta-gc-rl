# !/bin/bash

cd /cluster/home/anmari/gc_ttt
source venv/bin/activate

# Run meta-learning (REPTILE) with meta batch size 1 and inner loop steps 16

# Seed 0
# python meta_main.py meta_bc.yaml --meta_algorithm reptile --seed 0 --agent.inner_loop_steps 1 --agent.meta_batch_size 32 --finetune.num_steps 0 --train_steps 50000 --eval_interval 5000
# python meta_main.py meta_bc.yaml --meta_algorithm reptile --seed 0 --agent.inner_loop_steps 1 --agent.meta_batch_size 32 --finetune.num_steps 10 --train_steps 50000 --eval_interval 5000
# python meta_main.py meta_bc.yaml --meta_algorithm reptile --seed 0 --agent.inner_loop_steps 1 --agent.meta_batch_size 32 --finetune.num_steps 50 --train_steps 50000 --eval_interval 5000

# Seed 1
# python meta_main.py meta_bc.yaml --meta_algorithm reptile --seed 1 --agent.inner_loop_steps 1 --agent.meta_batch_size 32 --finetune.num_steps 0 --train_steps 50000 --eval_interval 5000
# python meta_main.py meta_bc.yaml --meta_algorithm reptile --seed 1 --agent.inner_loop_steps 1 --agent.meta_batch_size 32 --finetune.num_steps 10 --train_steps 50000 --eval_interval 5000
# python meta_main.py meta_bc.yaml --meta_algorithm reptile --seed 1 --agent.inner_loop_steps 1 --agent.meta_batch_size 32 --finetune.num_steps 50 --train_steps 50000 --eval_interval 5000

# Seed 2
python meta_main.py meta_bc.yaml --meta_algorithm reptile --seed 2 --agent.inner_loop_steps 1 --agent.meta_batch_size 32 --finetune.num_steps 0 --train_steps 50000 --eval_interval 5000
python meta_main.py meta_bc.yaml --meta_algorithm reptile --seed 2 --agent.inner_loop_steps 1 --agent.meta_batch_size 32 --finetune.num_steps 10 --train_steps 50000 --eval_interval 5000
python meta_main.py meta_bc.yaml --meta_algorithm reptile --seed 2 --agent.inner_loop_steps 1 --agent.meta_batch_size 32 --finetune.num_steps 50 --train_steps 50000 --eval_interval 5000
