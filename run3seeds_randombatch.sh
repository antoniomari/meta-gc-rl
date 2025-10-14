# !/bin/bash

cd /cluster/home/anmari/gc_ttt
source venv/bin/activate

# Run pretraining (no meta-learning)

# Seed 0
# python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 0 --train_steps 50000 --eval_interval 5000 --use_random_batch
# python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 5 --train_steps 50000 --eval_interval 5000 --use_random_batch
# python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 10 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 20 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 50 --train_steps 50000 --eval_interval 5000 --use_random_batch

# Seed 1
python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 0 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 5 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 10 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 20 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 50 --train_steps 50000 --eval_interval 5000 --use_random_batch

# Seed 2
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 0 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 5 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 10 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 20 --train_steps 50000 --eval_interval 5000 --use_random_batch
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 50 --train_steps 50000 --eval_interval 5000 --use_random_batch
