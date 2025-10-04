# !/bin/bash

cd /cluster/home/anmari/gc_ttt
source venv/bin/activate


# Run pretraining (no meta-learning)

# Seed 0
# python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 0
# python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 5
# python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 10
#python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 20
#python meta_main.py meta_bc.yaml --seed 0 --finetune.num_steps 50

# Seed 1
# python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 0
# python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 5
#python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 10
#python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 20
python meta_main.py meta_bc.yaml --seed 1 --finetune.num_steps 50

# Seed 2
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 0
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 5
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 10
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 20
python meta_main.py meta_bc.yaml --seed 2 --finetune.num_steps 50
