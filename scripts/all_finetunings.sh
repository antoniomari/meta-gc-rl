# Arguments for send_jobs_pretraining.sh
# config_file

# Expert GCBC
bash scripts/send_jobs_pretraining.sh config/pretraining/pointmaze_expert_bc.yaml
bash scripts/send_jobs_pretraining.sh config/pretraining/antmaze_expert_bc.yaml
bash scripts/send_jobs_pretraining.sh config/pretraining/humanoidmaze_expert_bc.yaml



# Arguments for send_jobs.sh
# config_file, restore_epoch, use_best_checkpoint, inner_lr, outer_lr, algo

# Humanoidmaze expert bc (inner lr 3e-04)
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 400000 True 3e-04 3e-04 reptile
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 400000 True 3e-04 3e-04 fomaml
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 400000 True 3e-04 3e-04 PT

# Humanoidmaze expert bc (inner lr 3e-05)
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 400000 True 3e-05 3e-04 reptile
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 400000 True 3e-05 3e-04 fomaml
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 400000 True 3e-05 3e-04 PT

# Humanoidmaze on 900_000
# Humanoidmaze expert bc (inner lr 3e-04)
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 900000 True 3e-04 3e-04 reptile
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 900000 True 3e-04 3e-04 fomaml
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 900000 True 3e-04 3e-04 PT

# Humanoidmaze expert bc (inner lr 3e-05)
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 900000 True 3e-05 3e-04 reptile
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 900000 True 3e-05 3e-04 fomaml
bash scripts/send_jobs.sh config/pretraining/humanoidmaze_expert_bc.yaml 900000 True 3e-05 3e-04 PT

# Antmaze expert bc
bash scripts/send_jobs.sh config/pretraining/antmaze_expert_bc.yaml 400000 True 3e-04 3e-04 reptile
bash scripts/send_jobs.sh config/pretraining/antmaze_expert_bc.yaml 400000 True 3e-04 3e-04 fomaml
bash scripts/send_jobs.sh config/pretraining/antmaze_expert_bc.yaml 400000 True 3e-04 3e-04 PT

# Antmaze expert bc (inner lr 3e-05)
bash scripts/send_jobs.sh config/pretraining/antmaze_expert_bc.yaml 400000 True 3e-05 3e-04 reptile
bash scripts/send_jobs.sh config/pretraining/antmaze_expert_bc.yaml 400000 True 3e-05 3e-04 fomaml
bash scripts/send_jobs.sh config/pretraining/antmaze_expert_bc.yaml 400000 True 3e-05 3e-04 PT

# Pointmaze expert bc
bash scripts/send_jobs.sh config/pretraining/pointmaze_expert_bc.yaml 400000 True 3e-04 3e-04 reptile
bash scripts/send_jobs.sh config/pretraining/pointmaze_expert_bc.yaml 400000 True 3e-04 3e-04 fomaml
bash scripts/send_jobs.sh config/pretraining/pointmaze_expert_bc.yaml 400000 True 3e-04 3e-04 PT

# Pointmaze expert bc (inner lr 3e-05)
bash scripts/send_jobs.sh config/pretraining/pointmaze_expert_bc.yaml 400000 True 3e-05 3e-04 reptile
bash scripts/send_jobs.sh config/pretraining/pointmaze_expert_bc.yaml 400000 True 3e-05 3e-04 fomaml
bash scripts/send_jobs.sh config/pretraining/pointmaze_expert_bc.yaml 400000 True 3e-05 3e-04 PT
