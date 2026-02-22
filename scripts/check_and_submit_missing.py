#!/usr/bin/env python3
"""Check for missing evaluation results and submit SLURM jobs to fill gaps.

Usage:
    python scripts/check_and_submit_missing.py <results_folder> [--dry-run]

Examples:
    python scripts/check_and_submit_missing.py results/humanoidmaze_stitch/iql_ddpgbc/RANDOM-humanoid_sti-ddpgbc-JT-1-1-lr0.0003-ilr0.0003_finetune_lr1e-03
    python scripts/check_and_submit_missing.py results/antmaze_stitch/iql_ddpgbc/RANDOM-ant_sti-ddpgbc-JT-1-1-lr0.0003-ilr0.0003_finetune_lr1e-04_TTT_actor --dry-run
"""

import argparse
import csv
import os
import re
import subprocess
import sys

EXPECTED_STEPS_FT = list(range(5000, 55000, 5000))  # 5k, 10k, ..., 50k
EXPECTED_STEPS_RANDOM = [400000]
EXPECTED_TTT_STEPS = [0, 10, 20, 50, 100, 200]
SEEDS = [0, 1, 2]
BASE_DIR = "/cluster/home/anmari/gc_ttt"


def parse_results_folder(results_folder):
    results_folder = results_folder.rstrip("/")
    if results_folder.startswith(BASE_DIR):
        results_folder = os.path.relpath(results_folder, BASE_DIR)

    parts = results_folder.split("/")
    try:
        idx = parts.index("results")
    except ValueError:
        print(f"Error: 'results' not found in path: {results_folder}")
        sys.exit(1)

    env = parts[idx + 1]
    model = parts[idx + 2]
    exp_name = parts[idx + 3]

    remaining = exp_name
    finetune_actor_only = False
    finetune_lr = None

    if remaining.endswith("_TTT_actor"):
        finetune_actor_only = True
        remaining = remaining[: -len("_TTT_actor")]

    traj_match = re.search(r"_traj_(\d+)$", remaining)
    if traj_match:
        remaining = remaining[: traj_match.start()]

    actor_loss_match = re.search(r"_actor_loss(.+)$", remaining)
    if actor_loss_match:
        remaining = remaining[: actor_loss_match.start()]

    ft_lr_match = re.search(r"_finetune_lr([0-9e.\-]+)$", remaining)
    if ft_lr_match:
        finetune_lr = ft_lr_match.group(1)
        remaining = remaining[: ft_lr_match.start()]

    base_exp_name = remaining

    if base_exp_name.startswith("FT"):
        expected_steps = EXPECTED_STEPS_FT
    else:
        expected_steps = EXPECTED_STEPS_RANDOM

    return {
        "env": env,
        "model": model,
        "exp_name": exp_name,
        "base_exp_name": base_exp_name,
        "finetune_lr": finetune_lr,
        "finetune_actor_only": finetune_actor_only,
        "expected_steps": expected_steps,
    }


def check_seed_csv(csv_path):
    present = set()
    if not os.path.exists(csv_path):
        return present
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            ttt = int(row["TTT_steps"])
            present.add((step, ttt))
    return present


def build_sbatch_script(params, seed, num_ttt_steps, restore_epoch=400000):
    env = params["env"]
    model = params["model"]
    base_exp_name = params["base_exp_name"]
    exp_name = params["exp_name"]
    finetune_lr = params["finetune_lr"]
    finetune_actor_only = params["finetune_actor_only"]

    restore_path = f"{BASE_DIR}/exp/{env}/{model}/{base_exp_name}/seed{seed}"
    output_file = f"results/{env}/{model}/{exp_name}/results_seed{seed}.csv"
    output_dir = os.path.dirname(output_file)
    log_dir = f"{BASE_DIR}/logs/eval_{env}"
    job_name = f"eval_{env}_s{seed}_ttt{num_ttt_steps}_e{restore_epoch}"

    optional_flags = ""
    if finetune_lr:
        optional_flags += f" --finetune_lr {finetune_lr}"
    if finetune_actor_only:
        optional_flags += " --finetune_actor_only"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --output={log_dir}/eval_%j.out
#SBATCH --error={log_dir}/eval_%j.err

cd {BASE_DIR}
source venv/bin/activate

mkdir -p "{output_dir}"

python eval_main.py \\
    --restore_path "{restore_path}" \\
    --restore_epoch {restore_epoch} \\
    --num_ttt_steps {num_ttt_steps} \\
    --seed {seed} \\
    --eval_episodes 50 \\
    --output_file "{output_file}" \\
    --num_trajectories 10{optional_flags}
"""
    return script, job_name


def main():
    parser = argparse.ArgumentParser(description="Check missing eval results and submit jobs")
    parser.add_argument("results_folder", help="Path to results folder")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    args = parser.parse_args()

    params = parse_results_folder(args.results_folder)

    print(f"Environment: {params['env']}")
    print(f"Model:       {params['model']}")
    print(f"Experiment:  {params['exp_name']}")
    print(f"Base name:   {params['base_exp_name']}")
    print(f"Finetune LR: {params['finetune_lr'] or 'default'}")
    print(f"Actor only:  {params['finetune_actor_only']}")
    print(f"Exp steps:   {params['expected_steps']}")
    print("=" * 60)

    results_folder = args.results_folder.rstrip("/")
    if not os.path.isabs(results_folder):
        results_folder = os.path.join(BASE_DIR, results_folder)

    log_dir = f"{BASE_DIR}/logs/eval_{params['env']}"
    os.makedirs(log_dir, exist_ok=True)

    total_missing = 0
    submitted = 0

    expected_steps = params["expected_steps"]

    for seed in SEEDS:
        csv_path = os.path.join(results_folder, f"results_seed{seed}.csv")
        present = check_seed_csv(csv_path)

        missing = []
        for step in expected_steps:
            for ttt in EXPECTED_TTT_STEPS:
                if (step, ttt) not in present:
                    missing.append((step, ttt))

        if missing:
            print(f"\nSeed {seed}: MISSING {len(missing)} entries:")
            for step, ttt in missing:
                print(f"  step={step}, TTT_steps={ttt}")
            for step, ttt in missing:
                script, job_name = build_sbatch_script(params, seed, ttt, restore_epoch=step)
                total_missing += 1

                if args.dry_run:
                    print(f"  [DRY RUN] Would submit: {job_name}")
                else:
                    result = subprocess.run(
                        ["sbatch"],
                        input=script,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        print(f"  Submitted: {job_name} -> {result.stdout.strip()}")
                        submitted += 1
                    else:
                        print(f"  FAILED: {job_name} -> {result.stderr.strip()}")
        else:
            print(f"\nSeed {seed}: Complete")

    print(f"\n{'=' * 60}")
    print(f"Total missing: {total_missing}")
    if not args.dry_run:
        print(f"Jobs submitted: {submitted}")
    else:
        print("(dry run - no jobs submitted)")


if __name__ == "__main__":
    main()
