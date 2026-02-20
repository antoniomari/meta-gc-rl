"""
Push the contents of gc_ttt/exp/ to a Hugging Face repository.

Usage:
    python scripts/push_to_hf.py --repo_id <your-hf-username>/<repo-name> [--private]

Example:
    python scripts/push_to_hf.py --repo_id anmari/gc_ttt_exp --private
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def main():

    parser = argparse.ArgumentParser(description="Push gc_ttt/exp/ to a Hugging Face repo.")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo id, e.g. 'username/repo-name'")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    parser.add_argument("--exp_dir", type=str, default=None, help="Path to exp/ folder (default: auto-detect)")
    parser.add_argument("--overwrite", action="store_true", help="Delete all existing remote files before uploading")
    args = parser.parse_args()

    if args.exp_dir is not None:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = Path(__file__).resolve().parent.parent / "exp"

    if not exp_dir.exists():
        raise FileNotFoundError(f"exp directory not found at {exp_dir}")

    print(f"Exp directory: {exp_dir}")
    print(f"Target repo:   {args.repo_id}")
    print(f"Private:       {args.private}")
    print(f"Overwrite:     {args.overwrite}")

    api = HfApi()

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repo '{args.repo_id}' ready.")

    print("Uploading folder (this may take a while)...")
    upload_kwargs = dict(
        folder_path=str(exp_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    if args.overwrite:
        upload_kwargs["delete_patterns"] = ["*"]
    api.upload_folder(**upload_kwargs)
    print(f"Done! View at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
