"""Upload this folder to the Hugging Face Hub as a dataset repository.

Usage:
    pip install huggingface_hub
    huggingface-cli login            # or set HF_TOKEN
    python scripts/push_to_hub.py Enzo-py/partial-prompting-ds [--public]
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("repo_id")
ap.add_argument("--public", action="store_true", help="Create the repo as public (private by default)")
args = ap.parse_args()

api = HfApi()
api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
api.upload_folder(
    folder_path=str(ROOT),
    repo_id=args.repo_id,
    repo_type="dataset",
    ignore_patterns=["**/__pycache__/**", "**/.DS_Store"],
    commit_message="Upload partial prompt study (pilot)",
)
print(f"https://huggingface.co/datasets/{args.repo_id}")
