#!/usr/bin/env python3
"""Download pre-trained checkpoints from Google Drive.

Usage
-----
# Download a specific checkpoint:
    uv run python scripts/download_checkpoint.py --policy vqbet --task door_opening

# Download all checkpoints for a policy:
    uv run python scripts/download_checkpoint.py --policy vqbet --task all

# Change output directory (default: checkpoints/):
    uv run python scripts/download_checkpoint.py --policy diffusion \
            --task reorientation --out-dir /data/checkpoints
"""

import argparse
import os
import sys

import gdown

TASK_GDRIVE_ID = {
    "vqbet": {
        "door_opening": "17fWsurnkp1-UtFg9BEK0t3nOWryrhh7g",
        "drawer_opening": "1B13lIdFeqXGnxAPUtsrtgKe5K5bI-pS0",
        "reorientation": "1Shhs8rMA8EIF46N7_-D8yES02kg_hK2w",
        "bag_pick_up": "1LpmdIQ7-pV7BIqiTFWzs70nK3JiMyRwz",
        "tissue_pick_up": "1tw03YyFUBM0nVEG_DRDVvvDed3ftU3dH",
    },
    "diffusion": {
        "door_opening": "1G8ZuhXnfDrZiugba9TktMPGc65K5NX0j",
        "drawer_opening": "1ETnWSjddHwsdp9xnWi19UGduukYvpy7m",
        "reorientation": "1ClyHWjhM9RpT18XB5DG_mupVPl-81E6N",
        "bag_pick_up": "1pqFwzXxV7Gm80r8gbtiHEa7dD9Z8GSAX",
        "tissue_pick_up": "1HPT1Vz82DAANn0B3NYe5N1iDxGlRVjhQ",
    },
}


def download(policy: str, task: str, out_dir: str) -> str:
    """Download checkpoint and return local path.  Skip if already exists."""
    gdrive_id = TASK_GDRIVE_ID[policy][task]
    dest_dir = os.path.join(out_dir, policy, task)
    dest_path = os.path.join(dest_dir, "checkpoint.pt")

    if os.path.isfile(dest_path):
        print(f"[skip] {dest_path} already exists")
        return dest_path

    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"[download] {policy}/{task} -> {dest_path}")
    gdown.download(url, dest_path, quiet=False)
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download izumi checkpoints from Google Drive")
    parser.add_argument(
        "--policy",
        choices=list(TASK_GDRIVE_ID.keys()),
        required=True,
        help="Policy type",
    )
    parser.add_argument(
        "--task",
        default="all",
        help=f"Task name or 'all'.  Available: {list(next(iter(TASK_GDRIVE_ID.values())).keys())}",
    )
    parser.add_argument(
        "--out-dir",
        default="checkpoints",
        help="Root directory for downloaded checkpoints (default: checkpoints/)",
    )
    args = parser.parse_args()

    tasks_for_policy = TASK_GDRIVE_ID[args.policy]

    if args.task == "all":
        tasks = list(tasks_for_policy.keys())
    elif args.task in tasks_for_policy:
        tasks = [args.task]
    else:
        print(f"Error: unknown task '{args.task}' for policy '{args.policy}'.")
        print(f"  Available tasks: {list(tasks_for_policy.keys())}")
        sys.exit(1)

    for task in tasks:
        path = download(args.policy, task, args.out_dir)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
