#!/usr/bin/env python3
"""
convert_kinova_to_lerobot.py

Convert Kinova Gen3 HDF5 episodes to LeRobot dataset format.
Compatible with LeRobot v3 API (physical-intelligence fork).

Usage:
    # Install leRobot first:
    pip install git+https://github.com/physical-intelligence/lerobot.git

    # Run conversion:
    python convert_kinova_to_lerobot.py \\
        --h5-dir ~/kinova_data/train_data_20260712_173158 \\
        --output-dir ~/lerobot_datasets/kinova_cube \\
        --fps 10
"""

import os
import sys
import time
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Please install LeRobot first:")
    print("  pip install git+https://github.com/physical-intelligence/lerobot.git")
    sys.exit(1)


def find_instruction(f):
    """Read instruction from attrs with multiple fallback paths."""
    for key in ["instruction", "task/language_instruction", "task/instruction"]:
        val = f.attrs.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "unnamed_task"


def convert(args):
    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    fps = args.fps

    if not h5_dir.exists():
        print(f"ERROR: {h5_dir} does not exist")
        sys.exit(1)

    h5_files = sorted(h5_dir.glob("episode_*.h5"))
    if not h5_files:
        print(f"No episode_*.h5 files found in {h5_dir}")
        sys.exit(1)

    print(f"Found {len(h5_files)} episodes")
    print(f"Output: {output_dir}")

    # Read first file to determine shapes
    with h5py.File(h5_files[0], "r") as f:
        img = f["obs/image"]
        T, H, W = img.shape

    # Remove output dir if exists
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=output_dir.name,
        robot_type="kinova_gen3",
        fps=fps,
        features={
            "observation.images.camera": {
                "dtype": "video",
                "shape": (3, H, W),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": [
                    "joint_1", "joint_2", "joint_3", "joint_4",
                    "joint_5", "joint_6", "joint_7", "gripper",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "dx", "dy", "dz",
                    "droll", "dpitch", "dyaw",
                    "gripper",
                ],
            },
        },
        root=output_dir.parent,
    )

    # Process each episode
    total_frames = 0
    t_start = time.time()

    for h5_path in tqdm(h5_files, desc="Episodes"):
        with h5py.File(h5_path, "r") as f:
            images = f["obs/image"][:]
            states = f["obs/proprio"][:]
            actions = f["action"][:]
            ts = f["timestamps"][:]
            instruction = find_instruction(f)

        # Use relative timestamps (0 = episode start)
        t0 = float(ts[0])
        timestamps = [float(t) - t0 for t in ts]

        Ti = len(images)
        for i in range(Ti):
            gray = images[i]
            rgb = np.stack([gray, gray, gray], axis=-1)

            dataset.add_frame({
                "observation.images.camera": Image.fromarray(rgb),
                "observation.state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "timestamp": timestamps[i],
                "task": instruction,
            })

        dataset.save_episode()
        total_frames += Ti

    dataset.consolidate()

    elapsed = time.time() - t_start
    print(f"\nDone: {len(h5_files)} episodes, {total_frames} frames, {elapsed:.1f}s")
    print(f"Dataset: {output_dir}")
    print(f"\n  View it:")
    print(f"    lerobot-dataset-viz --repo-id {output_dir.name} --root {output_dir.parent}")
    print(f"\n  Load in Python:")
    print(f"    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset")
    print(f"    ds = LeRobotDataset('{output_dir.name}', root='{output_dir.parent}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    convert(args)