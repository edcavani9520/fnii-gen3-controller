#!/usr/bin/env python3
"""
convert_kinova_to_lerobot.py

Convert Kinova Gen3 HDF5 episodes to LeRobot dataset format,
with automatic train/val split (90/10).
"""

import os, sys, time, argparse, random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Install LeRobot: pip install git+https://github.com/physical-intelligence/lerobot.git")
    sys.exit(1)


def find_instruction(f):
    for key in ["instruction", "task/language_instruction", "task/instruction"]:
        val = f.attrs.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "unnamed_task"


def convert(args):
    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    fps = args.fps
    random.seed(42)

    if not h5_dir.exists():
        print(f"ERROR: {h5_dir} not found")
        sys.exit(1)

    h5_files = sorted(h5_dir.glob("episode_*.h5"))
    if not h5_files:
        print(f"No episode_*.h5 files in {h5_dir}")
        sys.exit(1)

    n = len(h5_files)
    indices = list(range(n))
    random.shuffle(indices)
    n_train = max(int(n * 0.9), 1)
    n_val = n - n_train
    train_set = set(indices[:n_train])
    print(f"Found {n} episodes -> {n_train} train / {n_val} val")

    # Determine shapes
    with h5py.File(h5_files[0], "r") as f:
        _, H, W = f["obs/image"].shape

    if output_dir.exists():
        import shutil; shutil.rmtree(output_dir)

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=output_dir.name,
        robot_type="kinova_gen3",
        fps=fps,
        features={
            "observation.images.camera": {
                "dtype": "video", "shape": (3, H, W),
            },
            "observation.state": {
                "dtype": "float32", "shape": (8,),
                "names": [
                    "joint_1", "joint_2", "joint_3", "joint_4",
                    "joint_5", "joint_6", "joint_7", "gripper",
                ],
            },
            "action": {
                "dtype": "float32", "shape": (7,),
                "names": [
                    "dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper",
                ],
            },
        },
        root=output_dir.parent,
    )

    total_frames = 0
    t0 = time.time()

    for ep_num, h5_path in enumerate(tqdm(h5_files, desc="Episodes")):
        with h5py.File(h5_path, "r") as f:
            images = f["obs/image"][:]
            states = f["obs/proprio"][:]
            actions = f["action"][:]
            ts = f["timestamps"][:]
            instruction = find_instruction(f)

        t_offset = float(ts[0])
        for i in range(len(images)):
            rgb = np.stack([images[i]] * 3, axis=-1)
            dataset.add_frame({
                "observation.images.camera": Image.fromarray(rgb),
                "observation.state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "timestamp": float(ts[i]) - t_offset,
                "task": instruction,
            })
        dataset.save_episode()
        total_frames += len(images)

    dataset.consolidate()

    # Tag splits in episodes.parquet
    meta_file = output_dir / "meta" / "episodes.parquet"
    if meta_file.exists():
        import pandas as pd
        df = pd.read_parquet(meta_file)
        # df has columns: episode_index, from, to, length
        split_col = []
        for ep_idx in range(len(df)):
            split_col.append("train" if ep_idx in train_set else "val")
        df["split"] = split_col
        df.to_parquet(meta_file, index=False)

    elapsed = time.time() - t0
    print(f"\nDone: {n} episodes ({n_train} train / {n_val} val), "
          f"{total_frames} frames, {elapsed:.1f}s")
    print(f"Dataset: {output_dir}")
    print(f"\nopenpi config:")
    print(f"  data:")
    print(f"    repo_id: {output_dir.name}")
    print(f"    root: {output_dir.parent}")
    print(f"    train_split: train")
    print(f"    val_split: val")
    print(f"    image_key: observation.images.camera")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    convert(args)