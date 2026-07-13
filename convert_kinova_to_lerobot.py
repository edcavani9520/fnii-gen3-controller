#!/usr/bin/env python3
"""
convert_kinova_to_lerobot.py

Convert Kinova Gen3 HDF5 episodes to LeRobot dataset format.
Supports merging multiple collection directories.
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
    print("Install: pip install git+https://github.com/physical-intelligence/lerobot.git")
    sys.exit(1)


def find_instruction(f):
    for key in ["instruction", "task/language_instruction", "task/instruction"]:
        val = f.attrs.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "unnamed_task"


def collect_h5_files(dirs):
    """Scan multiple directories recursively for episode_*.h5 files."""
    files = []
    for d in dirs:
        p = Path(d).expanduser().resolve()
        if p.is_dir():
            found = sorted(p.rglob("episode_*.h5"))
            files.extend(found)
            print(f"  {p}: {len(found)} episodes")
        else:
            print(f"  [skip] {p}: not a directory")
    if not files:
        print("ERROR: no episode_*.h5 files found in any given directory")
        sys.exit(1)
    # Deduplicate by full path, preserve order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def convert(args):
    fps = args.fps
    output_dir = Path(args.output_dir).expanduser()
    random.seed(42)

    print("Scanning directories for HDF5 files...")
    h5_files = collect_h5_files(args.h5_dir)

    n = len(h5_files)
    indices = list(range(n))
    random.shuffle(indices)
    n_train = max(int(n * 0.9), 1)
    n_val = n - n_train
    train_set = set(indices[:n_train])
    print(f"\nTotal: {n} episodes -> {n_train} train / {n_val} val")

    # Determine shapes from first file
    with h5py.File(h5_files[0], "r") as f:
        _, H, W = f["obs/image"].shape

    if output_dir.exists():
        import shutil; shutil.rmtree(output_dir)

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
        root=output_dir,
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
                "task": str(instruction),
            })
        dataset.save_episode()
        total_frames += len(images)

    dataset.consolidate()

    # Write split info
    meta_file = output_dir / "meta" / "episodes.parquet"
    if meta_file.exists():
        import pandas as pd
        df = pd.read_parquet(meta_file)
        split_col = ["train" if i in train_set else "val" for i in range(len(df))]
        df["split"] = split_col
        df.to_parquet(meta_file, index=False)

    # Compatibility: generate tasks.jsonl for openpi (v2.1 format)
    tasks_parquet = output_dir / "meta" / "tasks.parquet"
    tasks_jsonl = output_dir / "meta" / "tasks.jsonl"
    if tasks_parquet.exists() and not tasks_jsonl.exists():
        try:
            tasks_df = pd.read_parquet(tasks_parquet)
            with open(tasks_jsonl, "w", encoding="utf-8") as f:
                for _, row in tasks_df.iterrows():
                    task_text = row.get("task", row.get("instruction", ""))
                    if pd.isna(task_text):
                        task_text = ""
                    f.write(json.dumps({"task": task_text}) + "\n")
            print(f"  tasks.jsonl written ({len(tasks_df)} tasks)")
        except Exception as e:
            print(f"  Warning: tasks.jsonl generation failed: {e}")

    elapsed = time.time() - t0
    print(f"\nDone: {n} episodes ({n_train} train / {n_val} val), "
          f"{total_frames} frames, {elapsed:.1f}s")
    print(f"Dataset: {output_dir}")
    print(f"\nopenpi config:")
    print(f"  data:")
    print(f"    repo_id: {output_dir.name}")
    print(f"    root: {str(output_dir.parent)}")
    print(f"    train_split: train")
    print(f"    val_split: val")
    print(f"    image_key: observation.images.camera")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Kinova Gen3 HDF5 to LeRobot dataset (supports merging multiple directories)"
    )
    parser.add_argument(
        "--h5-dir", nargs="+", required=True,
        help="One or more directories (recursively scanned for episode_*.h5)"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    convert(args)