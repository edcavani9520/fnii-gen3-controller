#!/usr/bin/env python
"""读取 LeRobot 数据集（twist 或 *_torque）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from dataset_utils import is_torque_dataset, list_episodes, load_dataset, resolve_dataset_path, to_numpy


def read_data(dataset_path: str | Path, episode_index: int) -> dict[str, Any]:
    meta = load_dataset(dataset_path)
    if episode_index < 0 or episode_index >= meta.meta.total_episodes:
        raise IndexError(
            f"episode_index={episode_index} out of range [0, {meta.meta.total_episodes})"
        )

    ds = load_dataset(dataset_path, episode_index)
    frames: dict[str, list] = {}
    for i in range(ds.num_frames):
        for key, value in ds[i].items():
            if key == "task":
                frames.setdefault("task", value)
                continue
            if key.startswith("observation.images."):
                continue
            frames.setdefault(key, []).append(to_numpy(value))

    stacked = {k: np.stack(v) for k, v in frames.items() if k != "task"}
    if "task" in frames:
        stacked["task"] = frames["task"]

    path = resolve_dataset_path(dataset_path)
    torque = is_torque_dataset(ds.features)
    result = {
        "dataset_name": path.name,
        "is_torque": torque,
        "episode_index": episode_index,
        "num_frames": ds.num_frames,
        "fps": ds.fps,
        "feature_names": {
            k: ds.features[k]["names"] for k in ("action", "observation.state") if k in ds.features
        },
        "data": stacked,
    }
    if torque:
        names = result["feature_names"]["observation.state"]
        j_idx = [names.index(f"joint_{i:02d}.position_deg") for i in range(1, 8)]
        states = stacked["observation.state"].astype(np.float64)
        actions = stacked["action"].astype(np.float64)
        result["targets"] = {
            "joints_deg": (states[:, j_idx] + actions[:, :7]).astype(np.float32),
            "gripper": np.clip(actions[:, 7], 0.0, 1.0).astype(np.float32),
        }
    return result


def main() -> None:
    dataset_name = os.environ["DATASET_NAME"]
    episode_index = int(os.environ["EPISODE_INDEX"])

    path = resolve_dataset_path(dataset_name)
    print(f"数据集: {path.name}")
    for ep in list_episodes(path):
        print(f"  ep {ep['episode_index']}: {ep['num_frames']} frames ({ep['length_s']:.1f}s)")

    data = read_data(path, episode_index)
    breakpoint()
    print(
        f"episode {episode_index}: {data['num_frames']} frames, "
        f"type={'torque' if data['is_torque'] else 'twist'}, "
        f"keys={list(data['data'].keys())}"
    )


if __name__ == "__main__":
    main()
