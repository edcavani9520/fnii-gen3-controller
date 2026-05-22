"""LeRobot 数据集路径与 repo_id 工具。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
HF_USER = os.environ.get("HF_USER", "local")

OBS_STATE = "observation.state"
ACTION = "action"


def repo_id_for_dataset(dataset_name: str) -> str:
    return f"{HF_USER}/kinova_{dataset_name.replace('-', '_')}"


def resolve_dataset_path(dataset_name_or_path: str | Path) -> Path:
    path = Path(dataset_name_or_path)
    if not path.is_absolute():
        path = DEFAULT_DATA_ROOT / path
    if not path.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {path}")
    if not (path / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"不是有效的 LeRobot 数据集: {path}")
    return path


def is_torque_dataset(features: dict[str, Any]) -> bool:
    names = features.get("action", {}).get("names", [])
    return any("delta_position" in str(n) for n in names)


def load_dataset(dataset_path: str | Path, episode_index: int | None = None) -> LeRobotDataset:
    path = resolve_dataset_path(dataset_path)
    episodes = [episode_index] if episode_index is not None else None
    return LeRobotDataset(
        repo_id_for_dataset(path.name),
        root=path,
        episodes=episodes,
        download_videos=False,
    )


def list_episodes(dataset_path: str | Path) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_path)
    info_path = resolve_dataset_path(dataset_path) / "meta" / "info.json"
    fps = json.loads(info_path.read_text())["fps"] if info_path.is_file() else ds.fps
    out = []
    for ep_idx in range(ds.meta.total_episodes):
        ep = ds.meta.episodes[ep_idx]
        n = int(ep["dataset_to_index"] - ep["dataset_from_index"])
        out.append({"episode_index": ep_idx, "num_frames": n, "length_s": n / fps})
    return out


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def joint_indices(names: list[str]) -> list[int]:
    return [names.index(f"joint_{i:02d}.position_deg") for i in range(1, 8)]


def gripper_index(names: list[str]) -> int:
    return names.index("gripper.position")
