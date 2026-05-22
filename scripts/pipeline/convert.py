#!/usr/bin/env python
"""将 twist 录制数据集转为关节差分 + 夹爪绝对值 action（输出 {name}_torque）。"""

from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from dataset_utils import OBS_STATE, joint_indices, repo_id_for_dataset
from lerobot.datasets import LeRobotDataset

ACTION_NAMES = [
    "joint_01.delta_position_deg",
    "joint_02.delta_position_deg",
    "joint_03.delta_position_deg",
    "joint_04.delta_position_deg",
    "joint_05.delta_position_deg",
    "joint_06.delta_position_deg",
    "joint_07.delta_position_deg",
    "gripper.position",
]


def _scalar_int(x: torch.Tensor | int | np.integer) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.item())
    if isinstance(x, np.generic):
        return int(x)
    return int(x)


def _task_name(meta, task_index: torch.Tensor | int | np.integer) -> str:
    return str(meta.tasks.iloc[_scalar_int(task_index)].name)


def _episode_joint_deltas(states: np.ndarray, joint_idx: list[int]) -> np.ndarray:
    q = states[:, joint_idx].astype(np.float64)
    unwrapped = np.unwrap(np.deg2rad(q), axis=0)
    dq = np.zeros_like(q)
    dq[:-1] = np.rad2deg(np.diff(unwrapped, axis=0))
    return dq.astype(np.float32)


def _episode_actions(
    states: np.ndarray, joint_idx: list[int], src_gripper: np.ndarray
) -> np.ndarray:
    dq = _episode_joint_deltas(states, joint_idx)
    grip = np.clip(src_gripper.astype(np.float64), 0.0, 1.0).astype(np.float32)
    return np.concatenate([dq, grip[:, np.newaxis]], axis=1)


def _vcodec(features: dict) -> str:
    for ft in features.values():
        if ft.get("dtype") != "video":
            continue
        codec = str(ft.get("info", {}).get("video.codec", "")).lower()
        if "av1" in codec:
            return "libsvtav1"
        if "h264" in codec or "264" in codec:
            return "libx264"
    return "libsvtav1"


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _image(x: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def convert_dataset(dataset_name: str, *, data_root: Path, force: bool = False) -> Path:
    src_root = data_root / dataset_name
    if not src_root.is_dir():
        raise FileNotFoundError(f"源数据集不存在: {src_root}")

    out_name = f"{dataset_name}_torque"
    dst_root = data_root / out_name
    if dst_root.exists():
        if not force:
            raise FileExistsError(f"目标已存在: {dst_root}（设置 FORCE=true 覆盖）")
        shutil.rmtree(dst_root)

    src_repo = repo_id_for_dataset(dataset_name)
    src = LeRobotDataset(src_repo, root=src_root, download_videos=False)

    new_features = copy.deepcopy(dict(src.features))
    new_features["action"] = {"dtype": "float32", "names": ACTION_NAMES, "shape": [8]}

    state_names = list(src.features[OBS_STATE]["names"])
    src_action_names = list(src.features["action"]["names"])
    j_idx = joint_indices(state_names)
    src_grip_idx = src_action_names.index("gripper.position")
    img_keys = [k for k in src.features if k.startswith("observation.images.")]

    dst = LeRobotDataset.create(
        repo_id=repo_id_for_dataset(out_name),
        fps=int(src.meta.fps),
        features=new_features,
        root=dst_root,
        robot_type=src.meta.robot_type,
        use_videos=True,
        vcodec=_vcodec(new_features),
    )

    for ep in range(src.meta.total_episodes):
        ep_ds = LeRobotDataset(src_repo, root=src_root, episodes=[ep], download_videos=True)
        n = len(ep_ds)
        states = np.zeros((n, len(state_names)), dtype=np.float32)
        src_gripper = np.zeros(n, dtype=np.float32)
        tasks: list[str] = []
        for i in range(n):
            raw = ep_ds.get_raw_item(i)
            states[i] = _to_numpy(raw[OBS_STATE]).astype(np.float32)
            src_gripper[i] = float(_to_numpy(raw["action"])[src_grip_idx])
            tasks.append(_task_name(ep_ds.meta, int(raw["task_index"])))
        actions = _episode_actions(states, j_idx, src_gripper)

        for i in range(n):
            frame = {OBS_STATE: states[i], "action": actions[i], "task": tasks[i]}
            for key in img_keys:
                frame[key] = _image(ep_ds[i][key])
            dst.add_frame(frame)

        dst.save_episode()
        print(f"  episode {ep}: {n} frames", flush=True)

    dst.finalize()
    print(f"完成: {dst_root}")
    return dst_root


def main() -> None:
    dataset_name = os.environ["DATASET_NAME"]
    data_root = Path(os.environ["DATASET_ROOT"])
    force = os.environ.get("FORCE", "false").lower() in ("1", "true", "yes")
    convert_dataset(dataset_name, data_root=data_root, force=force)


if __name__ == "__main__":
    main()
