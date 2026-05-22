#!/usr/bin/env python
"""力矩模式回放 *_torque 数据集：q_target = q + Δq，move_torque。"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_utils import ACTION, OBS_STATE, joint_indices, repo_id_for_dataset, to_numpy
from kinova_manage import KinovaManager
from lerobot.datasets import LeRobotDataset


def main() -> None:
    dataset_name = os.environ["DATASET_NAME"]
    episode_index = int(os.environ["EPISODE_INDEX"])
    data_root = Path(os.environ["DATASET_ROOT"])
    settle_s = float(os.environ["SETTLE_S"])
    robot_ip = os.environ["ROBOT_IP"]

    root = data_root / dataset_name
    if not root.is_dir():
        raise SystemExit(f"数据集不存在: {root}")

    ds = LeRobotDataset(
        repo_id_for_dataset(dataset_name),
        root=root,
        episodes=[episode_index],
        download_videos=False,
    )
    state_names = list(ds.features[OBS_STATE]["names"])
    j_idx = joint_indices(state_names)
    n, fps = len(ds), float(ds.meta.fps)
    print(f">>> {dataset_name} ep {episode_index}: {n} 帧 @ {fps} Hz")

    arm = KinovaManager(robot_ip)
    try:
        arm.init_torque()
        period = 1.0 / fps
        for i in range(n):
            t0 = time.perf_counter()
            item = ds[i]
            state = to_numpy(item[OBS_STATE]).astype(np.float64)
            act = to_numpy(item[ACTION]).astype(np.float64)
            q_tgt = (state[j_idx] + act[:7]).tolist()
            g_tgt = float(np.clip(act[7], 0.0, 1.0))
            arm.move_torque(q_tgt, gripper=g_tgt)
            if (i + 1) % max(1, int(fps)) == 0 or i == 0:
                print(f"  帧 {i + 1}/{n}", flush=True)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        time.sleep(settle_s)
        last = ds[n - 1]
        st = to_numpy(last[OBS_STATE]).astype(np.float64)
        act = to_numpy(last[ACTION]).astype(np.float64)
        out = arm.move_torque(
            (st[j_idx] + act[:7]).tolist(),
            gripper=float(np.clip(act[7], 0.0, 1.0)),
        )
        print(f"  当前关节(°): {out['q_deg'].round(2)}")
    except KeyboardInterrupt:
        print("\n中断", flush=True)
    finally:
        arm.shutdown_torque()


if __name__ == "__main__":
    main()
