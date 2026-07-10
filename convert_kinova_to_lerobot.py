#!/usr/bin/env python
"""convert_kinova_to_lerobot.py — 将 Kinova Gen3 h5 原始数据转换为 LeRobot 格式

用法:
  python convert_kinova_to_lerobot.py
  python convert_kinova_to_lerobot.py --h5 <path> --out <dataset_name>

输出位置: lerobot_data/<dataset_name>/
"""
import sys
import argparse
import cv2
import numpy as np
import h5py
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ===== 输出目录配置 =====
OUTPUT_ROOT = Path.home() / "lerobot_data"  # 所有 LeRobot 数据都放这里


def load_episode_h5(h5_path: str):
    """从 episode h5 提取关节角和 JPEG 帧"""
    with h5py.File(h5_path, "r") as f:
        rgb_bytes = f["camera/rgb"][:]             # (N,) object  ← JPEG bytes
        cam_ts = f["camera/timestamp"][:].astype(np.float64)
        joint_pos_deg = f["robot/joint_position"][:].astype(np.float64)  # (M, 7)
        robot_ts = f["robot/timestamp"][:].astype(np.float64)

    # 解码第一帧确认尺寸
    sample = cv2.imdecode(rgb_bytes[0], cv2.IMREAD_COLOR)
    assert sample is not None, "无法解码 JPEG 帧"
    H, W = sample.shape[:2]

    # 关节角 度 → 弧度
    joint_pos = np.deg2rad(joint_pos_deg)

    # 时间同步: camera 帧 → 最近的 robot 时间点
    sync = np.searchsorted(robot_ts, cam_ts)
    sync = np.clip(sync, 0, len(joint_pos) - 1)
    for i in range(len(cam_ts)):
        idx = sync[i]
        if idx > 0 and abs(cam_ts[i] - robot_ts[idx - 1]) < abs(cam_ts[i] - robot_ts[idx]):
            sync[i] = idx - 1

    print(f"  [h5] {len(rgb_bytes)} 帧, {W}x{H}, {len(joint_pos)} 个 robot 时间点")
    return rgb_bytes, joint_pos, sync, H, W


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """解码 JPEG 字节 → BGR uint8"""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # (H, W, 3), BGR


def main():
    parser = argparse.ArgumentParser(description="Kinova Gen3 h5 → LeRobot 格式")
    parser.add_argument("--h5", default="episode_0002.h5",
                        help="输入 h5 文件路径")
    parser.add_argument("--out", default="kinova_gen3_deblur_v1",
                        help="数据集名称 (输出到 lerobot_data/<name>/")
    parser.add_argument("--task", default="push the cup to the right",
                        help="语言指令")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        # 尝试在 deblurrin 项目目录下找
        alt = Path.home() / "Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin" / args.h5
        if alt.exists():
            h5_path = alt
        else:
            print(f"❌ 找不到 h5 文件: {args.h5}")
            sys.exit(1)

    repo_id = args.out
    output_dir = OUTPUT_ROOT / repo_id
    print(f"📥 输入: {h5_path}")
    print(f"📦 输出: {output_dir}")

    # 1. 加载 h5
    rgb_bytes, joint_pos, sync_indices, H, W = load_episode_h5(str(h5_path))
    num_frames = len(rgb_bytes)
    print(f"   有效帧: {num_frames}")

    # 2. 创建 LeRobot 数据集（用自定义输出路径）
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=OUTPUT_ROOT,  # ← 指定根目录，不依赖 HF_LEROBOT_HOME
        robot_type="kinova_gen3",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
    )

    # 3. 逐帧添加
    print("📝 写入帧...")
    for i in range(num_frames):
        img_bgr = decode_jpeg(rgb_bytes[i])
        if img_bgr is None:
            print(f"   ⚠️ 跳过第 {i} 帧: 解码失败")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224))

        ri = sync_indices[i]
        state = joint_pos[ri].astype(np.float32)
        next_idx = min(ri + 1, len(joint_pos) - 1)
        action = joint_pos[next_idx].astype(np.float32)

        dataset.add_frame({
            "image": img_rgb,
            "wrist_image": img_rgb.copy(),
            "state": state,
            "actions": action,
            "task": args.task,
        })

        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{num_frames}")

    # 4. 保存
    dataset.save_episode()
    print(f"\n✅ 转换完成!")
    print(f"   episodes: {dataset.num_episodes}")
    print(f"   frames:   {dataset.num_frames}")
    print(f"   保存路径: {output_dir}")
    print(f"\n下一步: 在 openpi 中训练时使用 repo_id='{repo_id}'")
    print(f"       数据路径: {output_dir}")


if __name__ == "__main__":
    main()
