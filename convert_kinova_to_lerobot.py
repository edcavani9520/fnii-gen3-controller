#!/usr/bin/env python
"""convert_kinova_to_lerobot.py — 将去模糊后的 Kinova Gen3 数据转换为 LeRobot 格式"""
import sys
import h5py
import cv2
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


def main():
    # ===== 路径配置 =====
    h5_path = "/home/kinova-1/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin/episode_0002.h5"
    deblur_dir = "/home/kinova-1/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin/deblur_output/episode_0002/deblurred"
    repo_id = "kinova_gen3_deblur_v1"  # 数据集名称，后面训练会用

    # ===== 检查输入文件 =====
    if not Path(h5_path).exists():
        print(f"❌ h5 文件不存在: {h5_path}")
        sys.exit(1)

    deblur_dir = Path(deblur_dir)
    if not deblur_dir.exists():
        print(f"❌ 去模糊帧目录不存在: {deblur_dir}")
        sys.exit(1)

    print(f"📂 输入: {h5_path}")
    print(f"📸 去模糊帧: {deblur_dir}/")

    # ===== 1. 加载 h5 =====
    print("📥 加载 h5...")
    h5 = h5py.File(h5_path, "r")
    joint_pos = np.deg2rad(h5["robot/joint_position"][:])  # (605, 7) 度 → 弧度
    cam_ts = h5["camera/timestamp"][:]                     # (368,)
    robot_ts = h5["robot/timestamp"][:]                    # (605,)
    num_frames = len(cam_ts)
    print(f"   关节数据: {joint_pos.shape[0]} 个时间点, 共 {num_frames} 帧")

    # 时间同步
    sync_indices = np.searchsorted(robot_ts, cam_ts)
    sync_indices = np.clip(sync_indices, 0, len(joint_pos) - 1)

    # ===== 2. 创建 LeRobot dataset =====
    print("📦 创建 LeRobot 数据集...")
    output_path = Path(HF_LEROBOT_HOME) / repo_id
    print(f"   输出路径: {output_path}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
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

    # ===== 3. 逐帧添加 =====
    print("📝 写入帧...")
    for i in range(num_frames):
        # 读取去模糊帧
        img_path = deblur_dir / f"step_{i:04d}.jpg"
        if not img_path.exists():
            print(f"   ⚠️ 第 {i} 帧图片不存在: {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"   ⚠️ 第 {i} 帧无法读取")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224))

        ri = sync_indices[i]
        state = joint_pos[ri].astype(np.float32)

        # action: 下一帧的 joint position（如果是最后一帧则用自己）
        next_idx = min(ri + 1, len(joint_pos) - 1)
        action = joint_pos[next_idx].astype(np.float32)

        dataset.add_frame({
            "image": img_rgb,
            "wrist_image": img_rgb.copy(),
            "state": state,
            "actions": action,
            "task": "push the cup to the right",
        })

        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{num_frames}")

    # ===== 4. 保存 =====
    print("💾 保存 episode...")
    dataset.save_episode()
    print(f"\n✅ 转换完成!")
    print(f"   episodes: {dataset.num_episodes}")
    print(f"   frames:   {dataset.num_frames}")
    print(f"   保存路径: {output_path}")
    print(f"\n下一步: 在 openpi 训练时使用 repo_id='{repo_id}'")


if __name__ == "__main__":
    main()
