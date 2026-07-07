#!/usr/bin/env python3
"""从 HDF5 采集数据中提取视频，输出 MP4 到同目录。"""
import sys
import os
import cv2
import h5py
import numpy as np


def extract_video(h5_path, fps=15):
    """将 HDF5 中的 JPEG 帧解码并保存为 MP4 视频。"""
    h5_dir = os.path.dirname(h5_path)
    stem = os.path.splitext(os.path.basename(h5_path))[0]

    with h5py.File(h5_path, 'r') as f:
        rgb_data = f['camera/rgb'][:]
        timestamps = f['camera/timestamp'][:]
        cam_fps = f.attrs.get('camera_fps', fps)
        exposure = f.attrs.get('camera_exposure_100us', '?')
        gain = f.attrs.get('camera_gain', '?')

    if len(rgb_data) == 0:
        print(f"  ⏭ {h5_path} — 无图像数据")
        return

    # 解码第一帧获取尺寸
    first_frame = cv2.imdecode(np.frombuffer(rgb_data[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = first_frame.shape[:2]

    # 视频写入
    video_path = os.path.join(h5_dir, f"{stem}_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, cam_fps, (w, h))

    writer.write(first_frame)
    for i in range(1, len(rgb_data)):
        frame = cv2.imdecode(np.frombuffer(rgb_data[i], dtype=np.uint8), cv2.IMREAD_COLOR)
        writer.write(frame)

    writer.release()

    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    print(f"  ✅ {video_path}")
    print(f"     {len(rgb_data)} frames, {w}x{h}, {cam_fps}fps, {duration:.1f}s"
          f"  |  exposure={exposure}×100µs  gain={gain}")


def process_directory(dir_path, fps=15):
    """扫描目录下所有 episode_*.h5 文件并提取视频。"""
    h5_files = sorted([f for f in os.listdir(dir_path)
                       if f.startswith('episode_') and f.endswith('.h5')])
    if not h5_files:
        print(f"❌ 目录中未找到 episode_*.h5 文件: {dir_path}")
        return

    print(f"📁 {dir_path}")
    print(f"   找到 {len(h5_files)} 个 HDF5 文件")
    for h5f in h5_files:
        extract_video(os.path.join(dir_path, h5f), fps)
    print(f"✅ 全部完成")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从 HDF5 采集数据提取视频")
    parser.add_argument("path", nargs="?", default=None,
                        help="HDF5 文件或目录路径（默认: ~/kinova_data_blur 下最新的采集目录）")
    parser.add_argument("--fps", type=int, default=15, help="输出视频帧率（默认 15）")
    args = parser.parse_args()

    if args.path:
        target = args.path
    else:
        # 默认取 ~/kinova_data_blur 下最新的采集目录
        base = os.path.expanduser("~/kinova_data_blur")
        if not os.path.isdir(base):
            print(f"❌ 默认目录不存在: {base}")
            sys.exit(1)
        dirs = sorted([d for d in os.listdir(base) if d.startswith("blur_collection_")])
        if not dirs:
            print(f"❌ ~/kinova_data_blur 下无 blur_collection_* 目录")
            sys.exit(1)
        target = os.path.join(base, dirs[-1])
        print(f"📂 自动选择最新目录: {target}")

    if os.path.isdir(target):
        process_directory(target, args.fps)
    elif os.path.isfile(target) and target.endswith('.h5'):
        extract_video(target, args.fps)
    else:
        print(f"❌ 无效路径: {target}")
