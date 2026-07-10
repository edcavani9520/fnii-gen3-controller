#!/usr/bin/env python3
"""
Kinova Gen3 训练数据集加载器
===============================
从 data_collector.py 输出的 HDF5 文件中加载 (obs, action) 对，
封装为 PyTorch Dataset + DataLoader，可直接用于行为克隆 / VLA 训练。

支持:
  - 单个 episode 加载
  - 多 episode 目录加载
  - 图像解码（可选的 resize / normalize）
  - 灵活的动作空间选择

用法:
  from kinova_dataset import KinovaEpisodeDataset, kinova_dataloader

  dataset = KinovaEpisodeDataset("~/kinova_data/train_data_20260709/")
  loader = kinova_dataloader(dataset, batch_size=32, shuffle=True)

  for batch in loader:
      obs_img = batch["obs"]["camera_0"]     # (B, H, W, 3) uint8
      obs_jp  = batch["obs"]["joint_pos"]     # (B, 7)
      action  = batch["action"]["eef_delta"]  # (B, 6)
      # ... 训练 ...
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ======================================================================
#  单个 Episode Dataset
# ======================================================================

class KinovaEpisodeDataset(Dataset):
    """
    从单个 HDF5 episode 文件中加载同步后的 (obs, action) 对。

    格式约定 (由 data_collector.py 生成):
      /obs/camera_0      (T, H, W, 3) uint8    — RGB 图像
      /obs/joint_pos     (T, 7) float64         — 关节位置
      /obs/joint_vel     (T, 7) float64         — 关节速度
      /obs/eef_pose      (T, 6) float64         — 末端位姿
      /obs/gripper_pos   (T, 1) float64         — 夹爪位置
      /action/eef_delta  (T, 6) float64         — delta 位姿
      /action/gripper    (T, 1) float64         — 夹爪动作
      /action/raw_twist  (T, 6) float64         — 原始速度指令
      /timestamps        (T,) float64
    """

    def __init__(
        self,
        h5_path: str,
        image_size: Optional[Tuple[int, int]] = None,  # (H, W) 可选 resize
        normalize_images: bool = False,                  # [-1, 1] 或 [0, 1]
        action_keys: List[str] = None,                   # 要包含的动作字段
        obs_keys: List[str] = None,                      # 要包含的观测字段
        transform=None,                                  # 自定义 transform
        load_images: bool = True,
    ):
        self.h5_path = str(h5_path)
        self.transform = transform
        self.load_images = load_images
        self.image_size = image_size
        self.normalize_images = normalize_images

        # 默认动作字段
        self.action_keys = action_keys or ["eef_delta", "gripper", "raw_twist"]
        # 默认观测字段（不含图像，单独处理）
        self.obs_keys = obs_keys or ["joint_pos", "joint_vel", "eef_pose", "gripper_pos"]

        # 打开文件（只读）并读取长度
        self._f = h5py.File(self.h5_path, "r")
        self._len = self._f["timestamps"].shape[0]

        if self._len == 0:
            raise ValueError(f"Episode {h5_path} 为空 (0 steps)")

        # 检查是否有 images 群组（兼容旧格式）
        self._has_images = "obs/camera_0" in self._f

        # 预读取非图像数据到内存（存为 numpy，getitem 时转 torch）
        self._obs_data = {}
        for key in self.obs_keys:
            ds_path = f"obs/{key}"
            if ds_path in self._f:
                self._obs_data[key] = self._f[ds_path][:]

        self._action_data = {}
        for key in self.action_keys:
            ds_path = f"action/{key}"
            if ds_path in self._f:
                self._action_data[key] = self._f[ds_path][:]

    @property
    def episode_id(self) -> int:
        return self._f.attrs.get("episode", -1)

    @property
    def language_instruction(self) -> str:
        """获取该 episode 的语言指令。"""
        return str(self._f.attrs.get("task/language_instruction", ""))

    @property
    def task_id(self) -> int:
        return int(self._f.attrs.get("task/id", -1))

    @property
    def camera_shape(self) -> Tuple[int, int, int]:
        if self._has_images and "camera_0" in self._f["obs"]:
            return self._f["obs/camera_0"].shape[1:]  # (H, W, 3)
        return (0, 0, 0)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        # 观测: 图像
        obs = {}
        if self.load_images and self._has_images:
            img = self._f["obs/camera_0"][idx]  # (H, W, 3) uint8
            if self.transform:
                img = self.transform(img)
            elif self.image_size:
                # 手动 resize
                from torchvision.transforms import functional as TF
                img_t = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)
                img_t = TF.resize(img_t, self.image_size)
                img = img_t.permute(1, 2, 0).numpy()
            if self.normalize_images:
                img = img.astype(np.float32) / 127.5 - 1.0
            obs["camera_0"] = torch.as_tensor(img, dtype=torch.uint8)
        else:
            obs["camera_0"] = self._obs_data.get("camera_0")

        # 观测: 标量（转 torch float，兼容 numpy 1.x/2.x）
        for key, arr in self._obs_data.items():
            obs[key] = torch.as_tensor(arr[idx], dtype=torch.float32)

        # 动作
        action = {}
        for key, arr in self._action_data.items():
            action[key] = torch.as_tensor(arr[idx], dtype=torch.float32)

        return {
            "obs": obs,
            "action": action,
            "language_instruction": self.language_instruction,
            "episode_id": self.episode_id,
        }

    def close(self):
        """手动关闭 HDF5 文件（通常由 Dataset 析构器处理）。"""
        if hasattr(self, "_f") and self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        """Pickle 支持（DataLoader 多进程需要）。"""
        state = self.__dict__.copy()
        state["_f"] = None  # HDF5 不能 pickle
        state["_h5_path_backup"] = self.h5_path
        return state

    def __setstate__(self, state):
        """Unpickle 时重新打开 HDF5。"""
        self.__dict__.update(state)
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")


# ======================================================================
#  多 Episode 数据集（目录级别）
# ======================================================================

class KinovaTrainDataset(Dataset):
    """
    包含多个 episode 的训练数据集。
    自动扫描目录下所有 episode_*.h5 文件。

    用法:
      dataset = KinovaTrainDataset("~/kinova_data/train_data_20260709/")
      loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Optional[Tuple[int, int]] = None,
        normalize_images: bool = False,
        action_keys: List[str] = None,
        obs_keys: List[str] = None,
        transform=None,
        load_images: bool = True,
        file_pattern: str = "episode_*.h5",
    ):
        data_dir = os.path.expanduser(data_dir)
        files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        if not files:
            raise FileNotFoundError(f"在 {data_dir} 下未找到 {file_pattern}")

        # 构建子数据集列表
        self.datasets = [
            KinovaEpisodeDataset(
                f, image_size=image_size,
                normalize_images=normalize_images,
                action_keys=action_keys,
                obs_keys=obs_keys,
                transform=transform,
                load_images=load_images,
            )
            for f in files
        ]
        self.total_steps = sum(len(d) for d in self.datasets)
        self.n_episodes = len(self.datasets)

        print(f"📚 加载 {self.n_episodes} 个 episode, 共 {self.total_steps} steps")
        print(f"   文件: {', '.join(os.path.basename(f) for f in files[:3])}{'...' if len(files)>3 else ''}")

        self._dataset = ConcatDataset(self.datasets)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def close(self):
        for d in self.datasets:
            d.close()

    def episode_stats(self) -> List[dict]:
        """返回每个 episode 的统计信息。"""
        stats = []
        for ds in self.datasets:
            stats.append({
                "episode": ds.episode_id,
                "file": os.path.basename(ds.h5_path),
                "steps": len(ds),
                "camera_shape": ds.camera_shape,
            })
        return stats


# ======================================================================
#  工具函数
# ======================================================================

def kinova_collate_fn(batch: List[dict]) -> dict:
    """合并 batch 中的 dict 结构。"""
    collated = {"obs": {}, "action": {}}
    # 观测
    for key in batch[0]["obs"]:
        vals = [b["obs"][key] for b in batch]
        if isinstance(vals[0], np.ndarray):
            collated["obs"][key] = torch.as_tensor(np.stack(vals, 0))
        else:
            collated["obs"][key] = torch.stack(vals, 0)
    # 动作
    for key in batch[0]["action"]:
        vals = [b["action"][key] for b in batch]
        collated["action"][key] = torch.stack(vals, 0)
    # 语言指令（整个 episode 共享，取第一条）
    collated["language_instruction"] = batch[0].get("language_instruction", "")
    collated["episode_id"] = [b.get("episode_id", -1) for b in batch]
    return collated


def kinova_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """创建标准化的 Kinova 数据 DataLoader。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=kinova_collate_fn,
        **kwargs,
    )


def inspect_episode(h5_path: str) -> dict:
    """检查一个 episode 文件的结构和信息。"""
    import json
    with h5py.File(h5_path, "r") as f:
        info = {"file": h5_path, "attrs": dict(f.attrs)}
        for path in ["/obs/camera_0", "/obs/joint_pos", "/obs/joint_vel",
                      "/obs/eef_pose", "/obs/gripper_pos",
                      "/action/eef_delta", "/action/gripper", "/action/raw_twist",
                      "/timestamps"]:
            if path in f:
                ds = f[path]
                info[path] = {"shape": ds.shape, "dtype": str(ds.dtype)}
        # 统计数据
        info["n_steps"] = f["timestamps"].shape[0]
        if info["n_steps"] > 0:
            duration = f["timestamps"][-1] - f["timestamps"][0]
            info["duration_s"] = round(duration, 2)
            info["avg_hz"] = round(info["n_steps"] / duration, 1) if duration > 0 else 0
        # 语言指令
        info["language_instruction"] = str(f.attrs.get("task/language_instruction", ""))
        info["task_id"] = int(f.attrs.get("task/id", -1))
        info["robot_type"] = str(f.attrs.get("robot_type", ""))
        info["control_mode"] = str(f.attrs.get("control_mode", ""))
        info["dataset_name"] = str(f.attrs.get("dataset_name", ""))
        info["date_collected"] = str(f.attrs.get("date_collected", ""))
    return info


# ======================================================================
#  CLI 用法
# ======================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kinova 训练数据集检查")
    parser.add_argument("path", nargs="?", default="~/kinova_data",
                        help="数据目录或单个 .h5 文件路径")
    parser.add_argument("--inspect", action="store_true", help="检查文件结构")
    parser.add_argument("--batch", type=int, default=4, help="测试 batch size")
    args = parser.parse_args()

    path = os.path.expanduser(args.path)

    if os.path.isdir(path):
        print(f"📂 扫描目录: {path}")
        dataset = KinovaTrainDataset(
            path, load_images=True, normalize_images=False
        )
        print(f"\n🧮 总步数: {len(dataset)}")
        print("\n📊 Episode 统计:")
        for s in dataset.episode_stats():
            print(f"   Episode {s['episode']:04d}: {s['steps']} steps, "
                  f"相机 {s['camera_shape']}")
        # 测试一个 batch
        print(f"\n🧪 测试 Dataloader (batch={args.batch})...")
        loader = kinova_dataloader(dataset, batch_size=args.batch, shuffle=False)
        batch = next(iter(loader))
        print(f"   obs/camera_0:  {batch['obs']['camera_0'].shape}  "
              f"dtype={batch['obs']['camera_0'].dtype}")
        print(f"   obs/joint_pos: {batch['obs']['joint_pos'].shape}")
        print(f"   obs/eef_pose:  {batch['obs']['eef_pose'].shape}")
        print(f"   action/eef_delta: {batch['action']['eef_delta'].shape}")
        print(f"   action/gripper:   {batch['action']['gripper'].shape}")
        print(f"   action/raw_twist: {batch['action']['raw_twist'].shape}")
        print("✅ 数据加载测试通过！")
        dataset.close()

    elif os.path.isfile(path):
        info = inspect_episode(path)
        print(f"📄 Episode 信息:")
        print(f"   文件: {info['file']}")
        print(f"   步数: {info['n_steps']}")
        print(f"   时长: {info.get('duration_s', '?')}s")
        print(f"   帧率: {info.get('avg_hz', '?')} Hz")
        print(f"   属性: {info['attrs']}")
        print(f"   数据集:")
        for k, v in info.items():
            if k.startswith("/"):
                print(f"     {k}: {v}")
    else:
        print(f"❌ 路径不存在: {path}")
