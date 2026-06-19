# Kinova Gen3 Easy Control 🚀

这是一个专为 Kinova Gen3 / G3L 机械臂设计的轻量化控制框架。本项目在官方 SDK 的基础上进行了二次开发，解决了原生驱动在笛卡尔空间控制时的抖动问题，并集成了手柄和 Apple Vision Pro 的遥操功能。

## 🌟 核心亮点

* **平滑的笛卡尔控制**：相比传统的逆解计算，本项目优化了控制逻辑，避免了动作“抖动”问题，实现平滑的笛卡尔空间移动。
* **多端遥操支持**：集成游戏手柄（Gamepad）和 Apple Vision Pro 遥操接口。
* **开箱即用**：内置了 `kortex_api` SDK，无需复杂的跨境网络安装。
* **详尽的接口注释**：提供 `.pyi` 存根文件，支持代码补全与函数说明。

---

## 📂 目录结构说明

| 文件/文件夹 | 说明 |
| --- | --- |
| **`kinova_manage.py`** | **核心控制代码**。包含所有的控制类和函数。 |
| **`kinova_manage.pyi`** | **函数说明文档**。专门为 `kinova_manage.py` 编写的接口解释，方便开发者查看。 |
| **`kortex_api/`** | Kinova 官方 SDK 包。本项目已内置，用户无需手动安装。 |
| **`Kinova_kortex2_Gen3_G3L/`** | 官方原始资源包。包含参考代码和示例。 |
| **`visionpro_control.py`** | 使用 Apple Vision Pro 进行机械臂遥操的主程序。 |
| **`gamepad_control_obs.py`** | 手柄控制脚本（含数据监测）。可在终端实时查看机械臂状态数据。 |
| **`gamepad_control.py`** | 基础手柄控制脚本（无数据反馈）。 |
| **`isbot/`** | 存档：前同事的旧版控制方法（基于力矩控制，不支持笛卡尔控制）。 |
| **`api_control/`** | 存档：存放旧版 README 及部分历史文件。 |
| **`requirements.txt`** | 环境依赖清单。 |
| **`sync.sh`** | GitHub 自动化同步脚本。 |

---

## 🛠️ 环境配置

本项目建议在 **Python 3.10** 环境下运行。

1. **创建并激活环境**：
```bash
conda create -n kinova_env python=3.10
conda activate kinova_env

```


2. **安装依赖**：
```bash
pip install -r requirements.txt

```



---

## 🕹️ 使用指南

### 1. 核心调用

如果你想在自己的项目中使用该控制框架，只需导入 `KinovaManager`：

```python
from kinova_manage import KinovaManager

# 初始化并连接机械臂
arm = KinovaManager()

```

### 2. 手柄控制

* **带数据监控**：`python gamepad_control_obs.py`
* **仅控制**：`python gamepad_control.py`

### 3. Vision Pro 遥操

确保 Vision Pro 与机械臂处于同一局域网下，运行：

```bash
python visionpro_control.py

```

---

## ⚠️ 注意事项

* **笛卡尔控制**：本项目通过优化算法规避了直接计算逆解导致的动作抖动，建议优先使用本框架提供的 `Cartesian` 相关函数。
* **SDK 依赖**：代码会自动调用同级目录下的 `kortex_api`，请勿删除或移动该文件夹。

---

## 🔄 开发与同步

如果你对代码进行了修改，可以使用内置脚本快速同步至 GitHub：

```bash
./sync.sh
