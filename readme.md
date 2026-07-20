# Kinova Gen3 Easy Control 🚀

这是一个专为 Kinova Gen3 / G3L 机械臂设计的轻量化控制框架。本项目在官方 SDK 的基础上进行了二次开发，解决了原生驱动在笛卡尔空间控制时的抖动问题，并集成了手柄和 Apple Vision Pro 的遥操功能。

## 🌟 核心亮点

* **平滑的笛卡尔控制**：相比传统的逆解计算，本项目优化了控制逻辑，避免了动作“抖动”问题，实现平滑的笛卡尔空间移动。
* **多端遥操支持**：集成游戏手柄（Gamepad）和 Apple Vision Pro 遥操接口。
* **开箱即用**：内置了 `kortex_api` SDK，无需复杂的跨境网络安装。
* **详尽的接口注释**：提供 `.pyi` 存根文件，支持代码补全与函数说明。

---

## 📂 目录结构说明

| 文件/文件夹                    | 说明                                                                         |
| ------------------------------ | ---------------------------------------------------------------------------- |
| **`kinova_manage.py`**         | **核心控制代码**。包含所有的控制类和函数。                                   |
| **`kinova_manage.pyi`**        | **函数说明文档**。专门为 `kinova_manage.py` 编写的接口解释，方便开发者查看。 |
| **`kortex_api/`**              | Kinova 官方 SDK 包。本项目已内置，用户无需手动安装。                         |
| **`Kinova_kortex2_Gen3_G3L/`** | 官方原始资源包。包含参考代码和示例。                                         |
| **`visionpro_control.py`**     | 使用 Apple Vision Pro 进行机械臂遥操的主程序。                               |
| **`gamepad_control_obs.py`**   | 手柄控制脚本（含数据监测）。可在终端实时查看机械臂状态数据。                 |
| **`gamepad_control.py`**       | 基础手柄控制脚本（无数据反馈）。                                             |
| **`isbot/`**                   | 存档：前同事的旧版控制方法（基于力矩控制，不支持笛卡尔控制）。               |
| **`api_control/`**             | 存档：存放旧版 README 及部分历史文件。                                       |
| **`requirements.txt`**         | 环境依赖清单。                                                               |
| **`sync.sh`**                  | GitHub 自动化同步脚本。                                                      |

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
pip install -r requirements-py.txt

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

### 4. π0.5 / OpenPI 真机推理正式版

真机运行需要开两个终端：终端 1 保持 OpenPI policy server 运行，终端 2 再启动 Kinova 控制脚本。

#### 终端 1：启动 OpenPI 推理服务

```bash
cd ~/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_kinova_lora \
  --policy.dir=/home/kinova-1/openpi/checkpoints/pi05_kinova_lora/kinova_cube_20260717_lora_gemma2b_fixed_5000_bs32/4999
```

看到下面日志后保持该终端不要关闭：

```bash
server listening on 0.0.0.0:8000
```

#### 终端 2：启动真机控制

```bash
cd ~/fnii-gen3-controller

python pi05_ws_control.py \
  --control-mode twist \
  --freq 5 \
  --action-steps 1 \
  --action-scale 0.7 \
  --min-ee-z 0.01 \
  --camera-drain-frames 1 \
  --log-every 1
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
```
---

## ⚡ Kinova 机械臂双网卡环境网络配置指南

适用于 Ubuntu 工作站 + Kinova Gen3 系列机械臂的开发环境，实现「有线直连机械臂 + 同时上网」的稳定并行网络。


### 一、机械臂原始默认参数

Kinova Gen3 / Gen3 Lite 出厂网络配置固定，未修改过的设备可直接使用以下参数连接：

| 参数项        | 默认值                |
| ------------- | --------------------- |
| 有线网静态 IP | `192.168.1.10`        |
| 子网掩码      | `255.255.255.0` (/24) |
| Web 后台地址  | `http://192.168.1.10` |
| 默认登录账号  | `admin`               |
| 默认登录密码  | `admin`               |



### 二、为什么要修改机械臂 IP？

#### 核心问题：双网卡同网段路由冲突

工作站同时存在两张工作网卡时：

- **WiFi 网卡**：连接实验室 / 家用路由器，默认网段通常为 `192.168.1.0/24`，负责访问公网
- **有线网卡**：直连机械臂，机械臂默认 IP 也在 `192.168.1.0/24` 网段

当两张网卡处于同一网段时，Linux 系统无法精准区分流量走向：

- 发往机械臂 `192.168.1.10` 的数据包可能被错误路由到 WiFi 网卡，导致连接失败
- 若两张网卡都配置了默认网关，会出现路由优先级冲突，直接导致公网访问中断

#### 解决方案：网段隔离

将机械臂迁移到独立的专用网段（本文档使用 `192.168.8.0/24`），与 WiFi 上网网段完全隔离：

- 公网流量自动走 WiFi 网卡及对应网关
- 机械臂流量仅通过专用有线网卡传输

两者互不干扰，实现「同时上网 + 控制机械臂」的稳定环境。


### 三、详细配置步骤

#### 步骤 1：临时同网段连通，进入机械臂后台

此步骤先关闭 WiFi 排除冲突，确保能正常登录机械臂管理页面。

1. 关闭工作站 **WiFi**（右上角网络图标 → 关闭 WiFi）

2. 打开系统设置 → **网络** → 找到对应有线连接，点击齿轮图标进入配置

3. 切换到 **IPv4** 选项卡，按如下参数配置：

   | 字段       | 值                |
   | ---------- | ----------------- |
   | IPv4 方式  | **手动 (Manual)** |
   | 地址       | `192.168.1.11`    |
   | 子网掩码   | `255.255.255.0`   |
   | **网关**   | **留空不填**      |
   | DNS 服务器 | 留空不填          |

   > 直连同网段无需网关，填入会引发后续路由冲突。

4. 点击 **应用**，断开并重连该有线网络使配置生效

5. 打开浏览器访问 `http://192.168.1.10`，输入账号 `admin` / 密码 `admin` 登录 Kortex Web 后台


#### 步骤 2：修改机械臂静态 IP 为独立网段

1. 在 Web 后台左侧导航栏依次进入 **Networks → Ethernet**

2. IPv4 模式保持 **Static（静态）**，修改核心参数：

   | 字段             | 值              |
   | ---------------- | --------------- |
   | IPv4 address     | `192.168.8.10`  |
   | IPv4 subnet mask | `255.255.255.0` |
   | IPv4 gateway     | **留空**        |

3. 点击 **Apply / Save** 保存配置

> 保存后机械臂会重启网络服务，浏览器会临时断连，属于正常现象。


#### 步骤 3：工作站同步配置，开启双网并行

1. 回到工作站有线网卡 IPv4 设置，同步修改参数：

   | 字段     | 值              |
   | -------- | --------------- |
   | 地址     | `192.168.8.11`  |
   | 子网掩码 | `255.255.255.0` |
   | **网关** | **务必留空**    |
   | DNS      | 留空            |

2. 点击 **应用**，断开并重连有线网络

3. 重新开启 WiFi，连接上网网络


### 四、配置有效性验证

#### 1. 机械臂连通性验证

```bash
ping 192.168.8.10
```

正常输出：持续返回 `64 bytes from 192.168.8.10` 格式的回复，按 `Ctrl+C` 终止。

#### 2. Web 后台访问验证

浏览器访问 `http://192.168.8.10`，可正常登录 Kortex Web App 管理页面。

#### 3. 双网并行验证

- **公网访问**：可正常打开网页、执行 `sudo apt update` 等网络命令
- **路由表检查**：执行 `ip route` 命令，正常状态为：
  - 仅一条 `default via xxx` 默认路由，指向 **WiFi 网卡**
  - 存在 `192.168.8.0/24 dev 有线网卡名` 静态路由条目

---

# ROS 2 Humble + ros2_kortex 安装指南

本项目支持通过 **ROS 2 Humble** + **ros2_kortex** 进行机械臂控制，提供话题订阅、rosbag 录制、可视化等功能，比原生 Kortex API 更适用于数据采集和机器人视觉任务。

---

## 1. 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Ubuntu 22.04 (Jammy) — **注意：仅支持 22.04** |
| **架构** | x86_64 |
| **Python** | 3.10（系统自带） |
| **机械臂** | Kinova Gen3 / Gen3 lite |

---

## 2. 安装 ROS 2 Humble

### 2.1 添加 ROS 2 软件源

```bash
# 设置 locale
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 添加 ROS 2 GPG key
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 添加 ROS 2 仓库
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update
```

### 2.2 安装 ROS 2 Humble Desktop

```bash
export DEBIAN_FRONTEND=noninteractive
sudo apt install -y ros-humble-desktop
```

安装约 **282 个 ROS 2 包**（rclpy、rviz2、ros2bag、TF2、control_msgs 等）。

### 2.3 安装编译工具

```bash
sudo apt install -y python3-colcon-common-extensions python3-vcstool python3-rosdep

# 初始化 rosdep
sudo rosdep init
rosdep update
```

### 2.4 配置环境变量

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

验证安装：
```bash
ros2 -h
python3 -c "import rclpy; print('rclpy OK')"
```

---

## 3. 安装 ros2_kortex

> **注意：** ROS 2 版本是独立的新仓库 `ros2_kortex`，**不是**旧版 ROS 1 的 `ros_kortex`。

### 3.1 创建工作区

```bash
export COLCON_WS=~/ws/ros2_kortex_ws
mkdir -p $COLCON_WS/src
cd $COLCON_WS
```

### 3.2 克隆（Humble 分支）

```bash
git clone -b humble https://github.com/Kinovarobotics/ros2_kortex.git src/ros2_kortex
```

支持的 ROS 2 发行版：

| ROS 2 版本 | 分支 | 状态 |
|------------|------|------|
| **Humble** | `humble` | ✅ Stable，有预编译二进制 |
| Jazzy | `jazzy` | ✅ Stable (source only) |
| Rolling | `main` | ⚠️ 不稳定 |

### 3.3 导入依赖

```bash
cd $COLCON_WS
vcs import src --skip-existing --input src/ros2_kortex/ros2_kortex.humble.repos
vcs import src --skip-existing --input src/ros2_kortex/ros2_kortex-not-released.humble.repos
```

### 3.4 安装系统依赖

```bash
rosdep install --ignore-src --from-paths src -y -r
```

会自动安装：`ros2-control`、`ros2-controllers`、`moveit-*`、`backward-ros`、`control-toolbox`、`kinematics-interface`、`realtime-tools`、`libcap-dev` 等。

### 3.5 编译

```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers 4
```

预期输出：成功编译 **10 个 kortex 包**（kortex_api、kortex_driver、kortex_bringup、kortex_description、MoveIt configs 等）。

### 3.6 配置工作区环境

```bash
echo "source ~/ws/ros2_kortex_ws/install/setup.bash" >> ~/.bashrc

# 可选：快捷别名
echo 'alias ros2kortex="source /opt/ros/humble/setup.bash && source ~/ws/ros2_kortex_ws/install/setup.bash"' >> ~/.bashrc

source ~/.bashrc
```

---

## 4. 使用指南

### 4.1 启动机械臂驱动

```bash
# 终端 1：启动驱动
ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=<机械臂 IP> \
  use_fake_hardware:=false

# 仿真模式（离线调试用）
ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=<机械臂 IP> \
  use_fake_hardware:=true
```

启动后自动打开 RViz2。

### 4.2 查看话题

```bash
# 另开一个终端
source /opt/ros/humble/setup.bash
source ~/ws/ros2_kortex_ws/install/setup.bash
ros2 topic list
```

关键话题：

| 话题 | 类型 | 说明 |
|------|------|------|
| `/joint_states` | `sensor_msgs/JointState` | 7 个关节角度 + 速度 |
| `/tf` | `tf2_msgs/TFMessage` | TF 变换树 |
| `/joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | 关节轨迹指令 |
| `/twist_controller/commands` | `geometry_msgs/TwistStamped` | 末端速度指令 |

### 4.3 关节位置控制

```bash
# 发送关节目标（2 秒内归零）
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "trajectory:
  joint_names: ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
  points:
  - positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_from_start: {sec: 2, nanosec: 0}"

# 多点轨迹
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "trajectory:
  joint_names: ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
  points:
  - positions: [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_from_start: {sec: 2, nanosec: 0}
  - positions: [0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0]
    time_from_start: {sec: 5, nanosec: 0}"
```

### 4.4 末端 Twist 控制

```bash
# 持续发送末端速度（Ctrl+C 停止）
ros2 topic pub /twist_controller/commands geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: 'tool_frame'}, twist: {linear: {x: 0.0, y: 0.0, z: 0.01}, angular: {x: 0.0, y: 0.0, z: 0.0}}}" \
  --rate 10
```

### 4.5 rosbag 数据录制

```bash
# 所有话题
ros2 bag record -a

# 指定话题
ros2 bag record /joint_states /tf /tf_static

# 指定输出目录
ros2 bag record -o motion_data /joint_states /tool_pose

# 查看 bag 信息
ros2 bag info motion_data/

# 回放
ros2 bag play motion_data/
```

### 4.6 查看关节状态

```bash
# 实时打印
ros2 topic echo /joint_states

# 一次性查看关节值
ros2 topic echo /joint_states --once --field position

# 查看发布频率
ros2 topic hz /joint_states
```

---

## 5. 常见问题

### Q: RViz2 弹出但无机械臂模型
A: `robot_state_publisher` 未正常启动，检查 launch 输出。

### Q: spawner 持续等待 controller_manager service
A: 驱动无法连接机械臂。检查：
- `ping <robot_ip>` 是否通
- 机械臂是否开机
- PC 和机械臂是否同一网段

### Q: ROS 2 实时调度警告
```
Could not enable FIFO RT scheduling policy
```
A: 正常警告，非实时 Linux 限制，不影响功能。

### Q: rosbag 录到 0 条消息
A: 录制前先 `ros2 topic list` 确认话题存在且有数据。

### Q: cv_bridge 报 numpy 版本错误
```
AttributeError: _ARRAY_API not found
```
A: cv_bridge 依赖 numpy 1.x，需要降级：
```bash
pip install 'numpy<2' 'opencv-python<5'
```

---

## 6. Python 编程示例

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointLogger(Node):
    def __init__(self):
        super().__init__('joint_logger')
        self.sub = self.create_subscription(
            JointState, '/joint_states', self.cb, 10)

    def cb(self, msg):
        joints = [f"{j:.3f}" for j in msg.position[:7]]
        self.get_logger().info(f'joints: [{joints}]')

rclpy.init()
rclpy.spin(JointLogger())
rclpy.shutdown()
```

---

## 7. 与原生 Kortex API 对比

| 特性 | 原生 Kortex API | ROS 2 + ros2_kortex |
|------|----------------|---------------------|
| 安装复杂度 | ✅ 简单 | ⚠️ 稍复杂 |
| 数据采集 | ❌ 手动记录 | ✅ rosbag 一键录制 |
| 时间同步 | ❌ 无 | ✅ message_filters |
| 可视化 | ❌ 无 | ✅ RViz2 |
| 多传感器融合 | ❌ 需自建 | ✅ 原生支持 |
| MoveIt 规划 | ❌ 无 | ✅ 内置配置 |

对于数据采集 + 视觉类项目，推荐 ROS 2。

---
<<<<<<< Updated upstream

## 8. 实时 RGB 去模糊 WS 推理

本仓库提供轻量入口 `ws_inference_realtime_deblur.py`。去模糊算法仍由独立
仓库维护；入口会加载该仓库的标准 WS 包装器，并自动把当前目录作为
`--controller-root`。标准包装器只替换相机取图步骤，策略收到的
`observation/image` 是实时 RGB Wiener 去模糊后的图像，其他 Pi05 控制与
WebSocket 推理流程保持不变。

公开仓库：

- Gen3 Controller：<https://github.com/edcavani9520/fnii-gen3-controller.git>
- RGB 去模糊：<https://github.com/edcavani9520/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin.git>

推荐克隆为同级目录：

```powershell
git clone https://github.com/edcavani9520/fnii-gen3-controller.git
git clone https://github.com/edcavani9520/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin.git
cd fnii-gen3-controller
python ws_inference_realtime_deblur.py `
  --deblur-root ../Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin `
  --ws-host localhost --ws-port 8000 `
  --K 0.01 --depth 0.5 --exposure 0.03 --fx 733.37 --fy 733.37
```

`--deblur-root` 只由当前轻量入口处理，其余参数原样转发给去模糊仓库中的
标准启动脚本。运行前需要安装两个仓库各自声明的依赖，包括 NumPy、SciPy、
OpenCV、Kinova Kortex、OpenPI/WS，以及支持同步客户端的 `websockets`。

---

=======
>>>>>>> Stashed changes
