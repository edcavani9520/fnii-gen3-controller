# Kinova Gen3 机器人遥控与管理系统 (Xbox Controller & Manager)

本仓库包含两个核心 Python 脚本，用于实现 Kinova Gen3 机械臂的实时手柄控制和结构化管理。

## 📂 文件说明

| 文件名 | 核心功能 | 控制模式 |
| --- | --- | --- |
| `gamepad_control.py` | 使用 Xbox 手柄实时遥控机械臂，支持 6 自由度运动及夹爪控制。 | **Velocity Control** (Twist Command) |
| `kinova_manage.py` | 提供机械臂状态获取、数据打印及点对点（关节/笛卡尔）运动的类封装。 | **Position Control** (Reach Pose/Joints) |

---

## 🛠 环境依赖

1. **硬件要求**：
* Kinova Gen3 机器人 (G3L)
* Xbox 无线/有线手柄 + 电脑接收器


2. **软件库**：
* `kortex_api`: Kinova 官方 SDK
* `pygame`: 用于处理手柄输入信号
* `utilities`: Kinova 示例代码附带的连接工具类



```bash
pip install pygame

```

---

## 🎮 gamepad_control.py：实时遥控指南

该脚本通过发送 `TwistCommand` 实现低延迟的速度控制，适合人工演示采集（Demonstration Learning）或环境探索。

### 1. 启动方式

确保手柄已连接并处于激活状态，机器人 IP 为 `192.168.8.10`：

```bash
python gamepad_control.py

```

### 2. 手柄按键映射 (Xbox)

| 类别     | 手柄部件         | 机器人动作          | 说明 |
| **平移** | **左摇杆 上/下** | **X 轴** 前/后      | 推上为前 |
|          | **左摇杆 左/右** | **Y 轴** 右/左      | **镜像模式**：推左向右，推右向左 |
|          | **右扳机 RT**    | **Z 轴** 垂直上升   | 线性映射速度 |
|          | **左扳机 LT**    | **Z 轴** 垂直下降   | 线性映射速度 |
| **姿态** | **右摇杆 上/下** | **Pitch** (绕 Y 轴) | 抬头/低头 |
|          | **右摇杆 左/右** | **Roll** (绕 X 轴)  | 左右倾斜 |
|          | **十字键 左/右** | **Yaw** (绕 Z 轴)   | 水平转动 |
| **执行** | **A 按钮**       | **夹爪关闭**        | 100% 闭合 |
|          | **B 按钮**       | **夹爪打开**        | 0% 开启 |
| **系统** | **Menu 按钮**    | **安全退出**        | 停止所有动作并断开连接 |

---

## 🏗 kinova_manage.py：API 调用细节

该脚本封装了 `KinovaManager` 类，适合集成到自动化流程或 RL 训练循环中。

### 1. 初始化与连接

```python
from kinova_manage import KinovaManager

arm = KinovaManager(ip_address="192.168.8.10")
arm.connect()

```

### 2. 核心函数说明

#### `get_status()`

* **功能**: 获取机器人当前的完整循环反馈数据。
* **返回**: `BaseCyclic_pb2.Feedback` 对象，包含末端位姿、关节角度、力矩、电流等。

#### `print_status(status)`

* **功能**: 将 `get_status()` 返回的原始数据格式化为易读的中英双语表格。

#### `move_angular(angles_and_gripper)`

* **参数**: `[j1, j2, j3, j4, j5, j6, j7, gripper_pos]`
* **单位**: 角度为度 (°)，夹爪为 0 (开) - 100 (关)。
* **细节**: 阻塞型调用，直到运动完成或超时（20s）。

#### `move_cartesian(pose_and_gripper)`

* **参数**: `[x, y, z, theta_x, theta_y, theta_z, dummy, gripper_pos]`
* **单位**: 坐标为米 (m)，旋转为欧拉角 (°)。
* **细节**: 基于 `CARTESIAN_REFERENCE_FRAME_BASE`（基座坐标系）运动。

---

## ⚠️ 安全注意事项

1. **死区 (Deadzone)**: `gamepad_control.py` 设置了 `0.1` 的死区，以防摇杆漂移导致机器人意外缓慢移动。
2. **紧急停止**:
* 遥控时按下 **Menu 按钮**。
* 代码运行中按下 **Ctrl+C**。
* 以上操作都会触发 `base.Stop()` 指令。


3. **坐标系**:
* 手柄控制默认使用 **Base Frame**。如果你需要相对于手爪（Tool Frame）移动，请在 `command.reference_frame` 中修改。



---

## 📝 开发备忘

* **频率控制**: 遥控循环运行在约 `20Hz` (`time.sleep(0.05)`)。对于高动态任务，可适当调低 sleep 时间，但需注意网络抖动。
* **镜像逻辑**: `linear_y` 的镜像映射是为了满足操作者站在机器人对面的视觉习惯，如需恢复直觉方向，请修改 `command.twist.linear_y = axis_0 * self.speed_limit`。