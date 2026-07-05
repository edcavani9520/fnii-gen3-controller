# ROS 2 Humble + ros2_kortex 基础调用指南

## 1. 环境

```bash
# source 环境（已加到 .bashrc）
source /opt/ros/humble/setup.bash
source ~/ws/ros2_kortex_ws/install/setup.bash

# 快捷别名
ros2kortex
```

## 2. 启动机械臂驱动

```bash
# 真实机械臂
ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=192.168.8.10 \
  use_fake_hardware:=false

# 仿真模式（离线调试用）
ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=192.168.8.10 \
  use_fake_hardware:=true
```

启动成功后会有：
- `ros2_control_node` — controller manager
- `robot_state_publisher` — URDF + TF
- `joint_state_broadcaster`, `joint_trajectory_controller`, `twist_controller`
- RViz2 仿真界面

## 3. 常用 ROS 2 命令

```bash
# 查看所有话题
ros2 topic list

# 查看话题内容（实时打印）
ros2 topic echo /joint_states
ros2 topic echo /tool_pose

# 查看话题信息（类型、发布者）
ros2 topic info /joint_states

# 查看话题频率
ros2 topic hz /joint_states

# 查看所有服务
ros2 service list

# 查看所有节点
ros2 node list

# 查看 TF 树
ros2 run tf2_tools view_frames.py
```

## 4. 机械臂控制

### 4.1 通过 Action 发送关节位置（推荐）

```bash
# 发送单点目标（2 秒内归零）
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "trajectory:
  joint_names: ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
  points:
  - positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_from_start: {sec: 2, nanosec: 0}"

# 发送多点轨迹（依次经过多个位姿）
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "trajectory:
  joint_names: ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
  points:
  - positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_from_start: {sec: 2, nanosec: 0}
  - positions: [0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_from_start: {sec: 5, nanosec: 0}"
```

> **关键：** YAML 格式必须用冒号+换行缩进，不能用 `{}` 花括号，否则 shell 会解析错误。
```

### 4.3 Twist 控制（末端速度）

```bash
# 发送末端线速度/角速度
ros2 topic pub /twist_controller/commands \
  geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: 'tool_frame'},
    twist: {linear: {x: 0.0, y: 0.0, z: 0.02},
            angular: {x: 0.0, y: 0.0, z: 0.0}}}" \
  --rate 10
```

> **安全提示：** Twist 控制是持续发送的，`--rate 10` 会持续发 10Hz，Ctrl+C 停止。

## 5. 数据录制（rosbag）

```bash
# 录制所有话题
ros2 bag record -a

# 只录需要的
ros2 bag record /joint_states /tool_pose /tf /tf_static

# 指定输出目录
ros2 bag record -o motion_data /joint_states /tool_pose

# 查看 bag 信息
ros2 bag info motion_data/

# 回放 bag
ros2 bag play motion_data/
```

### rosbag 输出结构

```
motion_data/
├── metadata.yaml         # 元数据（话题列表、时长、消息数）
└── motion_data_0.db3     # SQLite 数据库（实际数据）
```

## 6. Python 编程接口

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
import time

class KortexControl(Node):
    def __init__(self):
        super().__init__('kortex_control')
        
        # 订阅关节状态
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', 
            self.joint_callback, 10)
        
        # 订阅末端位姿
        self.pose_sub = self.create_subscription(
            PoseStamped, '/tool_pose',
            self.pose_callback, 10)
        
        # Action client 做轨迹控制
        self.traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')
        
        self.latest_joints = None
        self.latest_pose = None
    
    def joint_callback(self, msg):
        self.latest_joints = msg
    
    def pose_callback(self, msg):
        self.latest_pose = msg
    
    def move_to_joints(self, positions, duration=2.0):
        """发送关节目标位置"""
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'joint_7'
        ]
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal_msg.trajectory.points = [point]
        
        self.traj_client.wait_for_server()
        return self.traj_client.send_goal_async(goal_msg)


def main():
    rclpy.init()
    node = KortexControl()
    
    # 单线程 spin（用于简单采集）
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    # 采集 10 秒数据
    start = time.time()
    while time.time() - start < 10:
        executor.spin_once(timeout_sec=0.01)
        if node.latest_joints:
            print(f"Joints: {[round(j, 3) for j in node.latest_joints.position]}")
        if node.latest_pose:
            p = node.latest_pose.pose.position
            print(f"Pose: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7. 关键话题列表

| 话题 | 类型 | 说明 |
|------|------|------|
| `/joint_states` | `sensor_msgs/JointState` | 7 个关节角度 + 速度 |
| `/tool_pose` | `geometry_msgs/PoseStamped` | 末端位姿 |
| `/tf` | `tf2_msgs/TFMessage` | TF 变换树 |
| `/tf_static` | `tf2_msgs/TFMessage` | 静态 TF（world→base） |
| `/joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | 关节轨迹指令 |
| `/twist_controller/commands` | `geometry_msgs/TwistStamped` | 末端速度指令 |

## 8. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| RViz 弹出但无模型 | `robot_state_publisher` 未启动 | 检查 launch 输出 |
| `joint_state_broadcaster` 等待 service | controller_manager 未就绪 | 检查 robot_ip 是否正确 |
| rt 调度警告 | 非实时系统 | 可忽略，不影响功能 |
| rosbag 录到 0 条消息 | 录制时话题没在发布 | 先 `ros2 topic list` 确认 |
