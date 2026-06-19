from typing import List, Optional, Any, Union

'''
相关文件路径
本文件：/home/cuhk/Documents/visionpro-kinova-rl/robot_control/api_control/kinova_manage.pyi
对py文件：/home/cuhk/Documents/visionpro-kinova-rl/robot_control/api_control/kinova_manage.py
'''

# ===============================================================================
# Kinova Gen3 机器人状态结构详尽参考手册 (基于 Kortex API RefreshFeedback)
# ===============================================================================
# 访问示例: 
#   x = status.base.tool_pose_x 
#   j1_pos = status.actuators[0].position
#   grip = status.interconnect.gripper_feedback.motor[0].position
# ===============================================================================

"""
base {
  active_state_connection_identifier: 23
  active_state: ARMSTATE_SERVOING_MANUALLY_CONTROLLED
  arm_voltage: 24.124364852905273
  arm_current: 0.9299541711807251
  temperature_cpu: 54.999977111816406
  temperature_ambient: 36.375
  imu_acceleration_x: 0.08630575239658356
  imu_acceleration_y: -0.07141627371311188
  imu_acceleration_z: -9.556197166442871
  imu_angular_velocity_x: 3.193711757659912
  imu_angular_velocity_y: -0.18035158514976501
  imu_angular_velocity_z: 1.4630030393600464
  tool_pose_x: 0.3999621570110321
  tool_pose_y: 2.849023530870909e-06
  tool_pose_z: 0.2699083983898163
  tool_pose_theta_x: -179.98507690429688
  tool_pose_theta_y: -0.000762932701036334
  tool_pose_theta_z: 89.99980926513672
  tool_twist_linear_x: -8.20965287857689e-05
  tool_twist_linear_y: 2.324608203707612e-06
  tool_twist_linear_z: -8.235634595621377e-05
  tool_twist_angular_x: 0.0006292497273534536
  tool_twist_angular_y: 0.014186703599989414
  tool_twist_angular_z: -5.534363936021691e-07
  tool_external_wrench_force_x: -1.7827863693237305
  tool_external_wrench_force_y: 1.4760409593582153
  tool_external_wrench_force_z: -2.5812265872955322
  tool_external_wrench_torque_x: 0.23872852325439453
  tool_external_wrench_torque_y: -1.0745967626571655
  tool_external_wrench_torque_z: -0.0855925977230072
  commanded_tool_pose_x: 0.39996787905693054
  commanded_tool_pose_y: -2.8001591090287548e-06
  commanded_tool_pose_z: 0.26992854475975037
  commanded_tool_pose_theta_x: -179.9883575439453
  commanded_tool_pose_theta_y: -6.223928357940167e-05
  commanded_tool_pose_theta_z: 89.99903869628906
}
actuators {
  command_id: 2147524113
  status_flags: 33556528
  jitter_comm: 1987652891
  position: 356.87493896484375
  velocity: 0.00030315073672682047
  torque: -0.32055824995040894
  current_motor: -0.33042049407958984
  voltage: 23.35129737854004
  temperature_motor: 34.105690002441406
  temperature_core: 41.47410583496094
}
actuators {
  command_id: 2147589649
  status_flags: 33556528
  jitter_comm: 1987660789
  position: 15.231453895568848
  velocity: 0.003938477952033281
  torque: 12.737239837646484
  current_motor: -0.9104360342025757
  voltage: 23.32292366027832
  temperature_motor: 35.15385818481445
  temperature_core: 40.89493942260742
}
actuators {
  command_id: 2147655185
  status_flags: 33556528
  jitter_comm: 1987646449
  position: 184.0891571044922
  velocity: -0.00030337978387251496
  torque: -0.8916051387786865
  current_motor: 0.38680264353752136
  voltage: 23.30873680114746
  temperature_motor: 33.52641677856445
  temperature_core: 42.69841384887695
}
actuators {
  command_id: 2147720721
  status_flags: 33556528
  jitter_comm: 1987629086
  position: 260.16143798828125
  velocity: -0.01168150920420885
  torque: -7.970439434051514
  current_motor: 0.4754100441932678
  voltage: 23.209430694580078
  temperature_motor: 34.54961013793945
  temperature_core: 43.172691345214844
}
actuators {
  command_id: 2147786257
  status_flags: 33556528
  jitter_comm: 1987633630
  position: 357.7303771972656
  velocity: -0.0006063014734536409
  torque: -0.5848085880279541
  current_motor: 0.0362548828125
  voltage: 23.209430694580078
  temperature_motor: 40.232765197753906
  temperature_core: 45.74297332763672
}
actuators {
  command_id: 2147851793
  status_flags: 33556528
  jitter_comm: 1987651886
  position: 295.5670166015625
  velocity: 0.0009087661164812744
  torque: 0.5347604155540466
  current_motor: 0.024169921875
  voltage: 23.13849639892578
  temperature_motor: 40.615455627441406
  temperature_core: 45.28455352783203
}
actuators {
  command_id: 2147917329
  status_flags: 33556528
  jitter_comm: 1987619467
  position: 91.23336791992188
  velocity: 0.0006063014734536409
  torque: -0.16213254630565643
  current_motor: -0.0443115234375
  voltage: 23.294551849365234
  temperature_motor: 40.735496520996094
  temperature_core: 44.08000183105469
}
interconnect {
  feedback_id {
    identifier: 2147983107
  }
  status_flags: 268438544
  jitter_comm: 1987703061
  imu_acceleration_x: 0.04842180013656616
  imu_acceleration_y: -0.05081300064921379
  imu_acceleration_z: -9.6425142288208
  imu_angular_velocity_x: -0.4025000035762787
  imu_angular_velocity_y: -1.4612499475479126
  imu_angular_velocity_z: 0.13124999403953552
  voltage: 23.564062118530273
  temperature_core: 45.37255096435547
  gripper_feedback {
    status_flags: 13
    motor {
      motor_id: 1
      position: 0.4385974407196045
      voltage: 23.564062118530273
    }
  }
}
"""

class KinovaManager:
    """
    Kinova Gen3 机器人管理类，封装了 Kortex API 的常用控制逻辑。
    """

    def __init__(self, ip_address: str = "192.168.8.10") -> None:
        """
        初始化管理器
        :param ip_address: 机器人的 IP 地址，默认 192.168.8.10
        """
        ...

    def connect(self) -> None:
        """ 建立 TCP 连接并初始化 BaseClient 与 BaseCyclicClient """
        ...

    def disconnect(self) -> None:
        """ 安全断开连接并清理路由器环境 """
        ...

    def get_status(self) -> Any:
        """
        从机器人获取最新的实时反馈数据。
        :return: kortex_api.RefreshFeedback 对象，包含位姿、电流、扭矩等全量信息。
        """
        ...

    def print_status(self, status: Any) -> None:
        """
        在控制台打印格式化后的机器人核心信息（位姿、关节、夹爪、电力）。
        :param status: 由 get_status() 获取的反馈对象。
        """
        ...

    def control_gripper(self, value: float, dual_grip: bool = True) -> None:
        """
        直接控制夹爪。
        :param value: 
            - 若 dual_grip=True:  阈值信号 [0.0 ~ 1.0]。 >=0.5 为闭合，<0.5 为开启。
            - 若 dual_grip=False: 连续位置信号 [0.0 ~ 100.0]。 0.0 为全开，100.0 为全闭。
        :param dual_grip: 是否启用 0.5 阈值的二元化判定逻辑。
        """
        ...

    def move_angular(self, angles_and_gripper: List[float], dual_grip: bool = True) -> None:
        """
        关节空间绝对位置移动。
        :param angles_and_gripper: 形状为 (8,) 的列表 [J1, J2, J3, J4, J5, J6, J7, Gripper]
            - J1~J7: 角度范围约为 [-360.0, 360.0] (取决于机器人软限位)。
            - Gripper: 夹爪信号，范围参考 control_gripper 的 value 定义。
        :param dual_grip: 夹爪判定模式。
        """
        ...

    def move_cartesian(self, pose_and_gripper: List[float], dual_grip: bool = True, skip_gripper: bool = False) -> None:
        """
        笛卡尔空间绝对位置移动。
        :param pose_and_gripper: 形状为 (7,) 的列表 [X, Y, Z, ThetaX, ThetaY, ThetaZ, Gripper]
            - X, Y, Z: 单位为米 (m)。
            - ThetaX/Y/Z: 欧拉角 (RPY)，单位为度 (°)。
            - Gripper: 夹爪信号，范围参考 control_gripper 的 value 定义。
        :param dual_grip: 夹爪判定模式。
        :param skip_gripper: (内部参数) 是否跳过夹爪动作。建议外部调用保持 False。
        """
        ...

    def move_velocity(self, velocities: List[float], duration: int = 0) -> None:
        """
        笛卡尔空间速度控制 (非阻塞，适用于实时 RL Step)。
        :param velocities: 形状为 (6,) 的列表 [Vx, Vy, Vz, Wx, Wy, Wz]
            - Vx, Vy, Vz: 线速度，单位为米/秒 (m/s)。
            - Wx, Wy, Wz: 角速度，单位为度/秒 (°/s)。
        :param duration: 指令持续时间。
            - 0: 持续执行直到收到新指令或 Stop 指令。
            - >0: 执行指定的毫秒数后停止。
        """

    def move_relative(self, delta_pose: List[float], dual_grip: bool = True) -> None:
        """
        相对于当前位置的增量移动。
        :param delta_pose: 形状为 (7,) 的列表 [dX, dY, dZ, dTX, dTY, dTZ, GripperSignal]
            - dX, dY, dZ: 位移增量 (m)。
            - dTX, dTY, dTZ: 角度增量 (°)。
            - GripperSignal: 
                - 若 dual_grip=True:  触发信号。 >=0.5 则“翻转”当前状态（开变关，关变开）；<0.5 则保持现状。
                - 若 dual_grip=False: 目标位置 [0.0 ~ 100.0]。
        :param dual_grip: 决定 GripperSignal 是作为“翻转触发”还是“绝对目标”。
        """
        ...