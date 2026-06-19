import sys
import os

# 动态添加 utilities 路径
sys.path.insert(0, "/home/cuhk/Documents/visionpro-kinova-rl/Kinova-kortex2_Gen3_G3L/api_python/examples")
import utilities

from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

def get_arm_status(ip_address="192.168.8.10"):
    """
    获取机械臂当前的所有反馈状态（BaseCyclic Feedback）
    """
    # 模拟命令行参数，用于 utilities 解析
    class Args:
        def __init__(self):
            self.ip = ip_address
            self.username = "admin"
            self.password = "admin"
            self.port = 10000 # Kortex 默认 TCP 端口

    args = Args()

    try:
        # 使用 TCP 连接
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            # 创建循环反馈客户端
            base_cyclic = BaseCyclicClient(router)
            
            # 获取实时反馈（包含位姿、力矩、电流、温度等）
            current_status = base_cyclic.RefreshFeedback()
            
            return current_status
    except Exception as e:
        print(f"Error connecting to the arm at {ip_address}: {e}")
        return None
    
def print_status(status):
    if not status:
        print("无有效状态数据 / No valid status data.")
        return

    print("=" * 60)
    print(f"{'机械臂状态概览 / ARM STATUS OVERVIEW':^60}")
    print("=" * 60)

    # 1. 末端位姿 (Base / Tool Pose)
    base = status.base
    print(f"\n[末端位姿 / Tool Pose]")
    print(f"  坐标 (Cartesian): X: {base.tool_pose_x:7.3f} m, Y: {base.tool_pose_y:7.3f} m, Z: {base.tool_pose_z:7.3f} m")
    print(f"  姿态 (Rotation) : θx: {base.tool_pose_theta_x:7.2f}°, θy: {base.tool_pose_theta_y:7.2f}°, θz: {base.tool_pose_theta_z:7.2f}°")
    
    # 2. 外部受力 (External Wrench)
    print(f"\n[末端受力 / External Wrench]")
    print(f"  力 (Force)  : Fx: {base.tool_external_wrench_force_x:6.2f} N, Fy: {base.tool_external_wrench_force_y:6.2f} N, Fz: {base.tool_external_wrench_force_z:6.2f} N")
    print(f"  扭矩(Torque): Tx: {base.tool_external_wrench_torque_x:6.2f} Nm, Ty: {base.tool_external_wrench_torque_y:6.2f} Nm, Tz: {base.tool_external_wrench_torque_z:6.2f} Nm")

    # 3. 关节状态 (Actuators Status)
    print(f"\n[关节状态 / Actuators Status]")
    header = f"{'ID':>4} | {'角度/Pos (°)':>12} | {'力矩/Trq (Nm)':>12} | {'温度/Temp (°C)':>12}"
    print("-" * len(header))
    print(header)
    for i, actuator in enumerate(status.actuators):
        print(f"{i+1:>4} | {actuator.position:>12.2f} | {actuator.torque:>12.2f} | {actuator.temperature_core:>12.1f}")

    # 4. 夹爪状态 (Gripper)
    gripper = status.interconnect.gripper_feedback.motor
    if gripper:
        print(f"\n[夹爪状态 / Gripper Status]")
        # 夹爪位置通常 0-100，这里根据你的反馈显示位置
        print(f"  位置 (Position): {gripper[0].position:6.2f} % (0为全开/Open, 100为全关/Closed)")

    # 5. 系统健康 (System Health)
    print(f"\n[系统健康 / System Health]")
    print(f"  电压 (Voltage): {base.arm_voltage:5.2f} V  |  电流 (Current): {base.arm_current:5.2f} A")
    print(f"  CPU温度 (CPU Temp): {base.temperature_cpu:4.1f} °C")
    
    print("-" * 60)

# --- 测试代码 ---
if __name__ == "__main__":
    status = get_arm_status()
    print_status(status)
    # print(status.base.tool_pose_x)


''' status demo
base {
  active_state_connection_identifier: 4
  active_state: ARMSTATE_SERVOING_READY
  arm_voltage: 24.169273376464844
  arm_current: 0.8841744065284729
  temperature_cpu: 56.77482604980469
  temperature_ambient: 37.81197738647461
  imu_acceleration_x: 0.08321584761142731
  imu_acceleration_y: -0.050576791167259216
  imu_acceleration_z: -8.150874137878418
  imu_angular_velocity_x: 3.171170234680176
  imu_angular_velocity_y: -0.22051280736923218
  imu_angular_velocity_z: 1.5034761428833008
  tool_pose_x: 0.37847408652305603
  tool_pose_y: 0.03945393115282059
  tool_pose_z: 0.2316967099905014
  tool_pose_theta_x: 179.23329162597656
  tool_pose_theta_y: -0.6652464866638184
  tool_pose_theta_z: 93.70948791503906
  tool_twist_linear_x: 2.425988895993214e-06
  tool_twist_linear_y: 2.3646825297873875e-07
  tool_twist_linear_z: 1.4261031537898816e-06
  tool_twist_angular_x: 2.95878853648901e-05
  tool_twist_angular_y: -0.00030239473562687635
  tool_twist_angular_z: -1.9151974584019626e-07
  tool_external_wrench_force_x: -1.5155402421951294
  tool_external_wrench_force_y: -0.4452867805957794
  tool_external_wrench_force_z: -0.906146228313446
  tool_external_wrench_torque_x: 1.045444369316101
  tool_external_wrench_torque_y: -0.491803377866745
  tool_external_wrench_torque_z: 0.4887232184410095
  commanded_tool_pose_x: 0.37852099537849426
  commanded_tool_pose_y: 0.03946182131767273
  commanded_tool_pose_z: 0.23176279664039612
  commanded_tool_pose_theta_x: 179.2212677001953
  commanded_tool_pose_theta_y: -0.6660537123680115
  commanded_tool_pose_theta_z: 93.70902252197266
}
actuators {
  command_id: 2147542310
  status_flags: 33556496
  jitter_comm: 210206059
  position: 356.2059020996094
  torque: -0.34274446964263916
  current_motor: -0.01611328125
  voltage: 23.408044815063477
  temperature_motor: 36.02880859375
  temperature_core: 43.06772994995117
}
actuators {
  command_id: 2147607846
  status_flags: 33556496
  jitter_comm: 210211199
  position: 15.091864585876465
  torque: 11.970172882080078
  current_motor: -1.216552734375
  voltage: 23.393856048583984
  temperature_motor: 36.848976135253906
  temperature_core: 43.38521194458008
}
actuators {
  command_id: 2147673382
  status_flags: 33556496
  jitter_comm: 210208686
  position: 177.9479522705078
  torque: -0.5636883974075317
  current_motor: -0.011393810622394085
  voltage: 23.379671096801758
  temperature_motor: 35.62919998168945
  temperature_core: 44.60317611694336
}
actuators {
  command_id: 2147738918
  status_flags: 33556496
  jitter_comm: 210183983
  position: 251.9836883544922
  velocity: 0.00030383889679796994
  torque: -7.370142459869385
  current_motor: 0.845947265625
  voltage: 23.251989364624023
  temperature_motor: 37.29289627075195
  temperature_core: 46.06425857543945
}
actuators {
  command_id: 2147804454
  status_flags: 33556496
  jitter_comm: 210203100
  position: 358.7534484863281
  torque: -0.43960997462272644
  current_motor: -0.0845947265625
  voltage: 23.266178131103516
  temperature_motor: 42.00603103637695
  temperature_core: 48.31325149536133
}
actuators {
  command_id: 2147869990
  status_flags: 33556496
  jitter_comm: 210204215
  position: 304.45355224609375
  torque: 0.08952199667692184
  current_motor: 0.08066704869270325
  voltage: 23.19524383544922
  temperature_motor: 42.82861328125
  temperature_core: 48.21138381958008
}
actuators {
  command_id: 2147935526
  status_flags: 33556496
  jitter_comm: 210184616
  position: 88.09202575683594
  torque: 0.45318782329559326
  current_motor: -0.02450331673026085
  voltage: 23.32292366027832
  temperature_motor: 42.5184326171875
  temperature_core: 46.0
}
interconnect {
  feedback_id {
    identifier: 2147998924
  }
  status_flags: 268438544
  jitter_comm: 210255474
  imu_acceleration_x: 0.16021040081977844
  imu_acceleration_y: -0.025107599794864655
  imu_acceleration_z: -9.610233306884766
  imu_angular_velocity_x: 0.45500001311302185
  imu_angular_velocity_y: -1.347499966621399
  imu_angular_velocity_z: -0.4025000035762787
  voltage: 23.57695198059082
  temperature_core: 46.94117736816406
  gripper_feedback {
    status_flags: 13
    motor {
      motor_id: 1
      position: 99.12664031982422
      voltage: 23.57695198059082
    }
  }
}
'''