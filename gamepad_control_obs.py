import sys
import os
import time
import pygame
import threading
import numpy as np

# 动态添加 utilities 路径
sys.path.insert(0, "/home/kinova-1/Kinova-gen3/gen3-controller/Kinova_kortex2_Gen3_G3L/api_python/examples")
import utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2

class KinovaJoyTeleop:
    def __init__(self, ip="192.168.8.10"):
        self.ip = ip
        self.base = None
        self.base_cyclic = None
        self.router = None
        self.connection = None
        
        # 控制参数 / Control Parameters
        self.deadzone = 0.1      
        self.speed_limit = 0.1  
        self.turn_limit = 20.0   

        # 初始化 Pygame 手柄 / Initialize Joystick
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("❌ 未检测到 Xbox 手柄，请确保连接。")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        
        # 共享状态变量 / Shared State
        self.full_status = None
        self.running = False
        self.last_gripper_label = "IDLE"

    def connect(self):
        class Args:
            def __init__(self, ip):
                self.ip, self.username, self.password, self.port = ip, "admin", "admin", 10000
        self.connection = utilities.DeviceConnection.createTcpConnection(Args(self.ip))
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        print(f"✅ 已连接: {self.joy.get_name()}")
        os.system('clear')

    def _state_observer_thread(self):
        """后台线程：以 100Hz 频率拉取全量周期性反馈"""
        while self.running:
            try:
                # 获取包含 Base, Actuators, Interconnect 的全量反馈 (Cyclic Feedback)
                self.full_status = self.base_cyclic.RefreshFeedback()
                time.sleep(0.01) 
            except Exception:
                pass

    def display_full_dashboard(self):
        """综合仪表盘：手柄输入监控 + 机械臂全维度状态"""
        if not self.full_status:
            return

        # 移动光标到左上角实现无闪烁刷新
        sys.stdout.write("\033[H")
        
        b = self.full_status.base
        inter = self.full_status.interconnect
        
        # --- 标题与指令监控 / Title & Joy Monitor ---
        print(f"{' KINOVA 实时状态仪表盘 REAL-TIME DASHBOARD ':=^85}")
        axes = [self.joy.get_axis(i) for i in range(6)]
        print(f"[手柄输入] L轴(X/Y): {axes[1]:.2f}, {axes[0]:.2f} | R轴: {axes[3]:.2f}, {axes[4]:.2f} | 升降: {(self.joy.get_axis(5)-self.joy.get_axis(2)):.2f}")
        print(f"[指令动作] 夹爪: {self.last_gripper_label:<8} | 频率: 20Hz控制 / 100Hz采样")

        # --- 1. 末端位姿与指令参考 (跟随误差监控) ---
        print(f"\n[ 末端位姿与指令 / POSE & COMMAND ]")
        print(f"当前位姿/Actual  | XYZ(m): {b.tool_pose_x:7.4f}, {b.tool_pose_y:7.4f}, {b.tool_pose_z:7.4f}")
        print(f"                  | RPY(°): {b.tool_pose_theta_x:7.2f}, {b.tool_pose_theta_y:7.2f}, {b.tool_pose_theta_z:7.2f}")
        print(f"指令参考/Command | XYZ(m): {b.commanded_tool_pose_x:7.4f}, {b.commanded_tool_pose_y:7.4f}, {b.commanded_tool_pose_z:7.4f}")

        # --- 2. 末端速度与受力 (捻度 + Wrench) ---
        print(f"\n[ 运动捻度与受力 / TWIST & WRENCH ]")
        print(f"线速度/Linear(m/s) | X:{b.tool_twist_linear_x:7.3f}, Y:{b.tool_twist_linear_y:7.3f}, Z:{b.tool_twist_linear_z:7.3f}")
        print(f"角速度/Ang(°/s)    | X:{b.tool_twist_angular_x:7.2f}, Y:{b.tool_twist_angular_y:7.2f}, Z:{b.tool_twist_angular_z:7.2f}")
        print(f"力矩/Wrench(N,Nm)  | Fx:{b.tool_external_wrench_force_x:6.1f}, Fy:{b.tool_external_wrench_force_y:6.1f}, Fz:{b.tool_external_wrench_force_z:6.1f}")
        print(f"                   | Tx:{b.tool_external_wrench_torque_x:6.2f}, Ty:{b.tool_external_wrench_torque_y:6.2f}, Tz:{b.tool_external_wrench_torque_z:6.2f}")

        # --- 3. 关节执行器详情 ---
        print(f"\n[ 关节详情 / JOINT DETAILS ]")
        header = f"{'ID':<4} {'Pos(°)':<8} {'Vel(°/s)':<9} {'Trq(Nm)':<8} {'Cur(A)':<7} {'Tmp_C':<7} {'Jitter':<8}"
        print(header)
        print("-" * len(header))
        for i, act in enumerate(self.full_status.actuators):
            print(f"J{i+1:<3} {act.position:<8.1f} {act.velocity:<9.1f} {act.torque:<8.2f} "
                  f"{act.current_motor:<7.2f} {act.temperature_core:<7.1f} {act.jitter_comm:<8}")

        # --- 4. 夹爪、IMU 与系统健康 ---
        print(f"\n[ 夹爪与系统健康 / SYSTEM HEALTH ]")
        gripper = inter.gripper_feedback.motor
        grip_pos = gripper[0].position if gripper else 0.0
        print(f"夹爪位置/Gripper  | {grip_pos:6.2f}% | 状态: {b.active_state}")
        print(f"系统供电/Power    | {b.arm_voltage:4.1f} V / {b.arm_current:4.2f} A")
        print(f"系统温度/Temp     | CPU: {b.temperature_cpu:4.1f} °C | Ambient: {b.temperature_ambient:4.1f} °C")
        print(f"基座 IMU Acc      | X: {b.imu_acceleration_x:6.2f}, Y: {b.imu_acceleration_y:6.2f}, Z: {b.imu_acceleration_z:6.2f}")
        
        print(f"\n{' MENU键退出程序 / PRESS MENU TO EXIT ':=^85}")
        sys.stdout.flush()

    def run(self):
        self.running = True
        obs_thread = threading.Thread(target=self._state_observer_thread, daemon=True)
        obs_thread.start()

        try:
            while True:
                pygame.event.pump()
                if self.joy.get_button(7): break # Menu按钮

                # 1. 摇杆读取与死区处理
                a0, a1 = self.apply_deadzone(self.joy.get_axis(0)), self.apply_deadzone(self.joy.get_axis(1))
                a2, a3 = (self.joy.get_axis(2) + 1) / 2.0, self.apply_deadzone(self.joy.get_axis(3))
                a4, a5 = self.apply_deadzone(self.joy.get_axis(4)), (self.joy.get_axis(5) + 1) / 2.0
                hat = self.joy.get_hat(0)

                # 2. 构建 Twist 指令
                command = Base_pb2.TwistCommand()
                command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
                command.twist.linear_x = -a1 * self.speed_limit
                command.twist.linear_y = -a0 * self.speed_limit
                command.twist.linear_z = (a5 - a2) * self.speed_limit
                command.twist.angular_x = a3 * self.turn_limit
                command.twist.angular_y = -a4 * self.turn_limit
                command.twist.angular_z = -hat[0] * self.turn_limit

                # 3. 发送控制信号
                if any([abs(v) > 0.01 for v in [a0, a1, a5-a2, a3, a4, hat[0]]]):
                    self.base.SendTwistCommand(command)
                else:
                    self.base.Stop()

                # 4. 夹爪控制逻辑
                if self.joy.get_button(0): # A键
                    self.control_gripper(1.0)
                    self.last_gripper_label = "CLOSING"
                elif self.joy.get_button(1): # B键
                    self.control_gripper(0.0)
                    self.last_gripper_label = "OPENING"
                else:
                    self.last_gripper_label = "IDLE"

                # 5. 刷新综合仪表盘
                self.display_full_dashboard()
                time.sleep(0.05) # 20Hz 控制频率

        finally:
            self.cleanup()

    def apply_deadzone(self, value):
        return value if abs(value) > self.deadzone else 0.0

    def control_gripper(self, pos):
        try:
            gripper_command = Base_pb2.GripperCommand()
            gripper_command.mode = Base_pb2.GRIPPER_POSITION
            finger = gripper_command.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = pos
            self.base.SendGripperCommand(gripper_command)
        except: pass

    def cleanup(self):
        self.running = False
        if self.base: self.base.Stop()
        if self.connection: self.connection.__exit__(None, None, None)
        pygame.quit()
        print("\n✅ 安全退出 / Safety Exit Complete.")

if __name__ == "__main__":
    teleop = KinovaJoyTeleop()
    teleop.connect()
    teleop.run()