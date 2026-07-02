import sys
import os
import time
import pygame
import threading

# 动态添加 utilities 路径
sys.path.insert(0, "/home/kinova-1/Kinova-gen3/gen3-controller/Kinova_kortex2_Gen3_G3L/api_python/examples")



# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(current_dir, '..', 'api_python', 'examples')
sys.path.insert(0, os.path.abspath(examples_dir))
import utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

class KinovaJoyTeleop:
    def __init__(self, ip="192.168.8.10"):
        self.ip = ip
        self.base = None
        self.router = None
        self.connection = None
        
        # 控制参数
        self.deadzone = 0.1      # 摇杆死区
        self.speed_limit = 0.20  # 平移速度最大值 (m/s)
        self.turn_limit = 30.0   # 旋转速度最大值 (deg/s)

        # 初始化 Pygame 手柄
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("❌ 未检测到 Xbox 手柄，请确保连接。")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        
        # 内部状态记录
        self.current_pose = [0.0] * 6 # x, y, z, theta_x, theta_y, theta_z

    def connect(self):
        """建立机器人连接"""
        class Args:
            def __init__(self, ip):
                self.ip, self.username, self.password, self.port = ip, "admin", "admin", 10000
        self.connection = utilities.DeviceConnection.createTcpConnection(Args(self.ip))
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        print(f"✅ 已连接控制器: {self.joy.get_name()}")
        print("🚀 遥控模式启动！使用 Menu 键退出程序。\n")

    def apply_deadzone(self, value):
        return value if abs(value) > self.deadzone else 0.0

    def get_robot_pose(self):
        """获取并更新机器人当前末端位姿"""
        try:
            pose = self.base.GetMeasuredCartesianPose()
            self.current_pose = [pose.x, pose.y, pose.z, pose.theta_x, pose.theta_y, pose.theta_z]
        except Exception:
            pass

    def display_status(self, axes, hat, gripper_status):
        """在终端实时刷新显示状态"""
        # 构造手柄状态字符串
        joy_str = f"Joy -> X:{axes[1]:.2f} Y:{axes[0]:.2f} Z:{(axes[5]-axes[2]):.2f} | R:{axes[3]:.2f} P:{axes[4]:.2f} Y:{hat[0]:.2f} | Grip:{gripper_status}"
        # 构造机器人位姿字符串
        pose_str = f"Pose -> X:{self.current_pose[0]:.3f} Y:{self.current_pose[1]:.3f} Z:{self.current_pose[2]:.3f} | Roll:{self.current_pose[3]:.1f} Pitch:{self.current_pose[4]:.1f} Yaw:{self.current_pose[5]:.1f}"
        
        # \r 使光标回到行首，实现原地刷新
        sys.stdout.write(f"\r{joy_str}  ||  {pose_str}    ")
        sys.stdout.flush()

    def run(self):
        try:
            gripper_label = "IDLE"
            while True:
                pygame.event.pump()
                
                # 检查 Menu 键退出
                if self.joy.get_button(7):
                    print("\n\n停止程序...")
                    break 

                # 1. 读取手柄数据
                a0 = self.apply_deadzone(self.joy.get_axis(0)) # 左摇杆左右 (Y轴)
                a1 = self.apply_deadzone(self.joy.get_axis(1)) # 左摇杆上下 (X轴)
                a2 = (self.joy.get_axis(2) + 1) / 2.0          # LT下压
                a3 = self.apply_deadzone(self.joy.get_axis(3)) # 右摇杆左右 (Roll)
                a4 = self.apply_deadzone(self.joy.get_axis(4)) # 右摇杆上下 (Pitch)
                a5 = (self.joy.get_axis(5) + 1) / 2.0          # RT下压
                hat = self.joy.get_hat(0)                      # 十字键 (Yaw)

                # 2. 构建 Twist 指令
                command = Base_pb2.TwistCommand()
                command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
                command.duration = 0

                # --- 平移逻辑 ---
                command.twist.linear_x = -a1 * self.speed_limit   # 上推为正(前)
                command.twist.linear_y = -a0 * self.speed_limit   # 推左向右(镜像)
                command.twist.linear_z = (a5 - a2) * self.speed_limit # RT升，LT降

                # --- 旋转逻辑 ---
                command.twist.angular_x = a3 * self.turn_limit    # Roll
                command.twist.angular_y = -a4 * self.turn_limit   # Pitch
                command.twist.angular_z = -hat[0] * self.turn_limit # Yaw (十字键左右)

                # 3. 发送指令
                has_input = any([a0, a1, abs(a5-a2)>0.05, a3, a4, hat[0]!=0])
                if has_input:
                    self.base.SendTwistCommand(command)
                else:
                    self.base.Stop()

                # 4. 夹爪控制
                if self.joy.get_button(0): # A键
                    self.control_gripper(1.0)
                    gripper_label = "CLOSED"
                elif self.joy.get_button(1): # B键
                    self.control_gripper(0.0)
                    gripper_label = "OPENED"

                # 5. 获取反馈并显示
                self.get_robot_pose()
                self.display_status([a0, a1, a2, a3, a4, a5], hat, gripper_label)

                time.sleep(0.05) # 20Hz

        except Exception as e:
            print(f"\n运行时发生错误: {e}")
        finally:
            self.base.Stop()
            self.connection.__exit__(None, None, None)
            pygame.quit()
            print("\n👋 程序已安全退出。")

    def control_gripper(self, pos):
        """pos: 0.0 为全开，1.0 为全关"""
        try:
            gripper_command = Base_pb2.GripperCommand()
            gripper_command.mode = Base_pb2.GRIPPER_POSITION
            finger = gripper_command.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = pos
            self.base.SendGripperCommand(gripper_command)
        except:
            pass

if __name__ == "__main__":
    teleop = KinovaJoyTeleop()
    teleop.connect()
    teleop.run()