import sys
import os
import time

# 动态添加 utilities 路径
sys.path.insert(0, "/home/cuhk/Documents/visionpro-kinova-rl/Kinova-kortex2_Gen3_G3L/api_python/examples")
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
        self.speed_limit = 0.20  # 平移速度最大值 (m/s)
        self.turn_limit = 30.0   # 旋转速度最大值 (deg/s)
        self.current_pose = [0.0] * 6 # x, y, z, theta_x, theta_y, theta_z

    def connect(self):
        """建立机器人连接"""
        class Args:
            def __init__(self, ip):
                self.ip, self.username, self.password, self.port = ip, "admin", "admin", 10000
        self.connection = utilities.DeviceConnection.createTcpConnection(Args(self.ip))
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        # Twist/JointSpeed 等实时控制前需处于 Single-level servoing
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        print(f"✅ Connected to Kinova at {self.ip}")


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
            command = Base_pb2.TwistCommand()
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            command.duration = 0

            # 使用安全量级，先验证通路再逐步放大
            command.twist.linear_x = 0.03
            command.twist.linear_y = 0.00
            command.twist.linear_z = 0.00
            command.twist.angular_x = 0.0
            command.twist.angular_y = 0.0
            command.twist.angular_z = 5.0

            print("🚀 Sending twist command for 2 seconds...")
            self.base.SendTwistCommand(command)
            time.sleep(2.0)
            self.base.Stop()

            before = self.get_gripper_position()
            print(f"🤖 Gripper position before command: {before:.3f}")

            # 先开再关，便于肉眼确认夹爪确实在响应
            self.control_gripper(0.0)
            time.sleep(1.0)
            mid = self.get_gripper_position()
            print(f"🤖 Gripper position after OPEN : {mid:.3f}")

            self.control_gripper(1.0)
            time.sleep(1.0)
            after = self.get_gripper_position()
            print(f"🤖 Gripper position after CLOSE: {after:.3f}")
        except Exception as e:
            print(f"❌ 控制失败: {e}")
            print("排查建议: 1) 关闭其它正在连接机械臂的程序 2) Kinova Web App 退出 Jog/控制页面 3) 重新运行脚本")
            raise
        finally:
            try:
                if self.base:
                    self.base.Stop()
            finally:
                if self.connection:
                    self.connection.__exit__(None, None, None)
            print("\n👋 程序已安全退出。")

    def control_gripper(self, pos):
        """pos: 0.0 为全开，1.0 为全关"""
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = float(max(0.0, min(1.0, pos)))
        self.base.SendGripperCommand(gripper_command)

    def get_gripper_position(self):
        """读取夹爪当前位置(0.0~1.0)，读取失败则抛出异常。"""
        req = Base_pb2.GripperRequest()
        req.mode = Base_pb2.GRIPPER_POSITION
        meas = self.base.GetMeasuredGripperMovement(req)
        if len(meas.finger) == 0:
            raise RuntimeError("GetMeasuredGripperMovement 返回空 finger")
        return float(meas.finger[0].value)

if __name__ == "__main__":
    teleop = KinovaJoyTeleop()
    teleop.connect()
    teleop.run()