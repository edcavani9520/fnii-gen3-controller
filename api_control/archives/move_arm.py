import sys
import os
import time
import threading

# 动态添加 utilities 路径
sys.path.insert(0, "/home/cuhk/Documents/visionpro-kinova-rl/Kinova-kortex2_Gen3_G3L/api_python/examples")
import utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

class KinovaController:
    def __init__(self, ip_address="192.168.8.10"):
        self.ip = ip_address
        self.router = None
        self.base = None
        self.connection = None

        # 模拟命令行参数供 utilities 使用
        class Args:
            def __init__(self, ip):
                self.ip = ip
                self.username = "admin"
                self.password = "admin"
                self.port = 10000
        self.args = Args(self.ip)

    def connect(self):
        """建立长连接"""
        self.connection = utilities.DeviceConnection.createTcpConnection(self.args)
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        print(f"Connected to Kinova at {self.ip}")

    def disconnect(self):
        """关闭连接"""
        if self.connection:
            self.connection.__exit__(None, None, None)
            print("Disconnected.")

    def _check_for_end_or_abort(self, e):
        """动作监听闭包"""
        def check(notification, e=e):
            if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                e.set()
        return check

    def move_angular(self, angles_and_gripper):
        """
        输入: 8个数字 (7关节角度 + 1个夹爪位置)
        angles_and_gripper: [j1, j2, j3, j4, j5, j6, j7, gripper]
        角度单位: 度 (°), 夹爪单位: 0(开) - 100(关)
        """
        joint_angles = angles_and_gripper[:7]
        gripper_val = angles_and_gripper[7]

        print(f"Moving Angular to: {joint_angles}, Gripper: {gripper_val}%")
        
        # 1. 执行关节运动
        action = Base_pb2.Action()
        action.name = "Angular Movement"
        for i, angle in enumerate(joint_angles):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = angle

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        self.base.ExecuteAction(action)
        e.wait(20) # 等待结束
        self.base.Unsubscribe(notification_handle)

        # 2. 执行夹爪运动
        self.control_gripper(gripper_val)

    def move_cartesian(self, pose_and_gripper):
        """
        输入: 8个数字 (3个XYZ + 3个角度Theta + 1个夹爪) 
        注：你提到的4位角度通常指四元数，但Kortex基座API常用Theta X/Y/Z (Euler)，此处按Euler实现
        pose_and_gripper: [x, y, z, tx, ty, tz, gripper]
        """
        pos_x, pos_y, pos_z, theta_x, theta_y, theta_z, gripper_val = pose_and_gripper[0:7]
        # 注意：这里只取了前7个参数用于位姿，第8个参数是夹爪（由于你要求输入8位，这里多的一位如果是四元数第四位请调整）
        if len(pose_and_gripper) == 8:
            gripper_val = pose_and_gripper[7]

        print(f"Moving Cartesian to: Pos({pos_x}, {pos_y}, {pos_z}), Rot({theta_x}, {theta_y}, {theta_z}), Gripper: {gripper_val}%")

        action = Base_pb2.Action()
        action.name = "Cartesian Movement"
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = pos_x
        cartesian_pose.y = pos_y
        cartesian_pose.z = pos_z
        cartesian_pose.theta_x = theta_x
        cartesian_pose.theta_y = theta_y
        cartesian_pose.theta_z = theta_z

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        self.base.ExecuteAction(action)
        e.wait(20)
        self.base.Unsubscribe(notification_handle)

        # 执行夹爪运动
        self.control_gripper(gripper_val)

    def control_gripper(self, value):
        """控制夹爪 (0-100 映射到 0.0-1.0)"""
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = value / 100.0 # 转换百分比
        self.base.SendGripperCommand(gripper_command)
        time.sleep(0.5) # 给夹爪一点反应时间

def main():
    arm = KinovaController()
    arm.connect()

    try:
        # --- 数据组 A: 关节空间数据 (Angular) ---
        # 对应你给出的: ID 1-7的角度 + 夹爪位置 
        angular_data = [356.21, 0, 177.95, 251.98, 358.75, 304.45, 90, 0]
        
        print("\n--- Testing Angular Move ---")
        # arm.move_angular(angular_data)

        # --- 数据组 B: 笛卡尔空间数据 (Cartesian) ---
        # 对应你给出的: X, Y, Z, ThetaX, ThetaY, ThetaZ, 预留位0, 夹爪位置 
        # 输入格式修正为：[X, Y, Z, TX, TY, TZ, 0, Gripper]
        cartesian_data = [0.378, 0.039, 0.272, 179.23, -0.67, 93.71, 0.0, 0]
        
        print("\n--- Testing Cartesian Move ---")
        arm.move_cartesian(cartesian_data)

    finally:
        arm.disconnect()

if __name__ == "__main__":
    main()