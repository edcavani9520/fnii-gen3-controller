import sys
import time
import threading

# 动态添加 utilities 路径（保持你的原路径）
sys.path.insert(0, "/home/kinova-1/Kinova-gen3/gen3-controller/Kinova_kortex2_Gen3_G3L/api_python/examples")
import utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2


class KinovaJoyTeleop:
    def __init__(self, ip="192.168.8.10"):
        self.ip = ip
        self.base = None
        self.router = None
        self.connection = None
        
        self.speed_limit = 0.20
        self.turn_limit = 30.0
        self.current_pose = [0.0] * 6

    def connect(self):
        """建立机器人连接"""
        class Args:
            def __init__(self, ip):
                self.ip, self.username, self.password, self.port = ip, "admin", "admin", 10000
        self.connection = utilities.DeviceConnection.createTcpConnection(Args(self.ip))
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        # 注意：执行 reach_pose（高层动作）时，单层伺服模式可能不是必需的，
        # 但原脚本这样设置通常不影响 ExecuteAction，保留即可。
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        print(f"✅ Connected to Kinova at {self.ip}")

    def disconnect(self):
        """安全断开连接"""
        if self.base:
            try:
                self.base.Stop()
            except:
                pass
        if self.connection:
            try:
                self.connection.__exit__(None, None, None)
            except:
                pass
        self.connection = None
        self.router = None
        self.base = None
        print(" Disconnected.")

    def get_robot_pose(self):
        """获取并更新当前末端位姿"""
        try:
            pose = self.base.GetMeasuredCartesianPose()
            self.current_pose = [pose.x, pose.y, pose.z,
                                 pose.theta_x, pose.theta_y, pose.theta_z]
            return self.current_pose
        except Exception:
            return None

    # ------------- 新增：绝对笛卡尔运动 -------------
    def _wait_for_action(self):
        """等待高层动作完成（用于 reach_pose）"""
        e = threading.Event()

        def check(notification):
            if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                e.set()

        handle = self.base.OnNotificationActionTopic(check, Base_pb2.NotificationOptions())
        return e, handle

    def move_cartesian(self, target_pose):
        """
        执行绝对笛卡尔位姿运动（阻塞直到完成）
        target_pose: [x, y, z, theta_x, theta_y, theta_z, gripper]
                     gripper 范围 0.0（全开）~ 1.0（全关）
        """
        if len(target_pose) != 7:
            raise ValueError("target_pose 必须包含7个元素: x,y,z,rx,ry,rz,gripper")

        # 构造 reach_pose 动作
        action = Base_pb2.Action()
        tp = action.reach_pose.target_pose
        tp.x = float(target_pose[0])
        tp.y = float(target_pose[1])
        tp.z = float(target_pose[2])
        tp.theta_x = float(target_pose[3])
        tp.theta_y = float(target_pose[4])
        tp.theta_z = float(target_pose[5])

        # 启动动作并等待完成
        event, handle = self._wait_for_action()
        self.base.ExecuteAction(action)
        event.wait()                     # 阻塞，直到动作结束或异常中止
        self.base.Unsubscribe(handle)

        # 执行夹爪命令
        gripper_val = target_pose[6]
        self.control_gripper(gripper_val)
        time.sleep(0.5)  # 等待夹爪动作完成（可根据实际情况调整）

    # ------------- 以下为原有方法 -------------
    def control_gripper(self, pos):
        """pos: 0.0 全开，1.0 全关"""
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = float(max(0.0, min(1.0, pos)))
        self.base.SendGripperCommand(gripper_command)

    def get_gripper_position(self):
        """读取夹爪当前位置 (0.0~1.0)"""
        req = Base_pb2.GripperRequest()
        req.mode = Base_pb2.GRIPPER_POSITION
        meas = self.base.GetMeasuredGripperMovement(req)
        if len(meas.finger) == 0:
            raise RuntimeError("GetMeasuredGripperMovement 返回空 finger")
        return float(meas.finger[0].value)




# ==================== 用户需要填写的目标位姿序列 ====================
# 每个元素: [x(m), y(m), z(m), roll(°), pitch(°), yaw(°), gripper(0-1)]
TARGET_POSES = [
    [0.267, -0.065, 0.337, 5.092, -176.815, 38.278, 0.0],
    [0.409, -0.171, 0.217, 2.818,-176.587, 4.072, 0.0],
    [0.409, -0.171, 0.033, 2.803,-176.611, 4.052, 0.0],
    [0.409, -0.171, 0.033, 2.803,-176.611, 4.052, 0.5],
    [0.409, -0.171, 0.166, 2.803,-176.611, 4.052, 0.5],   
    [0.363, 0.278, 0.166, 3.009,-176.611, 4.052, 0.5],    
    [0.363, 0.278, 0.166, 3.009,-176.611, 4.052, 0.0] 
]

WAIT_BETWEEN = 1.0   # 两次运动间额外等待秒数（轨迹完成后）


def main():
    # 1. 创建对象并连接
    teleop = KinovaJoyTeleop()
    teleop.connect()

    try:
        # 2. 依次执行笛卡尔运动
        for i, target in enumerate(TARGET_POSES):
            print(f"\n--- 第 {i+1}/{len(TARGET_POSES)} 个目标 ---")
            t_start = time.perf_counter()
            teleop.move_cartesian(target)
            elapsed = (time.perf_counter() - t_start) * 1000
            print(f"  运动完成，耗时: {elapsed:.1f} ms")
            if WAIT_BETWEEN > 0 and i < len(TARGET_POSES) - 1:
                time.sleep(WAIT_BETWEEN)

        print("\n✅ 所有笛卡尔位姿序列执行完毕！")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断，正在安全退出...")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    main()