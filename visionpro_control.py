import sys
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from kinova_manage import KinovaManager
except ImportError:
    print("⚠️ 请确保 kinova_manage.py 在当前目录下")
    sys.exit(1)

from avp_stream import VisionProStreamer

# ================= 配置区域 =================
AVP_IP = "192.168.1.223"
ROBOT_IP = "192.168.8.10"

SCALE_FACTOR = 0.5       
MAX_LINEAR_VEL = 0.3     # m/s
MAX_ANGULAR_VEL = 40.0   # deg/s
PINCH_THRESHOLD = 0.02
NO_OBSERVATION_THRESHOLD = 0.0005

# 左手控制参数
LEFT_ROLL_CENTER = 3.0    # 虎口向右为基准 (rad)
LEFT_ROLL_DEADZONE = 0.4  # 死区，防止漂移
LEFT_ROLL_SENSITIVITY = 30.0 # 旋转灵敏度
# ===========================================

class VisionProTeleop:
    def __init__(self):
        print(f"Connecting to Vision Pro at {AVP_IP}...")
        self.avp = VisionProStreamer(ip=AVP_IP)
        self.avp.start_webrtc()
        
        print(f"Connecting to Kinova at {ROBOT_IP}...")
        self.arm = KinovaManager(ip_address=ROBOT_IP)
        self.arm.connect()
        
        self.clutch_engaged = False
        self.start_hand_pos = None
        self.start_robot_pos = None
        self.start_hand_rot = None  
        self.start_robot_rot = None 
        self.last_gripper_closed = None
        self.is_active = False

    def get_hand_data(self, data):
        matrix = data['right_wrist'][0] 
        pos = matrix[:3, 3]
        rot = R.from_matrix(matrix[:3, :3])
        return pos, rot

    def transform_pos_avp_to_robot(self, avp_delta):
        dx, dy, dz = avp_delta
        return np.array([dy, -dx, dz])

    def transform_rot_avp_to_robot(self, delta_rot_obj):
        # 处理右手姿态映射 (Pitch 和 Yaw)
        euler = delta_rot_obj.as_euler('xyz', degrees=True)
        # 排除原本的 Z 轴旋转(顺逆时针)，因为现在由左手控制
        robot_euler = [euler[1], -euler[0], 0] 
        return R.from_euler('xyz', robot_euler, degrees=True)

    def run(self):
        print("\n🚀 混合控制遥操作已启动")
        print("操作：左手捏合激活，右手控制位置+俯仰偏航，左手 Roll 控制顺逆时针")
        
        try:
            while True:
                r = self.avp.get_latest()
                if not r: continue

                # 1. 激活判断
                left_pinch = r['left_pinch_distance']
                if left_pinch >= NO_OBSERVATION_THRESHOLD:
                    self.is_active = left_pinch < PINCH_THRESHOLD
                
                curr_hand_pos, curr_hand_rot = self.get_hand_data(r)

                if self.is_active:
                    status = self.arm.get_status()
                    if not status: continue
                    b = status.base
                    
                    if not self.clutch_engaged:
                        print("🟢 控制激活")
                        self.clutch_engaged = True
                        self.start_hand_pos = curr_hand_pos
                        self.start_hand_rot = curr_hand_rot
                        self.start_robot_pos = np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z])
                        self.start_robot_rot = R.from_euler('xyz', [b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z], degrees=True)

                    # --- 2. 位置控制 (右手) ---
                    delta_hand_pos = curr_hand_pos - self.start_hand_pos
                    target_robot_pos = self.start_robot_pos + self.transform_pos_avp_to_robot(delta_hand_pos) * SCALE_FACTOR
                    curr_robot_pos = np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z])
                    lin_vel = (target_robot_pos - curr_robot_pos) * 2.5 

                    # --- 3. 姿态控制 (右手 Pitch/Yaw) ---
                    delta_rot_hand = curr_hand_rot * self.start_hand_rot.inv()
                    delta_rot_robot = self.transform_rot_avp_to_robot(delta_rot_hand)
                    target_robot_rot = delta_rot_robot * self.start_robot_rot
                    curr_robot_rot = R.from_euler('xyz', [b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z], degrees=True)
                    error_rot = target_robot_rot * curr_robot_rot.inv()
                    ang_vel_vec = error_rot.as_rotvec(degrees=True) * 2.0 

                    # --- 4. 顺逆时针旋转 (左手 Roll) ---
                    left_roll = r['left_wrist_roll']
                    if left_roll < 0:
                        left_roll += 2 * np.pi
                    # 计算相对于“虎口向右(3.0)”的偏移
                    # 向上通常 roll 减小 (趋向0或负)，向下趋向 1.5
                    roll_diff = left_roll - LEFT_ROLL_CENTER
                    
                    # 逻辑映射：
                    # 向上 (roll < 1.5) -> 顺时针 (+WZ)
                    # 向右 (roll ~ 3.0) -> 停止
                    # 向下 (roll ~ 1.5) -> 逆时针 (-WZ)
                    
                    w_z_override = 0.0
                    if abs(roll_diff) > LEFT_ROLL_DEADZONE:
                        # 如果 roll 值变小（向上），我们希望输出正的 w_z
                        # 这里用 3.0 减去当前值，这样向上(比如1.0)就会得到正数
                        w_z_override = (LEFT_ROLL_CENTER - left_roll) * LEFT_ROLL_SENSITIVITY


                    # --- 5. 夹爪控制 ---
                    right_pinch = r['right_pinch_distance']
                    if self.last_gripper_closed:
                        current_gripper_closed = right_pinch <= PINCH_THRESHOLD + 0.01
                    else:
                        current_gripper_closed = right_pinch <= PINCH_THRESHOLD - 0.01

                    if (current_gripper_closed != self.last_gripper_closed) and (right_pinch >= NO_OBSERVATION_THRESHOLD):
                        print(f"cur = {current_gripper_closed}, last = {self.last_gripper_closed}, right_pinch = {right_pinch}")
                        target_val = 100.0 if current_gripper_closed else 60
                        self.arm.control_gripper(target_val, dual_grip=False)
                        self.last_gripper_closed = current_gripper_closed

                    # --- 6. 速度合成与发送 ---
                    # 限速
                    lin_speed = np.linalg.norm(lin_vel)
                    if lin_speed > MAX_LINEAR_VEL: lin_vel = (lin_vel / lin_speed) * MAX_LINEAR_VEL
                    
                    # 最终速度向量 [Vx, Vy, Vz, Wx, Wy, Wz]
                    # 我们用 ang_vel_vec 的 X 和 Y 处理俯仰偏航，用 w_z_override 处理左手控制的旋转
                    velocities = [
                        lin_vel[0], lin_vel[1], lin_vel[2], 
                        ang_vel_vec[0], ang_vel_vec[1], w_z_override
                    ]
                    
                    # 全局角速度限速
                    ang_speed = np.linalg.norm(velocities[3:])
                    if ang_speed > MAX_ANGULAR_VEL:
                        scale = MAX_ANGULAR_VEL / ang_speed
                        velocities[3:] = [v * scale for v in velocities[3:]]

                    self.arm.move_velocity(velocities, duration_ms=100)

                else:
                    if self.clutch_engaged:
                        print("🔴 控制断开")
                        self.clutch_engaged = False
                        self.arm.move_velocity([0]*6, duration_ms=0)

                time.sleep(0.02)

        except KeyboardInterrupt:
            pass
        finally:
            self.arm.disconnect()

if __name__ == "__main__":
    VisionProTeleop().run()