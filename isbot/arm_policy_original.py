import queue
import time
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
from robot_controller.gen3.arm_controller import JointCompliantController
from robot_controller.gen3.kinova import TorqueControlledArm
import os,sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(parent_dir)
sys.path.append(parent_dir)
from configs.constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from robot_controller.ik_solver import IKSolver
import torch
import math

GRIPPER_CLOSE = 1
GRIPPER_OPEN = 0

trajectory = [
    [0.0,     0.0,     0.0,     0.0,    0.0,    0.0,    0.0],
    [-0.0690, -0.0225, -0.0667, 0.0983, 0.1287, 0.0487, 0.1147],
    [-0.1523, -0.0069, -0.0896, 0.2037, 0.2540, 0.1469, 0.2301],
    [-0.1577, 0.0665, -0.1632, 0.3111, 0.2301, 0.2527, 0.3456],
    [-0.1823, 0.1791, -0.2634, 0.4249, 0.3415, 0.3667, 0.4601],
    [-0.2306, 0.2968, -0.3438, 0.5415, 0.4058, 0.4823, 0.5745],
    [-0.2992, 0.4151, -0.4330, 0.6579, 0.5177, 0.5978, 0.6877],
    [-0.4170, 0.5303, -0.3474, 0.7708, 0.4063, 0.7110, 0.8024],
    [-0.4959, 0.6467, -0.2454, 0.8856, 0.2934, 0.8254, 0.9166],
    [-0.4153, 0.7635, -0.3365, 1.0002, 0.1812, 0.9402, 0.8016],
    [-0.3042, 0.8781, -0.4572, 1.1132, 0.2852, 1.0499, 0.9152],
    [-0.1925, 0.9774, -0.5223, 1.2273, 0.3872, 1.1648, 0.8000],
    [-0.1287, 1.0073, -0.6031, 1.3167, 0.4773, 1.0792, 0.6893],
    [-0.2057, 0.9557, -0.5447, 1.3808, 0.5936, 1.0676, 0.8043],
    [-0.1431, 0.9179, -0.6425, 1.4433, 0.4961, 0.9564, 0.7007],
    [-0.2305, 0.9230, -0.5378, 1.3593, 0.5733, 1.0712, 0.8141],
    [-0.1481, 0.9078, -0.6372, 1.4295, 0.5028, 0.9524, 0.7057]
]


JOINT_LIMITS_DEG = [
    (-180, 180),   # Joint 1（假设无限制）
    (-130, 120),   # Joint 2
    (-180, 180),   # Joint 3（假设无限制）
    (-150, 150),   # Joint 4
    (-180, 180),   # Joint 5（假设无限制）
    (-120, 120),   # Joint 6
    (-180, 180),   # Joint 7（假设无限制）
]


class Arm:
    def __init__(self):
        self.arm = TorqueControlledArm()
        self.arm.set_joint_limits(speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,)))
        self.command_queue = queue.Queue(1)
        self.controller = None
        self.ik_solver = IKSolver(ee_offset=0.12)

    def reset(self):
        # Stop low-level control
        if self.arm.cyclic_running:
            time.sleep(0.75)  # Wait for arm to stop moving
            self.arm.stop_cyclic()

        # Clear faults
        self.arm.clear_faults()

        # Reset arm configuration
        self.arm.open_gripper()
        # self.arm.retract()

        # Create new instance of controller
        self.controller = JointCompliantController(self.command_queue)

        # Start low-level control
        self.arm.init_cyclic(self.controller.control_callback)
        while not self.arm.cyclic_running:
            time.sleep(0.01)

    def execute_action_g(self, action):
        qpos = self.ik_solver.solve(action['arm_pos'], action['arm_quat'], self.arm.q)
        self.command_queue.put((qpos, action['gripper_pos'].item()))

    def execute_action_q(self, action):
        qpos = np.deg2rad(action['arm_pos'])
        self.command_queue.put((qpos, action['gripper_pos'].item()))

    def get_state(self):
        arm_gpos, arm_quat = self.arm.get_tool_pose()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        '''
        state = {
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'gripper_pos': np.array([self.arm.gripper_pos]),
        }
        '''
        arm_qpos = np.rad2deg(self.arm.q.copy())
        arm_qvel = np.rad2deg(self.arm.dq.copy())
        state = {
            'arm_qpos': arm_qpos,
            'arm_qvel': arm_qvel,
            'arm_gpos': arm_gpos,
            'arm_quat': arm_quat,
            'gripper_pos': np.array([self.arm.gripper_pos]),
        }
        return state

    def close(self):
        if self.arm.cyclic_running:
            time.sleep(0.75)  # Wait for arm to stop moving
            self.arm.stop_cyclic()
        self.arm.disconnect()

    def go_zero(self):
        home_pt = np.array([360.0, 360.0, 360.0, 360.0, 360.0, 360.0, 360.0])
        for i in range(200):   # 5s
            self.execute_action_q({
                'arm_pos': home_pt,
                'gripper_pos': np.array([GRIPPER_OPEN]),
            })
            time.sleep(0.05) # 20hz

    def change_gripper(self, gripper_valu = GRIPPER_OPEN):
        action = self.get_state()['arm_qpos'].copy()
        self.execute_action_q({
                'arm_pos' : action,
                'gripper_pos':np.array([gripper_valu]),
            })
            
def limit_joint_angles_deg(joint_angles_deg):
    # [0, 360] -> [-180, 180]
    def clamp_angle(angle_deg, min_deg, max_deg):
        """将角度先归一化再限幅"""
        angle_norm = (angle_deg + 180) % 360 - 180
        return np.clip(angle_norm, min_deg, max_deg)
    
    """输入: 角度数组 (单位: 度)，输出: 归一化并限幅后的角度数组"""
    limited = []
    for i, angle in enumerate(joint_angles_deg):
        min_limit, max_limit = JOINT_LIMITS_DEG[i]
        limited.append(clamp_angle(angle, min_limit, max_limit))
    return np.array(limited)

def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w])

def encode_joint_angles(theta_rad):
    """
    将7个关节角度(单位:度)转成7个sin和7个cos，拼接成14维向量
    """
    # theta_rad = np.deg2rad(arm_qpos_deg)  # 转成弧度
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    encoded = np.concatenate([sin_theta, cos_theta])
    return encoded

def decode_joint_angles(encoded):
    """
    将14维 sin+cos 编码还原成7个关节角度（单位：度）
    """
    sin_theta = encoded[:7]
    cos_theta = encoded[7:14]
    theta_rad = np.arctan2(sin_theta, cos_theta)
    theta_deg = np.rad2deg(theta_rad)
    return theta_deg

def run_policy(arm: Arm, policy, step:int, init_obs, gripper_status=GRIPPER_OPEN, if_print=False, err_threshold=0.0):
    obs = init_obs
    last_joint = init_obs[:14]
    diff_sum = 0

    for i in range(step): 
        # if gripper_status == GRIPPER_CLOSE:
        #     input('next')
        state = arm.get_state()
        # input('next')

        # 检测和上一次的位置偏差，很小就认为到达目标点，直接退出
        encode_joint = encode_joint_angles(np.deg2rad(state["arm_qpos"].copy()))
        diff_sum = np.sum(np.abs(encode_joint - last_joint))
        last_joint = encode_joint

        if i > 50 and diff_sum < err_threshold:
            break

        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float().to(device='cuda:0')
            action_output = policy(obs).to(device='cpu').detach().view(-1).numpy()
        obs = np.zeros(21)
        obs[:14] = encode_joint
        obs[14:21] = action_output[:].copy()

        action_delta = action_output*180/math.pi
        action_delta = np.mod(action_delta, 360) 
        action = state['arm_qpos'].copy() + action_delta
        action = np.mod(action, 360) 
        action = limit_joint_angles_deg(action)

        if if_print:
            print(f"step {i} and diff_sum is {diff_sum:.5f} ----------------------------------------")
            # print("state_sin: ", [f"{v:.2f}" for v in obs[:7]])
            # print("state_cos: ", [f"{v:.2f}" for v in obs[7:14]])
            print("state: ", [f"{v:.2f}" for v in state['arm_qpos'].copy()])
            print("action_delta: ", [f"{v:.2f}" for v in action_delta])
            print("action_abs: ", [f"{v:.2f}" for v in action])

        for _ in range(1):
            arm.execute_action_q({
                'arm_pos' : action,
                'gripper_pos':np.array([gripper_status]),
            })
            time.sleep(0.045)
    print(f"Policy executed {i} steps and last diff_sum is {diff_sum:.5f}")

    return obs

if __name__ == '__main__':
    import numpy as np
    from configs.constants import POLICY_CONTROL_PERIOD

    arm = Arm()
    arm.reset()
    policy_1 = torch.jit.load("/home/cuhk/kinova/ZMAI/IS_Bot/policy/pull_catkin/policy_pt1_as.pt", map_location="cuda:0")
    # policy_2 = torch.jit.load("/home/cuhk/kinova/ZMAI/IS_Bot/policy/pull_catkin/policy_pt2_s.pt", map_location="cuda:0")
    policy_2 = torch.jit.load("/home/cuhk/kinova/ZMAI/IS_Bot/policy/policy_pt2.pt", map_location="cuda:0")
    policy_3 = torch.jit.load("/home/cuhk/kinova/ZMAI/IS_Bot/policy/pull_catkin/policy_pt3_s.pt", map_location="cuda:0")
    policy_4 = torch.jit.load("/home/cuhk/kinova/ZMAI/IS_Bot/policy/policy_pt4_as.pt", map_location="cuda:0")
    policy_5 = torch.jit.load("/home/cuhk/kinova/ZMAI/IS_Bot/policy/policy_pt5.pt", map_location="cuda:0")
    policy_5_final = torch.jit.load("/home/cuhk/kinova/ZMAI/IS_Bot/policy/policy_pt5_final.pt", map_location="cuda:0")
    policy_6 = torch.jit.load("/home/cuhk/kinova/ZMAI/IS_Bot/policy/policy_pt6_s.pt", map_location="cuda:0")
    policy_7 = torch.jit.load("/home/cuhk/kinova/ZMAI/IS_Bot/policy/policy_pt7_as_longer.pt", map_location="cuda:0")
    
    arm.go_zero()
    print(f"Start pos is {[f'{v:.2f}' for v in arm.get_state()['arm_qpos'].copy()]}")
    input("Input enter to start!")

    init_obs = np.zeros(21)
    init_obs[7:14] = 1.0
    step = 200

    obs = run_policy(arm=arm, policy=policy_1, step=step, init_obs=init_obs, err_threshold=0.001)
    input('ready for pt2')
    # obs = run_policy(arm=arm, policy=policy_2, step=step, init_obs=obs, err_threshold=0.0003)
    obs = run_policy(arm=arm, policy=policy_2, step=150, init_obs=obs, err_threshold=0.0)
    print(f"Start pos is {[f'{v:.2f}' for v in arm.get_state()['arm_gpos'].copy()]}")

    input('ready for pt3 gripper close')
    arm.change_gripper(GRIPPER_CLOSE)

    input('ready for pt3')
    obs = run_policy(arm=arm, policy=policy_3, step=33, init_obs=obs, gripper_status=GRIPPER_CLOSE, err_threshold=0.001)

    input("ready to go home")
    state = arm.get_state()
    target = state['arm_gpos'].copy() + np.array([-0.1, 0, 0.2])
    for _ in range(100):
        arm.execute_action_g({
            'arm_pos': target,
            'arm_quat': state['arm_quat'].copy(),
            'gripper_pos': np.array([GRIPPER_OPEN]),
        })
        time.sleep(5 * POLICY_CONTROL_PERIOD)
    arm.go_zero()

    input('ready for pt4')
    obs = run_policy(arm=arm, policy=policy_4, step=95, init_obs=init_obs, err_threshold=0.005)
    
    input('ready for gripper close')
    arm.change_gripper(GRIPPER_CLOSE)
    
    input('ready for pt5')
    obs = run_policy(arm=arm, policy=policy_5, step=step, init_obs=obs, gripper_status=GRIPPER_CLOSE, err_threshold=0.001)

    input('ready for pt5 final')
    obs = run_policy(arm=arm, policy=policy_5_final, step=100, init_obs=obs, gripper_status=GRIPPER_CLOSE, err_threshold=0.0003)

    input('ready for pt5')
    obs = run_policy(arm=arm, policy=policy_5, step=step, init_obs=obs, err_threshold=0.002)

    input('ready for pt6')
    obs = run_policy(arm=arm, policy=policy_6, step=100, init_obs=obs, err_threshold=0.002)

    input('ready for pt7')
    obs = run_policy(arm=arm, policy=policy_7, step=50, init_obs=obs, err_threshold=0.001)

    print(f"Now pos is {[f'{v:.2f}' for v in arm.get_state()['arm_qpos'].copy()]}")
    input("Input enter to reset!")
    arm.go_zero()

