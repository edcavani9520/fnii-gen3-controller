import sys
import os
import queue
import time
import threading
import termios
import tty
from typing import Any

import numpy as np

from Kinova_kortex2_Gen3_G3L.api_python.examples import utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

# 与 test_kinova_manager 中 home 一致：7 关节 (°) + 夹爪 0–100
HOME_JOINTS = [360.00, 10.00, 180.00, 240.00, 180.00, 50.00, 270.00, 0.0]

_TORQUE_HOLD_HZ = 50.0
_TORQUE_SETTLE_S = 0.75
_ISBOT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isbot")


def _ensure_isbot_import_path() -> None:
    if _ISBOT_ROOT not in sys.path:
        sys.path.insert(0, _ISBOT_ROOT)


def _verify_torque_runtime_deps() -> None:
    try:
        import pinocchio as pin  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "缺少 pinocchio。请在本 conda 环境中执行:\n"
            "  pip uninstall pinocchio -y\n"
            "  conda install -c conda-forge pinocchio"
        ) from exc

    if not hasattr(pin, "buildModelFromUrdf"):
        ver = getattr(pin, "__version__", "?")
        raise ImportError(
            f"当前 pinocchio (version={ver}) 不是机器人学库，缺少 buildModelFromUrdf。\n"
            "请执行:\n"
            "  pip uninstall pinocchio -y\n"
            "  conda install -c conda-forge pinocchio"
        ) from None

    try:
        import ruckig  # noqa: F401
    except ImportError as exc:
        raise ImportError("缺少 ruckig。请执行: pip install ruckig") from exc


class KinovaManager:
    def __init__(self, ip_address="192.168.8.10"):
        self.ip = ip_address
        self.connection = None
        self.router = None
        self.base = None
        self.base_cyclic = None

        class Args:
            def __init__(self, ip):
                self.ip = ip
                self.username = "admin"
                self.password = "admin"
                self.port = 10000
        self.args = Args(self.ip)

        # 力矩循环会话（需先 init_torque()）
        self._torque_arm = None
        self._torque_command_queue: queue.Queue | None = None
        self._torque_hold_stop = threading.Event()
        self._torque_hold_thread: threading.Thread | None = None
        self._torque_hold_lock = threading.Lock()
        self._torque_hold_joints_deg: list[float] | None = None
        self._torque_hold_gripper = 0.0

    def connect(self):
        self.connection = utilities.DeviceConnection.createTcpConnection(self.args)
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        print(f"✅ Connected to Kinova at {self.ip}")

    def disconnect(self):
        if self._torque_arm is not None:
            self.shutdown_torque()
        if self.connection:
            try:
                self.connection.__exit__(None, None, None)
            except KeyboardInterrupt:
                pass
            self.connection = None
            self.router = None
            self.base = None
            self.base_cyclic = None
            print("❌ Disconnected.")

    def _require_torque_session(self):
        if self._torque_arm is None or self._torque_command_queue is None:
            raise RuntimeError("请先调用 init_torque()")
        return self._torque_arm

    def _read_torque_state(self, arm) -> dict[str, Any]:
        return {
            "q_deg": np.rad2deg(arm.q.copy()),
            "dq_deg": np.rad2deg(arm.dq.copy()),
            "tau": arm.tau.copy(),
            "gripper": float(arm.gripper_pos),
        }

    def _enqueue_torque_target(self, joints_deg: list[float], gripper: float) -> None:
        self._require_torque_session()
        qpos = np.deg2rad(np.asarray(joints_deg, dtype=np.float64))
        while not self._torque_command_queue.empty():
            try:
                self._torque_command_queue.get_nowait()
            except queue.Empty:
                break
        self._torque_command_queue.put((qpos, float(gripper)))

    def _torque_hold_loop(self, period: float) -> None:
        while not self._torque_hold_stop.is_set() and self._torque_arm is not None and self._torque_arm.cyclic_running:
            with self._torque_hold_lock:
                joints = self._torque_hold_joints_deg
                grip = self._torque_hold_gripper
            if joints is not None:
                try:
                    self._enqueue_torque_target(joints, grip)
                except RuntimeError:
                    break
            time.sleep(period)

    def _start_torque_hold(self, hz: float = _TORQUE_HOLD_HZ) -> None:
        self._torque_hold_stop.clear()
        if self._torque_hold_thread is not None and self._torque_hold_thread.is_alive():
            return
        period = 1.0 / hz
        self._torque_hold_thread = threading.Thread(
            target=self._torque_hold_loop, args=(period,), daemon=True
        )
        self._torque_hold_thread.start()

    def _stop_torque_hold(self) -> None:
        self._torque_hold_stop.set()
        if self._torque_hold_thread is not None:
            self._torque_hold_thread.join(timeout=1.0)
        self._torque_hold_stop.clear()

    def _reach_joint_angles_blocking(self, base, angles_deg: list[float]) -> None:
        """高层 ExecuteAction 阻塞到达目标关节角（进入 cyclic 前使用）。"""
        action = Base_pb2.Action()
        for i, angle in enumerate(angles_deg):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = float(angle)
        e = threading.Event()

        def check(notification):
            if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                e.set()

        handle = base.OnNotificationActionTopic(check, Base_pb2.NotificationOptions())
        base.ExecuteAction(action)
        e.wait()
        base.Unsubscribe(handle)

    def init_torque(self, *, home: bool = True, hold_hz: float = _TORQUE_HOLD_HZ) -> dict[str, Any]:
        """清故障 → 可选回 HOME_JOINTS → 启动力矩循环 + 50Hz 目标保持。返回当前关节状态。

        注意：使用项目 ``HOME_JOINTS`` 做高层到位，**不**调用固件参考动作 ``arm.home()``，
        避免 Kinova 内置 Home 与 ``HOME_JOINTS`` 不一致（尤其 J5/J7 可差 180°）导致进入 cyclic 后突然大幅运动。
        进入 cyclic 后首帧目标取**当前实测关节角**，不再重复下发 ``HOME_JOINTS`` 角度值。
        """
        if self._torque_arm is not None:
            self.shutdown_torque()
        if self.connection is not None:
            self.disconnect()

        _ensure_isbot_import_path()
        _verify_torque_runtime_deps()
        from robot_controller.gen3.arm_controller import JointCompliantController
        from robot_controller.gen3.kinova import TorqueControlledArm

        arm = TorqueControlledArm()
        arm.set_joint_limits(speed_limits=(30.0,) * 7, acceleration_limits=(80.0,) * 7)
        arm.clear_faults()
        if home:
            self._reach_joint_angles_blocking(arm.base, [float(x) for x in HOME_JOINTS[:7]])

        self._torque_command_queue = queue.Queue(1)
        controller = JointCompliantController(self._torque_command_queue)
        arm.init_cyclic(controller.control_callback)

        t0 = time.time()
        while not arm.cyclic_running:
            if time.time() - t0 > 5.0:
                arm.disconnect()
                raise TimeoutError("cyclic 未在 5s 内启动")
            time.sleep(0.005)

        self._torque_arm = arm
        st = self._read_torque_state(arm)
        joints = [float(x) for x in st["q_deg"]]
        self._torque_hold_gripper = float(st["gripper"])
        with self._torque_hold_lock:
            self._torque_hold_joints_deg = joints
        self._enqueue_torque_target(joints, self._torque_hold_gripper)
        self._start_torque_hold(hold_hz)
        return st

    def move_torque(
        self,
        joints_deg: np.ndarray | list[float],
        gripper: float = 0.0,
    ) -> dict[str, Any]:
        """非阻塞：仅更新目标并写入 cyclic 队列，立即返回当前反馈（不等轨迹结束）。

        与 ``move_angular``（ExecuteAction + wait）不同，单次调用应在亚毫秒级返回；
        50Hz 后台线程会持续刷新目标以防指令超时。
        """
        joints = [float(x) for x in joints_deg]
        with self._torque_hold_lock:
            self._torque_hold_joints_deg = joints
            self._torque_hold_gripper = float(gripper)
        self._enqueue_torque_target(joints, gripper)
        return self._read_torque_state(self._require_torque_session())

    def shutdown_torque(self) -> None:
        """停止力矩 cyclic 并断开（与 init_torque 配对使用）。"""
        self._stop_torque_hold()
        if self._torque_arm is not None and self._torque_arm.cyclic_running:
            time.sleep(_TORQUE_SETTLE_S)
            self._torque_arm.stop_cyclic()
        if self._torque_arm is not None:
            self._torque_arm.disconnect()
        self._torque_arm = None
        self._torque_command_queue = None
        with self._torque_hold_lock:
            self._torque_hold_joints_deg = None

    def _wait_for_action(self):
        e = threading.Event()
        def check(notification):
            if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                e.set()
        handle = self.base.OnNotificationActionTopic(check, Base_pb2.NotificationOptions())
        return e, handle

    def get_status(self):
        try:
            return self.base_cyclic.RefreshFeedback()
        except Exception as e:
            print(f"Error getting status: {e}")
            return None

    def print_status(self, status):
        """
        全面状态报告：包含末端位姿、捻度、受力、关节详情、系统健康度及夹爪。
        Full Status Report: Pose, Twist, Wrench, Actuators, System Health, and Gripper.
        """
        if not status:
            print("无状态信息 / No status information available.")
            return

        b = status.base
        print(f"\n{' 系统状态报告 SYSTEM STATUS REPORT ':=^85}")

        # 1. 末端位姿与指令参考 (Base Pose & Command Reference)
        print(f"\n[ 末端位姿与指令 / POSE & COMMAND ]")
        print(f"当前位姿/Actual  | XYZ(m): {b.tool_pose_x:7.4f}, {b.tool_pose_y:7.4f}, {b.tool_pose_z:7.4f}")
        print(f"                  | RPY(°): {b.tool_pose_theta_x:7.2f}, {b.tool_pose_theta_y:7.2f}, {b.tool_pose_theta_z:7.2f}")
        print(f"指令参考/Command | XYZ(m): {b.commanded_tool_pose_x:7.4f}, {b.commanded_tool_pose_y:7.4f}, {b.commanded_tool_pose_z:7.4f}")
        print(f"                  | RPY(°): {b.commanded_tool_pose_theta_x:7.2f}, {b.commanded_tool_pose_theta_y:7.2f}, {b.commanded_tool_pose_theta_z:7.2f}")

        # 2. 末端运动捻度 (Tool Twist - Linear & Angular Velocity)
        print(f"\n[ 末端捻度速度 / TOOL TWIST ]")
        print(f"线速度/Linear (m/s) | X: {b.tool_twist_linear_x:9.5f}, Y: {b.tool_twist_linear_y:9.5f}, Z: {b.tool_twist_linear_z:9.5f}")
        print(f"角速度/Angular(°/s) | X: {b.tool_twist_angular_x:9.5f}, Y: {b.tool_twist_angular_y:9.5f}, Z: {b.tool_twist_angular_z:9.5f}")

        # 3. 外部受力情况 (External Wrench / Force-Torque)
        print(f"\n[ 末端受力 / EXTERNAL WRENCH ]")
        print(f"力/Force (N)        | X: {b.tool_external_wrench_force_x:7.2f}, Y: {b.tool_external_wrench_force_y:7.2f}, Z: {b.tool_external_wrench_force_z:7.2f}")
        print(f"力矩/Torque (Nm)     | X: {b.tool_external_wrench_torque_x:7.2f}, Y: {b.tool_external_wrench_torque_y:7.2f}, Z: {b.tool_external_wrench_torque_z:7.2f}")

        # 4. 惯性测量单元 (Base IMU Data)
        print(f"\n[ 基座惯性单元 / BASE IMU DATA ]")
        print(f"加速度/Accel (m/s²) | X: {b.imu_acceleration_x:7.2f}, Y: {b.imu_acceleration_y:7.2f}, Z: {b.imu_acceleration_z:7.2f}")
        print(f"角速度/Gyro (°/s)   | X: {b.imu_angular_velocity_x:7.2f}, Y: {b.imu_angular_velocity_y:7.2f}, Z: {b.imu_angular_velocity_z:7.2f}")

        # 5. 关节执行器详情 (Actuators Detail)
        print(f"\n[ 关节详情 / JOINT DETAILS ]")
        header = f"{'ID':<4} {'Pos(°)':<9} {'Vel(°/s)':<10} {'Trq(Nm)':<9} {'Cur(A)':<8} {'Volt(V)':<8} {'Tmp_C':<8} {'Jitter':<10}"
        print(header)
        print("-" * len(header))
        for i, act in enumerate(status.actuators):
            # 这里的字段完全对应报文中的：position, velocity, torque, current_motor, voltage, temperature_core, jitter_comm
            print(f"J{i+1:<3} {act.position:<9.2f} {act.velocity:<10.3f} {act.torque:<9.2f} "
                  f"{act.current_motor:<8.2f} {act.voltage:<8.1f} {act.temperature_core:<8.1f} {act.jitter_comm:<10}")

        # 6. 夹爪与互联模块 (Gripper & Interconnect)
        print(f"\n[ 夹爪与外设 / GRIPPER & INTERCONNECT ]")
        inter = status.interconnect
        gripper = inter.gripper_feedback.motor
        # 对应报文：gripper_feedback -> motor -> position
        grip_pos = gripper[0].position if gripper else 0.0
        print(f"夹爪位置/Gripper     | Position: {grip_pos:6.2f}% (Voltage: {inter.voltage:4.1f}V)")
        print(f"末端 IMU Acc (m/s²)  | X: {inter.imu_acceleration_x:7.2f}, Y: {inter.imu_acceleration_y:7.2f}, Z: {inter.imu_acceleration_z:7.2f}")

        # 7. 系统健康度 (System Health)
        print(f"\n[ 系统健康度 / SYSTEM HEALTH ]")
        # 对应报文：active_state, arm_voltage, arm_current, temperature_cpu
        print(f"当前状态/Active State | {b.active_state}")
        print(f"主供电/Power Supply  | {b.arm_voltage:4.1f} V / {b.arm_current:4.2f} A")
        print(f"温度/Temperature     | CPU: {b.temperature_cpu:4.1f} °C, Ambient: {b.temperature_ambient:4.1f} °C")
        
        print(f"\n{'='*85}")

    def send_gripper_position(self, position: float) -> float:
        """发送夹爪目标位置（Kortex 0.0–1.0，与 ``gamepad_control_obs`` 一致）。"""
        target = max(0.0, min(1.0, float(position)))
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = target
        self.base.SendGripperCommand(cmd)
        return target

    def control_gripper(self, value, dual_grip=True):
        if dual_grip:
            target = 1.0 if value >= 0.5 else 0.0
        else:
            target = max(0.0, min(1.0, float(value) / 100.0))
        self.send_gripper_position(target)
        time.sleep(0.5)

    def move_angular(self, angles_and_gripper, dual_grip=True):
        angles = angles_and_gripper[:7]
        gripper_val = angles_and_gripper[7]
        action = Base_pb2.Action()
        for i, angle in enumerate(angles):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = angle
        e, handle = self._wait_for_action()
        self.base.ExecuteAction(action)
        e.wait()
        self.base.Unsubscribe(handle)
        self.control_gripper(gripper_val, dual_grip)

    def _send_joint_speeds_once(self, joint_speeds_deg_per_s: list[float], per_joint_duration_ms: int) -> None:
        """构建并发送一条 JointSpeeds（不阻塞、不等 ACTION_END）。"""
        cmd = Base_pb2.JointSpeeds()
        for i, val in enumerate(joint_speeds_deg_per_s):
            js = cmd.joint_speeds.add()
            js.joint_identifier = i
            js.value = float(val)
            js.duration = int(per_joint_duration_ms)
        self.base.SendJointSpeedsCommand(cmd)

    def move_angular_fast(
        self,
        joint_speeds_deg_per_s: list[float],
        *,
        hold_s: float = 0.0,
        fps: float = 50.0,
        per_joint_duration_ms: int = 0,
        stop_after: bool = True,
        verbose_each_send: bool = False,
    ) -> None:
        """
        关节空间高速流式控制：使用 SendJointSpeedsCommand（类比 gamepad 的 SendTwistCommand），
        不经过 ExecuteAction / reach_joint_angles，因此单条指令返回很快。

        Args:
            joint_speeds_deg_per_s: 长度 7，各关节角速度 (°/s)。
            hold_s: 为 0 时只发送一条速度指令后立即返回；>0 时按 fps 周期性重复发送，总时长 hold_s 秒。
            fps: hold_s>0 时的发送频率，默认 50Hz。
            per_joint_duration_ms: 写入每条 JointSpeed.duration；0 与官方示例一致，由后续指令覆盖。
            stop_after: hold_s>0 结束时是否 base.Stop()，防止速度指令在固件侧持续生效。
            verbose_each_send: 为 True 时打印每次 SendJointSpeedsCommand 与 sleep 的分段耗时（调试用）。
        """
        if len(joint_speeds_deg_per_s) != 7:
            raise ValueError(f"joint_speeds_deg_per_s 长度须为 7，收到 {len(joint_speeds_deg_per_s)}")

        eps = 1e-6
        if all(abs(v) < eps for v in joint_speeds_deg_per_s):
            self.base.Stop()
            return

        if hold_s <= 0.0:
            t_one = time.perf_counter()
            self._send_joint_speeds_once(joint_speeds_deg_per_s, per_joint_duration_ms)
            if verbose_each_send:
                print(f"      fast single send: {(time.perf_counter() - t_one) * 1000:.2f} ms")
            return

        if fps <= 0:
            raise ValueError("fps 必须 > 0")

        period = 1.0 / fps
        t_end = time.perf_counter() + hold_s
        n_iter = 0
        while time.perf_counter() < t_end:
            n_iter += 1
            t_rpc = time.perf_counter()
            self._send_joint_speeds_once(joint_speeds_deg_per_s, per_joint_duration_ms)
            dt_rpc = time.perf_counter() - t_rpc
            t_sl = time.perf_counter()
            time.sleep(period)
            dt_sl = time.perf_counter() - t_sl
            if verbose_each_send:
                print(
                    f"      iter {n_iter}: SendJointSpeeds {dt_rpc * 1000:.2f} ms, "
                    f"sleep {dt_sl * 1000:.2f} ms (target {period * 1000:.2f} ms)"
                )

        if stop_after:
            t_st = time.perf_counter()
            self.base.Stop()
            if verbose_each_send:
                print(f"      base.Stop(): {(time.perf_counter() - t_st) * 1000:.2f} ms")

    def move_cartesian(self, pose_and_gripper, dual_grip=True, skip_gripper=False):
        p = pose_and_gripper
        gripper_val = p[6]
        action = Base_pb2.Action()
        tp = action.reach_pose.target_pose
        tp.x, tp.y, tp.z = p[0], p[1], p[2]
        tp.theta_x, tp.theta_y, tp.theta_z = p[3], p[4], p[5]
        e, handle = self._wait_for_action()
        self.base.ExecuteAction(action)
        e.wait()
        self.base.Unsubscribe(handle)
        if not skip_gripper:
            self.control_gripper(gripper_val, dual_grip)

    def move_velocity(self, speeds, duration_ms=20000):
        """
        speeds: [vx, vy, vz, wx, wy, wz]
        duration_ms: 自动停止阈值，默认200ms（防止程序崩溃导致机器人撞墙）
        """
        # print(f"\nIn kinova_manage, sending velocity command: {speeds} for duration: {duration_ms} ms")
        command = Base_pb2.TwistCommand()
        # 建议使用 REFERENCE_FRAME_BASE，更符合 RL 的坐标系逻辑
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        
        # 设置自动停止阈值
        command.duration = duration_ms 
        
        twist = command.twist
        twist.linear_x, twist.linear_y, twist.linear_z = speeds[0], speeds[1], speeds[2]
        twist.angular_x, twist.angular_y, twist.angular_z = speeds[3], speeds[4], speeds[5]
        
        # 立即发送，不阻塞
        self.base.SendTwistCommand(command)

    def move_relative(self, delta_pose, dual_grip=True):
        current_feedback = self.get_status()
        if not current_feedback: return
        curr = current_feedback.base
        target = [
            curr.tool_pose_x + delta_pose[0],
            curr.tool_pose_y + delta_pose[1],
            curr.tool_pose_z + delta_pose[2],
            curr.tool_pose_theta_x + delta_pose[3],
            curr.tool_pose_theta_y + delta_pose[4],
            curr.tool_pose_theta_z + delta_pose[5],
            0.0
        ]
        self.move_cartesian(target, dual_grip=False, skip_gripper=True)
        gripper_input = delta_pose[6]
        if dual_grip:
            if gripper_input >= 0.5:
                current_grip_pos = current_feedback.interconnect.gripper_feedback.motor[0].position
                new_target = 100.0 if current_grip_pos < 50.0 else 0.0
                self.control_gripper(new_target, dual_grip=False)
        else:
            self.control_gripper(gripper_input, dual_grip=False)


def test_kinova_manager():
    import numpy as np
    arm = KinovaManager()
    arm.connect()
    try:
        status = arm.get_status()
        # arm.print_status(status)
        home_joints = list(HOME_JOINTS)

        # exit()
        # home_joints = [360.00, 0.00, 180.00, 241.29, 180.00, 61.82, 270.00, 0.0]
        # home_joints = [355.22, 4.94, 190.63, 241.29, 181.31, 51.82, 277.83, 100.0]

        # arm.move_cartesian(np.array([*CUP_PUT_POSE, 100.0]).tolist(), dual_grip=False)
        # arm.move_cartesian(np.array([*CAMERA_PUT_POSE, 85.0]).tolist(), dual_grip=False) # move to camera put position
        # arm.move_angular(home_joints, dual_grip=False)

        # arm.move_cartesian(np.array([0.25, 0.1, 0.3, 135, -90, 132, 100.0]).tolist(), dual_grip=False) # move to camera put position

        position_1 = np.array([0.28, 0, 0.19, 175, 5, 75, 100.0]).tolist()
        position_2 = np.array([0.28, 0, 0.29, 175, 5, 75, 100.0]).tolist()
        position_3 = np.array([0.28, 0, 0.59, 175, 5, 75, 100.0]).tolist()

        time_0 = time.time()
        arm.move_cartesian(position_1, dual_grip=False)
        time_1 = time.time()
        arm.move_cartesian(position_2, dual_grip=False)
        time_2 = time.time()
        arm.move_cartesian(position_3, dual_grip=False)
        time_3 = time.time()
        
        print(f"time_1 - time_0 = {time_1 - time_0}")
        print(f"time_2 - time_1 = {time_2 - time_1}")
        print(f"time_3 - time_2 = {time_3 - time_2}")
        print(f"total time = {time_3 - time_0}")
        
        exit()

        '''
        status = arm.get_status()
        arm.print_status(status)
        print(f"\ntime.time(): {time.time()}")
        action = np.array([0.00, 0.00, 0.02, 0.0, 0.0, 0.0])
        action = -action
        print(f"action = {action}")
        arm.move_velocity(action)  # 向上移动
        print(f"time.time() after sending velocity command: {time.time()}\n")
        time.sleep(4)
        arm.move_velocity([0.0]*6)  # 停止移动
        time.sleep(1)
        status = arm.get_status()
        arm.print_status(status)

        time.sleep(1)
        arm.move_cartesian(np.array([0.3, 0.07, 0.35, 174.2, 9.16, 91.68, 0.0]).tolist(), dual_grip=True)
        status = arm.get_status()
        arm.print_status(status)
        '''

    finally:
        arm.disconnect()

def _read_terminal_key() -> str:
    """Read one key or an arrow-key escape sequence (Linux tty)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return ch + ch2 + ch3
            return ch + ch2
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def test_kinova_manager2():
    """
    交互测试：先到 HOME_JOINTS（与 line-213 home 相同），
    然后 ←→ 切换关节索引，↑↓ 增减角度（关节单位：度；夹爪：0–100）。
    每次 move_angular（含关节轨迹等待与夹爪）打印 wall-clock 耗时。按 q 退出。
    """
    arm_deg_step = 5.0
    gripper_step = 5.0

    arm = KinovaManager()
    arm.connect()
    move_count = 0

    def move_and_log(targets: list[float], label: str) -> None:
        nonlocal move_count
        move_count += 1
        t0 = time.perf_counter()
        arm.move_angular(targets, dual_grip=False)
        elapsed_s = time.perf_counter() - t0
        print(
            f"\n  [#{move_count} {label}] move_angular wall time: {elapsed_s * 1000:.1f} ms "
            f"({elapsed_s:.3f} s)"
        )

    try:
        print("Moving to HOME_JOINTS (initial pose, same as test_kinova_manager home)...")
        move_and_log(list(HOME_JOINTS), "home")
        targets = list(HOME_JOINTS)
        idx = 0

        print(
            "\nControls: ← / →  switch joint (0–6 arm, 7 gripper)  |  ↑ / ↓  increase / decrease\n"
            f"Step: {arm_deg_step}° per key (gripper ±{gripper_step}). Press q to quit.\n"
        )

        while True:
            jname = f"joint_{idx}" if idx < 7 else "gripper"
            vals = ", ".join(f"{v:6.2f}" for v in targets)
            print(f"\r[{idx}] {jname:8s} | {vals}", end="", flush=True)

            key = _read_terminal_key()
            if key in ("q", "Q", "\x03"):  # q or Ctrl+C
                print("\nExit.")
                break
            if key == "\x1b[D":  # Left
                idx = (idx - 1) % 8
            elif key == "\x1b[C":  # Right
                idx = (idx + 1) % 8
            elif key == "\x1b[A":  # Up
                if idx < 7:
                    targets[idx] += arm_deg_step
                else:
                    targets[7] = min(100.0, targets[7] + gripper_step)
                move_and_log(targets, f"joint_{idx}+up" if idx < 7 else "gripper+up")
            elif key == "\x1b[B":  # Down
                if idx < 7:
                    targets[idx] -= arm_deg_step
                else:
                    targets[7] = max(0.0, targets[7] - gripper_step)
                move_and_log(targets, f"joint_{idx}+down" if idx < 7 else "gripper+down")
            elif key in ("\r", "\n"):
                continue
    finally:
        arm.disconnect()

def test_move_angular_fast():
    """
    先用 move_angular 回到 HOME_JOINTS（位置模式、慢但稳），再用手柄式关节速度流：
    ←→ 选关节 0–6；↑↓ 对当前关节发 ±hold_s 秒的角速度脉冲（move_angular_fast）；
    关节 7 为夹爪，仍用 control_gripper 微调目标。每次动作打印 wall time。
    """
    joint_speed_mag = 15.0  # °/s，单关节脉冲幅值
    hold_s = 0.2
    fps = 50.0
    gripper_step = 5.0

    arm = KinovaManager()
    arm.connect()
    move_count = 0

    def log_move_angular(targets: list[float], label: str) -> None:
        nonlocal move_count
        move_count += 1
        t0 = time.perf_counter()
        arm.move_angular(targets, dual_grip=False)
        dt = time.perf_counter() - t0
        print(f"\n  [#{move_count} {label}] move_angular: {dt * 1000:.1f} ms ({dt:.3f} s)")

    def log_move_angular_fast(speeds7: list[float], label: str) -> None:
        nonlocal move_count
        move_count += 1
        t0 = time.perf_counter()
        print(f"\n  [#{move_count} {label}] move_angular_fast detail (hold_s={hold_s}, fps={fps}):")
        arm.move_angular_fast(
            speeds7, hold_s=hold_s, fps=fps, stop_after=True, verbose_each_send=True
        )
        dt = time.perf_counter() - t0
        print(
            f"  [#{move_count} {label}] move_angular_fast TOTAL: {dt * 1000:.1f} ms ({dt:.3f} s)"
        )

    def log_gripper(val: float, label: str) -> None:
        nonlocal move_count
        move_count += 1
        t0 = time.perf_counter()
        arm.control_gripper(val, dual_grip=False)
        dt = time.perf_counter() - t0
        print(f"\n  [#{move_count} {label}] control_gripper: {dt * 1000:.1f} ms ({dt:.3f} s)")

    try:
        print("Phase 1: move_angular ·→ HOME_JOINTS ...")
        log_move_angular(list(HOME_JOINTS), "home")
        idx = 0
        grip_target = float(HOME_JOINTS[7])
        last_speeds = [0.0] * 7

        print(
            "\nPhase 2: move_angular_fast (joint speed bursts)\n"
            f"← / →  select joint 0–6 or gripper 7  |  ↑ / ↓  joint: ±{joint_speed_mag} °/s for {hold_s}s @ {fps:g}Hz\n"
            f"gripper: ±{gripper_step} (position). q quit.\n"
        )

        while True:
            jname = f"joint_{idx}" if idx < 7 else "gripper"
            if idx < 7:
                line2 = ", ".join(f"{s:6.1f}" for s in last_speeds)
                extra = f" | last burst °/s: [{line2}]"
            else:
                extra = f" | grip cmd %: {grip_target:5.1f}"
            print(f"\r[{idx}] {jname:8s}{extra}", end="", flush=True)

            key = _read_terminal_key()
            if key in ("q", "Q", "\x03"):
                print("\nExit.")
                arm.move_angular_fast([0.0] * 7, hold_s=0.0, stop_after=True)
                break
            if key == "\x1b[D":
                idx = (idx - 1) % 8
            elif key == "\x1b[C":
                idx = (idx + 1) % 8
            elif key == "\x1b[A":
                if idx < 7:
                    sp = [0.0] * 7
                    sp[idx] = joint_speed_mag
                    last_speeds = list(sp)
                    log_move_angular_fast(sp, f"j{idx}+speed+")
                else:
                    grip_target = min(100.0, grip_target + gripper_step)
                    log_gripper(grip_target, "gripper+")
            elif key == "\x1b[B":
                if idx < 7:
                    sp = [0.0] * 7
                    sp[idx] = -joint_speed_mag
                    last_speeds = list(sp)
                    log_move_angular_fast(sp, f"j{idx}+speed-")
                else:
                    grip_target = max(0.0, grip_target - gripper_step)
                    log_gripper(grip_target, "gripper-")
            elif key in ("\r", "\n"):
                continue
    finally:
        arm.disconnect()


def test_move_torque(
    *,
    step_deg: float = 5.0,
    settle_s: float = 0.5,
) -> None:
    """先回到 HOME，再逐关节 +step_deg / -step_deg 力矩控制冒烟测试。

    ``move_torque`` 本身非阻塞；``settle_s`` 仅用于等待机械臂物理到位，可设为 0。
    """
    home = [float(x) for x in HOME_JOINTS[:7]]
    gripper = float(HOME_JOINTS[7]) / 100.0

    arm = KinovaManager()
    try:
        print("init_torque (高层 HOME_JOINTS → 进入 cyclic 后保持当前实测角度)...")
        t_init = time.perf_counter()
        arm.init_torque(home=True)
        print(f"  init_torque: {(time.perf_counter() - t_init) * 1000:.1f} ms")
        print(f"HOME 关节(°): {np.round(home, 2)}")

        t0 = time.perf_counter()
        arm.move_torque(home, gripper=gripper)
        print(f"  move_torque HOME: {(time.perf_counter() - t0) * 1000:.3f} ms (非阻塞)")
        if settle_s > 0:
            time.sleep(settle_s)

        for j in range(7):
            plus = home.copy()
            plus[j] += step_deg
            t0 = time.perf_counter()
            arm.move_torque(plus, gripper=gripper)
            print(
                f"  J{j + 1} +{step_deg}° -> {plus[j]:.1f}°  "
                f"send {(time.perf_counter() - t0) * 1000:.3f} ms",
                flush=True,
            )
            if settle_s > 0:
                time.sleep(settle_s)

            minus = home.copy()
            minus[j] -= step_deg
            t0 = time.perf_counter()
            arm.move_torque(minus, gripper=gripper)
            print(
                f"  J{j + 1} -{step_deg}° -> {minus[j]:.1f}°  "
                f"send {(time.perf_counter() - t0) * 1000:.3f} ms",
                flush=True,
            )
            if settle_s > 0:
                time.sleep(settle_s)

            t0 = time.perf_counter()
            arm.move_torque(home, gripper=gripper)
            print(f"  J{j + 1} -> HOME  send {(time.perf_counter() - t0) * 1000:.3f} ms", flush=True)
            if settle_s > 0:
                time.sleep(settle_s)

        st = arm.move_torque(home, gripper=gripper)
        print(f"结束关节(°): {np.round(st['q_deg'], 2)}")
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        arm.shutdown_torque()


if __name__ == "__main__":
    test_move_torque()