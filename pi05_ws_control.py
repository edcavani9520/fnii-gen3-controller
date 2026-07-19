#!/usr/bin/env python3
"""
pi05_ws_control.py — π0.5 WebSocket 推理 + Kinova Gen3 真机控制

用法:
  python pi05_ws_control.py
  python pi05_ws_control.py --ws-host localhost --ws-port 8000 --robot-ip 192.168.8.10
  python pi05_ws_control.py --dry-run          # 只推理不连机械臂

流程:
  1. 连接 WebSocket 推理服务端 (openpi)
  2. 连接 Kinova Gen3
  3. 读取相机 + 关节状态 → 发送推理服务 → 接收动作 → 执行

依赖:
  pip install ~/openpi/packages/openpi-client opencv-python
"""

import argparse
import json
import os
import sys
import time
import threading
import signal
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ---------- Kortex SDK ----------
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "Kinova_kortex2_Gen3_G3L", "api_python", "examples"))
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2

# ---------- openpi WebSocket 客户端 ----------
sys.path.insert(0, str(Path.home() / "openpi" / "packages" / "openpi-client" / "src"))
import websockets.sync.client
from websockets.exceptions import ConnectionClosed
from openpi_client import msgpack_numpy

# ---------- OpenCV ----------
import cv2
import numpy as np

# ---------- IK Solver ----------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "isbot"))
from robot_controller.ik_solver import IKSolver


# ==================== 旋转工具函数 ====================

def rpy_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """RPY (度) → 四元数 [x, y, z, w]"""
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, cp, cy = np.cos([r * 0.5, p * 0.5, y * 0.5])
    sr, sp, sy = np.sin([r * 0.5, p * 0.5, y * 0.5])
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def angle_diff_deg(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Shortest signed angular difference in degrees."""
    return (target - current + 180.0) % 360.0 - 180.0


def _check_for_end_or_abort(e):
    def check(notification, e=e):
        if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            e.set()
    return check


class Pi05WebSocketControl:
    """π0.5 WebSocket 推理 + Kinova 真机控制"""

    def __init__(self, ws_host="localhost", ws_port=8000, robot_ip="192.168.8.10",
                 camera_id=0, prompt="Put the cube into the bowl",
                 dry_run=False, control_freq=10.0, action_steps=1,
                 max_pos_step=0.015, max_rot_step=1.0,
                 max_joint_speed=10.0, action_scale=1.0,
                 observe_only=False, control_mode="twist",
                 max_linear_speed=0.05, max_angular_speed=3.0,
                 log_every=5, auto_start=True,
                 start_pose_path=None, start_tolerance_deg=3.0,
                 min_ee_z=None, max_down_step=None,
                 camera_drain_frames=0):
        # ── 配置 ──
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.robot_ip = robot_ip
        self.camera_id = camera_id
        self.prompt = prompt
        self.dry_run = dry_run
        self.observe_only = observe_only
        self.dt = 1.0 / control_freq
        self.action_steps = max(1, int(action_steps))  # 每推理一次执行几步 (1=每步都推理)
        self.max_pos_step = float(max_pos_step)
        self.max_rot_step = float(max_rot_step)
        self.max_joint_speed = float(max_joint_speed)
        self.action_scale = float(action_scale)
        if control_mode not in ("twist", "ik"):
            raise ValueError(f"Unsupported control_mode={control_mode!r}; use 'twist' or 'ik'")
        self.control_mode = control_mode
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.log_every = max(1, int(log_every))
        self.auto_start = bool(auto_start)
        self.start_pose_path = start_pose_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "start_pose.json"
        )
        self.start_tolerance_deg = float(start_tolerance_deg)
        self.min_ee_z = None if min_ee_z is None else float(min_ee_z)
        self.max_down_step = None if max_down_step is None else abs(float(max_down_step))
        self.camera_drain_frames = max(0, int(camera_drain_frames))

        # ── Kortex 对象 ──
        self.base = None
        self.base_cyclic = None
        self.router = None
        self.connection = None

        # ── WebSocket 客户端 ──
        self.policy = None

        # ── 共享状态 (后台线程更新) ──
        self.full_status = None  # BaseCyclic feedback
        self.running = False

        # ── 动作 chunk 缓存 ──
        self.action_chunk = None
        self.action_idx = 0
        self.step_count = 0
        self.infer_count = 0
        self.prev_ee_pos = None
        self.prev_ee_rpy = None

        # ── IK 解算器 ──
        self.ik_solver = IKSolver(ee_offset=0.12)

        # ── 相机 ──
        self.cap = None

    def _connect_policy_socket(self):
        uri = f"ws://{self.ws_host}:{self.ws_port}"
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=None,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    # ==================== 连接 ====================

    def connect_robot(self):
        """建立 Kortex 连接 (同 gamepad_control.py 模式)"""
        if self.dry_run:
            print("[robot] Dry-run mode, skipping robot connection")
            return

        class Args:
            def __init__(self, ip):
                self.ip = ip
                self.username = "admin"
                self.password = "admin"
                self.port = 10000

        self.connection = utilities.DeviceConnection.createTcpConnection(Args(self.robot_ip))
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        print(f"[robot] ✅ Connected to Kinova Gen3 at {self.robot_ip}")

    def connect_websocket(self):
        """连接 openpi WebSocket 推理服务端"""
        uri = f"ws://{self.ws_host}:{self.ws_port}"
        print(f"[ws] Connecting to π0.5 server at {uri}...")
        self._packer = msgpack_numpy.Packer()
        self.policy, meta = self._connect_policy_socket()
        print(f"[ws] ✅ Connected. Metadata: {meta}")

    def infer_policy(self, obs):
        """Send one inference request, reconnecting once if the socket was closed."""
        data = self._packer.pack(obs)
        for attempt in range(2):
            try:
                self.policy.send(data)
                response = self.policy.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")
                return msgpack_numpy.unpackb(response)
            except ConnectionClosed as e:
                if attempt == 1:
                    raise
                print(f"[ws] ⚠ Connection closed during infer ({e}); reconnecting once...")
                try:
                    self.policy.close()
                except Exception:
                    pass
                self.policy, _ = self._connect_policy_socket()

    def connect_camera(self):
        """初始化相机"""
        if self.dry_run:
            print("[camera] Dry-run mode, skipping camera")
            return

        self.cap = cv2.VideoCapture(self.camera_id)
        # 尽量和 data_collector.py 的采集相机设置保持一致。
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 100)
        # 训练数据是 320x240 原始相机图；OpenPI 侧会再做 resize_with_pad 到 224x224。
        # 这里不要提前拉伸成 224x224，否则视觉几何会和训练分布不一致。
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        if not self.cap.isOpened():
            print(f"[camera] ⚠ Cannot open camera {self.camera_id}, will use dummy image")
        else:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[camera] ✅ Camera {self.camera_id} ready ({width}x{height})")

    # ==================== 状态读取 ====================

    def _state_observer(self):
        """后台线程: 100Hz 拉取全量反馈 (同 gamepad_control_obs.py)"""
        while self.running:
            try:
                self.full_status = self.base_cyclic.RefreshFeedback()
                time.sleep(0.01)
            except Exception:
                pass

    def get_joint_positions(self):
        """获取当前 7 关节角度 (度)"""
        if self.dry_run or self.full_status is None:
            return np.zeros(7, dtype=np.float32)
        return np.array([act.position for act in self.full_status.actuators], dtype=np.float32)

    def get_gripper_position(self):
        """获取夹爪位置 [0, 100]"""
        if self.dry_run or self.full_status is None:
            return 0.0
        try:
            fb = self.full_status.interconnect.gripper_feedback
            return fb.motor[0].position if fb.motor else 0.0
        except Exception:
            return 0.0

    def get_ee_pose(self):
        """获取末端位姿: position (m), RPY (度)"""
        if self.dry_run or self.full_status is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        b = self.full_status.base
        pos = np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z], dtype=np.float32)
        rpy = np.array([b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z], dtype=np.float32)
        return pos, rpy

    def get_camera_image(self):
        """读取一帧 RGB 相机图像，保持训练采集时的 320x240 比例。"""
        if self.cap is not None and self.cap.isOpened():
            for _ in range(self.camera_drain_frames):
                self.cap.grab()
            ret, frame = self.cap.read()
            if ret:
                if frame.shape[0] != 240 or frame.shape[1] != 320:
                    frame = cv2.resize(frame, (320, 240))
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.zeros((240, 320, 3), dtype=np.uint8)

    # ==================== 动作执行 ====================

    def send_joint_command(self, target_joints_deg):
        """
        发送关节速度指令，逼近目标位置。
        P 控制: speed = (target - current) * gain
        """
        # 获取当前关节角
        current = self.get_joint_positions()
        gain = 20.0  # 速度增益 (°/s per ° error)
        error = angle_diff_deg(np.asarray(target_joints_deg), current)
        speeds = np.clip(error * gain, -self.max_joint_speed, self.max_joint_speed)

        cmd = Base_pb2.JointSpeeds()
        for i in range(7):
            js = cmd.joint_speeds.add()
            js.joint_identifier = i
            js.value = float(speeds[i])
            js.duration = 0

        if np.all(np.abs(speeds) < 0.5):
            self.base.Stop()
        else:
            self.base.SendJointSpeedsCommand(cmd)

    def send_twist_command(self, delta_action):
        """把模型预测的单步 Cartesian delta 转成 Kinova base-frame twist 速度指令。"""
        linear = np.asarray(delta_action[:3], dtype=np.float32) / self.dt
        angular = np.asarray(delta_action[3:6], dtype=np.float32) / self.dt
        raw_linear = linear.copy()
        raw_angular = angular.copy()

        linear = np.clip(linear, -self.max_linear_speed, self.max_linear_speed)
        angular = np.clip(angular, -self.max_angular_speed, self.max_angular_speed)
        clipped = (
            not np.allclose(raw_linear, linear, rtol=0.0, atol=1e-6)
            or not np.allclose(raw_angular, angular, rtol=0.0, atol=1e-6)
        )

        if np.linalg.norm(linear) < 1e-4 and np.linalg.norm(angular) < 1e-3:
            self.base.Stop()
            return linear, angular, clipped

        cmd = Base_pb2.TwistCommand()
        cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        cmd.duration = 0
        cmd.twist.linear_x = float(linear[0])
        cmd.twist.linear_y = float(linear[1])
        cmd.twist.linear_z = float(linear[2])
        cmd.twist.angular_x = float(angular[0])
        cmd.twist.angular_y = float(angular[1])
        cmd.twist.angular_z = float(angular[2])
        self.base.SendTwistCommand(cmd)
        return linear, angular, clipped

    def sanitize_action(self, action):
        """Clamp model output to a small, safe per-control-step delta."""
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[-1] < 7 or not np.all(np.isfinite(action[:7])):
            raise ValueError(f"Invalid action from policy: {action}")

        raw = action.copy()
        action[:6] *= self.action_scale
        action[:3] = np.clip(action[:3], -self.max_pos_step, self.max_pos_step)
        action[3:6] = np.clip(action[3:6], -self.max_rot_step, self.max_rot_step)
        action[6] = np.clip(action[6], 0.0, 1.0)
        clipped = not np.allclose(raw[:7], action[:7], rtol=0.0, atol=1e-6)
        return action, clipped

    def apply_workspace_safety(self, action, ee_pos):
        """可选的桌面高度保护：限制继续向下的 z delta。"""
        action = np.asarray(action, dtype=np.float32).copy()
        clipped = False

        if self.max_down_step is not None and action[2] < -self.max_down_step:
            action[2] = -self.max_down_step
            clipped = True

        if self.min_ee_z is not None:
            allowed_dz = self.min_ee_z - float(ee_pos[2])
            if action[2] < allowed_dz:
                action[2] = max(0.0, allowed_dz)
                clipped = True

        return action, clipped

    def send_gripper_command(self, position_0_100):
        """发送夹爪位置指令"""
        try:
            gripper_cmd = Base_pb2.GripperCommand()
            gripper_cmd.mode = Base_pb2.GRIPPER_POSITION
            finger = gripper_cmd.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = position_0_100 / 100.0
            self.base.SendGripperCommand(gripper_cmd)
        except Exception as e:
            print(f"[gripper] ⚠ Error: {e}")

    def load_start_joint_angles(self):
        """读取 data_collector 使用的 start_pose.json 关节角。"""
        if not os.path.exists(self.start_pose_path):
            raise FileNotFoundError(f"start_pose.json not found: {self.start_pose_path}")
        with open(self.start_pose_path, "r", encoding="utf-8") as f:
            pose_data = json.load(f)
        angles = pose_data.get("joint_angles_deg")
        if not isinstance(angles, list) or len(angles) != 7:
            raise ValueError(f"Invalid joint_angles_deg in {self.start_pose_path}")
        return np.asarray(angles, dtype=np.float32)

    def is_at_joint_pose(self, target_joints_deg):
        """判断当前关节角是否已在目标附近，处理 0/360 度环绕。"""
        current = self.get_joint_positions()
        err = angle_diff_deg(np.asarray(target_joints_deg), current)
        max_err = float(np.max(np.abs(err)))
        return max_err <= self.start_tolerance_deg, current, err, max_err

    def move_to_joint_pose(self, target_joints_deg, name="StartPose", timeout=20.0):
        """通过 Kinova Action 移动到指定关节角。"""
        if self.dry_run or self.observe_only or self.base is None:
            return False

        self.base.Stop()
        time.sleep(0.1)
        try:
            self.base.ClearFaults()
        except Exception:
            pass

        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(servo_mode)

        action = Base_pb2.Action()
        action.name = name
        action.application_data = ""
        for i, val in enumerate(target_joints_deg):
            ja = action.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier = i
            ja.value = float(val)

        done = threading.Event()
        result = {"event": None}

        def check(notification):
            if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                result["event"] = notification.action_event
                done.set()

        handle = self.base.OnNotificationActionTopic(check, Base_pb2.NotificationOptions())
        try:
            self.base.ExecuteAction(action)
            notified = done.wait(timeout)
        finally:
            self.base.Unsubscribe(handle)

        self.action_chunk = None
        self.action_idx = 0
        self.prev_ee_pos = None
        self.prev_ee_rpy = None
        return notified and result["event"] == Base_pb2.ACTION_END

    def ensure_start_pose(self):
        """如果不在采集 start pose，则自动移动过去，并打开夹爪。"""
        if not self.auto_start:
            print("[start] Auto-start disabled")
            return
        if self.dry_run:
            print("[start] Dry-run mode, skipping start pose")
            return
        if self.observe_only:
            print("[start] Observe-only mode, not moving to start pose")
            return
        if self.base is None:
            return

        try:
            target = self.load_start_joint_angles()
        except Exception as e:
            print(f"[start] ⚠ Cannot load start pose: {e}")
            raise RuntimeError("Cannot load start pose") from e

        at_start, current, err, max_err = self.is_at_joint_pose(target)
        print(f"[start] Target joints: {' '.join(f'{v:.2f}' for v in target)}")
        print(f"[start] Current error max={max_err:.2f}° "
              f"({' '.join(f'{v:+.2f}' for v in err)})")

        if not at_start:
            print("[start] Not at start pose, moving there first...")
            finished = self.move_to_joint_pose(target, name="StartPose", timeout=20.0)
            if not finished:
                print("[start] ⚠ Start pose move timeout/abort; stopping before policy run")
                self.base.Stop()
                raise RuntimeError("Failed to reach start pose")
            time.sleep(0.5)
            print("[start] Start pose reached")
        else:
            print("[start] Already near start pose")

        # data_collector 中 gripper 0.0 表示打开，1.0 表示闭合。
        self.send_gripper_command(0.0)
        time.sleep(0.5)
        print("[start] Gripper opened")

    def go_to_home(self):
        """回到初始位姿"""
        if self.dry_run or self.base is None:
            return
        print("[robot] Moving to home pose...")

        # 先停止、清故障、设置伺服模式（参考 gamepad_control）
        self.base.Stop()
        time.sleep(0.1)

        # 清除故障
        try:
            self.base.ClearFaults()
        except:
            pass

        servoing_mode = Base_pb2.ServoingModeInformation()
        servoing_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(servoing_mode)

        action = Base_pb2.Action()
        action.name = "Home"
        home_angles = [0, 340, 180, 214, 0, 310, 90]
        for i, val in enumerate(home_angles):
            ja = action.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier = i
            ja.value = float(val)

        e = threading.Event()
        def check(notification, e=e):
            if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                e.set()
        handle = self.base.OnNotificationActionTopic(check, Base_pb2.NotificationOptions())
        self.base.ExecuteAction(action)
        e.wait(10.0)
        self.base.Unsubscribe(handle)
        self.send_gripper_command(0.0)
        print("[robot] ✓ Home pose reached")

    # ==================== 主循环 ====================

    def run(self):
        """主控制循环"""
        running = True

        def on_sigint(_, __):
            nonlocal running
            print("\n[stop] Interrupted...")
            running = False
        signal.signal(signal.SIGINT, on_sigint)

        # ── 连接 ──
        self.connect_websocket()
        self.connect_robot()
        self.connect_camera()

        # 不干跑模式时启动状态观测线程
        if not self.dry_run and self.base_cyclic is not None:
            self.running = True
            obs_thread = threading.Thread(target=self._state_observer, daemon=True)
            obs_thread.start()
            time.sleep(0.5)  # 等状态回传

        # ── 初始定位：和 data_collector 的 start_pose.json 对齐 ──
        self.ensure_start_pose()

        print(f"\n{'='*60}")
        print(f" π0.5 推理控制循环")
        print(f" 服务端: {self.ws_host}:{self.ws_port}")
        print(f" 频率:   {1/self.dt:.1f} Hz")
        print(f" 控制模式: {self.control_mode}")
        print(f" chunk执行步数: {self.action_steps}")
        print(f" 单步限幅: xyz≤{self.max_pos_step:.3f}m, rpy≤{self.max_rot_step:.1f}°, "
              f"joint_speed≤{self.max_joint_speed:.1f}°/s, action_scale={self.action_scale:.2f}")
        if self.control_mode == "twist":
            print(f" Twist限速: linear≤{self.max_linear_speed:.3f}m/s, "
                  f"angular≤{self.max_angular_speed:.1f}°/s")
        if self.min_ee_z is not None or self.max_down_step is not None:
            print(f" Z保护: min_ee_z={self.min_ee_z}, max_down_step={self.max_down_step}")
        if self.camera_drain_frames:
            print(f" 相机丢弃旧帧: {self.camera_drain_frames} frame(s)/step")
        print(f" 提示词: \"{self.prompt}\"")
        if self.dry_run:
            print(" [DRY-RUN] 仅推理不执行")
        if self.observe_only:
            print(" [OBSERVE-ONLY] 推理并打印动作，不下发机械臂/夹爪命令")
        if self.auto_start:
            print(f" Start定位: {self.start_pose_path} "
                  f"(tol={self.start_tolerance_deg:.1f}°)")
        print(f"{'='*60}\n")

        # ── 主循环 ──
        while running:
            loop_start = time.monotonic()

            # 1. 观测
            image = self.get_camera_image()
            cv2.imshow("π0.5 Camera Feed", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[stop] Q pressed")
                break
            joint_pos = self.get_joint_positions()
            gripper = self.get_gripper_position()
            state = np.concatenate([joint_pos, np.array([gripper / 100.0])]).astype(np.float32)
            ee_pos, ee_rpy = self.get_ee_pose()
            if self.prev_ee_pos is None:
                actual_delta = None
            else:
                actual_delta = np.concatenate([
                    ee_pos - self.prev_ee_pos,
                    angle_diff_deg(ee_rpy, self.prev_ee_rpy),
                ])

            # 2. 推理（按 action_steps 消耗 chunk，默认每步重新推理）
            chunk_exec_limit = 0 if self.action_chunk is None else min(
                self.action_chunk.shape[0], self.action_steps
            )
            if self.action_chunk is None or self.action_idx >= chunk_exec_limit:
                # KinovaInputs 需要 wrist_image（但会 mask 掉），补上 dummy
                obs = {
                    "observation/image": image,
                    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                    "observation/state": state,
                    "prompt": self.prompt,
                }
                t0 = time.monotonic()
                outputs = self.infer_policy(obs)
                infer_ms = (time.monotonic() - t0) * 1000

                self.action_chunk = np.asarray(outputs["actions"], dtype=np.float32)  # (action_horizon, 7)
                if self.action_chunk.ndim != 2 or self.action_chunk.shape[1] < 7:
                    raise ValueError(f"Policy returned bad actions shape: {self.action_chunk.shape}")
                self.action_idx = 0
                self.infer_count += 1

                timing = outputs.get("server_timing", outputs.get("policy_timing", {}))
                print(f"[INFER #{self.infer_count}] "
                      f"chunk={self.action_chunk.shape} "
                      f"client={infer_ms:.0f}ms "
                      f"server={timing.get('infer_ms', 0):.0f}ms "
                      f"joint0={joint_pos[0]:.1f}°")

            # 3. 取动作
            action = self.action_chunk[self.action_idx]
            self.action_idx += 1
            self.step_count += 1
            safe_action, clipped = self.sanitize_action(action)
            safe_action, workspace_clipped = self.apply_workspace_safety(safe_action, ee_pos)
            clipped = clipped or workspace_clipped

            # 4. 执行
            # Kinova policy output is already unnormalized Cartesian delta:
            # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper].
            twist_linear = None
            twist_angular = None
            twist_clipped = False
            if not self.dry_run and not self.observe_only and self.base is not None:
                if self.control_mode == "twist":
                    twist_linear, twist_angular, twist_clipped = self.send_twist_command(safe_action)
                else:
                    # 计算目标位姿 = 当前位置 + 模型输出的增量
                    target_pos = ee_pos + safe_action[0:3]
                    target_rpy = ee_rpy + safe_action[3:6]
                    target_quat = rpy_to_quaternion(
                        target_rpy[0], target_rpy[1], target_rpy[2]
                    )

                    # IK 解算: target pose → joint angles (rad)
                    curr_qpos_rad = np.deg2rad(joint_pos)
                    target_joints_rad = self.ik_solver.solve(
                        target_pos, target_quat, curr_qpos_rad
                    )
                    target_joints_deg = np.rad2deg(target_joints_rad)

                    # 发送关节位置命令
                    self.send_joint_command(target_joints_deg)

                # 发送夹爪命令 (action[6] ∈ [0, 1] → 映射到 [0, 100])
                gripper_cmd = float(safe_action[6] * 100.0)
                self.send_gripper_command(gripper_cmd)

            # 打印进度
            if self.step_count % self.log_every == 1:
                clip_label = " CLIPPED" if clipped or twist_clipped else ""
                print(f"  [step {self.step_count}] "
                      f"raw=({' '.join(f'{v:.4f}' for v in action[:3])})m/"
                      f"({' '.join(f'{v:.3f}' for v in action[3:6])})° "
                      f"safe=({' '.join(f'{v:.4f}' for v in safe_action[:3])})m/"
                      f"({' '.join(f'{v:.3f}' for v in safe_action[3:6])})° "
                      f"gripper={safe_action[6]:.2f} ee_z={ee_pos[2]:.4f}m{clip_label}")
                if actual_delta is not None:
                    print(f"           actual_delta="
                          f"({' '.join(f'{v:.4f}' for v in actual_delta[:3])})m/"
                          f"({' '.join(f'{v:.3f}' for v in actual_delta[3:6])})°")
                if twist_linear is not None:
                    print(f"           twist_cmd="
                          f"({' '.join(f'{v:.3f}' for v in twist_linear)})m/s/"
                          f"({' '.join(f'{v:.2f}' for v in twist_angular)})°/s")

            self.prev_ee_pos = ee_pos.copy()
            self.prev_ee_rpy = ee_rpy.copy()

            # 5. 维持频率
            elapsed = time.monotonic() - loop_start
            sleep = self.dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

        # ── 清理 ──
        self.cleanup()

    def cleanup(self):
        print("\n[cleanup] Shutting down...")
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if not self.dry_run and self.base is not None:
            # self.go_to_home()
            self.base.Stop()
            if self.connection:
                self.connection.__exit__(None, None, None)
        print("[cleanup] ✓ Done")


def main():
    parser = argparse.ArgumentParser(description="π0.5 WebSocket + Kinova Gen3 真机控制")
    parser.add_argument("--ws-host", default="localhost", help="openpi 推理服务地址")
    parser.add_argument("--ws-port", type=int, default=8000, help="openpi 推理服务端口")
    parser.add_argument("--robot-ip", default="192.168.8.10", help="机械臂 IP")
    parser.add_argument("--camera-id", type=int, default=0, help="相机设备号")
    parser.add_argument("--prompt", default="Put the cube into the bowl", help="语言指令")
    parser.add_argument("--dry-run", action="store_true", help="不连机械臂不连相机")
    parser.add_argument("--observe-only", action="store_true",
                        help="连接机械臂/相机读取真实观测，但只打印动作不执行")
    parser.add_argument("--freq", type=float, default=10.0, help="控制频率 (Hz)")
    parser.add_argument("--action-steps", type=int, default=1,
                        help="每推理一次执行几步 (1=每步都推理)")
    parser.add_argument("--max-pos-step", type=float, default=0.015,
                        help="每个控制周期允许的最大 XYZ 增量 (m)")
    parser.add_argument("--max-rot-step", type=float, default=1.0,
                        help="每个控制周期允许的最大 RPY 增量 (度)")
    parser.add_argument("--max-joint-speed", type=float, default=10.0,
                        help="关节速度限幅 (度/秒)")
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="执行前对模型前 6 维 action 乘的比例")
    parser.add_argument("--control-mode", choices=["twist", "ik"], default="twist",
                        help="twist=按采集方式下发笛卡尔速度; ik=旧版 IK+关节速度")
    parser.add_argument("--max-linear-speed", type=float, default=0.05,
                        help="twist 模式下末端线速度限幅 (m/s)")
    parser.add_argument("--max-angular-speed", type=float, default=3.0,
                        help="twist 模式下末端角速度限幅 (度/秒)")
    parser.add_argument("--log-every", type=int, default=5,
                        help="每隔多少个控制周期打印一次动作诊断")
    parser.add_argument("--skip-start-position", action="store_true",
                        help="启动后不自动移动到 start_pose.json")
    parser.add_argument("--start-pose-path", default=None,
                        help="start_pose.json 路径，默认使用当前脚本同目录下的文件")
    parser.add_argument("--start-tolerance-deg", type=float, default=3.0,
                        help="判断已在 start 位置的最大关节角误差 (度)")
    parser.add_argument("--min-ee-z", type=float, default=None,
                        help="末端 Z 最低高度保护 (m)。低于此高度时禁止继续向下")
    parser.add_argument("--max-down-step", type=float, default=None,
                        help="单周期最大向下位移限制 (m)，例如 0.004")
    parser.add_argument("--camera-drain-frames", type=int, default=0,
                        help="每次取图前丢弃多少帧旧图，降低缓存延迟；USB相机可试 1")
    args = parser.parse_args()

    ctrl = Pi05WebSocketControl(
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        robot_ip=args.robot_ip,
        camera_id=args.camera_id,
        prompt=args.prompt,
        dry_run=args.dry_run,
        control_freq=args.freq,
        action_steps=args.action_steps,
        max_pos_step=args.max_pos_step,
        max_rot_step=args.max_rot_step,
        max_joint_speed=args.max_joint_speed,
        action_scale=args.action_scale,
        observe_only=args.observe_only,
        control_mode=args.control_mode,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        log_every=args.log_every,
        auto_start=not args.skip_start_position,
        start_pose_path=args.start_pose_path,
        start_tolerance_deg=args.start_tolerance_deg,
        min_ee_z=args.min_ee_z,
        max_down_step=args.max_down_step,
        camera_drain_frames=args.camera_drain_frames,
    )
    ctrl.run()


if __name__ == "__main__":
    main()
