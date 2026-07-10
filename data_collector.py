#!/usr/bin/env python3
"""
Kinova Gen3 数据采集脚本 - 训练用格式输出
=============================================
基于 gamepad 遥操作，同步录制 obs/action 对，直接可用作 VLA 行为克隆训练。

输出格式（每 episode 一个 HDF5）:
  obs/
    camera_0      (T_sync,) uint8 (vlen)       — JPEG-encoded RGB 图像（对齐到动作步频）
    joint_pos     (T_sync, 7) float64       — 关节位置
    joint_vel     (T_sync, 7) float64       — 关节速度
    eef_pose      (T_sync, 6) float64       — 末端位姿 [x,y,z,θx,θy,θz]
    gripper_pos   (T_sync, 1) float64       — 夹爪开合度 [0=开, 1=关]
  action/
    eef_delta     (T_sync, 6) float64       — 末端 delta [dx,dy,dz,dθx,dθy,dθz]
    gripper       (T_sync, 1) float64       — 夹爪动作 [0=不动, -1=开, +1=关]
    raw_twist     (T_sync, 6) float64       — 原始速度指令 [vx,vy,vz,wx,wy,wz]
  timestamps      (T_sync,) float64         — 每步时间戳
  meta/  (attrs)
    episode, start_time, robot_ip, camera_fps, action_hz, ...

T_sync = camera 帧数（对齐到 action_hz，默认 25Hz）。
每个 step = (obs_t, action_t) 对, action_t 是从 obs_t 到 obs_{t+1} 的指令。

控制映射（与 gamepad_control.py 一致）：
  左摇杆        → XY 平移
  LT/RT         → Z 轴升降
  右摇杆        → Roll / Pitch
  十字键左右    → Yaw
  A / B         → 夹爪关/开
  Y             → 开始录制；再按 Y 保存停止，按 X 删除
  Menu(按钮7)   → 退出程序

用法:
  python3 data_collector.py
"""

import sys
import os

# 兼容 protobuf 4.x + 旧版 kortex _pb2 桩代码
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import threading
import queue
import datetime
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import pygame
import cv2
import h5py

# ---------- Kinova API ----------
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Kinova_kortex2_Gen3_G3L", "api_python", "examples"
))
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2


# ======================================================================
#  配置
# ======================================================================
@dataclass
class Config:
    # ---- 机械臂 ----
    robot_ip: str = "192.168.8.10"
    speed_limit: float = 0.20      # m/s
    turn_limit: float = 20.0       # °/s
    deadzone: float = 0.1

    # ---- 摄像头 ----
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 25

    # ---- 采集 ----
    output_root: str = os.path.expanduser("~/kinova_data")
    robot_sample_hz: int = 100    # 机器人状态采样频率
    action_hz: int = 25           # 输出 obs/action 对的频率（与 camera 对齐）
    hdf5_compression: str = "gzip"
    hdf5_compression_opts: int = 4

    # ---- 任务描述（VLA language instruction）----
    task: str = ""                     # 命令行通过 --task 指定，为空则在录制时交互输入
    task_id: int = 0                   # 任务编号

    # ---- 录制控制 ----
    record_button: int = 3        # Y
    delete_button: int = 2        # X → 停止并删除视频
    exit_button: int = 7          # Menu

    # ---- 图像质量 ----
    jpeg_quality: int = 95        # 存储用 JPEG 压缩，加载时解压为 uint8


# ======================================================================
#  数据采集器 — 输出训练用格式
# ======================================================================
class KinovaTrainDataCollector:
    """
    采集（obs, action）对用于 VLA / 行为克隆训练。

    设计要点：
      - robot 状态 @ 100Hz, camera @ 25Hz, 动作 @ 25Hz
      - 保存时以 camera 时间戳为锚点，对齐最近的 robot 状态和动作
      - 输出 HDF5: obs/ + action/ 分组，每个时间步一对
      - 自动计算 eef_delta 和 gripper 动作
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()

        # ---- Kinova 连接 ----
        self.base: Optional[BaseClient] = None
        self.base_cyclic: Optional[BaseCyclicClient] = None
        self.router = None
        self.connection = None

        # ---- 手柄 ----
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("❌ 未检测到手柄")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        # ---- 摄像头 ----
        self.cap: Optional[cv2.VideoCapture] = None

        # ===== 三路缓冲 =====
        self._robot_q: queue.Queue = queue.Queue(maxsize=5000)
        self._camera_q: queue.Queue = queue.Queue(maxsize=2000)
        self._action_q: queue.Queue = queue.Queue(maxsize=5000)

        # ---- 录制状态 ----
        self._recording = False
        self._episode = 0
        self._task_counter = 0          # 自动递增任务编号
        self._output_dir = ""
        self._hdf5_file: Optional[h5py.File] = None
        self._running = threading.Event()
        self._running.set()

        # ---- 边沿检测 ----
        self._prev_y = False
        self._prev_x = False
        self._gripper_label = "IDLE"
        self._last_gripper_cmd = 0.0   # 上次夹爪命令

        # ---- 动作缓冲（用于计算 delta） ----
        self._prev_pose = None  # 前一步的末端位姿

        print(f"🎮 手柄已连接: {self.joy.get_name()}")

    # ================================================================
    #  连接
    # ================================================================
    def connect(self):
        class Args:
            def __init__(self, ip):
                self.ip = ip
                self.username = "admin"
                self.password = "admin"
                self.port = 10000
        args = Args(self.cfg.robot_ip)
        self.connection = utilities.DeviceConnection.createTcpConnection(args)
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        print(f"✅ 已连接 Kinova @ {self.cfg.robot_ip}")

    def connect_camera(self):
        self.cap = cv2.VideoCapture(self.cfg.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 无法打开摄像头 /dev/video{self.cfg.camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("❌ 摄像头无数据")
        actual_h, actual_w = frame.shape[:2]
        print(f"📷 摄像头已打开: {actual_w}x{actual_h} @ {self.cfg.camera_fps} FPS")

    # ================================================================
    #  后台采集线程（三路并行）
    # ================================================================

    def _robot_poll_thread(self):
        """100Hz 机器人状态 + 动作（手柄指令）。"""
        period = 1.0 / self.cfg.robot_sample_hz
        while self._running.is_set():
            t0 = time.perf_counter()
            try:
                feedback = self.base_cyclic.RefreshFeedback()
                ts = time.time()
                obs = self._snapshot_robot_state(feedback, ts)
                self._robot_q.put_nowait(obs)
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

    def _action_poll_thread(self):
        """记录手柄指令（动作），频率对齐 robot 采样。"""
        period = 1.0 / self.cfg.robot_sample_hz
        while self._running.is_set():
            t0 = time.perf_counter()
            try:
                axes, hat, buttons = self._read_gamepad()
                ts = time.time()
                action = self._snapshot_action(axes, hat, buttons, ts)
                self._action_q.put_nowait(action)
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

    def _camera_capture_thread(self):
        """摄像头采集线程（原生帧率）。"""
        min_interval = 1.0 / self.cfg.camera_fps
        last_cap = 0.0
        while self._running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._running.clear()
                break
            now = time.time()
            if now - last_cap >= min_interval:
                last_cap = now
                self._camera_q.put_nowait({
                    "timestamp": now,
                    "frame": frame.copy(),
                })
            time.sleep(0.001)

    # ================================================================
    #  快照函数
    # ================================================================

    @staticmethod
    def _snapshot_robot_state(feedback, timestamp: float) -> dict:
        b = feedback.base
        inter = feedback.interconnect
        n_act = len(feedback.actuators)
        joint_pos = np.zeros(n_act)
        joint_vel = np.zeros(n_act)
        joint_torque = np.zeros(n_act)
        for i, act in enumerate(feedback.actuators):
            joint_pos[i] = act.position
            joint_vel[i] = act.velocity
            joint_torque[i] = act.torque
        gripper_motors = inter.gripper_feedback.motor
        gripper_pos = float(gripper_motors[0].position) if gripper_motors else 0.0
        return {
            "timestamp": timestamp,
            "eef_pose": np.array([
                b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
                b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z
            ], dtype=np.float64),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
            "gripper_pos": gripper_pos,
        }

    def _read_gamepad(self):
        pygame.event.pump()
        a0 = self._deadzone(self.joy.get_axis(0))
        a1 = self._deadzone(self.joy.get_axis(1))
        a2 = (self.joy.get_axis(2) + 1) / 2.0
        a3 = self._deadzone(self.joy.get_axis(3))
        a4 = self._deadzone(self.joy.get_axis(4))
        a5 = (self.joy.get_axis(5) + 1) / 2.0
        hat = self.joy.get_hat(0)
        buttons = {i: self.joy.get_button(i) for i in range(self.joy.get_numbuttons())}
        return (a0, a1, a2, a3, a4, a5), hat, buttons

    def _deadzone(self, v):
        return v if abs(v) > self.cfg.deadzone else 0.0

    def _snapshot_action(self, axes, hat, buttons, timestamp: float) -> dict:
        """打包当前手柄指令为动作向量。"""
        a0, a1, a2, a3, a4, a5 = axes
        # Twist 指令 [vx, vy, vz, wx, wy, wz]
        twist = np.array([
            -a1 * self.cfg.speed_limit,
            -a0 * self.cfg.speed_limit,
            (a5 - a2) * self.cfg.speed_limit,
            a3 * self.cfg.turn_limit,
            -a4 * self.cfg.turn_limit,
            -hat[0] * self.cfg.turn_limit,
        ], dtype=np.float64)

        # 夹爪指令
        gripper_cmd = self._last_gripper_cmd
        if buttons.get(0):  # A → 关闭
            gripper_cmd = 1.0
        elif buttons.get(1):  # B → 打开
            gripper_cmd = 0.0
        self._last_gripper_cmd = gripper_cmd

        return {
            "timestamp": timestamp,
            "twist": twist,
            "gripper_cmd": np.array([gripper_cmd], dtype=np.float64),
        }

    # ================================================================
    #  手柄控制（主循环调用）
    # ================================================================

    def _send_twist_from_action(self, action_dict: dict):
        """从动作字典中提取 twist 并发送给机器人。"""
        twist = action_dict["twist"]
        cmd = Base_pb2.TwistCommand()
        cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        cmd.duration = 0
        cmd.twist.linear_x = twist[0]
        cmd.twist.linear_y = twist[1]
        cmd.twist.linear_z = twist[2]
        cmd.twist.angular_x = twist[3]
        cmd.twist.angular_y = twist[4]
        cmd.twist.angular_z = twist[5]

        has_input = np.any(np.abs(twist) > 0.001)
        if has_input:
            self.base.SendTwistCommand(cmd)
        else:
            self.base.Stop()

    def _send_gripper(self, pos: float):
        try:
            cmd = Base_pb2.GripperCommand()
            cmd.mode = Base_pb2.GRIPPER_POSITION
            finger = cmd.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = float(pos)
            self.base.SendGripperCommand(cmd)
        except Exception:
            pass

    # ================================================================
    #  HDF5 输出 — 训练用格式
    # ================================================================

    def _open_episode(self, episode: int):
        self._ensure_output_dir()
        fname = os.path.join(self._output_dir, f"episode_{episode:04d}.h5")
        f = h5py.File(fname, "w")

        # === obs 分组 ===
        grp_obs = f.create_group("obs")
        # camera_0: JPEG-encoded RGB frames (variable-length)
        grp_obs.create_dataset(
            "camera_0", shape=(0,), maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype('uint8')),
            compression=self.cfg.hdf5_compression,
            compression_opts=self.cfg.hdf5_compression_opts,
        )
        for name, dim in [("joint_pos", 7), ("joint_vel", 7),
                          ("eef_pose", 6), ("gripper_pos", 1)]:
            grp_obs.create_dataset(
                name, shape=(0, dim), maxshape=(None, dim), dtype=np.float64,
                compression=self.cfg.hdf5_compression,
            )

        # === action 分组 ===
        grp_act = f.create_group("action")
        for name, dim in [("eef_delta", 6), ("gripper", 1), ("raw_twist", 6)]:
            grp_act.create_dataset(
                name, shape=(0, dim), maxshape=(None, dim), dtype=np.float64,
                compression=self.cfg.hdf5_compression,
            )

        # === 时间戳 ===
        f.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)

        # === VLA 元数据（language instruction, task info） ===
        task_desc = self.cfg.task or "unnamed"
        f.attrs["episode"] = episode
        f.attrs["task/id"] = self.cfg.task_id
        f.attrs["task/language_instruction"] = task_desc
        f.attrs["dataset_name"] = "kinova_gen3_teleop"
        f.attrs["robot_type"] = "Kinova Gen3"
        f.attrs["control_mode"] = "twist_velocity"
        f.attrs["camera_names"] = "['cam_0']"
        f.attrs["start_time"] = time.time()
        f.attrs["robot_ip"] = self.cfg.robot_ip
        f.attrs["speed_limit"] = self.cfg.speed_limit
        f.attrs["turn_limit"] = self.cfg.turn_limit
        f.attrs["camera_fps"] = self.cfg.camera_fps
        f.attrs["camera_width"] = self.cfg.camera_width
        f.attrs["camera_height"] = self.cfg.camera_height
        f.attrs["action_hz"] = self.cfg.action_hz
        f.attrs["robot_sample_hz"] = self.cfg.robot_sample_hz
        f.attrs["jpeg_quality"] = self.cfg.jpeg_quality
        f.attrs["date_collected"] = datetime.datetime.now().isoformat()

        self._hdf5_file = f
        print(f"\n📝 开始录制 → {fname}")
        print(f"   🏷️  task: {task_desc}")

        # 重置 delta 计算缓存
        self._prev_pose = None

    def _sync_and_write(self):
        """
        核心对齐函数：以 camera 时间戳为锚点，同步 robot 状态和动作，
        计算 eef_delta，写入 HDF5。
        """
        if self._hdf5_file is None:
            return

        f = self._hdf5_file

        # 1. drain 所有缓冲
        robot_items = self._drain(self._robot_q)
        camera_items = self._drain(self._camera_q)
        action_items = self._drain(self._action_q)

        if not camera_items or not robot_items or not action_items:
            return

        # 2. 转 numpy 方便搜索
        cam_ts = np.array([c["timestamp"] for c in camera_items])
        robot_ts = np.array([r["timestamp"] for r in robot_items])
        action_ts = np.array([a["timestamp"] for a in action_items])

        # === 一次性扩展所有数据集 ===
        cur = f["timestamps"].shape[0]
        n_total = len(camera_items)
        for ds_name in ["timestamps", "obs/camera_0", "obs/joint_pos", "obs/joint_vel",
                        "obs/eef_pose", "obs/gripper_pos",
                        "action/eef_delta", "action/gripper", "action/raw_twist"]:
            ds = f[ds_name]
            ds.resize((cur + n_total, *ds.shape[1:]))

        # 3. 为每个 camera 帧找到最近的 robot state 和 action
        for ci, cam_item in enumerate(camera_items):
            cam_t = cam_item["timestamp"]

            # 找最近的 robot 状态
            ri = np.argmin(np.abs(robot_ts - cam_t))
            r_item = robot_items[ri] if ri < len(robot_items) else robot_items[-1]

            # 找最近的动作
            ai = np.argmin(np.abs(action_ts - cam_t))
            a_item = action_items[ai] if ai < len(action_items) else action_items[-1]

            # === obs ===
            obs_pose = r_item["eef_pose"]
            obs_jp = r_item["joint_pos"]
            obs_jv = r_item["joint_vel"]
            obs_gp = np.array([r_item["gripper_pos"]], dtype=np.float64)

            # === action（通过 delta 计算） ===
            if self._prev_pose is not None:
                # eef_delta = 当前位姿 - 上一步位姿
                eef_delta = obs_pose - self._prev_pose
            else:
                # 第一步：上一帧没有信息，先用 twist * dt 估计
                eef_delta = np.zeros(6, dtype=np.float64)

            self._prev_pose = obs_pose.copy()

            # gripper action: 检测变化
            if f["obs/gripper_pos"].shape[0] > 0:
                gripper_prev = f["obs/gripper_pos"][-1]
                gripper_action = obs_gp - gripper_prev[0]  # +1=关, -1=开, 0=不动
            else:
                # 第一帧无历史数据，gripper动作为0
                gripper_action = np.array([0.0])
            gripper_action = np.clip(gripper_action, -1.0, 1.0)

            # === 写入 ===
            idx = cur + ci
            f["timestamps"][idx] = cam_t

            # obs: 图像（JPEG 编码保存）
            frame = cam_item["frame"]  # BGR from cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret, buf = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            f["obs/camera_0"][idx] = np.frombuffer(buf, dtype=np.uint8) if ret else b""
            f["obs/joint_vel"][idx] = obs_jv
            f["obs/eef_pose"][idx] = obs_pose
            f["obs/gripper_pos"][idx] = obs_gp

            # action
            f["action/eef_delta"][idx] = eef_delta
            f["action/gripper"][idx] = gripper_action
            f["action/raw_twist"][idx] = a_item["twist"]

        f.flush()
    def _close_episode(self):
        """关闭当前 episode（写剩余 + 计算）。"""
        self._sync_and_write()
        if self._hdf5_file is not None:
            elapsed = time.time() - self._hdf5_file.attrs["start_time"]
            n = self._hdf5_file["timestamps"].shape[0]
            self._hdf5_file.close()
            self._hdf5_file = None
            hz = n / (elapsed + 1e-6)
            print(f"💾 Episode {self._episode:04d} 已保存: "
                  f"{n} steps, {elapsed:.1f}s, {hz:.1f} Hz")

    def _ensure_output_dir(self):
        if not self._output_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self.cfg.output_root, f"train_data_{ts}")
        os.makedirs(self._output_dir, exist_ok=True)

    @staticmethod
    def _drain(q):
        items = []
        while not q.empty():
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                break
        return items

    # ================================================================
    #  主循环
    # ================================================================

    def run(self):
        try:
            # 启动三条后台线程
            threads = [
                threading.Thread(target=self._robot_poll_thread, daemon=True),
                threading.Thread(target=self._camera_capture_thread, daemon=True),
                threading.Thread(target=self._action_poll_thread, daemon=True),
            ]
            for t in threads:
                t.start()
            print("🚀 后台采集线程已启动 (robot 100Hz | camera 25Hz | action 100Hz)")

            display_counter = 0
            rec_label = "■ IDLE"

            while self._running.is_set():
                # === 1. 读取手柄（主循环 20Hz） ===
                axes, hat, buttons = self._read_gamepad()

                # 退出检测
                if buttons.get(self.cfg.exit_button):
                    print("\n⏹ 退出程序...")
                    break

                # 录制控制：Y 开始 / Y 保存 / X 删除
                y_pressed = bool(buttons.get(self.cfg.record_button))
                x_pressed = bool(buttons.get(self.cfg.delete_button))

                if y_pressed and not self._prev_y:
                    if not self._recording:
                        self._start_recording()
                        rec_label = "● REC"
                    else:
                        # Y 再次按下 → 停止并保存
                        self._stop_recording()
                        rec_label = "■ IDLE"

                if x_pressed and not self._prev_x:
                    if self._recording:
                        # X 按下 → 停止并删除
                        self._stop_recording(delete=True)
                        rec_label = "■ IDLE"

                self._prev_y = y_pressed
                self._prev_x = x_pressed

                # 夹爪
                if buttons.get(0):
                    self._send_gripper(1.0)
                    self._gripper_label = "CLOSED"
                elif buttons.get(1):
                    self._send_gripper(0.0)
                    self._gripper_label = "OPENED"

                # 构建并发送 twist（从当前手柄状态）
                twist_now = np.array([
                    -axes[1] * self.cfg.speed_limit,
                    -axes[0] * self.cfg.speed_limit,
                    (axes[5] - axes[2]) * self.cfg.speed_limit,
                    axes[3] * self.cfg.turn_limit,
                    -axes[4] * self.cfg.turn_limit,
                    -hat[0] * self.cfg.turn_limit,
                ], dtype=np.float64)
                has_input = np.any(np.abs(twist_now) > 0.001)
                if has_input:
                    cmd = Base_pb2.TwistCommand()
                    cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
                    cmd.duration = 0
                    cmd.twist.linear_x = twist_now[0]
                    cmd.twist.linear_y = twist_now[1]
                    cmd.twist.linear_z = twist_now[2]
                    cmd.twist.angular_x = twist_now[3]
                    cmd.twist.angular_y = twist_now[4]
                    cmd.twist.angular_z = twist_now[5]
                    self.base.SendTwistCommand(cmd)
                else:
                    self.base.Stop()

                # === 2. 录制同步 ===
                if self._recording:
                    display_counter += 1
                    if display_counter % 5 == 0:
                        self._sync_and_write()
                else:
                    self._drain(self._robot_q)
                    self._drain(self._camera_q)
                    self._drain(self._action_q)
                    self._prev_pose = None  # 重置

                # === 3. 状态显示 ===
                display_counter += 1
                if display_counter % 5 == 0:
                    self._print_status(axes, hat, rec_label)

                time.sleep(0.05)  # 20Hz

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _start_recording(self):
        """开始新 episode。"""
        self._episode += 1
        self._task_counter += 1
        self.cfg.task_id = self._task_counter

        self._drain(self._robot_q)
        self._drain(self._camera_q)
        self._drain(self._action_q)
        self._prev_pose = None
        self._open_episode(self._episode)
        self._recording = True

    def _stop_recording(self, delete=False):
        self._recording = False
        if delete:
            # 保存文件名后关闭再删除
            fname = self._hdf5_file.filename if self._hdf5_file is not None else None
            self._close_episode()
            if fname and os.path.exists(fname):
                os.remove(fname)
                print(f"Deleted episode {self._episode:04d}: {fname}")
        else:
            self._close_episode()

    def _print_status(self, axes, hat, rec_label):
        a0, a1, _, a3, a4, _ = axes
        joy_str = (f"X:{a1:+5.2f} Y:{a0:+5.2f} Z:{axes[5]-axes[2]:+5.2f} | "
                   f"R:{a3:+5.2f} P:{a4:+5.2f} Y:{hat[0]:+2.0f} | "
                   f"Grip:{self._gripper_label}")
        # 如果录制中，显示当前步骤数
        extra = ""
        if self._recording and self._hdf5_file is not None:
            n = self._hdf5_file["timestamps"].shape[0]
            extra = f" | steps:{n}"
        sys.stdout.write(
            f"\r{rec_label}  {joy_str}  "
            f"| Q:{self._robot_q.qsize():<4} ActQ:{self._action_q.qsize():<4} "
            f"CamQ:{self._camera_q.qsize():<4}{extra}     "
        )
        sys.stdout.flush()


    def _cleanup(self):
        self._running.clear()
        if self._recording:
            self._recording = False
            self._close_episode()
        if self.base:
            try:
                self.base.Stop()
            except Exception:
                pass
        if self.connection:
            try:
                self.connection.__exit__(None, None, None)
            except Exception:
                pass
        if self.cap:
            self.cap.release()
        pygame.quit()
        cv2.destroyAllWindows()
        print("\n👋 安全退出。")
        if self._output_dir:
            print(f"\n📁 数据保存位置:\n   {self._output_dir}")
            print(f"   格式: obs/(camera_0, joint_pos, joint_vel, eef_pose, gripper_pos) + action/(eef_delta, gripper, raw_twist)")


# ======================================================================
#  Entry
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kinova Gen3 训练数据采集器")
    parser.add_argument("--task", type=str, required=True,
                        help="任务描述（language instruction），如 --task 'pick up the red cup'")
    parser.add_argument("--ip", type=str, default="192.168.8.10",
                        help="机械臂 IP 地址")
    args = parser.parse_args()

    cfg = Config()
    cfg.task = args.task
    cfg.robot_ip = args.ip

    print("=" * 60)
    print("  Kinova Gen3 训练数据采集器")
    if cfg.task:
        print(f"  🏷️  任务: {cfg.task}")
    print("  输出格式: obs/action 对 + language instruction, 直接可用于 VLA 训练")
    print("=" * 60)
    print()
    print("  控制映射:")
    print("    左摇杆 → XY 平移    右摇杆 → Roll / Pitch")
    print("    LT/RT  → Z 轴升降   十字键 → Yaw")
    print("    A 关夹爪  B 开夹爪   Y 开始录制 / Y 保存 / X 删除")
    print("    Menu 退出")
    print(f"  输出: ~/kinova_data/train_data_<timestamp>/episode_XXXX.h5")

    collector = KinovaTrainDataCollector(cfg=cfg)
    try:
        collector.connect()
        collector.connect_camera()
        collector.run()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
