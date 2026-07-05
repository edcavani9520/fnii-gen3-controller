#!/usr/bin/env python3
"""
Kinova Gen3 数据采集脚本
=========================
基于 gamepad 遥操作，同步录制：
  - 机器人状态（关节角/速度/力矩、末端位姿/捻度/受力、夹爪、IMU 等）
  - 摄像头 RGB 图像

输出：HDF5 文件，按 episode 组织。

控制映射（与 gamepad_control_obs.py 一致）：
  左摇杆        → XY 平移
  LT/RT         → Z 轴升降
  右摇杆        → Roll / Pitch
  十字键左右    → Yaw
  A / B         → 夹爪关/开
  Y             → 开始/停止 录制 (toggle)
  Menu(按钮7)   → 退出程序

用法:
  python3 data_collector.py
"""

import sys
import os
import time
import threading
import queue
import datetime
from typing import Optional

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
#  配置（可根据需要修改）
# ======================================================================
class Config:
    # ---- 机械臂 ----
    robot_ip = "192.168.8.10"
    speed_limit = 0.20      # m/s
    turn_limit = 20.0       # °/s
    deadzone = 0.1

    # ---- 摄像头 ----
    camera_id = 0           # /dev/video0
    camera_width = 640
    camera_height = 480
    camera_fps = 25

    # ---- 采集 ----
    output_root = os.path.expanduser("~/kinova_data")     # 数据根目录
    robot_sample_hz = 100                                  # 机器人状态采样频率 (与反馈频率一致)
    camera_sample_hz = 25                                  # 相机采样频率
    hdf5_compression = "gzip"                              # HDF5 压缩算法
    hdf5_compression_opts = 4                              # 压缩等级

    # ---- 录制控制 ----
    record_button = 3       # Y 按钮 toggle 录制
    exit_button = 7         # Menu 退出


# ======================================================================
#  数据采集器
# ======================================================================
class KinovaDataCollector:
    """同步采集 Kinova 机械臂状态 + 摄像头图像，保存为 HDF5。"""

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()

        # ---- 机械臂连接 ----
        self.base: Optional[BaseClient] = None
        self.base_cyclic: Optional[BaseCyclicClient] = None
        self.router = None
        self.connection = None

        # ---- 手柄 ----
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("❌ 未检测到手柄，请连接后重试。")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        # ---- 摄像头 ----
        self.cap: Optional[cv2.VideoCapture] = None

        # ---- 采集缓冲 ----
        self._robot_queue: queue.Queue = queue.Queue(maxsize=5000)
        self._camera_queue: queue.Queue = queue.Queue(maxsize=2000)
        self._recording = False
        self._episode = 0
        self._output_dir = ""
        self._hdf5_file: Optional[h5py.File] = None
        self._running = threading.Event()
        self._running.set()

        # ---- 手柄按钮映射缓存 ----
        self._buttons = {}
        self._prev_y = False  # 边沿检测

        # ---- 夹爪状态 ----
        self._gripper_label = "IDLE"

        print(f"🎮 手柄已连接: {self.joy.get_name()}")

    # ================================================================
    #  连接
    # ================================================================
    def connect(self):
        """连接 Kinova 机械臂。"""
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
        """打开摄像头。"""
        self.cap = cv2.VideoCapture(self.cfg.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 无法打开摄像头 /dev/video{self.cfg.camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)
        # 实际读一帧确认
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("❌ 摄像头无数据")
        actual_h, actual_w = frame.shape[:2]
        print(f"📷 摄像头已打开: {actual_w}x{actual_h} @ {self.cfg.camera_fps} FPS")

    # ================================================================
    #  后台采集线程
    # ================================================================
    def _robot_poll_thread(self):
        """100Hz 机器人状态轮询线程。"""
        hz = self.cfg.robot_sample_hz
        period = 1.0 / hz
        while self._running.is_set():
            t_start = time.perf_counter()
            try:
                feedback = self.base_cyclic.RefreshFeedback()
                ts = time.time()
                snapshot = self._snapshot_robot_state(feedback, ts)
                self._robot_queue.put_nowait(snapshot)
            except Exception:
                pass
            elapsed = time.perf_counter() - t_start
            sleep = max(0.0, period - elapsed)
            time.sleep(sleep)

    def _camera_capture_thread(self):
        """摄像机采集线程（以相机原生帧率抓取，采样到 ~25Hz）。"""
        min_interval = 1.0 / self.cfg.camera_sample_hz
        last_cap = 0.0
        while self._running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._running.clear()
                break
            now = time.time()
            # 限流
            if now - last_cap >= min_interval:
                last_cap = now
                self._camera_queue.put_nowait({
                    "timestamp": now,
                    "frame": frame.copy(),
                })
            else:
                # 丢帧 — 摄像头通常比需求快
                pass
            # 不要忙等，稍微休息一下减少 CPU
            time.sleep(0.001)

    @staticmethod
    def _snapshot_robot_state(feedback, timestamp: float) -> dict:
        """将 Kortex Cyclic 反馈打包为 dict。"""
        b = feedback.base
        inter = feedback.interconnect

        # 关节数据
        n_act = len(feedback.actuators)
        joint_pos = np.zeros(n_act)
        joint_vel = np.zeros(n_act)
        joint_torque = np.zeros(n_act)
        joint_current = np.zeros(n_act)
        joint_voltage = np.zeros(n_act)
        joint_temp = np.zeros(n_act)
        for i, act in enumerate(feedback.actuators):
            joint_pos[i] = act.position
            joint_vel[i] = act.velocity
            joint_torque[i] = act.torque
            joint_current[i] = act.current_motor
            joint_voltage[i] = act.voltage
            joint_temp[i] = act.temperature_core

        # 夹爪
        gripper_motors = inter.gripper_feedback.motor
        gripper_pos = float(gripper_motors[0].position) if gripper_motors else 0.0

        return {
            "timestamp": timestamp,
            # 末端位姿
            "tool_pose": np.array([
                b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
                b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z
            ], dtype=np.float64),
            "tool_pose_cmd": np.array([
                b.commanded_tool_pose_x, b.commanded_tool_pose_y, b.commanded_tool_pose_z,
                b.commanded_tool_pose_theta_x, b.commanded_tool_pose_theta_y,
                b.commanded_tool_pose_theta_z
            ], dtype=np.float64),
            # 末端捻度（速度）
            "tool_twist": np.array([
                b.tool_twist_linear_x, b.tool_twist_linear_y, b.tool_twist_linear_z,
                b.tool_twist_angular_x, b.tool_twist_angular_y, b.tool_twist_angular_z
            ], dtype=np.float64),
            # 末端外力/力矩
            "tool_wrench": np.array([
                b.tool_external_wrench_force_x, b.tool_external_wrench_force_y,
                b.tool_external_wrench_force_z,
                b.tool_external_wrench_torque_x, b.tool_external_wrench_torque_y,
                b.tool_external_wrench_torque_z
            ], dtype=np.float64),
            # 关节
            "joint_position": joint_pos,
            "joint_velocity": joint_vel,
            "joint_torque": joint_torque,
            "joint_current": joint_current,
            "joint_voltage": joint_voltage,
            "joint_temperature": joint_temp,
            # 夹爪
            "gripper_position": gripper_pos,
            # IMU (基座)
            "base_imu_accel": np.array([
                b.imu_acceleration_x, b.imu_acceleration_y, b.imu_acceleration_z
            ], dtype=np.float64),
            "base_imu_gyro": np.array([
                b.imu_angular_velocity_x, b.imu_angular_velocity_y, b.imu_angular_velocity_z
            ], dtype=np.float64),
            # 电源
            "arm_voltage": b.arm_voltage,
            "arm_current": b.arm_current,
            "temperature_cpu": b.temperature_cpu,
            "temperature_ambient": b.temperature_ambient,
        }

    # ================================================================
    #  手柄输入
    # ================================================================
    def apply_deadzone(self, v: float) -> float:
        return v if abs(v) > self.cfg.deadzone else 0.0

    def read_gamepad(self) -> tuple:
        """读取手柄当前状态，返回 (axes, hat, buttons)。"""
        pygame.event.pump()
        a0 = self.apply_deadzone(self.joy.get_axis(0))
        a1 = self.apply_deadzone(self.joy.get_axis(1))
        a2 = (self.joy.get_axis(2) + 1) / 2.0
        a3 = self.apply_deadzone(self.joy.get_axis(3))
        a4 = self.apply_deadzone(self.joy.get_axis(4))
        a5 = (self.joy.get_axis(5) + 1) / 2.0
        hat = self.joy.get_hat(0)
        buttons = {i: self.joy.get_button(i) for i in range(self.joy.get_numbuttons())}
        return (a0, a1, a2, a3, a4, a5), hat, buttons

    def send_twist(self, axes, hat):
        """发送 Twist 指令。"""
        a0, a1, a2, a3, a4, a5 = axes
        command = Base_pb2.TwistCommand()
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = 0
        command.twist.linear_x = -a1 * self.cfg.speed_limit
        command.twist.linear_y = -a0 * self.cfg.speed_limit
        command.twist.linear_z = (a5 - a2) * self.cfg.speed_limit
        command.twist.angular_x = a3 * self.cfg.turn_limit
        command.twist.angular_y = -a4 * self.cfg.turn_limit
        command.twist.angular_z = -hat[0] * self.cfg.turn_limit

        has_input = any([
            abs(v) > 0.01
            for v in [a0, a1, a5 - a2, a3, a4, hat[0]]
        ])
        if has_input:
            self.base.SendTwistCommand(command)
        else:
            self.base.Stop()

    def control_gripper(self, pos: float):
        """pos: 0.0 全开, 1.0 全关"""
        try:
            cmd = Base_pb2.GripperCommand()
            cmd.mode = Base_pb2.GRIPPER_POSITION
            finger = cmd.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = pos
            self.base.SendGripperCommand(cmd)
        except Exception:
            pass

    # ================================================================
    #  HDF5 文件管理
    # ================================================================
    def _open_episode(self, episode: int):
        """创建新 episode 的 HDF5 文件。"""
        self._ensure_output_dir()
        fname = os.path.join(self._output_dir, f"episode_{episode:04d}.h5")
        f = h5py.File(fname, "w")

        # --- 创建数据集（可扩展） ---
        # 机器人状态：初始为无限可扩展
        f.create_dataset(
            "robot/timestamp", shape=(0,), maxshape=(None,), dtype=np.float64,
            compression=self.cfg.hdf5_compression,
            compression_opts=self.cfg.hdf5_compression_opts,
        )
        f.create_dataset(
            "robot/tool_pose", shape=(0, 6), maxshape=(None, 6), dtype=np.float64,
            compression=self.cfg.hdf5_compression,
        )
        f.create_dataset(
            "robot/tool_pose_cmd", shape=(0, 6), maxshape=(None, 6), dtype=np.float64,
        )
        f.create_dataset(
            "robot/tool_twist", shape=(0, 6), maxshape=(None, 6), dtype=np.float64,
        )
        f.create_dataset(
            "robot/tool_wrench", shape=(0, 6), maxshape=(None, 6), dtype=np.float64,
        )
        f.create_dataset(
            "robot/joint_position", shape=(0, 7), maxshape=(None, 7), dtype=np.float64,
        )
        f.create_dataset(
            "robot/joint_velocity", shape=(0, 7), maxshape=(None, 7), dtype=np.float64,
        )
        f.create_dataset(
            "robot/joint_torque", shape=(0, 7), maxshape=(None, 7), dtype=np.float64,
        )
        f.create_dataset(
            "robot/joint_current", shape=(0, 7), maxshape=(None, 7), dtype=np.float64,
        )
        f.create_dataset(
            "robot/joint_voltage", shape=(0, 7), maxshape=(None, 7), dtype=np.float64,
        )
        f.create_dataset(
            "robot/joint_temperature", shape=(0, 7), maxshape=(None, 7), dtype=np.float64,
        )
        f.create_dataset(
            "robot/gripper_position", shape=(0,), maxshape=(None,), dtype=np.float64,
        )
        f.create_dataset(
            "robot/base_imu_accel", shape=(0, 3), maxshape=(None, 3), dtype=np.float64,
        )
        f.create_dataset(
            "robot/base_imu_gyro", shape=(0, 3), maxshape=(None, 3), dtype=np.float64,
        )
        f.create_dataset(
            "robot/arm_voltage", shape=(0,), maxshape=(None,), dtype=np.float64,
        )
        f.create_dataset(
            "robot/arm_current", shape=(0,), maxshape=(None,), dtype=np.float64,
        )
        f.create_dataset(
            "robot/temperature_cpu", shape=(0,), maxshape=(None,), dtype=np.float64,
        )
        f.create_dataset(
            "robot/temperature_ambient", shape=(0,), maxshape=(None,), dtype=np.float64,
        )

        # --- 图像 ---
        f.create_dataset(
            "camera/timestamp", shape=(0,), maxshape=(None,), dtype=np.float64,
        )
        # 用 uint8 存储 JPEG 压缩后的图像（节省空间）
        f.create_dataset(
            "camera/rgb", shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('uint8')),
            compression=self.cfg.hdf5_compression,
            compression_opts=self.cfg.hdf5_compression_opts,
        )

        # --- 元数据 ---
        f.attrs["episode"] = episode
        f.attrs["start_time"] = time.time()
        f.attrs["robot_ip"] = self.cfg.robot_ip
        f.attrs["speed_limit"] = self.cfg.speed_limit
        f.attrs["turn_limit"] = self.cfg.turn_limit
        f.attrs["camera_fps"] = self.cfg.camera_fps
        f.attrs["robot_sample_hz"] = self.cfg.robot_sample_hz
        f.attrs["camera_sample_hz"] = self.cfg.camera_sample_hz

        self._hdf5_file = f
        print(f"\n📝 开始录制 → {fname}")

    def _flush_episode(self):
        """将缓冲数据写入 HDF5。"""
        if self._hdf5_file is None:
            return

        f = self._hdf5_file

        # --- 写机器人数据 ---
        robot_items = []
        while not self._robot_queue.empty():
            try:
                robot_items.append(self._robot_queue.get_nowait())
            except queue.Empty:
                break

        if robot_items:
            n = len(robot_items)
            keys = [
                ("robot/timestamp", "timestamp"),
                ("robot/tool_pose", "tool_pose"),
                ("robot/tool_pose_cmd", "tool_pose_cmd"),
                ("robot/tool_twist", "tool_twist"),
                ("robot/tool_wrench", "tool_wrench"),
                ("robot/joint_position", "joint_position"),
                ("robot/joint_velocity", "joint_velocity"),
                ("robot/joint_torque", "joint_torque"),
                ("robot/joint_current", "joint_current"),
                ("robot/joint_voltage", "joint_voltage"),
                ("robot/joint_temperature", "joint_temperature"),
                ("robot/gripper_position", "gripper_position"),
                ("robot/base_imu_accel", "base_imu_accel"),
                ("robot/base_imu_gyro", "base_imu_gyro"),
                ("robot/arm_voltage", "arm_voltage"),
                ("robot/arm_current", "arm_current"),
                ("robot/temperature_cpu", "temperature_cpu"),
                ("robot/temperature_ambient", "temperature_ambient"),
            ]
            for h5path, key in keys:
                data = np.array([r[key] for r in robot_items])
                ds = f[h5path]
                cur = ds.shape[0]
                ds.resize((cur + n, *ds.shape[1:]))
                ds[cur:] = data

        # --- 写图像 ---
        camera_items = []
        while not self._camera_queue.empty():
            try:
                camera_items.append(self._camera_queue.get_nowait())
            except queue.Empty:
                break

        if camera_items:
            n = len(camera_items)
            cam_ts = []
            cam_bytes = []
            for item in camera_items:
                cam_ts.append(item["timestamp"])
                # JPEG 编码
                ret, buf = cv2.imencode(".jpg", item["frame"], [
                    cv2.IMWRITE_JPEG_QUALITY, 90
                ])
                if ret:
                    cam_bytes.append(buf.tobytes())
                else:
                    cam_bytes.append(b"")

            ts_ds = f["camera/timestamp"]
            img_ds = f["camera/rgb"]
            cur = ts_ds.shape[0]
            ts_ds.resize((cur + n,))
            ts_ds[cur:] = np.array(cam_ts)
            img_ds.resize((cur + n,))
            for i, b in enumerate(cam_bytes):
                img_ds[cur + i] = np.frombuffer(b, dtype=np.uint8)

        f.flush()

    def _close_episode(self):
        """关闭当前 episode。"""
        # 写剩余缓冲
        self._flush_episode()
        if self._hdf5_file is not None:
            elapsed = time.time() - self._hdf5_file.attrs["start_time"]
            robot_count = self._hdf5_file["robot/timestamp"].shape[0]
            cam_count = self._hdf5_file["camera/timestamp"].shape[0]
            self._hdf5_file.close()
            self._hdf5_file = None
            print(
                f"💾 Episode {self._episode:04d} 已保存: "
                f"{elapsed:.1f}s, "
                f"robot={robot_count} samples (~{robot_count / (elapsed + 1e-6):.0f} Hz), "
                f"camera={cam_count} frames (~{cam_count / (elapsed + 1e-6):.0f} Hz)"
            )

    def _ensure_output_dir(self):
        """创建以当前时间命名的输出根目录。"""
        if not self._output_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self.cfg.output_root, f"collection_{ts}")
        os.makedirs(self._output_dir, exist_ok=True)

    # ================================================================
    #  主循环
    # ================================================================
    def run(self):
        try:
            # 启动后台线程
            robot_thread = threading.Thread(target=self._robot_poll_thread, daemon=True)
            camera_thread = threading.Thread(target=self._camera_capture_thread, daemon=True)
            robot_thread.start()
            camera_thread.start()
            print("🚀 后台采集线程已启动")

            # --- 状态显示缓冲区（每 5 帧刷新一次显示信息） ---
            display_counter = 0
            rec_label = "■ IDLE"

            while self._running.is_set():
                # 1. 读手柄
                axes, hat, buttons = self.read_gamepad()

                # 2. 检测退出
                if buttons.get(self.cfg.exit_button):
                    print("\n⏹ 退出程序...")
                    break

                # 3. 录制 toggle (Y 按钮上升沿)
                y_pressed = bool(buttons.get(self.cfg.record_button))
                if y_pressed and not self._prev_y:
                    if self._recording:
                        self._stop_recording()
                        rec_label = "■ IDLE"
                    else:
                        self._start_recording()
                        rec_label = "● REC"
                self._prev_y = y_pressed

                # 4. 夹爪
                if buttons.get(0):  # A
                    self.control_gripper(1.0)
                    self._gripper_label = "CLOSED"
                elif buttons.get(1):  # B
                    self.control_gripper(0.0)
                    self._gripper_label = "OPENED"

                # 5. 发送 Twist
                self.send_twist(axes, hat)

                # 6. 录制：周期性 flush 缓冲
                if self._recording:
                    display_counter += 1
                    if display_counter % 5 == 0:
                        self._flush_episode()
                else:
                    # 不录制时清空缓冲，避免堆积
                    self._drain_queues()

                # 7. 终端状态显示
                display_counter += 1
                if display_counter % 5 == 0:
                    self._print_status(axes, hat, rec_label)

                time.sleep(0.05)  # 20Hz 控制

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _start_recording(self):
        """开始新 episode 录制。"""
        self._episode += 1
        self._drain_queues()  # 清空可能残留的旧数据
        self._open_episode(self._episode)
        self._recording = True

    def _stop_recording(self):
        """停止当前录制。"""
        self._recording = False
        self._close_episode()

    def _drain_queues(self):
        """清空采集缓冲。"""
        for q in (self._robot_queue, self._camera_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _print_status(self, axes, hat, rec_label):
        """刷新单行终端显示。"""
        a0, a1, _, a3, a4, _ = axes
        joy_str = (
            f"X:{a1:+5.2f} Y:{a0:+5.2f} Z:{axes[5]-axes[2]:+5.2f} | "
            f"R:{a3:+5.2f} P:{a4:+5.2f} Y:{hat[0]:+2.0f} | "
            f"Grip:{self._gripper_label}"
        )
        # 取最后一帧机器人数据
        latest = None
        if not self._robot_queue.empty():
            try:
                # peek 最新（不弹出）
                pass
            except Exception:
                pass

        sys.stdout.write(
            f"\r{rec_label}  {joy_str}  "
            f"| Q:{self._robot_queue.qsize():<4}/{self._robot_queue.maxsize} "
            f"CamQ:{self._camera_queue.qsize():<4}/{self._camera_queue.maxsize}     "
        )
        sys.stdout.flush()

    def _cleanup(self):
        """清理：停止录制、断开连接、释放资源。"""
        self._running.clear()

        if self._recording:
            self._recording = False
            self._close_episode()

        # 停止机械臂
        if self.base:
            try:
                self.base.Stop()
            except Exception:
                pass

        # 断开连接
        if self.connection:
            try:
                self.connection.__exit__(None, None, None)
            except Exception:
                pass

        # 关闭摄像头
        if self.cap:
            self.cap.release()

        pygame.quit()
        cv2.destroyAllWindows()
        print("\n👋 安全退出。")

        # 输出汇总
        print("\n📁 数据保存位置:")
        print(f"   {self._output_dir}" if self._output_dir else "   (无数据)")


# ======================================================================
#  Entry
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Kinova Gen3 数据采集器 (Gamepad Teleop + Recording)")
    print("=" * 60)
    print()
    print("  控制映射:")
    print("    左摇杆 → XY 平移    右摇杆 → Roll / Pitch")
    print("    LT/RT  → Z 轴升降   十字键 → Yaw")
    print("    A 关夹爪  B 开夹爪   Y → 开始/停止录制")
    print("    Menu 退出")
    print()

    collector = KinovaDataCollector()
    try:
        collector.connect()
        collector.connect_camera()
        collector.run()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
