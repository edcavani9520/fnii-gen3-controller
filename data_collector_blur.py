#!/usr/bin/env python3
"""
Kinova Gen3 模糊数据采集脚本（专为 Deblur 算法测试）
===================================================
基于 gamepad_control.KinovaJoyTeleop 遥操作，录制时相机故意设长曝光产生运动模糊。

相机参数：
  - 1280×720, MJPG, 15 fps
  - 手动曝光, exposure ≈ 50 ms (值 500)
  - Gain 压低至 1~2，确保弱光下也能糊

控制映射（与 gamepad_control.py 一致）：
  左摇杆        → XY 平移
  LT/RT         → Z 轴升降
  右摇杆        → Roll / Pitch
  十字键左右    → Yaw
  A / B         → 夹爪关/开
  Y             → 开始/停止 录制 (toggle)
  Menu(按钮7)   → 退出程序

用法:
  python3 data_collector_blur.py
"""

import sys
import os
import time
import threading
import queue
import datetime
from typing import Optional

import numpy as np
import cv2
import h5py
import pygame

# ---------- 复用 gamepad_control 的遥操作 ----------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamepad_control import KinovaJoyTeleop

from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2


# ======================================================================
#  配置 — 模糊采集专用
# ======================================================================
class BlurConfig:
    # ---- 机械臂 ----
    robot_ip = "192.168.8.10"
    speed_limit = 0.20      # m/s
    turn_limit = 20.0       # °/s
    deadzone = 0.1

    # ---- 摄像头（故意长曝光 + 低 Gain 产生运动模糊） ----
    camera_id = 0
    camera_width = 1280
    camera_height = 720
    camera_fps = 15

    # 曝光控制（手动模式，1ms 轻度模糊）
    camera_manual_exposure = True
    camera_exposure = 100              # 100 × 100µs = 1ms
    camera_gain = 1
    camera_brightness = 0              # 亮度压低避免过曝
    camera_contrast = 1
    camera_saturation = 69

    # ---- 采集 ----
    output_root = os.path.expanduser("~/kinova_data_blur")
    robot_sample_hz = 100
    camera_sample_hz = 15              # 与相机帧率匹配
    hdf5_compression = "gzip"
    hdf5_compression_opts = 4

    # ---- 录制控制 ----
    record_button = 3       # Y
    exit_button = 7         # Menu


# ======================================================================
#  数据采集器（Blur 版）
# ======================================================================
class KinovaBlurDataCollector:
    """长曝光版本：录制带强运动模糊的图像 + 机器人状态，供 deblur 算法使用。"""

    def __init__(self, cfg: BlurConfig = None):
        self.cfg = cfg or BlurConfig()

        # ---- 遥操作 ----
        self.teleop = KinovaJoyTeleop(ip=self.cfg.robot_ip)
        self.teleop.speed_limit = self.cfg.speed_limit
        self.teleop.turn_limit = self.cfg.turn_limit
        self.teleop.deadzone = self.cfg.deadzone

        self.base = None
        self.base_cyclic = None

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

        self._prev_y = False
        self._gripper_label = "IDLE"

    # ================================================================
    #  连接
    # ================================================================
    def connect(self):
        self.teleop.connect()
        self.base = self.teleop.base
        self.base_cyclic = self.teleop.base_cyclic
        print(f"✅ 已连接 Kinova @ {self.cfg.robot_ip}")

    def connect_camera(self):
        """打开摄像头并写入长曝光参数。"""
        self.cap = cv2.VideoCapture(self.cfg.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 无法打开摄像头 /dev/video{self.cfg.camera_id}")

        # 1. 先设分辨率 / 帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)

        # 2. 强制 MJPG 编码（否则 720p 下 YUYV 只有 10fps）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # 3. 读一帧确认
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("❌ 摄像头无数据")

        # 4. 写入长曝光参数（需在出流后设置才生效）
        if self.cfg.camera_manual_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)           # 手动模式
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.cfg.camera_exposure)
            self.cap.set(cv2.CAP_PROP_GAIN, self.cfg.camera_gain)

        # 5. 其他图像参数
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.cfg.camera_brightness)
        self.cap.set(cv2.CAP_PROP_CONTRAST, self.cfg.camera_contrast)
        self.cap.set(cv2.CAP_PROP_SATURATION, self.cfg.camera_saturation)

        # 6. 读第二帧确认参数生效
        ret, frame = self.cap.read()
        actual_h, actual_w = frame.shape[:2]
        actual_fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        actual_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"📷 摄像头已打开: {actual_w}x{actual_h} @ {actual_fps_val:.0f} FPS")
        print(f"   曝光: {actual_exposure:.0f} × 100µs = {actual_exposure / 10000 * 1000:.1f} ms  |  Gain: {self.cap.get(cv2.CAP_PROP_GAIN):.1f}")

    # ================================================================
    #  后台采集线程
    # ================================================================
    def _robot_poll_thread(self):
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
        min_interval = 1.0 / self.cfg.camera_sample_hz
        last_cap = 0.0
        while self._running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._running.clear()
                break
            now = time.time()
            if now - last_cap >= min_interval:
                last_cap = now
                self._camera_queue.put_nowait({
                    "timestamp": now,
                    "frame": frame.copy(),
                })
            time.sleep(0.001)

    @staticmethod
    def _snapshot_robot_state(feedback, timestamp: float) -> dict:
        b = feedback.base
        inter = feedback.interconnect
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
        gripper_motors = inter.gripper_feedback.motor
        gripper_pos = float(gripper_motors[0].position) if gripper_motors else 0.0
        return {
            "timestamp": timestamp,
            "tool_pose": np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
                                   b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z], dtype=np.float64),
            "tool_pose_cmd": np.array([b.commanded_tool_pose_x, b.commanded_tool_pose_y, b.commanded_tool_pose_z,
                                        b.commanded_tool_pose_theta_x, b.commanded_tool_pose_theta_y,
                                        b.commanded_tool_pose_theta_z], dtype=np.float64),
            "tool_twist": np.array([b.tool_twist_linear_x, b.tool_twist_linear_y, b.tool_twist_linear_z,
                                    b.tool_twist_angular_x, b.tool_twist_angular_y, b.tool_twist_angular_z], dtype=np.float64),
            "tool_wrench": np.array([b.tool_external_wrench_force_x, b.tool_external_wrench_force_y,
                                     b.tool_external_wrench_force_z,
                                     b.tool_external_wrench_torque_x, b.tool_external_wrench_torque_y,
                                     b.tool_external_wrench_torque_z], dtype=np.float64),
            "joint_position": joint_pos, "joint_velocity": joint_vel, "joint_torque": joint_torque,
            "joint_current": joint_current, "joint_voltage": joint_voltage, "joint_temperature": joint_temp,
            "gripper_position": gripper_pos,
            "base_imu_accel": np.array([b.imu_acceleration_x, b.imu_acceleration_y, b.imu_acceleration_z], dtype=np.float64),
            "base_imu_gyro": np.array([b.imu_angular_velocity_x, b.imu_angular_velocity_y, b.imu_angular_velocity_z], dtype=np.float64),
            "arm_voltage": b.arm_voltage, "arm_current": b.arm_current,
            "temperature_cpu": b.temperature_cpu, "temperature_ambient": b.temperature_ambient,
        }

    # ================================================================
    #  游戏手柄控制（委托给 teleop）
    # ================================================================
    def _handle_gamepad(self):
        axes, hat, buttons = self.teleop.read_gamepad_state()
        if buttons.get(self.cfg.exit_button):
            return "exit", axes, hat, buttons
        y_pressed = bool(buttons.get(self.cfg.record_button))
        if y_pressed and not self._prev_y:
            if self._recording:
                self._stop_recording()
            else:
                self._start_recording()
        self._prev_y = y_pressed
        if buttons.get(0):
            self.teleop.control_gripper(1.0)
            self._gripper_label = "CLOSED"
        elif buttons.get(1):
            self.teleop.control_gripper(0.0)
            self._gripper_label = "OPENED"
        self.teleop.send_twist(axes, hat)
        return "ok", axes, hat, buttons

    # ================================================================
    #  HDF5 文件管理
    # ================================================================
    def _open_episode(self, episode: int):
        self._ensure_output_dir()
        fname = os.path.join(self._output_dir, f"episode_{episode:04d}.h5")
        f = h5py.File(fname, "w")
        for path, shape in [
            ("robot/timestamp", (0,)), ("robot/tool_pose", (0, 6)), ("robot/tool_pose_cmd", (0, 6)),
            ("robot/tool_twist", (0, 6)), ("robot/tool_wrench", (0, 6)),
            ("robot/joint_position", (0, 7)), ("robot/joint_velocity", (0, 7)), ("robot/joint_torque", (0, 7)),
            ("robot/joint_current", (0, 7)), ("robot/joint_voltage", (0, 7)), ("robot/joint_temperature", (0, 7)),
            ("robot/gripper_position", (0,)),
            ("robot/base_imu_accel", (0, 3)), ("robot/base_imu_gyro", (0, 3)),
            ("robot/arm_voltage", (0,)), ("robot/arm_current", (0,)),
            ("robot/temperature_cpu", (0,)), ("robot/temperature_ambient", (0,)),
        ]:
            f.create_dataset(path, shape=shape, maxshape=(None, *shape[1:]), dtype=np.float64,
                             compression=self.cfg.hdf5_compression)
        f.create_dataset("camera/timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
        f.create_dataset("camera/rgb", shape=(0,), maxshape=(None,),
                         dtype=h5py.special_dtype(vlen=np.dtype('uint8')),
                         compression=self.cfg.hdf5_compression, compression_opts=self.cfg.hdf5_compression_opts)
        f.attrs["episode"] = episode
        f.attrs["start_time"] = time.time()
        f.attrs["robot_ip"] = self.cfg.robot_ip
        f.attrs["speed_limit"] = self.cfg.speed_limit
        f.attrs["turn_limit"] = self.cfg.turn_limit
        f.attrs["camera_fps"] = self.cfg.camera_fps
        f.attrs["camera_exposure_100us"] = self.cfg.camera_exposure
        f.attrs["camera_gain"] = self.cfg.camera_gain
        f.attrs["robot_sample_hz"] = self.cfg.robot_sample_hz
        f.attrs["camera_sample_hz"] = self.cfg.camera_sample_hz
        self._hdf5_file = f
        print(f"\n📝 [BLUR] 开始录制 → {fname}")

    def _flush_episode(self):
        if self._hdf5_file is None:
            return
        f = self._hdf5_file
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
                ("robot/tool_pose", "tool_pose"), ("robot/tool_pose_cmd", "tool_pose_cmd"),
                ("robot/tool_twist", "tool_twist"), ("robot/tool_wrench", "tool_wrench"),
                ("robot/joint_position", "joint_position"), ("robot/joint_velocity", "joint_velocity"),
                ("robot/joint_torque", "joint_torque"), ("robot/joint_current", "joint_current"),
                ("robot/joint_voltage", "joint_voltage"), ("robot/joint_temperature", "joint_temperature"),
                ("robot/gripper_position", "gripper_position"),
                ("robot/base_imu_accel", "base_imu_accel"), ("robot/base_imu_gyro", "base_imu_gyro"),
                ("robot/arm_voltage", "arm_voltage"), ("robot/arm_current", "arm_current"),
                ("robot/temperature_cpu", "temperature_cpu"), ("robot/temperature_ambient", "temperature_ambient"),
            ]
            for h5path, key in keys:
                data = np.array([r[key] for r in robot_items])
                ds = f[h5path]
                cur = ds.shape[0]
                ds.resize((cur + n, *ds.shape[1:]))
                ds[cur:] = data
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
                ret, buf = cv2.imencode(".jpg", item["frame"], [cv2.IMWRITE_JPEG_QUALITY, 90])
                cam_bytes.append(buf.tobytes() if ret else b"")
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
        self._flush_episode()
        if self._hdf5_file is not None:
            elapsed = time.time() - self._hdf5_file.attrs["start_time"]
            robot_count = self._hdf5_file["robot/timestamp"].shape[0]
            cam_count = self._hdf5_file["camera/timestamp"].shape[0]
            self._hdf5_file.close()
            self._hdf5_file = None
            print(f"💾 Episode {self._episode:04d} 保存: "
                  f"{elapsed:.1f}s, robot={robot_count}, camera={cam_count}")

    def _ensure_output_dir(self):
        if not self._output_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self.cfg.output_root, f"blur_collection_{ts}")
        os.makedirs(self._output_dir, exist_ok=True)

    # ================================================================
    #  主循环
    # ================================================================
    def run(self):
        try:
            robot_thread = threading.Thread(target=self._robot_poll_thread, daemon=True)
            camera_thread = threading.Thread(target=self._camera_capture_thread, daemon=True)
            robot_thread.start()
            camera_thread.start()
            print("🚀 后台采集线程已启动")
            rec_label = "■ IDLE"
            display_counter = 0
            while self._running.is_set():
                status, axes, hat, buttons = self._handle_gamepad()
                if status == "exit":
                    print("\n⏹ 退出程序...")
                    break
                if self._recording:
                    display_counter += 1
                    if display_counter % 5 == 0:
                        self._flush_episode()
                else:
                    self._drain_queues()
                display_counter += 1
                if display_counter % 5 == 0:
                    rec_label = "● REC (BLUR)" if self._recording else "■ IDLE"
                    self._print_status(axes, hat, rec_label)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _start_recording(self):
        self._episode += 1
        self._drain_queues()
        self._open_episode(self._episode)
        self._recording = True

    def _stop_recording(self):
        self._recording = False
        self._close_episode()

    def _drain_queues(self):
        for q in (self._robot_queue, self._camera_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _print_status(self, axes, hat, rec_label):
        a0, a1, _, a3, a4, _ = axes
        joy_str = (f"X:{a1:+5.2f} Y:{a0:+5.2f} Z:{axes[5]-axes[2]:+5.2f} | "
                   f"R:{a3:+5.2f} P:{a4:+5.2f} Y:{hat[0]:+2.0f} | Grip:{self._gripper_label}")
        sys.stdout.write(f"\r{rec_label}  {joy_str}  "
                         f"| Q:{self._robot_queue.qsize():<4}/{self._robot_queue.maxsize} "
                         f"CamQ:{self._camera_queue.qsize():<4}/{self._camera_queue.maxsize}     ")
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
        if self.teleop.connection:
            try:
                self.teleop.connection.__exit__(None, None, None)
            except Exception:
                pass
        if self.cap:
            self.cap.release()
        pygame.quit()
        cv2.destroyAllWindows()
        print("\n👋 安全退出 (Blur Collector)")
        if self._output_dir:
            print(f"\n📁 数据保存位置:\n   {self._output_dir}")


# ======================================================================
#  Entry
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Kinova Gen3 模糊数据采集器 (Blur + Teleop)")
    print("=" * 60)
    print()
    print("  控制映射:")
    print("    左摇杆 → XY 平移    右摇杆 → Roll / Pitch")
    print("    LT/RT  → Z 轴升降   十字键 → Yaw")
    print("    A 关夹爪  B 开夹爪   Y → 开始/停止录制")
    print("    Menu 退出")
    print(f"  📷 曝光: {BlurConfig.camera_exposure * 100} µs  |  Gain: {BlurConfig.camera_gain}")
    print()

    collector = KinovaBlurDataCollector()
    try:
        collector.connect()
        collector.connect_camera()
        collector.run()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
