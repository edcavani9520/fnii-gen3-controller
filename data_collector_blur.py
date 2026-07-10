#!/usr/bin/env python3
"""
Kinova Gen3 模糊数据采集脚本（训练用格式）
=============================================
基于 gamepad 遥操作 + 长曝光相机产生运动模糊。
输出格式与 data_collector.py 一致：obs/action 对。

控制映射（与 data_collector.py 一致）：
  左摇杆        → XY 平移
  LT/RT         → Z 轴升降
  右摇杆        → Roll / Pitch
  十字键左右    → Yaw
  A / B         → 夹爪关/开
  Y             → 开始/停止 录制 (toggle)
  Menu(按钮7)   → 退出程序

用法:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 data_collector_blur.py
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

import numpy as np
import cv2
import h5py
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamepad_control import KinovaJoyTeleop
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2


# ======================================================================
#  配置
# ======================================================================
class BlurConfig:
    robot_ip = "192.168.8.10"
    speed_limit = 0.20
    turn_limit = 20.0
    deadzone = 0.1

    # 摄像头 — 长曝光制造模糊
    camera_id = 0
    camera_width = 1280
    camera_height = 720
    camera_fps = 15
    camera_manual_exposure = True
    camera_exposure = 100        # 1ms
    camera_gain = 1
    camera_brightness = 0
    camera_contrast = 1
    camera_saturation = 69

    output_root = os.path.expanduser("~/kinova_data_blur")
    robot_sample_hz = 100
    action_hz = 15               # 与 camera fps 对齐
    hdf5_compression = "gzip"
    hdf5_compression_opts = 4

    task = ""        # 命令行通过 --task 指定
    task_id = 0

    record_button = 3   # Y
    exit_button = 7     # Menu


# ======================================================================
#  采集器 — Blur 版（输出训练用格式）
# ======================================================================
class KinovaBlurTrainCollector:
    def __init__(self, cfg: BlurConfig = None):
        self.cfg = cfg or BlurConfig()
        self.teleop = KinovaJoyTeleop(ip=self.cfg.robot_ip)
        self.teleop.speed_limit = self.cfg.speed_limit
        self.teleop.turn_limit = self.cfg.turn_limit
        self.teleop.deadzone = self.cfg.deadzone
        self.base = None
        self.base_cyclic = None
        self.cap: Optional[cv2.VideoCapture] = None

        self._robot_q = queue.Queue(maxsize=5000)
        self._camera_q = queue.Queue(maxsize=2000)
        self._action_q = queue.Queue(maxsize=5000)
        self._recording = False
        self._episode = 0
        self._output_dir = ""
        self._hdf5_file = None
        self._running = threading.Event()
        self._running.set()
        self._prev_y = False
        self._gripper_label = "IDLE"
        self._last_gripper_cmd = 0.0
        self._prev_pose = None

    def connect(self):
        self.teleop.connect()
        self.base = self.teleop.base
        self.base_cyclic = self.teleop.base_cyclic
        print(f"✅ 已连接 Kinova @ {self.cfg.robot_ip}")

    def connect_camera(self):
        self.cap = cv2.VideoCapture(self.cfg.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 无法打开摄像头 /dev/video{self.cfg.camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("❌ 摄像头无数据")
        if self.cfg.camera_manual_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.cfg.camera_exposure)
            self.cap.set(cv2.CAP_PROP_GAIN, self.cfg.camera_gain)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.cfg.camera_brightness)
        self.cap.set(cv2.CAP_PROP_CONTRAST, self.cfg.camera_contrast)
        self.cap.set(cv2.CAP_PROP_SATURATION, self.cfg.camera_saturation)
        ret, frame = self.cap.read()
        h, w = frame.shape[:2]
        print(f"📷 摄像头已打开: {w}x{h} @ {self.cfg.camera_fps} FPS")
        print(f"   曝光: {self.cfg.camera_exposure * 100} µs  |  Gain: {self.cfg.camera_gain}")

    # ---- 后台线程 ----
    def _robot_poll_thread(self):
        period = 1.0 / self.cfg.robot_sample_hz
        while self._running.is_set():
            t0 = time.perf_counter()
            try:
                fb = self.base_cyclic.RefreshFeedback()
                self._robot_q.put_nowait(self._snapshot_robot(fb, time.time()))
            except Exception:
                pass
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _action_poll_thread(self):
        period = 1.0 / self.cfg.robot_sample_hz
        while self._running.is_set():
            t0 = time.perf_counter()
            try:
                axes, hat, buttons = self._read_gamepad()
                self._action_q.put_nowait(
                    self._snapshot_action(axes, hat, buttons, time.time()))
            except Exception:
                pass
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _camera_capture_thread(self):
        interval = 1.0 / self.cfg.camera_fps
        last = 0.0
        while self._running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._running.clear()
                break
            now = time.time()
            if now - last >= interval:
                last = now
                self._camera_q.put_nowait({"timestamp": now, "frame": frame.copy()})
            time.sleep(0.001)

    # ---- 数据快照 ----
    @staticmethod
    def _snapshot_robot(fb, ts):
        b, inter = fb.base, fb.interconnect
        n = len(fb.actuators)
        jp, jv, jt = np.zeros(n), np.zeros(n), np.zeros(n)
        for i, a in enumerate(fb.actuators):
            jp[i], jv[i], jt[i] = a.position, a.velocity, a.torque
        gm = inter.gripper_feedback.motor
        gp = float(gm[0].position) if gm else 0.0
        return {
            "timestamp": ts,
            "eef_pose": np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
                                   b.tool_pose_theta_x, b.tool_pose_theta_y,
                                   b.tool_pose_theta_z], dtype=np.float64),
            "joint_pos": jp, "joint_vel": jv, "joint_torque": jt,
            "gripper_pos": gp,
        }

    def _read_gamepad(self):
        axes, hat, buttons = self.teleop.read_gamepad_state()
        return axes, hat, buttons

    def _snapshot_action(self, axes, hat, buttons, ts):
        a0, a1, a2, a3, a4, a5 = axes
        twist = np.array([
            -a1 * self.cfg.speed_limit, -a0 * self.cfg.speed_limit,
            (a5 - a2) * self.cfg.speed_limit,
            a3 * self.cfg.turn_limit, -a4 * self.cfg.turn_limit,
            -hat[0] * self.cfg.turn_limit,
        ], dtype=np.float64)
        gc = self._last_gripper_cmd
        if buttons.get(0):
            gc = 1.0
        elif buttons.get(1):
            gc = 0.0
        self._last_gripper_cmd = gc
        return {"timestamp": ts, "twist": twist, "gripper_cmd": np.array([gc], dtype=np.float64)}

    # ---- HDF5 输出（训练用格式） ----
    def _open_episode(self, episode):
        self._ensure_output_dir()
        fname = os.path.join(self._output_dir, f"episode_{episode:04d}.h5")
        f = h5py.File(fname, "w")
        grp_obs = f.create_group("obs")
        grp_obs.create_dataset("camera_0", shape=(0, self.cfg.camera_height, self.cfg.camera_width, 3),
                                maxshape=(None, self.cfg.camera_height, self.cfg.camera_width, 3),
                                dtype=np.uint8, compression=self.cfg.hdf5_compression,
                                compression_opts=self.cfg.hdf5_compression_opts)
        for name, dim in [("joint_pos", 7), ("joint_vel", 7), ("eef_pose", 6), ("gripper_pos", 1)]:
            grp_obs.create_dataset(name, shape=(0, dim), maxshape=(None, dim), dtype=np.float64,
                                   compression=self.cfg.hdf5_compression)
        grp_act = f.create_group("action")
        for name, dim in [("eef_delta", 6), ("gripper", 1), ("raw_twist", 6)]:
            grp_act.create_dataset(name, shape=(0, dim), maxshape=(None, dim), dtype=np.float64,
                                   compression=self.cfg.hdf5_compression)
        f.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)
        task_desc = self.cfg.task if self.cfg.task else self._pending_task_desc or "unnamed"
        self._pending_task_desc = None

        f.attrs["episode"] = episode
        f.attrs["task/id"] = self.cfg.task_id
        f.attrs["task/language_instruction"] = task_desc
        f.attrs["dataset_name"] = "kinova_gen3_blur"
        f.attrs["robot_type"] = "Kinova Gen3"
        f.attrs["control_mode"] = "twist_velocity"
        f.attrs["camera_names"] = "['cam_0']"
        f.attrs["start_time"] = time.time()
        f.attrs["robot_ip"] = self.cfg.robot_ip
        f.attrs["camera_fps"] = self.cfg.camera_fps
        f.attrs["camera_width"] = self.cfg.camera_width
        f.attrs["camera_height"] = self.cfg.camera_height
        f.attrs["camera_exposure_100us"] = self.cfg.camera_exposure
        f.attrs["camera_gain"] = self.cfg.camera_gain
        f.attrs["action_hz"] = self.cfg.action_hz
        f.attrs["robot_sample_hz"] = self.cfg.robot_sample_hz
        f.attrs["date_collected"] = datetime.datetime.now().isoformat()
        self._hdf5_file = f
        self._prev_pose = None
        print(f"\n📝 [BLUR] 开始录制 → {fname}")
        print(f"   🏷️  task: {task_desc}")

    def _sync_and_write(self):
        if self._hdf5_file is None:
            return
        f = self._hdf5_file
        robot_items, camera_items, action_items = self._drain_all()

        if not camera_items:
            return

        cam_ts = np.array([c["timestamp"] for c in camera_items])
        robot_ts = np.array([r["timestamp"] for r in robot_items])
        action_ts = np.array([a["timestamp"] for a in action_items])

        for cam_item in camera_items:
            cam_t = cam_item["timestamp"]
            ri = np.argmin(np.abs(robot_ts - cam_t))
            ai = np.argmin(np.abs(action_ts - cam_t))
            r = robot_items[min(ri, len(robot_items)-1)]
            a = action_items[min(ai, len(action_items)-1)]

            obs_pose = r["eef_pose"]
            if self._prev_pose is not None:
                eef_delta = obs_pose - self._prev_pose
            else:
                eef_delta = np.zeros(6, dtype=np.float64)
            self._prev_pose = obs_pose.copy()

            obs_gp = np.array([r["gripper_pos"]], dtype=np.float64)
            gripper_prev = (f["obs/gripper_pos"][-1] if f["obs/gripper_pos"].shape[0] > 0
                           else np.array([0.0]))
            gripper_act = np.clip(np.array([obs_gp[0] - gripper_prev[0]]), -1.0, 1.0)

            cur = f["timestamps"].shape[0]
            for ds_name in ["timestamps", "obs/camera_0", "obs/joint_pos", "obs/joint_vel",
                            "obs/eef_pose", "obs/gripper_pos",
                            "action/eef_delta", "action/gripper", "action/raw_twist"]:
                f[ds_name].resize((cur + 1, *f[ds_name].shape[1:]))

            f["timestamps"][cur] = cam_t
            frame_rgb = cv2.cvtColor(cam_item["frame"], cv2.COLOR_BGR2RGB)
            f["obs/camera_0"][cur] = frame_rgb
            f["obs/joint_pos"][cur] = r["joint_pos"]
            f["obs/joint_vel"][cur] = r["joint_vel"]
            f["obs/eef_pose"][cur] = obs_pose
            f["obs/gripper_pos"][cur] = obs_gp
            f["action/eef_delta"][cur] = eef_delta
            f["action/gripper"][cur] = gripper_act
            f["action/raw_twist"][cur] = a["twist"]
        f.flush()

    def _close_episode(self):
        self._sync_and_write()
        if self._hdf5_file is not None:
            elapsed = time.time() - self._hdf5_file.attrs["start_time"]
            n = self._hdf5_file["timestamps"].shape[0]
            self._hdf5_file.close()
            self._hdf5_file = None
            print(f"💾 [BLUR] Episode {self._episode:04d}: {n} steps, {elapsed:.1f}s")

    def _ensure_output_dir(self):
        if not self._output_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self.cfg.output_root, f"blur_train_{ts}")
        os.makedirs(self._output_dir, exist_ok=True)

    def _drain_all(self):
        def drain(q):
            items = []
            while not q.empty():
                try:
                    items.append(q.get_nowait())
                except queue.Empty:
                    break
            return items
        return drain(self._robot_q), drain(self._camera_q), drain(self._action_q)

    # ---- 主循环 ----
    def run(self):
        try:
            threads = [
                threading.Thread(target=self._robot_poll_thread, daemon=True),
                threading.Thread(target=self._camera_capture_thread, daemon=True),
                threading.Thread(target=self._action_poll_thread, daemon=True),
            ]
            for t in threads:
                t.start()
            print("🚀 后台采集线程已启动")
            display_counter = 0

            while self._running.is_set():
                axes, hat, buttons = self._read_gamepad()
                if buttons.get(self.cfg.exit_button):
                    print("\n⏹ 退出...")
                    break
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

                if self._recording:
                    display_counter += 1
                    if display_counter % 5 == 0:
                        self._sync_and_write()
                else:
                    self._drain_all()
                    self._prev_pose = None

                display_counter += 1
                if display_counter % 5 == 0:
                    label = "● BLUR REC" if self._recording else "■ IDLE"
                    a0, a1, _, a3, a4, _ = axes
                    joy_str = (f"X:{a1:+5.2f} Y:{a0:+5.2f} Z:{axes[5]-axes[2]:+5.2f} | "
                               f"R:{a3:+5.2f} P:{a4:+5.2f} Y:{hat[0]:+2.0f} | "
                               f"Grip:{self._gripper_label}")
                    sys.stdout.write(f"\r{label}  {joy_str}  "
                                     f"| Q:{self._robot_q.qsize():<4} "
                                     f"CamQ:{self._camera_q.qsize():<4}     ")
                    sys.stdout.flush()
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    # 暂存交互输入的任务描述
    _pending_task_desc = None
    _task_counter = 0

    def _start_recording(self):
        self._episode += 1
        self._task_counter += 1
        self.cfg.task_id = self._task_counter

        if not self.cfg.task and self._task_counter == 1:
            if self.base:
                try: self.base.Stop()
                except: pass
            print()
            desc = input(f"  📋 任务描述 (所有 episode 共用): ").strip()
            self.cfg.task = desc if desc else f"task_{self._task_counter:03d}"

        self._pending_task_desc = None
        self._drain_all()
        self._prev_pose = None
        self._open_episode(self._episode)
        self._recording = True

    def _stop_recording(self):
        self._recording = False
        self._close_episode()

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
        print("\n👋 安全退出 (Blur)")
        if self._output_dir:
            print(f"📁 数据位置: {self._output_dir}")


# ======================================================================
#  Entry
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Kinova Gen3 模糊数据采集 (训练格式)")
    print("=" * 60)
    print(f"  📷 曝光: {BlurConfig.camera_exposure * 100} µs  Gain: {BlurConfig.camera_gain}")
    print()

    collector = KinovaBlurTrainCollector()
    try:
        collector.connect()
        collector.connect_camera()
        collector.run()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
