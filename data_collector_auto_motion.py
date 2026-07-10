#!/usr/bin/env python3
"""
Kinova Gen3 自动快速横向运动采集脚本（训练用格式）
=====================================================
不依赖手柄，自动执行预设的快速短距离横向运动。
输出格式与 data_collector.py 一致：obs/action 对。

安全说明：
  - 机械臂会从当前位置开始运动
  - 请确保运动空间无遮挡
  - 按 Ctrl+C 随时中断

用法:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 data_collector_auto_motion.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Kinova_kortex2_Gen3_G3L", "api_python", "examples"
))
import utilities


# ======================================================================
#  配置
# ======================================================================
class AutoMotionConfig:
    robot_ip = "192.168.8.10"
    motion_direction = 'x'
    motion_speed = 0.40
    motion_duration = 0.25
    pre_delay = 0.5
    post_delay = 0.5
    record_duration = 3.0
    repeat_count = 5
    repeat_interval = 3.0

    # 摄像头
    camera_id = 0
    camera_width = 1280
    camera_height = 720
    camera_fps = 15
    camera_manual_exposure = True
    camera_exposure = 100
    camera_gain = 1
    camera_brightness = 0
    camera_contrast = 1
    camera_saturation = 69

    output_root = os.path.expanduser("~/kinova_data_automotion")
    robot_sample_hz = 100
    action_hz = 15
    hdf5_compression = "gzip"
    hdf5_compression_opts = 4


# ======================================================================
#  自动运动采集器（训练用格式）
# ======================================================================
class AutoMotionTrainCollector:
    def __init__(self, cfg: AutoMotionConfig = None):
        self.cfg = cfg or AutoMotionConfig()
        self.base = None
        self.base_cyclic = None
        self.router = None
        self.connection = None
        self.cap = None

        self._robot_q = queue.Queue(maxsize=5000)
        self._camera_q = queue.Queue(maxsize=2000)
        self._action_q = queue.Queue(maxsize=5000)
        self._running = threading.Event()
        self._running.set()
        self._episode = 0
        self._task_counter = 0
        self._hdf5_file = None
        self._output_dir = ""
        self._prev_pose = None

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
            raise RuntimeError(f"❌ 摄像头 /dev/video{self.cfg.camera_id} 打开失败")
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
        print(f"📷 摄像头: {w}x{h} @ {self.cfg.camera_fps}fps, "
              f"曝光={self.cfg.camera_exposure}×100µs, gain={self.cfg.camera_gain}")

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

    def _action_gen_thread(self):
        """
        为自动运动生成预设动作序列：
        前 pre_delay 秒无动作 → motion_duration 秒运动 → post_delay 秒无动作
        """
        period = 1.0 / self.cfg.robot_sample_hz
        pre_steps = int(self.cfg.pre_delay / period)
        motion_steps = int(self.cfg.motion_duration / period)
        post_steps = int(self.cfg.post_delay / period)

        vx = self.cfg.motion_speed if 'x' in self.cfg.motion_direction else 0.0
        vy = self.cfg.motion_speed if 'y' in self.cfg.motion_direction else 0.0

        step = 0
        total = pre_steps + motion_steps + post_steps
        while self._running.is_set() and step < total:
            t0 = time.perf_counter()
            if step < pre_steps:
                twist = np.zeros(6, dtype=np.float64)
            elif step < pre_steps + motion_steps:
                twist = np.array([vx, vy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                twist = np.zeros(6, dtype=np.float64)
            self._action_q.put_nowait({
                "timestamp": time.time(),
                "twist": twist,
                "gripper_cmd": np.array([0.0], dtype=np.float64),
            })
            step += 1
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

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

    # ---- 运动执行 ----
    def _send_twist(self, twist):
        cmd = Base_pb2.TwistCommand()
        cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        cmd.duration = 0
        cmd.twist.linear_x = twist[0]
        cmd.twist.linear_y = twist[1]
        cmd.twist.linear_z = twist[2]
        cmd.twist.angular_x = twist[3]
        cmd.twist.angular_y = twist[4]
        cmd.twist.angular_z = twist[5]
        self.base.SendTwistCommand(cmd)

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
        task_desc = self.cfg.task if self.cfg.task else f"auto_motion_{self.cfg.motion_direction}"

        f.attrs["episode"] = episode
        f.attrs["task/id"] = self._task_counter
        f.attrs["task/language_instruction"] = task_desc
        f.attrs["dataset_name"] = "kinova_gen3_automotion"
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
        f.attrs["motion_direction"] = self.cfg.motion_direction
        f.attrs["motion_speed"] = self.cfg.motion_speed
        f.attrs["motion_duration_s"] = self.cfg.motion_duration
        f.attrs["action_hz"] = self.cfg.action_hz
        f.attrs["date_collected"] = __import__('datetime').datetime.now().isoformat()
        self._hdf5_file = f
        self._prev_pose = None
        print(f"\n📝 [AUTO] 开始录制 → {fname}")
        print(f"   🏷️  task: {task_desc}")

    def _sync_and_write(self):
        if self._hdf5_file is None:
            return
        f = self._hdf5_file

        def drain(q):
            items = []
            while not q.empty():
                try:
                    items.append(q.get_nowait())
                except queue.Empty:
                    break
            return items

        robot_items, camera_items, action_items = drain(self._robot_q), drain(self._camera_q), drain(self._action_q)
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
            print(f"💾 [AUTO] Episode {self._episode:04d}: {n} steps, {elapsed:.1f}s")

    def _ensure_output_dir(self):
        if not self._output_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self.cfg.output_root, f"automotion_train_{ts}")
        os.makedirs(self._output_dir, exist_ok=True)

    # ---- 主流程 ----
    def run(self):
        try:
            robot_thread = threading.Thread(target=self._robot_poll_thread, daemon=True)
            camera_thread = threading.Thread(target=self._camera_capture_thread, daemon=True)
            robot_thread.start()
            camera_thread.start()
            print("🚀 后台线程已启动")
            time.sleep(0.5)

            vx_str = f"vx={self.cfg.motion_speed}" if 'x' in self.cfg.motion_direction else ""
            vy_str = f"vy={self.cfg.motion_speed}" if 'y' in self.cfg.motion_direction else ""
            speed_str = " + ".join(s for s in [vx_str, vy_str] if s)

            print(f"\n🔁 {self.cfg.repeat_count} 轮, 方向 {self.cfg.motion_direction.upper()}")
            print(f"   速度: {speed_str} m/s  |  持续 {self.cfg.motion_duration*1000:.0f}ms")
            print(f"   间隔 {self.cfg.repeat_interval}s  |  Ctrl+C 中断\n")

            for repeat in range(1, self.cfg.repeat_count + 1):
                if not self._running.is_set():
                    break
                print(f"\n{'='*50}")
                print(f"  第 {repeat}/{self.cfg.repeat_count} 轮")

                # 启动动作线程
                action_thread = threading.Thread(target=self._action_gen_thread, daemon=True)
                action_thread.start()

                self._episode += 1
                self._prev_pose = None
                self._open_episode(self._episode)

                # 动态跟随动作线程
                total_dur = self.cfg.pre_delay + self.cfg.motion_duration + self.cfg.post_delay
                poll_period = 0.05  # 50ms
                poll_steps = int(total_dur / poll_period)

                for _ in range(poll_steps):
                    if not self._running.is_set() or not action_thread.is_alive():
                        break
                    # 从动作缓冲中消费最新的 twist
                    a_items = []
                    while not self._action_q.empty():
                        try:
                            a_items.append(self._action_q.get_nowait())
                        except queue.Empty:
                            break
                    if a_items:
                        latest = a_items[-1]["twist"]
                        self._send_twist(latest)
                        # 将未消费的动作放回（写 episode 时要用）
                        for item in a_items:
                            self._action_q.put_nowait(item)
                    self._sync_and_write()
                    time.sleep(poll_period)

                self._send_twist(np.zeros(6))
                self._close_episode()

                if repeat < self.cfg.repeat_count:
                    print(f"   等待 {self.cfg.repeat_interval:.0f}s...")
                    time.sleep(self.cfg.repeat_interval)

            print(f"\n✅ 全部 {self.cfg.repeat_count} 轮完成")

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _cleanup(self):
        self._running.clear()
        if self._hdf5_file is not None:
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
        cv2.destroyAllWindows()
        print("👋 安全退出")
        if self._output_dir:
            print(f"📁 数据位置: {self._output_dir}")


# ======================================================================
#  Entry
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Kinova Gen3 自动运动采集 (训练格式)")
    print("=" * 60)
    cfg = AutoMotionConfig()
    vx = f"vx={cfg.motion_speed}" if 'x' in cfg.motion_direction else ""
    vy = f"vy={cfg.motion_speed}" if 'y' in cfg.motion_direction else ""
    print(f"  运动: {cfg.motion_direction.upper()} ({vx} {vy}), {cfg.motion_duration*1000:.0f}ms")
    print(f"  重复: {cfg.repeat_count} 轮, 间隔 {cfg.repeat_interval}s")
    print()

    collector = AutoMotionTrainCollector()
    try:
        collector.connect()
        collector.connect_camera()
        collector.run()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
