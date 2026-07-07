#!/usr/bin/env python3
"""
Kinova Gen3 自动快速横向运动采集脚本（无手柄，用于 Deblur 测试）
==============================================================
不依赖手柄，自动执行预设的快速短距离横向运动，同步录制图像 + 机器人状态。

运动模式：
  - 沿 X 轴正方向快速移动一段短距离
  - 高速低时长，保证画面产生运动模糊

用法:
  python3 data_collector_auto_motion.py

安全说明：
  - 机械臂会从当前位置开始运动
  - 请确保运动空间无遮挡
  - 按 Ctrl+C 随时中断
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
    # ---- 机械臂 ----
    robot_ip = "192.168.8.10"

    # ---- 运动参数 ----
    # 运动方向: 'x', 'y', 'xy'（同时 X+Y）
    motion_direction = 'x'
    # 平移速度 (m/s) — 越大模糊越强
    motion_speed = 0.40
    # 运动时长 (秒)
    motion_duration = 0.25
    # 运动前等待 (秒)，给系统稳定时间
    pre_delay = 0.5
    # 运动后等待 (秒)，捕捉完整模糊轨迹
    post_delay = 0.5
    # 每轮总录制时长 (秒)
    record_duration = 3.0

    # ---- 运动次数 ----
    repeat_count = 5
    repeat_interval = 3.0          # 轮次间隔（秒）

    # ---- 摄像头（与 blur 版一致） ----
    camera_id = 0
    camera_width = 1280
    camera_height = 720
    camera_fps = 15
    camera_manual_exposure = True
    camera_exposure = 100          # 1ms
    camera_gain = 1
    camera_brightness = 0
    camera_contrast = 1
    camera_saturation = 69

    # ---- 采集 ----
    output_root = os.path.expanduser("~/kinova_data_automotion")
    robot_sample_hz = 100
    camera_sample_hz = 15
    hdf5_compression = "gzip"
    hdf5_compression_opts = 4


# ======================================================================
#  自动运动采集器
# ======================================================================
class AutoMotionCollector:
    def __init__(self, cfg: AutoMotionConfig = None):
        self.cfg = cfg or AutoMotionConfig()

        # ---- 机械臂连接 ----
        self.base: Optional[BaseClient] = None
        self.base_cyclic: Optional[BaseCyclicClient] = None
        self.router = None
        self.connection = None

        # ---- 摄像头 ----
        self.cap: Optional[cv2.VideoCapture] = None

        # ---- 采集缓冲 ----
        self._robot_queue: queue.Queue = queue.Queue(maxsize=5000)
        self._camera_queue: queue.Queue = queue.Queue(maxsize=2000)
        self._running = threading.Event()
        self._running.set()
        self._episode = 0
        self._hdf5_file: Optional[h5py.File] = None
        self._output_dir = ""

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

    # ================================================================
    #  后台线程
    # ================================================================
    def _robot_poll_thread(self):
        period = 1.0 / self.cfg.robot_sample_hz
        while self._running.is_set():
            t0 = time.perf_counter()
            try:
                fb = self.base_cyclic.RefreshFeedback()
                self._robot_queue.put_nowait(self._snapshot(fb, time.time()))
            except Exception:
                pass
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _camera_capture_thread(self):
        interval = 1.0 / self.cfg.camera_sample_hz
        last = 0.0
        while self._running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._running.clear()
                break
            now = time.time()
            if now - last >= interval:
                last = now
                self._camera_queue.put_nowait({"timestamp": now, "frame": frame.copy()})
            time.sleep(0.001)

    @staticmethod
    def _snapshot(fb, ts):
        b, inter = fb.base, fb.interconnect
        n = len(fb.actuators)
        jp = np.zeros(n); jv = np.zeros(n); jt = np.zeros(n)
        jc = np.zeros(n); jvl = np.zeros(n); jtm = np.zeros(n)
        for i, a in enumerate(fb.actuators):
            jp[i]=a.position; jv[i]=a.velocity; jt[i]=a.torque
            jc[i]=a.current_motor; jvl[i]=a.voltage; jtm[i]=a.temperature_core
        gm = inter.gripper_feedback.motor
        gp = float(gm[0].position) if gm else 0.0
        return {
            "timestamp": ts,
            "tool_pose": np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
                                   b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z]),
            "tool_twist": np.array([b.tool_twist_linear_x, b.tool_twist_linear_y, b.tool_twist_linear_z,
                                    b.tool_twist_angular_x, b.tool_twist_angular_y, b.tool_twist_angular_z]),
            "tool_wrench": np.array([b.tool_external_wrench_force_x, b.tool_external_wrench_force_y,
                                     b.tool_external_wrench_force_z,
                                     b.tool_external_wrench_torque_x, b.tool_external_wrench_torque_y,
                                     b.tool_external_wrench_torque_z]),
            "joint_position": jp, "joint_velocity": jv, "joint_torque": jt,
            "joint_current": jc, "joint_voltage": jvl, "joint_temperature": jtm,
            "gripper_position": gp,
            "arm_voltage": b.arm_voltage, "arm_current": b.arm_current,
            "temperature_cpu": b.temperature_cpu, "temperature_ambient": b.temperature_ambient,
        }

    # ================================================================
    #  运动执行
    # ================================================================
    def _send_twist(self, vx=0.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0):
        cmd = Base_pb2.TwistCommand()
        cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        cmd.duration = 0
        cmd.twist.linear_x = vx
        cmd.twist.linear_y = vy
        cmd.twist.linear_z = vz
        cmd.twist.angular_x = wx
        cmd.twist.angular_y = wy
        cmd.twist.angular_z = wz
        self.base.SendTwistCommand(cmd)

    def _stop_arm(self):
        self.base.Stop()

    def execute_burst(self):
        """执行一次快速横向运动。"""
        cfg = self.cfg
        vx, vy = 0.0, 0.0

        if 'x' in cfg.motion_direction:
            vx = cfg.motion_speed
        if 'y' in cfg.motion_direction:
            vy = cfg.motion_speed

        print(f"    🚀 快速移动 → vx={vx:.2f} vy={vy:.2f} m/s, 持续 {cfg.motion_duration*1000:.0f}ms")
        self._send_twist(vx=vx, vy=vy)
        time.sleep(cfg.motion_duration)
        self._stop_arm()
        print(f"    🛑 停止")

    # ================================================================
    #  HDF5
    # ================================================================
    def _open_episode(self, episode):
        self._ensure_output_dir()
        fname = os.path.join(self._output_dir, f"episode_{episode:04d}.h5")
        f = h5py.File(fname, "w")
        for path, shape in [
            ("robot/timestamp", (0,)), ("robot/tool_pose", (0, 6)),
            ("robot/tool_twist", (0, 6)), ("robot/tool_wrench", (0, 6)),
            ("robot/joint_position", (0, 7)), ("robot/joint_velocity", (0, 7)),
            ("robot/joint_torque", (0, 7)), ("robot/joint_current", (0, 7)),
            ("robot/joint_voltage", (0, 7)), ("robot/joint_temperature", (0, 7)),
            ("robot/gripper_position", (0,)),
            ("robot/arm_voltage", (0,)), ("robot/arm_current", (0,)),
            ("robot/temperature_cpu", (0,)), ("robot/temperature_ambient", (0,)),
        ]:
            f.create_dataset(path, shape=shape, maxshape=(None, *shape[1:]),
                             dtype=np.float64, compression=self.cfg.hdf5_compression)
        f.create_dataset("camera/timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
        f.create_dataset("camera/rgb", shape=(0,), maxshape=(None,),
                         dtype=h5py.special_dtype(vlen=np.dtype('uint8')),
                         compression=self.cfg.hdf5_compression, compression_opts=self.cfg.hdf5_compression_opts)
        f.attrs["episode"] = episode
        f.attrs["start_time"] = time.time()
        f.attrs["robot_ip"] = self.cfg.robot_ip
        f.attrs["motion_direction"] = self.cfg.motion_direction
        f.attrs["motion_speed"] = self.cfg.motion_speed
        f.attrs["motion_duration_s"] = self.cfg.motion_duration
        f.attrs["camera_exposure_100us"] = self.cfg.camera_exposure
        f.attrs["camera_gain"] = self.cfg.camera_gain
        f.attrs["camera_fps"] = self.cfg.camera_fps
        print(f"\n📝 开始录制 → {fname}")
        self._hdf5_file = f

    def _flush(self):
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
            pairs = [
                ("robot/timestamp", "timestamp"), ("robot/tool_pose", "tool_pose"),
                ("robot/tool_twist", "tool_twist"), ("robot/tool_wrench", "tool_wrench"),
                ("robot/joint_position", "joint_position"), ("robot/joint_velocity", "joint_velocity"),
                ("robot/joint_torque", "joint_torque"), ("robot/joint_current", "joint_current"),
                ("robot/joint_voltage", "joint_voltage"), ("robot/joint_temperature", "joint_temperature"),
                ("robot/gripper_position", "gripper_position"),
                ("robot/arm_voltage", "arm_voltage"), ("robot/arm_current", "arm_current"),
                ("robot/temperature_cpu", "temperature_cpu"), ("robot/temperature_ambient", "temperature_ambient"),
            ]
            for h5p, key in pairs:
                data = np.array([r[key] for r in robot_items])
                ds = f[h5p]
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
            ts_list, img_list = [], []
            for item in camera_items:
                ts_list.append(item["timestamp"])
                ret, buf = cv2.imencode(".jpg", item["frame"], [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_list.append(buf.tobytes() if ret else b"")
            ds_ts = f["camera/timestamp"]
            ds_img = f["camera/rgb"]
            cur = ds_ts.shape[0]
            ds_ts.resize((cur + n,)); ds_ts[cur:] = np.array(ts_list)
            ds_img.resize((cur + n,))
            for i, b in enumerate(img_list):
                ds_img[cur + i] = np.frombuffer(b, dtype=np.uint8)
        f.flush()

    def _close_episode(self):
        self._flush()
        if self._hdf5_file is not None:
            elapsed = time.time() - self._hdf5_file.attrs["start_time"]
            rc = self._hdf5_file["robot/timestamp"].shape[0]
            cc = self._hdf5_file["camera/timestamp"].shape[0]
            self._hdf5_file.close()
            self._hdf5_file = None
            print(f"💾 Episode 保存: {elapsed:.1f}s, robot={rc}, camera={cc}")

    def _ensure_output_dir(self):
        if not self._output_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self.cfg.output_root, f"automotion_{ts}")
        os.makedirs(self._output_dir, exist_ok=True)

    # ================================================================
    #  主流程
    # ================================================================
    def run(self):
        try:
            robot_thread = threading.Thread(target=self._robot_poll_thread, daemon=True)
            camera_thread = threading.Thread(target=self._camera_capture_thread, daemon=True)
            robot_thread.start()
            camera_thread.start()
            print("🚀 后台采集线程已启动")
            time.sleep(0.5)

            print(f"\n🔁 将执行 {self.cfg.repeat_count} 轮快速运动")
            print(f"   方向: {self.cfg.motion_direction.upper()}  速度: {self.cfg.motion_speed} m/s"
                  f"  时长: {self.cfg.motion_duration*1000:.0f}ms")
            print(f"   每轮录制 {self.cfg.record_duration:.0f}s, 间隔 {self.cfg.repeat_interval:.0f}s")
            print("   ⚠️ 确保运动空间无遮挡, Ctrl+C 随时中断\n")

            for repeat in range(1, self.cfg.repeat_count + 1):
                if not self._running.is_set():
                    break
                print(f"\n{'='*50}")
                print(f"  第 {repeat}/{self.cfg.repeat_count} 轮")

                # 打开新 episode
                self._episode += 1
                self._drain_queues()
                self._open_episode(self._episode)

                # 稳定期
                time.sleep(self.cfg.pre_delay)

                # 执行快速运动
                self.execute_burst()

                # 等待捕捉完整模糊轨迹
                time.sleep(self.cfg.post_delay)

                # 关闭 episode
                self._close_episode()

                # 轮次间隔
                if repeat < self.cfg.repeat_count:
                    print(f"   等待 {self.cfg.repeat_interval:.0f}s 后下一轮...")
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

    def _drain_queues(self):
        for q in (self._robot_queue, self._camera_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

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
            print(f"📁 数据保存位置:\n   {self._output_dir}")


# ======================================================================
#  Entry
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Kinova Gen3 自动横向运动采集 (Deblur 测试)")
    print("=" * 60)
    print()
    cfg = AutoMotionConfig()
    print(f"  运动: {cfg.motion_direction.upper()} 方向, {cfg.motion_speed} m/s, {cfg.motion_duration*1000:.0f}ms")
    print(f"  重复: {cfg.repeat_count} 轮, 间隔 {cfg.repeat_interval}s")
    print(f"  相机: 曝光 {cfg.camera_exposure}×100µs, gain={cfg.camera_gain}")
    print()

    collector = AutoMotionCollector()
    try:
        collector.connect()
        collector.connect_camera()
        collector.run()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
