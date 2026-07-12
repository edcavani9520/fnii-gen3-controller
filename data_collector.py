#!/usr/bin/env python3
"""
Kinova Gen3 数据采集器 - π0.5 训练格式
=========================================
10Hz 同步采集：同一时间戳捕获图像 + 机器人本体状态 + 动作指令
输出 HDF5 文件，格式兼容 OpenPI / π0.5 训练框架

数据结构（每条任务片段 episode 独立一个HDF5文件）
  obs/
    image     (T, H, W) uint8         — 单通道灰度图像（分辨率640×480）
    proprio   (T, 8) float64          — [7个关节角度, 夹爪当前开度]
  action      (T, 7) float64          — [Δx,Δy,Δz,Δrx,Δry,Δrz, 夹爪目标值]
  timestamps  (T,) float64            — 每一帧对应的Unix时间戳

手柄控制映射：
  左摇杆        X/Y平面平移
  LT / RT       Z轴下降 / Z轴上升
  右摇杆        滚转(Roll) / 俯仰(Pitch)
  十字键左右    偏航(Yaw)旋转
  A / B         夹爪闭合 / 夹爪打开
  Y             开始 / 停止录制片段
  X             删除当前正在录制的片段
  LB            预留：机械臂回原点
  Menu菜单键    退出程序

启动示例：
  python data_collector.py --task "place the block into the bowl"
"""

import sys
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import datetime
from typing import Optional
from dataclasses import dataclass

import numpy as np
import pygame
import cv2
import h5py

# 导入Kortex Gen3 API路径
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Kinova_kortex2_Gen3_G3L", "api_python", "examples"
))
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2


# ======================================================================
# 全局配置参数
# ======================================================================
@dataclass
class Config:
    # ---- 机械臂参数 ----
    robot_ip: str = "192.168.8.10"
    speed_limit: float = 0.20       # 直线运动最大速度 m/s
    turn_limit: float = 20.0        # 旋转最大速度 °/s
    deadzone: float = 0.1           # 摇杆输入死区

    # ---- 相机参数（采集分辨率必须和模型推理保持一致）----
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 10            # 相机硬件帧率

    # ---- 数据集采集参数 ----
    output_root: str = os.path.expanduser("~/kinova_data")
    sample_hz: int = 10             # 整体同步采样频率10Hz

    # ---- 任务信息 ----
    task: str = ""
    task_id: int = 0

    # ---- 手柄按键编号映射 ----
    record_button: int = 3          # Y键：录制开关
    delete_button: int = 2          # X键：舍弃片段
    exit_button: int = 7            # Menu菜单键：退出程序


# ======================================================================
# 主采集类：Kinova遥操作数据采集
# ======================================================================
class KinovaTrainDataCollector:

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()

        # ---- Kinova机器人通信句柄 ----
        self.base: Optional[BaseClient] = None
        self.base_cyclic: Optional[BaseCyclicClient] = None
        self.router = None
        self.connection = None

        # ---- 游戏手柄初始化 ----
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("未检测到游戏手柄设备")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        # ---- 相机采集对象 ----
        self.cap: Optional[cv2.VideoCapture] = None

        # ---- 录制状态变量 ----
        self._recording = False         # 是否正在录制
        self._episode = 0               # 片段序号
        self._task_counter = 0          # 任务计数
        self._output_dir = ""           # 当前数据集文件夹
        self._hdf5_file: Optional[h5py.File] = None
        self._running = True            # 主循环开关

        # ---- 按键边沿检测（防止长按反复触发） ----
        self._prev_y = False
        self._prev_x = False

        # ---- 差分action计算缓存 ----
        self._prev_eef_pose: Optional[np.ndarray] = None
        self._last_gripper_cmd: float = -1.0   # -1代表指令未初始化

        print(f"成功识别手柄: {self.joy.get_name()}")

    # ================================================================
    # 建立与机械臂TCP连接
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
        print(f"已连接 Kinova Gen3 机械臂 @ {self.cfg.robot_ip}")

    def connect_camera(self):
        """初始化相机，固定曝光、白平衡、对焦，保证数据集光照一致性"""
        self.cap = cv2.VideoCapture(self.cfg.camera_id)

        # 关闭自动曝光、自动白平衡、自动对焦
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # V4L2驱动：0.25=手动曝光
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)

        # 设置图像分辨率与帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)

        # 测试读取一帧图像验证相机
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("相机读取图像失败，请检查摄像头")
        actual_h, actual_w = frame.shape[:2]
        print(f"相机初始化完成：{actual_w}×{actual_h} 目标帧率 {self.cfg.camera_fps} FPS")

    # ================================================================
    # 读取手柄所有轴、十字键、按键原始输入
    # ================================================================
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
        """摇杆死区过滤，微小输入置零避免漂移"""
        return v if abs(v) > self.cfg.deadzone else 0.0

    # ================================================================
    # 同步采集单帧观测：图像 + 机器人反馈数据
    # ================================================================
    def _sync_step(self) -> Optional[dict]:
        """
        采集一组同步观测数据
        返回字典包含：时间戳、灰度图、关节角度、夹爪开度、末端笛卡尔位姿
        """
        # 读取图像并转为灰度
        ret, frame = self.cap.read()
        if not ret:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # (H, W) uint8

        # 获取机械臂实时反馈数据
        try:
            fb = self.base_cyclic.RefreshFeedback()
        except Exception:
            return None
        b = fb.base
        inter = fb.interconnect

        # 7个关节当前角度
        joint_pos = np.array([act.position for act in fb.actuators], dtype=np.float64)

        # 夹爪实时开度
        gripper_pos = 0.0
        if inter.gripper_feedback.motor:
            gripper_pos = float(inter.gripper_feedback.motor[0].position)

        # 末端执行器笛卡尔位姿 XYZ(米) RxRyRz(角度)
        eef_pose = np.array([
            b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
            b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z
        ], dtype=np.float64)

        return {
            "timestamp": time.time(),
            "image": gray,               # (H, W) uint8
            "joint_pos": joint_pos,      # (7,) float64
            "gripper_pos": gripper_pos,  # 标量
            "eef_pose": eef_pose,        # (6,) float64
        }

    # ================================================================
    # 根据摇杆输入发送速度扭控指令Twist
    # ================================================================
    def _send_twist_if_needed(self, axes, hat) -> bool:
        a0, a1, a2, a3, a4, a5 = axes
        twist = np.array([
            -a1 * self.cfg.speed_limit,
            -a0 * self.cfg.speed_limit,
            (a5 - a2) * self.cfg.speed_limit,
            a3 * self.cfg.turn_limit,
            -a4 * self.cfg.turn_limit,
            -hat[0] * self.cfg.turn_limit,
        ], dtype=np.float64)
        has_input = np.any(np.abs(twist) > 0.001)
        if has_input:
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
        else:
            self.base.Stop()
        return has_input

    def _send_gripper(self, pos: float):
        """发送夹爪位置指令 0=全开 1=全闭合"""
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
    # 创建新的Episode HDF5文件，初始化数据集
    # ================================================================
    def _open_episode(self, episode: int):
        self._ensure_output_dir()
        fname = os.path.join(self._output_dir, f"episode_{episode:04d}.h5")
        f = h5py.File(fname, "w")

        H, W = self.cfg.camera_height, self.cfg.camera_width

        # 观测分组 obs
        grp_obs = f.create_group("obs")
        grp_obs.create_dataset(
            "image", shape=(0, H, W), maxshape=(None, H, W),
            dtype=np.uint8, compression="gzip", compression_opts=4,
        )
        grp_obs.create_dataset(
            "proprio", shape=(0, 8), maxshape=(None, 8),
            dtype=np.float64,
        )

        # Action数据集
        f.create_dataset(
            "action", shape=(0, 7), maxshape=(None, 7),
            dtype=np.float64,
        )

        # 时间戳序列
        f.create_dataset(
            "timestamps", shape=(0,), maxshape=(None,),
            dtype=np.float64,
        )

        # 文件元信息属性
        task_desc = self.cfg.task or "unnamed_task"
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
        f.attrs["camera_width"] = W
        f.attrs["camera_height"] = H
        f.attrs["sample_hz"] = self.cfg.sample_hz
        f.attrs["date_collected"] = datetime.datetime.now().isoformat()
        f.attrs["success"] = True

        self._hdf5_file = f
        self._prev_eef_pose = None   # 新建片段，重置末端位姿缓存
        print(f"\n开始录制片段：{fname}")
        print(f"   当前任务指令：{task_desc}")

    def _write_step(self, ts, image, proprio, action):
        """向HDF5追加一帧完整数据"""
        f = self._hdf5_file
        if f is None:
            return

        cur = f["timestamps"].shape[0]

        # 动态扩容数据集长度
        f["timestamps"].resize((cur + 1,))
        f["obs/image"].resize((cur + 1, f["obs/image"].shape[1], f["obs/image"].shape[2]))
        f["obs/proprio"].resize((cur + 1, 8))
        f["action"].resize((cur + 1, 7))

        # 写入数据
        f["timestamps"][cur] = ts
        f["obs/image"][cur] = image
        f["obs/proprio"][cur] = proprio
        f["action"][cur] = action
        f.flush()

    def _close_episode(self):
        """关闭当前HDF5文件，打印片段统计信息"""
        if self._hdf5_file is not None:
            elapsed = time.time() - self._hdf5_file.attrs["start_time"]
            n = self._hdf5_file["timestamps"].shape[0]
            self._hdf5_file.close()
            self._hdf5_file = None
            hz = n / (elapsed + 1e-6)
            print(f"片段 {self._episode:04d} 保存完成："
                  f"{n} 帧, {elapsed:.1f}秒, 实际采集 {hz:.1f} Hz")

    def _ensure_output_dir(self):
        """首次运行自动创建带时间戳的数据根目录"""
        if not self._output_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self.cfg.output_root, f"train_data_{ts}")
        os.makedirs(self._output_dir, exist_ok=True)

    # ================================================================
    # 主采集循环
    # ================================================================
    def run(self):
        try:
            print("数据采集主循环启动，目标采样频率 10Hz")
            print("  Y键：开始/停止录制片段 | X键：舍弃当前片段")
            print("  Menu菜单键：退出程序\n")

            display_counter = 0
            rec_label = "空闲 IDLE"

            while self._running:
                # 1. 获取手柄输入
                axes, hat, buttons = self._read_gamepad()

                # 菜单按键触发退出
                if buttons.get(self.cfg.exit_button):
                    print("\n检测到退出按键，准备结束程序...")
                    break

                # 2. 录制启停边沿检测逻辑
                y_pressed = bool(buttons.get(self.cfg.record_button))
                x_pressed = bool(buttons.get(self.cfg.delete_button))

                if y_pressed and not self._prev_y:
                    if not self._recording:
                        self._start_recording()
                        rec_label = "录制中 REC"
                    else:
                        self._stop_recording()
                        rec_label = "空闲 IDLE"

                if x_pressed and not self._prev_x:
                    if self._recording:
                        self._stop_recording(delete=True)
                        rec_label = "空闲 IDLE"

                self._prev_y = y_pressed
                self._prev_x = x_pressed

                # 3. 更新夹爪目标指令状态
                if buttons.get(0):
                    self._last_gripper_cmd = 1.0
                elif buttons.get(1):
                    self._last_gripper_cmd = 0.0

                # 4. 下发机器人运动控制指令（优先下发保证操控实时性）
                has_input = self._send_twist_if_needed(axes, hat)
                if buttons.get(0):
                    self._send_gripper(1.0)
                elif buttons.get(1):
                    self._send_gripper(0.0)

                # 5. 同步采集观测数据
                data = self._sync_step()
                if data is None:
                    continue

                # 6. 计算7维Action：末端位姿差分 + 夹爪目标值
                if self._prev_eef_pose is not None:
                    delta_pose = data["eef_pose"] - self._prev_eef_pose
                    # 角度差值解卷绕，限制范围 [-180°,180°]（Kinova角度单位为度）
                    for i in [3, 4, 5]:
                        if delta_pose[i] > 180:
                            delta_pose[i] -= 360
                        elif delta_pose[i] < -180:
                            delta_pose[i] += 360
                else:
                    delta_pose = np.zeros(6, dtype=np.float64)
                self._prev_eef_pose = data["eef_pose"].copy()

                # 兜底：若无夹爪指令则使用当前实际开度
                gripper_target = self._last_gripper_cmd
                if gripper_target < 0:
                    gripper_target = data["gripper_pos"]

                action_7d = np.concatenate([delta_pose, [gripper_target]])
                proprio_8d = np.concatenate([data["joint_pos"], [data["gripper_pos"]]])

                # 7. 录制状态下写入样本
                if self._recording:
                    self._write_step(
                        data["timestamp"],
                        data["image"],
                        proprio_8d,
                        action_7d,
                    )
                else:
                    self._prev_eef_pose = None   # 非录制清空缓存，防止跨片段差分污染

                # 控制台状态栏定期刷新
                display_counter += 1
                if display_counter % 5 == 0:
                    self._print_status(axes, hat, rec_label)

                # 维持固定10Hz周期
                time.sleep(1.0 / self.cfg.sample_hz)

        except KeyboardInterrupt:
            print("\n收到键盘中断 Ctrl+C")
        except Exception as e:
            print(f"\n程序运行异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    # ================================================================
    # 开启新片段录制
    # ================================================================
    def _start_recording(self):
        self._episode += 1
        self._task_counter += 1
        self.cfg.task_id = self._task_counter
        self._prev_eef_pose = None
        self._last_gripper_cmd = -1.0
        self._open_episode(self._episode)
        self._recording = True

    def _stop_recording(self, delete=False):
        """停止录制；delete=True代表舍弃当前片段并删除文件"""
        self._recording = False
        if delete:
            fname = self._hdf5_file.filename if self._hdf5_file is not None else None
            self._close_episode()
            if fname and os.path.exists(fname):
                os.remove(fname)
                print(f"已舍弃片段 {self._episode:04d}，删除文件：{fname}")
        else:
            self._close_episode()

    # ================================================================
    # 终端实时状态打印
    # ================================================================
    def _print_status(self, axes, hat, rec_label):
        a0, a1, _, a3, a4, _ = axes
        if self._last_gripper_cmd > 0.5:
            gripper_str = "闭合"
        elif self._last_gripper_cmd >= 0:
            gripper_str = "打开"
        else:
            gripper_str = "待机"

        joy_str = (f"X:{a1:+5.2f} Y:{a0:+5.2f} Z:{axes[5]-axes[2]:+5.2f} | "
                   f"滚转:{a3:+5.2f} 俯仰:{a4:+5.2f} 偏航:{hat[0]:+2.0f} | "
                   f"夹爪:{gripper_str}")
        extra = ""
        if self._recording and self._hdf5_file is not None:
            n = self._hdf5_file["timestamps"].shape[0]
            extra = f" | 帧数:{n}"
        sys.stdout.write(f"\r{rec_label}  {joy_str}{extra}     ")
        sys.stdout.flush()

    # ================================================================
    # 资源统一释放
    # ================================================================
    def _cleanup(self):
        self._running = False
        if self._recording:
            self._recording = False
            self._close_episode()
        # 停止机械臂运动
        if self.base:
            try:
                self.base.Stop()
            except Exception:
                pass
        # 关闭TCP连接
        if self.connection:
            try:
                self.connection.__exit__(None, None, None)
            except Exception:
                pass
        # 释放摄像头
        if self.cap:
            self.cap.release()
        pygame.quit()
        cv2.destroyAllWindows()
        print("\n程序资源释放完成，安全退出")
        if self._output_dir:
            print(f"\n数据集存储目录：\n   {self._output_dir}")
            print("   数据结构：obs/(image, proprio) + action + timestamps")


# ======================================================================
# 程序入口
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kinova Gen3 π0.5 遥操作数据集采集工具")
    parser.add_argument("--task", type=str, required=True,
                        help="当前任务语言指令，示例：--task 'place the block into the bowl'")
    parser.add_argument("--ip", type=str, default="192.168.8.10",
                        help="机械臂IP地址")
    args = parser.parse_args()

    cfg = Config()
    cfg.task = args.task
    cfg.robot_ip = args.ip

    print("=" * 60)
    print("  Kinova Gen3 π0.5 训练数据采集器")
    if cfg.task:
        print(f"  当前任务指令：{cfg.task}")
    print("  数据集结构：obs/(image, proprio) + action + language instruction")
    print("  采集模式：10Hz 图像+机器人状态同步采样")
    print("=" * 60)
    print()
    print("  手柄控制说明：")
    print("    左摇杆     → X/Y平面平移")
    print("    LT / RT    → Z轴下降 / Z轴上升")
    print("    右摇杆     → Roll滚转 / Pitch俯仰")
    print("    十字键 ←/→ → Yaw偏航旋转")
    print("    A键        → 夹爪闭合")
    print("    B键        → 夹爪打开")
    print("    Y键        → 开始 / 停止录制片段")
    print("    X键        → 舍弃当前录制片段")
    print("    LB按键      → 预留：机械臂回原点")
    print("    Menu菜单键 → 退出程序")
    print(f"  文件输出路径：~/kinova_data/train_data_<时间戳>/episode_XXXX.h5")

    collector = KinovaTrainDataCollector(cfg=cfg)
    try:
        collector.connect()
        collector.connect_camera()
        collector.run()
    except RuntimeError as e:
        print(f"初始化失败：{e}")
        sys.exit(1)