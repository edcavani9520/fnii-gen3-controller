#!/usr/bin/env python3
"""
Kinova Gen3 ���ݲɼ��� - ��0.5 ѵ����ʽ
=========================================
10Hz ͬ���ɼ���ͬһʱ���ץͼ+������״̬+����
��� HDF5 ��ʽ���� OpenPI / ��0.5 ѵ����·

�����ʽ��ÿ episode һ�� HDF5��:
  obs/
    image     (T, H, W) uint8         �� ��ͨ���Ҷ�ͼ��640��480��
    proprio   (T, 8) float64          �� [7�ؽڽǶ�, ��צ����]
  action      (T, 7) float64          �� [��x,��y,��z,��rx,��ry,��rz, gripper_target]
  timestamps  (T,) float64            �� ÿ��ʱ���

����ӳ��:
  ��ҡ��        �� XY ƽ��
  LT/RT         �� Z ������
  ��ҡ��        �� Roll / Pitch
  ʮ�ּ�����    �� Yaw
  A / B         �� ��צ��/��
  Y             �� ��ʼ¼�ƣ��ٰ� Y ����ֹͣ���� X ɾ��
  LB(��ť4)     �� �ص�ԭ��
  Menu(��ť7)   �� �˳�����

�÷�:
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

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Kinova_kortex2_Gen3_G3L", "api_python", "examples"
))
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2


# ======================================================================
#  ����
# ======================================================================
@dataclass
class Config:
    # ---- ��е�� ----
    robot_ip: str = "192.168.8.10"
    speed_limit: float = 0.20       # m/s
    turn_limit: float = 20.0        # ��/s
    deadzone: float = 0.1

    # ---- ����������������ɼ�=�����----
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 10            # 10Hz

    # ---- �ɼ� ----
    output_root: str = os.path.expanduser("~/kinova_data")
    sample_hz: int = 10             # ͬ���ɼ� 10Hz

    # ---- �������� ----
    task: str = ""
    task_id: int = 0

    # ---- ¼�ƿ��� ----
    record_button: int = 3          # Y
    delete_button: int = 2          # X
    exit_button: int = 7            # Menu


# ======================================================================
#  ���ݲɼ���
# ======================================================================
class KinovaTrainDataCollector:

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()

        # ---- Kinova ���� ----
        self.base: Optional[BaseClient] = None
        self.base_cyclic: Optional[BaseCyclicClient] = None
        self.router = None
        self.connection = None

        # ---- �ֱ� ----
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("δ��⵽�ֱ�")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        # ---- ��� ----
        self.cap: Optional[cv2.VideoCapture] = None

        # ---- ¼��״̬ ----
        self._recording = False
        self._episode = 0
        self._task_counter = 0
        self._output_dir = ""
        self._hdf5_file: Optional[h5py.File] = None
        self._running = True

        # ---- ���ؼ�� ----
        self._prev_y = False
        self._prev_x = False

        # ---- delta ���㻺�� ----
        self._prev_eef_pose: Optional[np.ndarray] = None
        self._last_gripper_cmd: float = -1.0   # -1 = δ��ʼ��

        print(f"�ֱ�������: {self.joy.get_name()}")

    # ================================================================
    #  ����
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
        print(f"������ Kinova @ {self.cfg.robot_ip}")

    def connect_camera(self):
        """�����������ȫ���Զ�����"""
        self.cap = cv2.VideoCapture(self.cfg.camera_id)

        # ---- �����Զ����� ----
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = off (V4L2)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)           # �ֶ��̶�
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)             # ���Զ���ƽ��
        self.cap.set(cv2.CAP_PROP_GAIN, 0)                # �̶�����
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)           # ���Զ��Խ�
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)               # �̶�����
        # �����ȫ�ֿ���������˴�����������

        # ---- �ֱ��� & ֡�� ----
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)

        # ---- ��֤ ----
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("���������")
        actual_h, actual_w = frame.shape[:2]
        print(f"����Ѵ�: {actual_w}x{actual_h} @ {self.cfg.camera_fps} FPS��������������")

    # ================================================================
    #  �ֱ���ȡ
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
        return v if abs(v) > self.cfg.deadzone else 0.0

    # ================================================================
    #  ͬ���ɼ������ģ���ʱ��� ͼ+������+������
    # ================================================================

    def _sync_step(self, axes, hat, buttons) -> Optional[dict]:
        """
        ��ͬһ��ʱ�������ɣ�
        1. ��ȡ�Ҷ�ͼ
        2. ��ȡ������״̬���ؽڽǡ�ĩ��λ�ˡ���צ��
        3. ���� twist ָ�������
        4. ���� {timestamp, image, joint_pos, gripper_pos, eef_pose}
        """
        # ---- 1. ��� ----
        ret, frame = self.cap.read()
        if not ret:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # (H, W) uint8

        # ---- 2. ������ ----
        try:
            fb = self.base_cyclic.RefreshFeedback()
        except Exception:
            return None
        b = fb.base
        inter = fb.interconnect

        # �ؽ�λ�ã�Gen3 �� 7 ���ؽڣ�
        joint_pos = np.array([act.position for act in fb.actuators], dtype=np.float64)

        # ��צ����
        gripper_pos = 0.0
        if inter.gripper_feedback.motor:
            gripper_pos = float(inter.gripper_feedback.motor[0].position)

        # ĩ�˵ѿ���λ��
        eef_pose = np.array([
            b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
            b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z
        ], dtype=np.float64)

        # ---- 3. ���� twist �������� ----
        twist = np.array([
            -axes[1] * self.cfg.speed_limit,
            -axes[0] * self.cfg.speed_limit,
            (axes[5] - axes[2]) * self.cfg.speed_limit,
            axes[3] * self.cfg.turn_limit,
            -axes[4] * self.cfg.turn_limit,
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

        # ---- 4. ���ͼ�צ ----
        if buttons.get(0):       # A = �ر�
            self._send_gripper(1.0)
        elif buttons.get(1):     # B = ��
            self._send_gripper(0.0)

        return {
            "timestamp": time.time(),
            "image": gray,               # (H, W) uint8
            "joint_pos": joint_pos,      # (7,) float64
            "gripper_pos": gripper_pos,  # scalar
            "eef_pose": eef_pose,        # (6,) float64
        }

    # ================================================================
    #  ������ָ��
    # ================================================================

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
    #  HDF5 ���
    # ================================================================

    def _open_episode(self, episode: int):
        self._ensure_output_dir()
        fname = os.path.join(self._output_dir, f"episode_{episode:04d}.h5")
        f = h5py.File(fname, "w")

        H, W = self.cfg.camera_height, self.cfg.camera_width

        # === obs/ ===
        grp_obs = f.create_group("obs")
        grp_obs.create_dataset(
            "image", shape=(0, H, W), maxshape=(None, H, W),
            dtype=np.uint8, compression="gzip", compression_opts=4,
        )
        grp_obs.create_dataset(
            "proprio", shape=(0, 8), maxshape=(None, 8),
            dtype=np.float64,
        )

        # === action������ dataset��===
        f.create_dataset(
            "action", shape=(0, 7), maxshape=(None, 7),
            dtype=np.float64,
        )

        # === timestamps ===
        f.create_dataset(
            "timestamps", shape=(0,), maxshape=(None,),
            dtype=np.float64,
        )

        # === attributes ===
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
        f.attrs["camera_width"] = W
        f.attrs["camera_height"] = H
        f.attrs["sample_hz"] = self.cfg.sample_hz
        f.attrs["date_collected"] = datetime.datetime.now().isoformat()
        f.attrs["success"] = True    # ռλ��������ϴʱ��������

        self._hdf5_file = f
        self._prev_eef_pose = None   # �� episode ���� delta ����
        print(f"\n��ʼ¼�� �� {fname}")
        print(f"   task: {task_desc}")

    def _write_step(self, ts, image, proprio, action):
        """д��һ֡ͬ�����ݵ� HDF5"""
        f = self._hdf5_file
        if f is None:
            return

        cur = f["timestamps"].shape[0]

        # ��չ���� dataset
        f["timestamps"].resize((cur + 1,))
        f["obs/image"].resize((cur + 1, f["obs/image"].shape[1], f["obs/image"].shape[2]))
        f["obs/proprio"].resize((cur + 1, 8))
        f["action"].resize((cur + 1, 7))

        # д��
        f["timestamps"][cur] = ts
        f["obs/image"][cur] = image
        f["obs/proprio"][cur] = proprio
        f["action"][cur] = action
        f.flush()

    def _close_episode(self):
        """�رյ�ǰ episode"""
        if self._hdf5_file is not None:
            elapsed = time.time() - self._hdf5_file.attrs["start_time"]
            n = self._hdf5_file["timestamps"].shape[0]
            self._hdf5_file.close()
            self._hdf5_file = None
            hz = n / (elapsed + 1e-6)
            print(f"Episode {self._episode:04d} �ѱ���: "
                  f"{n} steps, {elapsed:.1f}s, {hz:.1f} Hz")

    def _ensure_output_dir(self):
        if not self._output_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self.cfg.output_root, f"train_data_{ts}")
        os.makedirs(self._output_dir, exist_ok=True)

    # ================================================================
    #  ��ѭ��
    # ================================================================

    def run(self):
        try:
            print("ͬ���ɼ���� @ 10Hz")
            print("  Y �� ��ʼ¼��  �ٰ� Y �� ����ֹͣ  �� X �� ɾ��ֹͣ")
            print("  LB �� �ص�ԭ��  Menu �� �˳�\n")

            display_counter = 0
            rec_label = "�� IDLE"

            while self._running:
                # === 1. ��ȡ�ֱ� ===
                axes, hat, buttons = self._read_gamepad()

                # �˳�
                if buttons.get(self.cfg.exit_button):
                    print("\n�˳�����...")
                    break

                # === 2. ¼�ƿ��� ===
                y_pressed = bool(buttons.get(self.cfg.record_button))
                x_pressed = bool(buttons.get(self.cfg.delete_button))

                if y_pressed and not self._prev_y:
                    if not self._recording:
                        self._start_recording()
                        rec_label = "�� REC"
                    else:
                        self._stop_recording()
                        rec_label = "�� IDLE"

                if x_pressed and not self._prev_x:
                    if self._recording:
                        self._stop_recording(delete=True)
                        rec_label = "�� IDLE"

                self._prev_y = y_pressed
                self._prev_x = x_pressed

                # === 3. ��¼��צ��ͼ ===
                if buttons.get(0):
                    self._last_gripper_cmd = 1.0
                elif buttons.get(1):
                    self._last_gripper_cmd = 0.0

                # === 4. ͬ���ɼ�����ʱ��� ͼ+������+���� ===
                data = self._sync_step(axes, hat, buttons)
                if data is None:
                    continue

                # === 5. ���� 7 ά���� ===
                # delta pose
                if self._prev_eef_pose is not None:
                    delta_pose = data["eef_pose"] - self._prev_eef_pose
                    # Unwrap theta to [-180, 180] (Kinova returns degrees)
                    for i in [3, 4, 5]:
                        if delta_pose[i] > 180:
                            delta_pose[i] -= 360
                        elif delta_pose[i] < -180:
                            delta_pose[i] += 360
                else:
                    delta_pose = np.zeros(6, dtype=np.float64)
                self._prev_eef_pose = data["eef_pose"].copy()

                # gripper target
                gripper_target = self._last_gripper_cmd
                if gripper_target < 0:          # �״�δ��ʼ��
                    gripper_target = data["gripper_pos"]

                action_7d = np.concatenate([delta_pose, [gripper_target]])
                proprio_8d = np.concatenate([data["joint_pos"], [data["gripper_pos"]]])

                # === 6. д�� HDF5 ===
                if self._recording:
                    self._write_step(
                        data["timestamp"],
                        data["image"],
                        proprio_8d,
                        action_7d,
                    )
                else:
                    self._prev_eef_pose = None   # ��¼��ʱ����

                # === 7. ״̬��ʾ ===
                display_counter += 1
                if display_counter % 5 == 0:
                    self._print_status(axes, hat, rec_label)

                # === 8. 10Hz ===
                time.sleep(1.0 / self.cfg.sample_hz)

        except KeyboardInterrupt:
            print("\n�û��ж�")
        except Exception as e:
            print(f"\n����: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    # ================================================================
    #  ¼����������
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
        self._recording = False
        if delete:
            fname = self._hdf5_file.filename if self._hdf5_file is not None else None
            self._close_episode()
            if fname and os.path.exists(fname):
                os.remove(fname)
                print(f"��ɾ�� episode {self._episode:04d}: {fname}")
        else:
            self._close_episode()

    # ================================================================
    #  ״̬��ʾ
    # ================================================================

    def _print_status(self, axes, hat, rec_label):
        a0, a1, _, a3, a4, _ = axes
        gripper_str = "CLOSED" if self._last_gripper_cmd > 0.5 else "OPENED" if self._last_gripper_cmd >= 0 else "?"
        joy_str = (f"X:{a1:+5.2f} Y:{a0:+5.2f} Z:{axes[5]-axes[2]:+5.2f} | "
                   f"R:{a3:+5.2f} P:{a4:+5.2f} Y:{hat[0]:+2.0f} | "
                   f"Grip:{gripper_str}")
        extra = ""
        if self._recording and self._hdf5_file is not None:
            n = self._hdf5_file["timestamps"].shape[0]
            extra = f" | steps:{n}"
        sys.stdout.write(f"\r{rec_label}  {joy_str}{extra}     ")
        sys.stdout.flush()

    # ================================================================
    #  ����
    # ================================================================

    def _cleanup(self):
        self._running = False
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
        print("\n��ȫ�˳�")
        if self._output_dir:
            print(f"\n���ݱ���λ��:\n   {self._output_dir}")
            print("   ��ʽ: obs/(image, proprio) + action + timestamps")


# ======================================================================
#  Entry
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kinova Gen3 ��0.5 ѵ�����ݲɼ���")
    parser.add_argument("--task", type=str, required=True,
                        help="������������ --task 'place the block into the bowl'")
    parser.add_argument("--ip", type=str, default="192.168.8.10",
                        help="��е�� IP ��ַ")
    args = parser.parse_args()

    cfg = Config()
    cfg.task = args.task
    cfg.robot_ip = args.ip

    print("=" * 60)
    print("  Kinova Gen3 ��0.5 ѵ�����ݲɼ���")
    if cfg.task:
        print(f"  ����: {cfg.task}")
    print("  �����ʽ: obs/(image, proprio) + action + language instruction")
    print("  ����: 10Hz ͬ���ɼ�")
    print("=" * 60)
    print()
    print("  ����ӳ��:")
    print("    ��ҡ�� �� XY ƽ��    ��ҡ�� �� Roll / Pitch")
    print("    LT/RT  �� Z ������   ʮ�ּ� �� Yaw")
    print("    A �ؼ�צ  B ����צ   Y ��ʼ¼�� / Y ���� / X ɾ��")
    print("    LB �ص�ԭ��   Menu �˳�")
    print(f"  ���: ~/kinova_data/train_data_<timestamp>/episode_XXXX.h5")

    collector = KinovaTrainDataCollector(cfg=cfg)
    try:
        collector.connect()
        collector.connect_camera()
        collector.run()
    except RuntimeError as e:
        print(f"{e}")
        sys.exit(1)
