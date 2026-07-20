#!/usr/bin/env python3
"""
Real-time deblur + pi0.5 WebSocket inference + Kinova Gen3 control.

This script keeps the same policy/robot interface as pi05_ws_control.py:
  observation/image        RGB uint8, 320x240
  observation/wrist_image  dummy image, masked by the Kinova policy
  observation/state        7 joints + gripper
  prompt                   language instruction

The only difference is that the camera frame is deblurred before it is sent to
the OpenPI policy server. Deblur uses the current Kinova tool_pose/tool_twist
feedback to estimate a motion PSF, then applies a lightweight Wiener deconvolution
on the luminance channel.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pi05_ws_control import Pi05WebSocketControl


DEBLUR_ROOT = Path(
    os.environ.get(
        "KINOVA_DEBLUR_ROOT",
        str(Path.home() / "Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin"),
    )
).expanduser()

if str(DEBLUR_ROOT) not in sys.path:
    sys.path.insert(0, str(DEBLUR_ROOT))

try:
    from joint_deblur import compute_psf_from_pose, wiener_deconvolution, tv_deconv, euler_zyx_to_rotmat
    from robot_configs import get_hand_eye
except Exception as exc:
    raise ImportError(
        "Cannot import deblur modules. Make sure the deblur repository exists at:\n"
        f"  {DEBLUR_ROOT}\n"
        "or set KINOVA_DEBLUR_ROOT=/path/to/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin"
    ) from exc


class RealtimeDeblurPi05Control(Pi05WebSocketControl):
    """Pi05 controller that deblurs camera frames before policy inference."""

    def __init__(
        self,
        *args,
        deblur=True,
        deblur_method="wiener",
        deblur_k=0.03,
        deblur_tv_lam=0.002,
        deblur_exposure=0.03,
        deblur_depth=0.5,
        deblur_auto_depth=False,
        deblur_table_z=0.0,
        deblur_fx=733.37,
        deblur_fy=733.37,
        deblur_hand_eye="kinova-gen3",
        deblur_min_pixels=0.5,
        deblur_max_pixels=25.0,
        angular_units="deg",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.deblur = bool(deblur)
        self.deblur_method = deblur_method
        self.deblur_k = float(deblur_k)
        self.deblur_tv_lam = float(deblur_tv_lam)
        self.deblur_exposure = float(deblur_exposure)
        self.deblur_depth = float(deblur_depth)
        self.deblur_auto_depth = bool(deblur_auto_depth)
        self.deblur_table_z = float(deblur_table_z)
        self.deblur_fx = float(deblur_fx)
        self.deblur_fy = float(deblur_fy)
        self.deblur_min_pixels = float(deblur_min_pixels)
        self.deblur_max_pixels = float(deblur_max_pixels)
        self.angular_units = angular_units
        self.hand_eye = get_hand_eye(deblur_hand_eye) if deblur_hand_eye else None
        self.last_deblur_info = None

    def _tool_pose_twist(self):
        if self.dry_run or self.full_status is None:
            return None, None
        b = self.full_status.base
        pose = np.array(
            [
                b.tool_pose_x,
                b.tool_pose_y,
                b.tool_pose_z,
                b.tool_pose_theta_x,
                b.tool_pose_theta_y,
                b.tool_pose_theta_z,
            ],
            dtype=np.float32,
        )
        twist = np.array(
            [
                b.tool_twist_linear_x,
                b.tool_twist_linear_y,
                b.tool_twist_linear_z,
                b.tool_twist_angular_x,
                b.tool_twist_angular_y,
                b.tool_twist_angular_z,
            ],
            dtype=np.float32,
        )
        if self.angular_units == "deg":
            twist[3:6] = np.deg2rad(twist[3:6])
        return pose, twist

    def _estimate_depth(self, pose):
        if not self.deblur_auto_depth:
            return self.deblur_depth

        r_ee = euler_zyx_to_rotmat(pose[3:])
        if self.hand_eye is not None:
            cam_pos = pose[:3] + r_ee @ self.hand_eye.t
            r_cam = r_ee @ self.hand_eye.R
        else:
            cam_pos = pose[:3]
            r_cam = r_ee

        opt_axis = r_cam @ np.array([0.0, 0.0, 1.0])
        opt_z = float(opt_axis[2])
        return max(abs((self.deblur_table_z - float(cam_pos[2])) / max(abs(opt_z), 0.01)), 0.02)

    def _deblur_rgb(self, rgb):
        pose, twist = self._tool_pose_twist()
        if not self.deblur or pose is None or twist is None:
            self.last_deblur_info = ("off", 0.0, 0.0, 0.0, 0.0)
            return rgb

        h, w = rgb.shape[:2]
        depth = self._estimate_depth(pose)
        try:
            psf, (du, dv) = compute_psf_from_pose(
                pose,
                twist,
                depth,
                fx=self.deblur_fx,
                fy=self.deblur_fy,
                cx=w // 2,
                cy=h // 2,
                exposure_time=self.deblur_exposure,
                hand_eye=self.hand_eye,
            )
        except Exception as exc:
            self.last_deblur_info = ("error", 0.0, 0.0, depth, 0.0)
            if self.step_count % self.log_every == 0:
                print(f"[deblur] ⚠ PSF failed: {exc}")
            return rgb

        motion_px = float(np.hypot(du, dv))
        if motion_px < self.deblur_min_pixels:
            self.last_deblur_info = ("skip", float(du), float(dv), depth, motion_px)
            return rgb

        if motion_px > self.deblur_max_pixels:
            scale = self.deblur_max_pixels / motion_px
            psf, (du, dv) = compute_psf_from_pose(
                pose,
                twist * scale,
                depth,
                fx=self.deblur_fx,
                fy=self.deblur_fy,
                cx=w // 2,
                cy=h // 2,
                exposure_time=self.deblur_exposure,
                hand_eye=self.hand_eye,
            )
            motion_px = float(np.hypot(du, dv))

        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0]
        if self.deblur_method == "tv":
            y_deblur = tv_deconv(y, psf, lam=self.deblur_tv_lam)
        else:
            y_deblur = wiener_deconvolution(y, psf, K=self.deblur_k)
        ycrcb[:, :, 0] = y_deblur
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        self.last_deblur_info = ("on", float(du), float(dv), depth, motion_px)
        return out

    def get_camera_image(self):
        rgb = super().get_camera_image()
        t0 = time.monotonic()
        out = self._deblur_rgb(rgb)
        deblur_ms = (time.monotonic() - t0) * 1000.0
        if self.step_count % self.log_every == 0 and self.last_deblur_info is not None:
            status, du, dv, depth, motion_px = self.last_deblur_info
            print(
                f"[deblur] {status} {deblur_ms:.0f}ms "
                f"du={du:+.2f}px dv={dv:+.2f}px |motion|={motion_px:.2f}px depth={depth:.3f}m"
            )
        return out

    def run(self):
        print(
            "[deblur] realtime deblur "
            f"enabled={self.deblur} method={self.deblur_method} K={self.deblur_k} "
            f"exposure={self.deblur_exposure}s depth={self.deblur_depth}m "
            f"auto_depth={self.deblur_auto_depth} angular_units={self.angular_units}"
        )
        super().run()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Realtime deblur + π0.5 WebSocket + Kinova Gen3 真机控制"
    )

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

    parser.add_argument("--no-deblur", action="store_true",
                        help="关闭实时去模糊，只保留普通 pi05_ws_control 行为")
    parser.add_argument("--deblur-method", choices=["wiener", "tv"], default="wiener",
                        help="实时去模糊方法；真机建议 wiener")
    parser.add_argument("--deblur-k", type=float, default=0.03,
                        help="Wiener 去卷积 K，越小越锐但越容易振铃")
    parser.add_argument("--deblur-tv-lam", type=float, default=0.002,
                        help="TV 去卷积 lambda，仅 --deblur-method tv 使用")
    parser.add_argument("--deblur-exposure", type=float, default=0.03,
                        help="相机曝光时间估计 (秒)")
    parser.add_argument("--deblur-depth", type=float, default=0.5,
                        help="固定物距估计 (m)")
    parser.add_argument("--deblur-auto-depth", action="store_true",
                        help="根据 tool pose 和 --deblur-table-z 自动估计物距")
    parser.add_argument("--deblur-table-z", type=float, default=0.0,
                        help="auto-depth 使用的桌面 Z 高度 (m)")
    parser.add_argument("--deblur-fx", type=float, default=733.37,
                        help="相机 fx 像素焦距")
    parser.add_argument("--deblur-fy", type=float, default=733.37,
                        help="相机 fy 像素焦距")
    parser.add_argument("--deblur-hand-eye", default="kinova-gen3",
                        help="robot_configs.py 中的手眼标定名称；可设为空字符串禁用")
    parser.add_argument("--deblur-min-pixels", type=float, default=0.5,
                        help="估计运动模糊低于该像素值时跳过去模糊")
    parser.add_argument("--deblur-max-pixels", type=float, default=25.0,
                        help="限制用于去模糊的最大像素位移，避免强振铃")
    parser.add_argument("--angular-units", choices=["deg", "rad"], default="deg",
                        help="Kinova tool_twist angular 单位；Kortex 通常为 deg/s")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    ctrl = RealtimeDeblurPi05Control(
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
        deblur=not args.no_deblur,
        deblur_method=args.deblur_method,
        deblur_k=args.deblur_k,
        deblur_tv_lam=args.deblur_tv_lam,
        deblur_exposure=args.deblur_exposure,
        deblur_depth=args.deblur_depth,
        deblur_auto_depth=args.deblur_auto_depth,
        deblur_table_z=args.deblur_table_z,
        deblur_fx=args.deblur_fx,
        deblur_fy=args.deblur_fy,
        deblur_hand_eye=args.deblur_hand_eye,
        deblur_min_pixels=args.deblur_min_pixels,
        deblur_max_pixels=args.deblur_max_pixels,
        angular_units=args.angular_units,
    )
    ctrl.run()


if __name__ == "__main__":
    main()
