#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kinova Gen3 的 LeRobot 适配层（当前为骨架实现）。

LeRobot 在 ``lerobot-record`` / ``lerobot-teleoperate`` 等流程中会调用本类：
先 ``connect``，循环里 ``get_observation`` → 人类/策略 ``send_action``，结束时 ``disconnect``。

运行环境需能 ``import kinova_manage``（例如事先 ``export PYTHONPATH=...`` 包含 ``kinova_manage.py`` 所在目录）。
"""

from __future__ import annotations

import time
from functools import cached_property

from kinova_manage import HOME_JOINTS, KinovaManager

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_not_connected

from ..robot import Robot
from .config_kinova_gen3 import KinovaGen3RobotConfig


class KinovaGen3(Robot):
    """
    Kinova Gen3（Kortex）与 LeRobot ``Robot`` 接口的桥接类。

    机械臂通信委托 ``KinovaManager``（在 ``__init__`` 里按 ``config.ip`` 创建）；连接/断开见 ``connect`` / ``disconnect``。

    标量观测键及 ``observation.state`` 分量顺序由 ``_PROPRIO_FEATURE_KEYS`` 定义（关节位/速、末端位姿、夹爪）；
    动作为 ``_ACTION_FEATURE_KEYS``：基座系 6 维末端速度 + 夹爪位置（``send_action`` → ``TwistCommand``）。
    图像键与 ``self.cameras`` 一致（默认 ``head`` / ``wrist``）。
    """

    config_class = KinovaGen3RobotConfig
    name = "kinova_gen3"

    # 7 维动作：基座系笛卡尔末端速度 (m/s, °/s) + 夹爪位置 0.0–1.0
    _ACTION_FEATURE_KEYS: tuple[str, ...] = (
        "ee.twist_linear_x",
        "ee.twist_linear_y",
        "ee.twist_linear_z",
        "ee.twist_angular_x",
        "ee.twist_angular_y",
        "ee.twist_angular_z",
        "gripper.position",
    )

    # 全部为 float；数据集里合并为 ``observation.state`` 时顺序与本元组一致
    _PROPRIO_FEATURE_KEYS: tuple[str, ...] = (
        *[f"joint_{i:02d}.position_deg" for i in range(1, 8)],
        *[f"joint_{i:02d}.velocity_deg_s" for i in range(1, 8)],
        "ee.pose_x",
        "ee.pose_y",
        "ee.pose_z",
        "ee.pose_theta_x",
        "ee.pose_theta_y",
        "ee.pose_theta_z",
        "gripper.position",
    )

    def __init__(self, config: KinovaGen3RobotConfig) -> None:
        """
        保存配置、按配置创建相机对象、初始化内部连接状态。

        ``self.cameras`` 会被 ``lerobot-record`` 用来决定图像写入线程数等，需与 ``observation_features`` 里
        的图像键一致。

        ``KinovaManager`` 仅在此处构造一次；实际建链在 ``connect()`` 里调用 ``self._manager.connect()``。
        """
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._connected: bool = False
        self._require_cameras: bool = True
        self._manager = KinovaManager(self.config.ip)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        描述一条观测里有哪些字段、类型或图像形状。

        LeRobot 用它在建数据集时生成 ``features`` 元数据；键名需与 ``get_observation()`` 返回字典的键一致。
        标量键见 ``_PROPRIO_FEATURE_KEYS``；图像键与 ``self.cameras`` 一致。
        """
        feats: dict[str, type | tuple] = {k: float for k in self._PROPRIO_FEATURE_KEYS}
        for name, cam in self.cameras.items():
            if cam.height is not None and cam.width is not None:
                feats[name] = (cam.height, cam.width, 3)
        return feats

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        描述 ``send_action(action)`` 里 ``action`` 有哪些键及类型。

        基座坐标系笛卡尔速度 + 夹爪位置；单位与 ``send_action`` / Kortex ``TwistCommand`` 一致。
        """
        return {k: float for k in self._ACTION_FEATURE_KEYS}

    @property
    def is_connected(self) -> bool:
        """
        是否已与机器人（及本类管理的相机）建立可用连接。

        LeRobot 在部分流程里会查询；未连接时不应调用 ``get_observation`` / ``send_action``。
        """
        if not self._connected:
            return False
        if self._require_cameras and self.cameras and not all(
            cam.is_connected for cam in self.cameras.values()
        ):
            return False
        return True

    def connect(self, calibrate: bool = True, *, connect_cameras: bool = True) -> None:
        """
        建立与机械臂（及相机）的通信，并可选择是否走校准流程。

        Args:
            calibrate: 为 True 时，若子类支持校准且未校准，可在此触发校准逻辑。
            connect_cameras: 为 False 时仅连接机械臂（适合纯运动学测试）。

        机械臂侧调用 ``self._manager.connect()``；相机侧按 ``config.cameras`` 逐个 ``connect``。
        """
        _ = calibrate  # Gen3：校准入口见 ``calibrate()``，此处不调用
        if self._connected:
            return
        self._require_cameras = connect_cameras
        self._manager.connect()
        if connect_cameras:
            try:
                for cam in self.cameras.values():
                    cam.connect()
            except Exception:
                self._manager.disconnect()
                raise
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        """
        是否已完成校准。

        Gen3 若不用 LeRobot 那套 Motor 校准文件，可恒为 True；若需要零点/限位等，在此反映真实状态。
        """
        return True

    def calibrate(self) -> None:
        """
        执行一次校准并写回 ``self.calibration`` / 校准文件（若适用）。

        Kinova 若无需与 SO100 相同的校准流程，可保持空实现或仅日志提示。
        """
        pass

    def configure(self) -> None:
        """
        连接后的一次性参数设置（控制模式、频率、保护参数等）。

        LeRobot 在 ``connect`` 成功后会调用；适合放 Kortex 的高层级模式切换、笛卡尔/关节混用配置等。
        """
        pass

    @classmethod
    def _feedback_to_proprio_dict(cls, status) -> dict[str, float]:
        """从 ``KinovaManager.get_status()`` / ``RefreshFeedback`` 提取标量观测（不含图像），键顺序与 ``_PROPRIO_FEATURE_KEYS`` 一致。"""
        b = status.base
        raw: dict[str, float] = {}
        for i in range(7):
            act = status.actuators[i]
            raw[f"joint_{i + 1:02d}.position_deg"] = float(act.position)
            raw[f"joint_{i + 1:02d}.velocity_deg_s"] = float(act.velocity)
        raw["ee.pose_x"] = float(b.tool_pose_x)
        raw["ee.pose_y"] = float(b.tool_pose_y)
        raw["ee.pose_z"] = float(b.tool_pose_z)
        raw["ee.pose_theta_x"] = float(b.tool_pose_theta_x)
        raw["ee.pose_theta_y"] = float(b.tool_pose_theta_y)
        raw["ee.pose_theta_z"] = float(b.tool_pose_theta_z)
        grip_motor = status.interconnect.gripper_feedback.motor
        raw["gripper.position"] = float(grip_motor[0].position) if grip_motor else 0.0
        return {k: raw[k] for k in cls._PROPRIO_FEATURE_KEYS}

    def _clip_twist_action(self, action: RobotAction) -> dict[str, float]:
        """解析并限幅 6 维 Twist + 夹爪；缺失键按 0 处理。"""
        cfg = self.config
        out: dict[str, float] = {}
        for i, key in enumerate(self._ACTION_FEATURE_KEYS[:6]):
            raw = float(action.get(key, 0.0))
            limit = cfg.max_linear_velocity_m_s if i < 3 else cfg.max_angular_velocity_deg_s
            out[key] = max(-limit, min(limit, raw))
        grip = max(0.0, min(1.0, float(action.get("gripper.position", 0.0))))
        out["gripper.position"] = grip
        return out

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        读取当前一帧观测：Kortex 周期反馈中的关节/末端/夹爪 + 各路相机图像。

        键与 ``observation_features`` 一致；标量为 Python float；图像为 ``uint8`` HWC。
        """
        status = self._manager.get_status()
        if status is None or len(status.actuators) < 7:
            raise RuntimeError(
                "KinovaGen3.get_observation: get_status() 无效或关节数不足 7，请检查 Kortex 连接。"
            )
        obs: RobotObservation = {**self._feedback_to_proprio_dict(status)}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        return obs

    def _zero_twist_action(self) -> dict[str, float]:
        return {k: 0.0 for k in self._ACTION_FEATURE_KEYS}

    @check_if_not_connected
    def stop_twist(self) -> None:
        """末端笛卡尔速度归零并 ``base.Stop()``，用于条间复位前清掉 Twist 指令。"""
        for _ in range(3):
            self.send_action(self._zero_twist_action())
        if getattr(self._manager, "base", None) is not None:
            self._manager.base.Stop()

    @check_if_not_connected
    def reset_to_home(self) -> None:
        """
        条间复位：先停 Twist，再经 ``KinovaManager.move_angular`` 回 home 关节位。

        目标关节来自 ``config.home_joints``，未配置则用 ``kinova_manage.HOME_JOINTS``。
        """
        self.stop_twist()
        home = self.config.home_joints if self.config.home_joints is not None else HOME_JOINTS
        if len(home) < 8:
            raise ValueError(f"home_joints 需要 7 关节 + 夹爪共 8 个数，当前 len={len(home)}")
        self._manager.move_angular(home)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        把 LeRobot 传来的 ``action`` 转成 Kortex 指令并下发；返回值建议为实际发送到硬件的动作（含限幅后）。

        机械臂：基座系 ``SendTwistCommand``（与 ``kinova_manage.move_velocity`` / gamepad 一致）。
        夹爪：``GRIPPER_POSITION``，目标 0.0–1.0（与观测 ``gripper.position`` 一致）。
        """
        clipped = self._clip_twist_action(action)
        speeds = [
            clipped["ee.twist_linear_x"],
            clipped["ee.twist_linear_y"],
            clipped["ee.twist_linear_z"],
            clipped["ee.twist_angular_x"],
            clipped["ee.twist_angular_y"],
            clipped["ee.twist_angular_z"],
        ]
        dz = self.config.velocity_deadzone
        if all(abs(v) < dz for v in speeds):
            self._manager.base.Stop()
        else:
            self._manager.move_velocity(speeds, duration_ms=self.config.twist_duration_ms)

        clipped["gripper.position"] = self._manager.send_gripper_position(clipped["gripper.position"])
        return clipped

    def disconnect(self) -> None:
        """
        断开连接并释放资源：停臂、关相机、关 Router 等。

        LeRobot 在上下文退出或录制结束时会调用。
        """
        if getattr(self._manager, "base", None) is not None:
            try:
                self._manager.base.Stop()
            except Exception:
                pass
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        self._manager.disconnect()
        self._connected = False


def test_send_action_twist_sequence() -> None:
    """顺序测试 send_action：±vx/vy/vz、±wx/wy/wz，每段 hold_s。"""
    cfg = KinovaGen3RobotConfig()
    linear_speed = cfg.max_linear_velocity_m_s
    angular_speed = cfg.max_angular_velocity_deg_s
    hold_s = 1.4
    hz = 5.0

    robot = KinovaGen3(cfg)
    robot.connect(connect_cameras=False)
    zero = {k: 0.0 for k in KinovaGen3._ACTION_FEATURE_KEYS}
    steps: list[tuple[str, dict[str, float]]] = [
        (f"vx=+{linear_speed:g}", {"ee.twist_linear_x": linear_speed}),
        (f"vy=+{linear_speed:g}", {"ee.twist_linear_y": linear_speed}),
        (f"vz=+{linear_speed:g}", {"ee.twist_linear_z": linear_speed}),
        (f"vx=-{linear_speed:g}", {"ee.twist_linear_x": -linear_speed}),
        (f"vy=-{linear_speed:g}", {"ee.twist_linear_y": -linear_speed}),
        (f"vz=-{linear_speed:g}", {"ee.twist_linear_z": -linear_speed}),
        (f"wx=+{angular_speed:g}", {"ee.twist_angular_x": angular_speed}),
        (f"wx=-{angular_speed:g}", {"ee.twist_angular_x": -angular_speed}),
        (f"wy=+{angular_speed:g}", {"ee.twist_angular_y": angular_speed}),
        (f"wy=-{angular_speed:g}", {"ee.twist_angular_y": -angular_speed}),
        (f"wz=+{angular_speed:g}", {"ee.twist_angular_z": angular_speed}),
        (f"wz=-{angular_speed:g}", {"ee.twist_angular_z": -angular_speed}),
    ]

    try:
        period = 1.0 / hz
        for label, twist in steps:
            action = {**zero, **twist}
            print(f"\n>>> {label} ({hold_s}s)")
            t_end = time.perf_counter() + hold_s
            while time.perf_counter() < t_end:
                t0 = time.perf_counter()
                robot.send_action(action)
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
            robot.send_action(zero)
        print("\n✅ 完成")
    except KeyboardInterrupt:
        print("\n⚠️ 中断")
    finally:
        robot.send_action(zero)
        robot.disconnect()


if __name__ == "__main__":
    test_send_action_twist_sequence()