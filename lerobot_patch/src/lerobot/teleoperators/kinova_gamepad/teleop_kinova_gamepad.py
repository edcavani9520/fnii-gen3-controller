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
Kinova Gen3 专用 Xbox 手柄遥操作。

轴映射与 ``gamepad_control_obs.py`` 一致；夹爪 A=1.0 / B=0.0（Kortex 0–1）。
"""

from __future__ import annotations

import time
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_kinova_gamepad import KinovaGamepadTeleopConfig

ACTION_KEYS: tuple[str, ...] = (
    "ee.twist_linear_x",
    "ee.twist_linear_y",
    "ee.twist_linear_z",
    "ee.twist_angular_x",
    "ee.twist_angular_y",
    "ee.twist_angular_z",
    "gripper.position",
)


def _deadzone(value: float, threshold: float) -> float:
    return value if abs(value) > threshold else 0.0


class KinovaGamepadTeleop(Teleoperator):
    config_class = KinovaGamepadTeleopConfig
    name = "kinova_gamepad"

    def __init__(self, config: KinovaGamepadTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self._joystick = None
        self._gripper = 0.0
        self._terminate_episode = False
        self._rerecord_episode = False
        self._start_next_episode = False
        self._stop_recording = False

    @property
    def action_features(self) -> dict[str, type]:
        return {k: float for k in ACTION_KEYS}

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        require_package("pygame", extra="gamepad")
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("未检测到手柄，请连接 Xbox 手柄后重试。")
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

    def _process_button_events(self) -> None:
        """边沿触发：Y 结束本条 | X 重录 | RB 开始下一条 | Menu 停止整次录制。"""
        import pygame

        for event in pygame.event.get():
            if event.type != pygame.JOYBUTTONDOWN:
                continue
            if event.button == 3:
                self._terminate_episode = True
            elif event.button == 2:
                self._rerecord_episode = True
            elif event.button == 5:
                self._start_next_episode = True
            elif event.button == 7:
                self._stop_recording = True

    def consume_start_next_episode(self) -> bool:
        if self._start_next_episode:
            self._start_next_episode = False
            return True
        return False

    def clear_start_next_episode(self) -> None:
        self._start_next_episode = False

    def consume_stop_recording(self) -> bool:
        if self._stop_recording:
            self._stop_recording = False
            return True
        return False

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        import pygame

        pygame.event.pump()
        self._process_button_events()
        joy = self._joystick
        cfg = self.config
        dz = cfg.deadzone

        a0 = _deadzone(joy.get_axis(0), dz)
        a1 = _deadzone(joy.get_axis(1), dz)
        a2 = (joy.get_axis(2) + 1.0) / 2.0
        a3 = _deadzone(joy.get_axis(3), dz)
        a4 = _deadzone(joy.get_axis(4), dz)
        a5 = (joy.get_axis(5) + 1.0) / 2.0
        hat_x = joy.get_hat(0)[0]

        lin = cfg.max_linear_velocity_m_s
        ang = cfg.max_angular_velocity_deg_s

        if joy.get_button(0):
            self._gripper = 1.0
        elif joy.get_button(1):
            self._gripper = 0.0

        return {
            "ee.twist_linear_x": -a1 * lin,
            "ee.twist_linear_y": -a0 * lin,
            "ee.twist_linear_z": (a5 - a2) * lin,
            "ee.twist_angular_x": a3 * ang,
            "ee.twist_angular_y": -a4 * ang,
            "ee.twist_angular_z": -hat_x * ang,
            "gripper.position": self._gripper,
        }

    def is_exit_requested(self) -> bool:
        if self._joystick is None:
            return False
        import pygame

        pygame.event.pump()
        return bool(self._joystick.get_button(7))

    def get_teleop_events(self) -> dict[str, Any]:
        self._process_button_events()
        terminate = self._terminate_episode
        rerecord = self._rerecord_episode
        self._terminate_episode = False
        self._rerecord_episode = False
        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: terminate or rerecord,
            TeleopEvents.SUCCESS: terminate and not rerecord,
            TeleopEvents.RERECORD_EPISODE: rerecord,
        }

    def disconnect(self) -> None:
        if self._joystick is None:
            return
        import pygame

        self._joystick.quit()
        pygame.joystick.quit()
        pygame.quit()
        self._joystick = None

    @property
    def is_connected(self) -> bool:
        return self._joystick is not None

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict) -> None:
        pass


def _test_with_robot() -> None:
    from lerobot.robots.kinova_gen3 import KinovaGen3, KinovaGen3RobotConfig

    hz = 20.0
    period = 1.0 / hz
    zero = {k: 0.0 for k in ACTION_KEYS}

    teleop = KinovaGamepadTeleop(KinovaGamepadTeleopConfig())
    robot = KinovaGen3(KinovaGen3RobotConfig())

    teleop.connect()
    robot.connect(connect_cameras=False)
    print(f"手柄: {teleop._joystick.get_name()}")
    print("Menu 退出 | A/B 夹爪 | Y 结束本条 | X 重录 | RB 开始下一条(复位后)")

    try:
        while not teleop.is_exit_requested():
            t0 = time.perf_counter()
            action = teleop.get_action()
            sent = robot.send_action(action)
            print(f"gripper={sent['gripper.position']:.2f}  twist_max={max(abs(sent[k]) for k in ACTION_KEYS[:6]):.3f}")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        robot.send_action(zero)
        robot.disconnect()
        teleop.disconnect()
        print("已安全退出")


if __name__ == "__main__":
    _test_with_robot()
