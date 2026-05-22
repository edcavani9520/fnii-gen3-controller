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

"""Kinova Gen3 Xbox 手柄遥操作配置（与 ``gamepad_control_obs.py`` 默认量级一致）。"""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("kinova_gamepad")
@dataclass
class KinovaGamepadTeleopConfig(TeleoperatorConfig):
    deadzone: float = 0.1
    max_linear_velocity_m_s: float = 0.1
    max_angular_velocity_deg_s: float = 20.0
