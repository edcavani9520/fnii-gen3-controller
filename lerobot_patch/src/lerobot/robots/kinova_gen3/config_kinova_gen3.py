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
Kinova Gen3 在 CLI / YAML 中的配置项定义。

命令行示例：

``--robot.type=kinova_gen3 --robot.ip=192.168.x.x``

仅双 UVC 相机、且未手写 ``--robot.cameras`` 时，会用 ``head_camera_index`` / ``wrist_camera_index``
及下方默认分辨率/帧率构造 OpenCV 相机（与 ``test/0515/camer_info.txt`` 中「当前生效」一致）。
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def _default_dual_uvc_cameras(cfg: "KinovaGen3RobotConfig") -> dict[str, OpenCVCameraConfig]:
    """顶部 + 腕部双路 OpenCV（V4L2），索引与格式来自 ``cfg``。"""
    common = dict(
        fps=cfg.camera_fps,
        width=cfg.camera_width,
        height=cfg.camera_height,
        fourcc=cfg.camera_fourcc,
        backend=Cv2Backends.V4L2,
    )
    return {
        "head": OpenCVCameraConfig(index_or_path=cfg.head_camera_index, **common),
        "wrist": OpenCVCameraConfig(index_or_path=cfg.wrist_camera_index, **common),
    }


@RobotConfig.register_subclass("kinova_gen3")
@dataclass
class KinovaGen3RobotConfig(RobotConfig):
    """
    Kinova Gen3 的 draccus 配置类；字段会映射为 ``--robot.<字段名>``。

    基类 ``RobotConfig`` 还提供 ``id``（多机区分）、``calibration_dir``（校准文件目录）等通用项。
    """

    # 机械臂 Kortex API 所在 IP（与 ``kinova_manage.KinovaManager`` 默认一致）；端口一般为 10000
    ip: str = "192.168.8.10"

    # 笛卡尔 Twist 限幅（与 KinovaGamepadTeleop / gamepad_control_obs 默认一致）
    max_linear_velocity_m_s: float = 0.1
    max_angular_velocity_deg_s: float = 20.0
    # 单条 Twist 有效时长 (ms)；须大于录制控制周期（5fps → 200ms/帧），默认 400ms
    twist_duration_ms: int = 400
    velocity_deadzone: float = 0.01

    # 条间复位目标关节角 (°) + 夹爪；为 None 时使用 ``kinova_manage.HOME_JOINTS``
    home_joints: list[float] | None = None

    # V4L2 / OpenCV 设备序号：见 ``test/0515/camer_info.txt``（顶部 video0、腕部 video2 为本机示例）
    head_camera_index: int = 0
    wrist_camera_index: int = 2

    # 与 camer_info 中「当前生效」一致的默认采集参数（640×480 @ 25fps，YUYV）
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 25
    camera_fourcc: str | None = "YUYV"

    # 若为空字典，则在 ``__post_init__`` 中按上面索引与参数自动填 ``head`` / ``wrist``；
    # 若自行传入，则完全使用自定义 ``cameras``，不再自动追加。
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self.cameras:
            self.cameras = _default_dual_uvc_cameras(self)
        super().__post_init__()
