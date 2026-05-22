#!/usr/bin/env python
"""Kinova 多 episode 录制：Y 结束本条 → 无限复位等待 → RB 开始下一条。"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from lerobot.common.control_utils import init_keyboard_listener, is_headless
from lerobot.datasets import LeRobotDataset, VideoEncodingManager, safe_stop_image_writer
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import Robot, make_robot_from_config
from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.kinova_gen3 import KinovaGen3, KinovaGen3RobotConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.kinova_gamepad import KinovaGamepadTeleop, KinovaGamepadTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

HF_USER = os.environ.get("HF_USER", "local")
REPO_ID = os.environ.get("REPO_ID", f"{HF_USER}/kinova_05172020")
DATA_ROOT = Path(
    os.environ.get("DATA_ROOT", "/home/cuhk/Documents/KinovaGen3_Easy_Control/data/05172020")
)
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "10"))
EPISODE_TIME_S = float(os.environ.get("EPISODE_TIME_S", "60"))
FPS = int(os.environ.get("FPS", "5"))
SINGLE_TASK = os.environ.get("SINGLE_TASK", "Kinova gamepad demo")
ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.8.10")
DISPLAY_DATA = os.environ.get("DISPLAY_DATA", "false").lower() in ("1", "true", "yes")
PLAY_SOUNDS = os.environ.get("PLAY_SOUNDS", "false").lower() in ("1", "true", "yes")
MAX_LINEAR_VELOCITY_M_S = float(os.environ.get("MAX_LINEAR_VELOCITY_M_S", "0.1"))
MAX_ANGULAR_VELOCITY_DEG_S = float(os.environ.get("MAX_ANGULAR_VELOCITY_DEG_S", "20"))
TWIST_DURATION_MS = int(os.environ.get("TWIST_DURATION_MS", "400"))


def _say(text: str, *, blocking: bool = False) -> None:
    """终端日志；默认不播语音（无音响时可设 PLAY_SOUNDS=true 开启）。"""
    print(text, flush=True)
    log_say(text, play_sounds=PLAY_SOUNDS, blocking=blocking)

_CAM_COMMON = dict(
    width=640,
    height=480,
    fps=25,
    fourcc="YUYV",
    warmup_s=5,
    backend=Cv2Backends.V4L2,
)
CAMERAS = {
    "head": OpenCVCameraConfig(index_or_path="/dev/video0", **_CAM_COMMON),
    "wrist": OpenCVCameraConfig(index_or_path="/dev/video2", **_CAM_COMMON),
}


def _apply_kinova_gamepad_events(
    teleop: KinovaGamepadTeleop, events: dict, *, recording: bool = True
) -> None:
    if teleop.consume_stop_recording():
        events["stop_recording"] = True
        events["exit_early"] = True
        return
    if not recording:
        return
    ev = teleop.get_teleop_events()
    if ev[TeleopEvents.RERECORD_EPISODE]:
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif ev[TeleopEvents.TERMINATE_EPISODE]:
        events["exit_early"] = True


@safe_stop_image_writer
def kinova_control_loop(
    robot: Robot,
    teleop: KinovaGamepadTeleop,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    dataset: LeRobotDataset | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
) -> None:
    """单条录制或复位等待共用控制循环；control_time_s=None 表示无限等待直到 RB/→。"""
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"dataset fps ({dataset.fps}) != requested fps ({fps})")

    control_interval = 1 / fps
    timestamp = 0.0
    start_t = time.perf_counter()

    while control_time_s is None or timestamp < control_time_s:
        loop_start = time.perf_counter()

        if events["stop_recording"]:
            break

        _apply_kinova_gamepad_events(teleop, events, recording=dataset is not None)
        if events["exit_early"]:
            events["exit_early"] = False
            break
        if events["rerecord_episode"]:
            break
        if control_time_s is None:
            if teleop.consume_stop_recording():
                events["stop_recording"] = True
                break
            if teleop.consume_start_next_episode():
                break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = None
        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        act = teleop.get_action()
        act_processed = teleop_action_processor((act, obs))
        robot_action = robot_action_processor((act_processed, obs))
        robot.send_action(robot_action)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, act_processed, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed)

        dt_s = time.perf_counter() - loop_start
        sleep_s = control_interval - dt_s
        if sleep_s < 0 and dataset is not None:
            logging.warning(
                "Loop slower (%.1f Hz) than target %d Hz.", 1 / dt_s if dt_s > 0 else 0, fps
            )
        precise_sleep(max(sleep_s, 0.0))
        timestamp = time.perf_counter() - start_t


def kinova_reset_and_wait(
    *,
    robot: Robot,
    teleop: KinovaGamepadTeleop,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    display_data: bool,
    prompt: str,
) -> bool:
    """停 Twist → 回 HOME → 等待 RB。若 Menu/ESC 停止则返回 False。"""
    _say("Stopping twist and moving to home")
    if isinstance(robot, KinovaGen3):
        robot.reset_to_home()
    _say(prompt)
    teleop.clear_start_next_episode()
    kinova_control_loop(
        robot=robot,
        teleop=teleop,
        events=events,
        fps=fps,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        dataset=None,
        control_time_s=None,
        display_data=display_data,
    )
    return not events["stop_recording"]


def main() -> None:
    init_logging()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    robot_cfg = KinovaGen3RobotConfig(
        ip=ROBOT_IP,
        cameras=CAMERAS,
        max_linear_velocity_m_s=MAX_LINEAR_VELOCITY_M_S,
        max_angular_velocity_deg_s=MAX_ANGULAR_VELOCITY_DEG_S,
        twist_duration_ms=TWIST_DURATION_MS,
    )
    teleop_cfg = KinovaGamepadTeleopConfig(
        max_linear_velocity_m_s=MAX_LINEAR_VELOCITY_M_S,
        max_angular_velocity_deg_s=MAX_ANGULAR_VELOCITY_DEG_S,
    )
    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)
    if not isinstance(teleop, KinovaGamepadTeleop):
        raise TypeError("teleop must be kinova_gamepad")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    resume = DATA_ROOT.joinpath("data").is_dir() and DATA_ROOT.joinpath("meta/info.json").is_file()
    if resume:
        dataset = LeRobotDataset.resume(REPO_ID, root=DATA_ROOT)
        print(f"续录: {DATA_ROOT}")
    else:
        if DATA_ROOT.exists():
            import shutil

            print(f"删除残缺目录: {DATA_ROOT}")
            shutil.rmtree(DATA_ROOT)
        from datetime import datetime

        stamped_repo = f"{REPO_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset = LeRobotDataset.create(
            stamped_repo,
            FPS,
            root=DATA_ROOT,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
        )

    if DISPLAY_DATA:
        init_rerun(session_name="kinova_record")

    listener, events = init_keyboard_listener()
    robot.connect()
    teleop.connect()

    print(
        "\n手柄: Y=结束本条 | X=重录(回HOME) | RB=开始/下一条 | Menu=停止全部\n"
        "键盘: →=跳过等待 | ←=重录 | ESC=停止全部\n"
        "流程: 启动后先回 HOME → RB 开始第一条 → 每条结束后再回 HOME → RB 下一条\n"
    )

    recorded = 0
    try:
        with VideoEncodingManager(dataset):
            if not kinova_reset_and_wait(
                robot=robot,
                teleop=teleop,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                display_data=DISPLAY_DATA,
                prompt="Initial reset done. Arrange scene, press RB to start episode 1",
            ):
                raise SystemExit(0)

            while recorded < NUM_EPISODES and not events["stop_recording"]:
                events["exit_early"] = False
                _say(f"Recording episode {dataset.num_episodes} (target {recorded + 1}/{NUM_EPISODES})")
                kinova_control_loop(
                    robot=robot,
                    teleop=teleop,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_S,
                    single_task=SINGLE_TASK,
                    display_data=DISPLAY_DATA,
                )

                if events["rerecord_episode"]:
                    _say("Re-record episode (discarded, not saved)")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    if events["stop_recording"]:
                        if isinstance(robot, KinovaGen3):
                            robot.reset_to_home()
                        break
                    if not kinova_reset_and_wait(
                        robot=robot,
                        teleop=teleop,
                        events=events,
                        fps=FPS,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        display_data=DISPLAY_DATA,
                        prompt="Re-record: reset done. Arrange scene, press RB to record again",
                    ):
                        break
                    continue

                if not dataset.has_pending_frames():
                    _say("本条无数据，未保存（录制太短或误触 Menu？）")
                    dataset.clear_episode_buffer()
                    if events["stop_recording"]:
                        if isinstance(robot, KinovaGen3):
                            robot.reset_to_home()
                        break
                    if not kinova_reset_and_wait(
                        robot=robot,
                        teleop=teleop,
                        events=events,
                        fps=FPS,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        display_data=DISPLAY_DATA,
                        prompt="No frames saved. Press RB to try again",
                    ):
                        break
                    continue

                dataset.save_episode()
                recorded += 1
                _say(f"Saved episode {recorded}/{NUM_EPISODES}")

                if isinstance(robot, KinovaGen3):
                    robot.reset_to_home()

                if events["stop_recording"]:
                    break

                if recorded >= NUM_EPISODES:
                    break

                if not kinova_reset_and_wait(
                    robot=robot,
                    teleop=teleop,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    display_data=DISPLAY_DATA,
                    prompt="Reset environment. Arrange objects, press RB for next episode",
                ):
                    break
    finally:
        _say("Stop recording", blocking=True)
        if dataset.has_pending_frames():
            try:
                dataset.save_episode()
                recorded += 1
                _say(f"Saved in-progress episode on exit ({recorded} total)")
            except Exception as exc:
                logging.warning("Could not save in-progress episode on exit: %s", exc)
                dataset.clear_episode_buffer()
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener:
            listener.stop()
        _say("Exiting")
        print(f"已保存 {recorded} 条 episode → {DATA_ROOT}")


if __name__ == "__main__":
    main()
