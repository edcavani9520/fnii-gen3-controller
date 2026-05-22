# LeRobot Kinova 扩展

1. 将 `src/lerobot/robots/kinova_gen3` 与 `src/lerobot/teleoperators/kinova_gamepad` 复制到你本地 LeRobot 安装对应的 `src/lerobot/` 路径下。
2. 将 `patches/` 中的 diff 应用到 `robots/utils.py` 与 `teleoperators/utils.py`（或手动加入其中的 `kinova_gen3` / `kinova_gamepad` 分支）。
3. 按 LeRobot 文档完成 `uv sync` 或 `pip install -e .`。

数据采集脚本见 `scripts/readme.md`。
