# openpi 训练配置说明

## 添加训练配置

在 `openpi/configs/config.py` 的 `_CONFIGS` 列表末尾添加下面的 TrainConfig（在最后一个 config 后面加一个逗号，然后粘贴下面内容）:

## 数据集 key 映射说明

LeRobotLiberoDataConfig 的 repack_transform 会做以下映射：

  LeRobot 数据集字段          → 模型内部字段
  ──────────────────────────────────────────────
  observation.images.camera   → image
  observation.state           → state
  action                      → actions
  task (language instruction) → prompt

这个映射在 openpi 的 data_config.py 里已有定义，只需在 create 时设置 repo_id 即可。