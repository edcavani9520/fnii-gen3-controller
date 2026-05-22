# Kinova 数据采集与回放

```
scripts/
├── record.sh   →  pipeline/record.py
├── convert.sh  →  pipeline/convert.py
├── replay.sh   →  pipeline/replay.py
├── read.sh     →  pipeline/read.py
└── pipeline/   （实现代码，勿直接调用）
```

流程：**采集（twist）→ 转换（关节 Δ + 夹爪目标）→ 回放（力矩）**

## 1. 采集

编辑 `scripts/record.sh` 顶部参数，或：

```bash
DATASET_NAME=0517-220612 bash scripts/record.sh
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `DATASET_NAME` | 当前时间 | 输出目录名 |
| `NUM_EPISODES` | 20 | 条数 |
| `EPISODE_TIME_S` | 60 | 每条最长秒数 |
| `FPS` | 5 | 录制帧率 |
| `MAX_LINEAR_VELOCITY_M_S` | 0.1 | 手柄线速度上限 (m/s) |
| `TWIST_DURATION_MS` | 400 | Twist 指令有效时长 (ms)，5fps 下须 >200 |
| `SINGLE_TASK` | Kinova gamepad demo | 任务标签 |

## 2. 转换

输出 `*_torque` 的 action 为 8 维：前 7 维是相邻帧关节角差分（度），第 8 维 `gripper.position` 直接沿用源 twist 数据的夹爪目标（0.0–1.0）。

编辑 `scripts/convert.sh` 或：

```bash
bash scripts/convert.sh 0517-220612
FORCE=true bash scripts/convert.sh 0517-220612
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `DATASET_NAME` | 0517-220612 | 源数据集（输出加 `_torque`） |
| `FORCE` | false | 覆盖已有输出 |

## 3. 回放

编辑 `scripts/replay.sh` 或：

```bash
bash scripts/replay.sh 0517-220612_torque 0
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `DATASET_NAME` | 0517-220612_torque | 数据集名 |
| `EPISODE_INDEX` | 0 | episode 序号 |
| `SETTLE_S` | 2.0 | 回放结束等待秒数 |
| `ROBOT_IP` | 192.168.8.10 | 机械臂 IP（`_common.sh`） |

## 4. 查看数据

```bash
bash scripts/read.sh 0517-220612_torque 0
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `DATASET_NAME` | 0517-220612_torque | 数据集名 |
| `EPISODE_INDEX` | 0 | episode 序号 |
