# Isaac Lab 多车路径规划避障 Demo

基于 Isaac Lab、PyTorch 和 skrl MAPPO/IPPO 实现的多阿克曼车强化学习在线避障 Demo。

默认在 `20 × 20m` 封闭场地中生成 4 辆车、3 个静态圆柱障碍物和 1 个动态圆柱障碍物。
每次环境重置都会重新采样车辆起点、独立目标点和障碍物位置。项目不包含 A* 等全局规划器，
车辆完全根据目标相对位置、LiDAR 和邻车状态输出在线控制动作。

## 功能概览

- 多车共享策略，默认使用 MAPPO 集中训练、分布执行。
- 支持切换为 IPPO。
- 每辆车拥有独立的 120 线 LiDAR。
- 使用最近 5 帧 LiDAR 提取时序特征。
- 随机生成起点、目标点、静态障碍物和动态障碍物。
- 任意车辆发生碰撞时结束整个环境。
- 记录成功率、碰撞率和车辆到达率。

## 默认场景

| 项目 | 默认值 |
| --- | --- |
| 场地大小 | `20 × 20m` |
| 车辆数量 | `4` |
| 静态障碍物 | `3` |
| 动态障碍物 | `1` |
| 目标点最小间距 | `4m` |
| 单车起点与目标最小距离 | `8m` |
| 目标到达半径 | `0.6m` |
| 控制频率 | `30Hz` |
| 最大 episode 长度 | `1000` 个控制步，约 `33.3s` |

## 项目结构

```text
.
├── README.md                         # 标准项目说明
├── 用法说明.md                       # 日常命令速查
├── logs/                             # 训练日志与 checkpoint
└── isaaclab_obstacle_avoidance_task/
    ├── scripts/
    │   ├── visualize_scene.py        # 场景检查
    │   └── skrl/
    │       ├── train.py              # 训练入口
    │       ├── play.py               # 播放与评估入口
    │       ├── model.py              # 特征提取器、Policy 和 Value
    │       ├── MAPPO.py              # MAPPO 实现
    │       └── runner.py             # skrl 组件构建
    └── source/obstacle_avoidance_task/
        ├── config/extension.toml
        └── obstacle_avoidance_task/
            ├── assets/               # 车辆 URDF 与网格
            └── tasks/direct/obstacle_avoidance_marl/
                ├── obstacle_avoidance_env_cfg.py
                ├── obstacle_avoidance_env.py
                ├── car_cfg.py
                └── utils.py
```

`logs/`、`outputs/` 和 `__pycache__/` 等运行生成物已在 `.gitignore` 中忽略。

## 安装

运行命令前必须进入 Isaac Lab Conda 环境：

```bash
source /home/zhangl/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
cd /home/zhangl/Isaaclab20260520/obstacle_avoidance
```

首次运行或工程路径变化后，注册 editable 包：

```bash
python -m pip install -e isaaclab_obstacle_avoidance_task/source/obstacle_avoidance_task
```

检查实际导入位置：

```bash
python -c "import obstacle_avoidance_task; print(obstacle_avoidance_task.__file__)"
```

输出路径应位于当前 `obstacle_avoidance` 工程中。

## 场景检查

使用零动作检查场景生成、目标点和障碍物：

```bash
python isaaclab_obstacle_avoidance_task/scripts/visualize_scene.py \
  --task Obstacle-Avoidance-Marl-Direct-v0 \
  --num_envs 1 \
  --robot_nums 4
```

使用轻微随机动作检查车辆控制：

```bash
python isaaclab_obstacle_avoidance_task/scripts/visualize_scene.py \
  --task Obstacle-Avoidance-Marl-Direct-v0 \
  --num_envs 1 \
  --robot_nums 4 \
  --random_actions \
  --action_scale 0.2
```

## 训练

MAPPO 训练：

```bash
python isaaclab_obstacle_avoidance_task/scripts/skrl/train.py \
  --task Obstacle-Avoidance-Marl-Direct-v0 \
  --algorithm MAPPO \
  --num_envs 128 \
  --robot_nums 4 \
  --headless
```

快速冒烟测试：

```bash
python isaaclab_obstacle_avoidance_task/scripts/skrl/train.py \
  --task Obstacle-Avoidance-Marl-Direct-v0 \
  --num_envs 2 \
  --robot_nums 4 \
  --max_iterations 1 \
  --headless
```

默认训练总步数为 `5,000,000`，每 `10,000` 步保存一次 checkpoint。
日志和模型保存到：

```text
logs/obstacle_avoidance_marl/时间戳_mappo/
```

## 播放与评估

播放指定模型：

```bash
python isaaclab_obstacle_avoidance_task/scripts/skrl/play.py \
  --task Obstacle-Avoidance-Marl-Direct-v0 \
  --num_envs 1 \
  --robot_nums 4 \
  --checkpoint /path/to/agent_xxx.pt
```

无界面评估 100 个 episode：

```bash
python isaaclab_obstacle_avoidance_task/scripts/skrl/play.py \
  --task Obstacle-Avoidance-Marl-Direct-v0 \
  --num_envs 32 \
  --robot_nums 4 \
  --eval_episodes 100 \
  --checkpoint /path/to/agent_xxx.pt \
  --headless
```

## 观测与模型

每辆车每帧局部观测由以下特征组成：

| 特征 | 维度 |
| --- | ---: |
| LiDAR 距离 | `120` |
| 目标相对位置与方位角 | `3` |
| 本车速度与转向角 | `2` |
| 每辆邻车的相对位置、朝向、距离、速度和转向角 | `6` |

默认 4 辆车时：

```text
单帧局部观测维度 = 120 + 3 + 2 + 6 × 3 = 143
5 帧局部观测维度 = 143 × 5 = 715
MAPPO 全局状态维度 = 715 × 4 = 2860
```

`model.py` 中的特征提取流程：

1. 最近 5 帧 LiDAR 作为 `Conv1d` 输入通道，经过两层卷积得到 140 维雷达特征。
2. 最新帧目标与本车状态提供 5 维特征。
3. 最新帧邻车状态提供 18 维特征。
4. 拼接得到 163 维特征，分别输入 Policy 和 Value 网络。

四辆车共享同一个 Policy 和 Value。MAPPO Value 使用全部车辆观测形成的集中式状态；
IPPO Value 只使用当前车辆局部观测。

## 奖励函数

单车每步奖励：

```text
奖励 = 目标距离进度 + 近障惩罚 + 动作变化惩罚 + 时间惩罚 + 到达奖励 + 碰撞惩罚
```

| 奖励项 | 默认设置 |
| --- | ---: |
| 目标距离进度系数 | `8.0` |
| 近障阈值 | `1.5m` |
| 近障惩罚系数 | `0.8` |
| 动作平滑惩罚系数 | `0.05` |
| 每步时间惩罚 | `-0.02` |
| 首次到达奖励 | `+20` |
| 碰撞惩罚 | `-20` |

车辆首次进入目标半径后会停车等待其他车辆。若同一步同时到达和碰撞，则按碰撞失败处理。

## 终止条件

- 任意车辆与墙体、障碍物或其他车辆碰撞：环境终止，判定失败。
- 全部车辆到达各自目标：环境终止，判定成功。
- 达到 `1000` 个控制步：环境截断。

训练日志记录 `success_rate`、`collision_rate` 和 `arrival_rate`。

## 查看日志与模型

启动 TensorBoard：

```bash
tensorboard --logdir logs/obstacle_avoidance_marl --port 6006
```

查找最新 checkpoint：

```bash
find logs/obstacle_avoidance_marl -type f -name 'agent_*.pt' \
  -printf '%T@ %p\n' | sort -nr | head -1
```

`best_agent.pt` 表示训练指标最优时保存的模型，不一定是时间上最新的模型。

## 常见问题

### 找不到任务

若出现：

```text
gymnasium.error.NameNotFound: Environment `Obstacle-Avoidance-Marl-Direct` doesn't exist
```

通常表示 Python 导入了其他位置的同名包。重新执行 editable 安装，并检查实际导入路径。

### 最新日志目录没有模型

训练每 `10,000` 步保存一次模型。训练在保存节点前结束时，日志目录存在，但不会生成 checkpoint。

### Hydra 参数解析错误

多行命令中的反斜杠 `\` 必须是该行最后一个字符，后面不能带空格。

### Isaac Sim 无法启动

先确认已经执行 `conda activate isaaclab`，再检查 GPU：

```bash
nvidia-smi
```

