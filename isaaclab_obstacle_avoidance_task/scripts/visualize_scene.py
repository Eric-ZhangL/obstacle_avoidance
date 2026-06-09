"""启动多车避障场景，用零动作或随机动作进行可视化检查。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 优先导入当前工程，避免其他 editable 安装覆盖本任务。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "obstacle_avoidance_task"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Obstacle-Avoidance-Marl-Direct-v0 before training.")
parser.add_argument("--task", type=str, default="Obstacle-Avoidance-Marl-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--robot_nums", type=int, default=4)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--random_actions", action="store_true", default=False)
parser.add_argument("--action_scale", type=float, default=0.2)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# 复制多行命令时可能混入纯空白参数，Hydra 无法解析这类参数。
sys.argv = [sys.argv[0]] + [arg for arg in hydra_args if arg.strip()]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import obstacle_avoidance_task.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg: DirectMARLEnvCfg, _agent_cfg: dict):
    """创建环境并运行少量动作，用于肉眼检查机器人、目标和障碍物。"""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.robot_nums = args_cli.robot_nums
    if hasattr(env_cfg, "rebuild"):
        env_cfg.rebuild()

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        actions = {}
        for agent in env.unwrapped.possible_agents:
            shape = (env.unwrapped.num_envs, env.unwrapped.cfg.action_space)
            if args_cli.random_actions:
                action = torch.randn(shape, device=env.unwrapped.device).clamp(-1.0, 1.0) * args_cli.action_scale
            else:
                action = torch.zeros(shape, device=env.unwrapped.device)
            actions[agent] = action
        env.step(actions)
        time.sleep(0.001)
        step += 1
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
