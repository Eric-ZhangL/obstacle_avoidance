"""使用 skrl 训练 Isaac Lab 多车避障任务。"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# 优先导入当前工程，避免其他 editable 安装覆盖本任务。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "obstacle_avoidance_task"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Obstacle-Avoidance-Marl-Direct-v0 with skrl MAPPO/IPPO.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--robot_nums", type=int, default=None)
parser.add_argument("--task", type=str, default="Obstacle-Avoidance-Marl-Direct-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch"])
parser.add_argument("--algorithm", type=str, default="MAPPO", choices=["MAPPO", "IPPO"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
# 复制多行命令时可能混入纯空白参数，Hydra 无法解析这类参数。
sys.argv = [sys.argv[0]] + [arg for arg in hydra_args if arg.strip()]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import obstacle_avoidance_task.tasks  # noqa: F401
from model import PolicyModel, ValueModel
from runner import Runner


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    """创建环境、模型和 Runner。"""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.robot_nums is not None:
        env_cfg.robot_nums = args_cli.robot_nums
    if hasattr(env_cfg, "rebuild"):
        env_cfg.rebuild()
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["agent"]["class"] = args_cli.algorithm
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    log_root = os.path.abspath(os.path.join("logs", "obstacle_avoidance_marl"))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.algorithm.lower()}"
    log_dir = os.path.join(log_root, run_name)
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"[INFO] 训练日志目录: {log_dir}")
    print(f"[INFO] 模型保存目录: {os.path.join(log_dir, 'checkpoints')}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    device = torch.device(env.device)
    models = {}
    first_agent = env.possible_agents[0]
    obs_space = env.observation_spaces[first_agent]
    state_space = env.state_spaces[first_agent]
    action_space = env.action_spaces[first_agent]
    shared_policy = PolicyModel(obs_space, state_space, action_space, device=device, env_cfg=env_cfg)
    shared_value = ValueModel(
        obs_space,
        state_space,
        action_space,
        device=device,
        env_cfg=env_cfg,
        centralized=args_cli.algorithm == "MAPPO",
    )
    for agent in env.possible_agents:
        models[agent] = {"policy": shared_policy, "value": shared_value}
    agent_cfg["models"] = models

    start_time = time.time()
    runner = Runner(env, agent_cfg)
    if args_cli.checkpoint:
        runner.agent.load(retrieve_file_path(args_cli.checkpoint))
    runner.run()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    checkpoints = sorted(Path(log_dir, "checkpoints").glob("agent_*.pt"), key=lambda path: path.stat().st_mtime)
    if checkpoints:
        print(f"[INFO] 最新保存模型: {checkpoints[-1]}")
    else:
        checkpoint_interval = agent_cfg["agent"]["experiment"]["checkpoint_interval"]
        print(f"[WARN] 本次训练未保存模型，需要至少训练到第 {checkpoint_interval} 个控制步。")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
