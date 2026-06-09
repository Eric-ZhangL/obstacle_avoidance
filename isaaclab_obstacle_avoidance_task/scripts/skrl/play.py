"""加载 skrl checkpoint 测试多车避障任务。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 优先导入当前工程，避免其他 editable 安装覆盖本任务。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "obstacle_avoidance_task"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Obstacle-Avoidance-Marl-Direct-v0 checkpoint.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--robot_nums", type=int, default=None)
parser.add_argument("--task", type=str, default="Obstacle-Avoidance-Marl-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch"])
parser.add_argument("--algorithm", type=str, default="MAPPO", choices=["MAPPO", "IPPO"])
parser.add_argument("--eval_episodes", type=int, default=0, help="统计多少个 episode 后退出；0 表示只可视化不自动退出。")
parser.add_argument("--real-time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# 复制多行命令时可能混入纯空白参数，Hydra 无法解析这类参数。
sys.argv = [sys.argv[0]] + [arg for arg in hydra_args if arg.strip()]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import obstacle_avoidance_task.tasks  # noqa: F401
from model import PolicyModel, ValueModel
from runner import Runner


def _episode_values(info: dict, key: str) -> list[float]:
    """从 Isaac Lab extras 中取出已结束环境的 episode 指标。"""
    episode_info = info.get("episode", {})
    value = episode_info.get(key)
    finished = episode_info.get("finished")
    if value is None or finished is None or not torch.is_tensor(value) or not torch.is_tensor(finished):
        return []
    value = value.detach().cpu().flatten()
    finished = finished.detach().cpu().bool().flatten()
    return value[finished].float().tolist()


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.robot_nums is not None:
        env_cfg.robot_nums = args_cli.robot_nums
    if hasattr(env_cfg, "rebuild"):
        env_cfg.rebuild()
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed
    agent_cfg["agent"]["class"] = args_cli.algorithm
    # 播放时不写训练日志或 checkpoint，避免生成容易误认为新模型的空目录。
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    device = torch.device(env.device)
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
    agent_cfg["models"] = {
        agent: {"policy": shared_policy, "value": shared_value}
        for agent in env.possible_agents
    }

    runner = Runner(env, agent_cfg)
    runner.agent.load(retrieve_file_path(args_cli.checkpoint))
    obs, _ = env.reset()
    timestep = 0
    eval_successes = []
    eval_collisions = []
    eval_arrival_rates = []
    while simulation_app.is_running():
        with torch.inference_mode():
            states = env.state() if hasattr(env, "state") else None
            outputs = runner.agent.act(obs, states, timestep=timestep, timesteps=0)
            # 播放时优先使用均值动作，便于复现实验效果；没有均值动作时回退到采样动作。
            actions = {
                agent: outputs[-1][agent].get("mean_actions", outputs[0][agent])
                for agent in env.possible_agents
            }
            obs, _, terminated, truncated, info = env.step(actions)
            eval_successes.extend(_episode_values(info, "success"))
            eval_collisions.extend(_episode_values(info, "collision"))
            eval_arrival_rates.extend(_episode_values(info, "arrival_rate"))
            if args_cli.eval_episodes > 0 and len(eval_successes) >= args_cli.eval_episodes:
                count = args_cli.eval_episodes
                success_rate = sum(eval_successes[:count]) / count
                collision_rate = sum(eval_collisions[:count]) / count
                arrival_rate = sum(eval_arrival_rates[:count]) / count
                print(
                    "[Eval] "
                    f"episodes={count} "
                    f"success_rate={success_rate:.3f} "
                    f"collision_rate={collision_rate:.3f} "
                    f"arrival_rate={arrival_rate:.3f}"
                )
                break
            if any(v.any().item() for v in terminated.values()) or any(v.any().item() for v in truncated.values()):
                obs, _ = env.reset()
        timestep += 1
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
