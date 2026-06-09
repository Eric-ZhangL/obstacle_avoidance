"""注册多车路径规划避障任务。"""

import gymnasium as gym

from . import agents

gym.register(
    id="Obstacle-Avoidance-Marl-Direct-v0",
    entry_point=f"{__name__}.obstacle_avoidance_env:ObstacleAvoidanceMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obstacle_avoidance_env_cfg:ObstacleAvoidanceMarlEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
