"""skrl 模型：复用雷达时序特征提取器进行多车避障。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class ObstacleAvoidanceBackbone(nn.Module):
    """提取单车局部观测特征，供策略网络和价值网络分别使用。"""

    def __init__(self, env_cfg):
        """根据环境观测结构创建雷达卷积层，并计算输出特征维度。"""
        super().__init__()
        self.history_len = env_cfg.history_len
        self.lidar_rays = env_cfg.lidar_rays
        self.obs_dim = env_cfg.per_robot_obs_dim
        # 当前帧中的目标相对位姿为 3 维，本车速度和转角为 2 维。
        self.goal_vehicle_dim = 5
        self.neighbor_dim = 6 * max(env_cfg.robot_nums - 1, 0)
        self.flat_obs_dim = env_cfg.observation_space

        # 将 5 帧雷达作为 Conv1d 输入通道，沿雷达射线方向提取时序特征。
        self.conv1 = nn.Conv1d(in_channels=self.history_len, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=5, kernel_size=3, stride=2)
        # 默认 120 条射线经过两层卷积后长度为 28，因此雷达特征为 5 × 28 = 140 维。
        conv_out = ((self.lidar_rays - 5) // 2 + 1 - 3) // 2 + 1
        self.laser_feat_dim = 5 * conv_out
        self.output_dim = self.laser_feat_dim + self.goal_vehicle_dim + self.neighbor_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """融合历史雷达特征与最新一帧的目标、本车和邻车状态。"""
        x = obs.reshape(-1, self.history_len, self.obs_dim)
        laser_seq = x[:, :, : self.lidar_rays]
        laser_feat = F.relu(self.conv1(laser_seq))
        laser_feat = F.relu(self.conv2(laser_feat)).reshape(obs.shape[0], -1)

        # 非雷达状态只使用最新帧，避免重复输入变化较慢的信息。
        last_frame = x[:, -1]
        goal_vehicle = last_frame[:, self.lidar_rays : self.lidar_rays + self.goal_vehicle_dim]
        neighbor = last_frame[:, self.lidar_rays + self.goal_vehicle_dim :]
        return torch.cat([laser_feat, goal_vehicle, neighbor], dim=-1)


class PolicyModel(GaussianMixin, Model):
    """共享策略网络，输出归一化加速度和转向角速度的高斯分布。"""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        env_cfg,
        clip_actions=False,
        clip_mean_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        role="",
    ):
        """初始化 skrl 高斯策略接口、局部特征提取器和动作网络。"""
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role=role,
        )
        self.backbone = ObstacleAvoidanceBackbone(env_cfg).to(device)
        self.net = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
        ).to(device)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)
        # 每个动作维度共享所有车辆和环境的可学习探索方差。
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -1.0))

    def compute(self, inputs, role=""):
        """skrl 模型钩子：根据局部观测返回动作均值和对数标准差。"""
        obs = inputs["observations"]
        mean = torch.tanh(self.net(self.backbone(obs)))
        return mean, {"log_std": self.log_std_parameter}


class ValueModel(DeterministicMixin, Model):
    """共享价值网络；MAPPO 评估全局状态，IPPO 评估单车局部观测。"""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        env_cfg,
        centralized: bool = True,
        clip_actions=False,
        role="",
    ):
        """初始化 skrl 确定性价值接口、特征提取器和标量价值网络。"""
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)
        self.num_agents = env_cfg.robot_nums
        self.centralized = centralized
        self.backbone = ObstacleAvoidanceBackbone(env_cfg).to(device)
        self.net = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

    def compute(self, inputs, role=""):
        """skrl 模型钩子：根据 MAPPO 或 IPPO 输入计算状态价值。"""
        if self.centralized:
            # 全局状态按车辆拆分，共用骨干提取特征后取均值，得到排列不敏感的团队表示。
            states = inputs["states"]
            batch_size = states.shape[0]
            obs = states.reshape(batch_size * self.num_agents, self.backbone.flat_obs_dim)
            embedding = self.backbone(obs).reshape(batch_size, self.num_agents, -1).mean(dim=1)
        else:
            # IPPO 只使用当前智能体的局部观测。
            embedding = self.backbone(inputs["observations"])
        return self.net(embedding), {}
