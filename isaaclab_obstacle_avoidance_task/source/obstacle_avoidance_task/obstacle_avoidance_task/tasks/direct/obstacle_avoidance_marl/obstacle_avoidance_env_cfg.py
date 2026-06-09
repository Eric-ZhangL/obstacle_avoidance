"""多车路径规划避障 DirectMARLEnv 配置。"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sim import PreviewSurfaceCfg, SimulationCfg, SphereCfg
from isaaclab.utils import configclass

from .car_cfg import CAR_CFG


@configclass
class ObstacleAvoidanceMarlEnvCfg(DirectMARLEnvCfg):
    """20×20 米方形场景中的多车路径规划避障配置。"""

    dt = 1.0 / 60.0
    decimation = 2
    episode_length_s = 1000 * dt * decimation
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=0.0, replicate_physics=True)

    # 多智能体与 skrl 空间。
    robot_nums: int = 4
    action_space: int = 2
    state_space: int = -1
    possible_agents: list[str] = []
    action_spaces: dict[str, int] = {}
    observation_spaces: dict[str, int] = {}

    # 每帧观测：雷达 + 目标相对位姿 + 本车状态 + 邻车状态。
    history_len: int = 5
    lidar_rays: int = 120
    lidar_max_range: float = 8.0
    goal_range: float = 30.0
    per_robot_obs_dim: int = 0
    observation_space: int = 0

    # 车辆与动作积分参数。
    max_linear_vel: float = 2.0
    max_steering_angle: float = 0.523
    max_acc: float = 5.0
    max_steer_rate: float = 3.1416
    collision_radius: float = 0.58
    lidar_collision_half_length: float = 0.50
    lidar_collision_half_width: float = 0.28
    lidar_collision_margin: float = 0.05
    reach_goal_threshold: float = 0.6

    # 20×20 米封闭方形场景。
    arena_x_range: tuple[float, float] = (0.0, 20.0)
    arena_y_range: tuple[float, float] = (0.0, 20.0)
    wall_thickness: float = 0.5
    wall_height: float = 1.5

    # 起点和终点均随机生成，并保证彼此留有基础安全距离。
    spawn_wall_margin: float = 0.9
    robot_spawn_clearance: float = 1.4
    min_goal_separation: float = 4.0
    min_start_goal_distance: float = 8.0
    position_sample_retries: int = 128

    # 少量静态/动态障碍。
    static_obstacle_count: int = 3
    static_obstacle_radius: float = 0.5
    static_obstacle_height: float = 1.5
    dynamic_obstacle_count: int = 1
    dynamic_obstacle_radius: float = 0.45
    dynamic_obstacle_height: float = 1.5
    dynamic_speed_min: float = 0.3
    dynamic_speed_max: float = 0.9
    obstacle_clearance_margin: float = 0.2
    obstacle_sample_retries: int = 32

    # 简化奖励：靠近目标、避障、到达、碰撞、时间成本和动作平滑。
    progress_reward_scale: float = 8.0
    obstacle_d_thresh: float = 1.5
    obstacle_penalty_scale: float = 0.8
    collision_penalty: float = -20.0
    arrive_reward: float = 20.0
    time_penalty: float = -0.02
    smooth_action_weight: float = 0.05
    log_window_size: int = 100

    # 以下配置由 rebuild 根据 robot_nums 动态生成。
    robot_cfgs: dict[str, ArticulationCfg] = {}
    lidars: dict[str, MultiMeshRayCasterCfg] = {}
    goal_markers_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/navigationGoals",
        markers={},
    )

    def rebuild(self):
        """根据机器人数量重建空间和机器人/雷达配置。"""
        self.per_robot_obs_dim = self.lidar_rays + 3 + 2 + 6 * max(self.robot_nums - 1, 0)
        self.observation_space = self.history_len * self.per_robot_obs_dim
        self.state_space = self.observation_space * self.robot_nums
        self.possible_agents = [f"robot_{i}" for i in range(self.robot_nums)]
        self.action_spaces = {agent: self.action_space for agent in self.possible_agents}
        self.observation_spaces = {agent: self.observation_space for agent in self.possible_agents}

        self.robot_cfgs = {}
        self.lidars = {}
        for i, agent in enumerate(self.possible_agents):
            self.robot_cfgs[agent] = CAR_CFG.replace(prim_path=f"/World/envs/env_.*/{agent}")
            # 雷达参数可以共用，但目标列表必须排除本车，因此每辆车仍需独立配置和实例。
            targets = [
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"/World/envs/env_.*/Wall_{wall_id}")
                for wall_id in range(4)
            ]
            for j in range(self.robot_nums):
                if i != j:
                    targets.append(
                        MultiMeshRayCasterCfg.RaycastTargetCfg(
                            prim_expr=f"/World/envs/env_.*/robot_{j}/base/collisions"
                        )
                    )
            for obs_id in range(self.static_obstacle_count):
                targets.append(
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr=f"/World/envs/env_.*/StaticObstacle_{obs_id}"
                    )
                )
            for obs_id in range(self.dynamic_obstacle_count):
                targets.append(
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr=f"/World/envs/env_.*/DynamicObstacle_{obs_id}"
                    )
                )
            self.lidars[agent] = MultiMeshRayCasterCfg(
                prim_path=f"/World/envs/env_.*/{agent}/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
                mesh_prim_paths=targets,
                ray_alignment="base",
                max_distance=self.lidar_max_range,
                debug_vis=False,
                pattern_cfg=patterns.LidarPatternCfg(
                    channels=1,
                    horizontal_fov_range=(-180.0, 179.9),
                    vertical_fov_range=(0.0, 0.0),
                    horizontal_res=360.0 / self.lidar_rays,
                ),
            )

        markers = {
            f"goal_{i}": SphereCfg(
                radius=0.18,
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
            for i in range(self.robot_nums)
        }
        self.goal_markers_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/navigationGoals",
            markers=markers,
        )
