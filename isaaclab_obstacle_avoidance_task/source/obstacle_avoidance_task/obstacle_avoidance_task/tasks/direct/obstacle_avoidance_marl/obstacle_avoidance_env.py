"""Isaac Lab 多车路径规划避障强化学习环境。"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import MultiMeshRayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .obstacle_avoidance_env_cfg import ObstacleAvoidanceMarlEnvCfg
from .utils import action_to_wheel_commands, euler_to_quaternion, quat_to_yaw, vec_to_body, wrap_to_pi


class ObstacleAvoidanceMarlEnv(DirectMARLEnv):
    """四车在 20×20 米方形区域内独立导航的 DirectMARLEnv。"""

    cfg: ObstacleAvoidanceMarlEnvCfg

    def __init__(self, cfg: ObstacleAvoidanceMarlEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化环境配置、车辆控制状态、观测历史和训练指标。"""
        cfg.rebuild()
        super().__init__(cfg, render_mode, **kwargs)
        self._steering_indices = [0, 1, 2, 3]
        self._drive_indices = [4, 5, 6, 7]

        shape = (self.num_envs, self.cfg.robot_nums)
        self.velocity = torch.zeros(shape, device=self.device)
        self.steering = torch.zeros(shape, device=self.device)
        self.prev_distance = torch.zeros(shape, device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, self.cfg.robot_nums, 3), device=self.device)
        self.reached_goal = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.obs_history = torch.zeros(
            (self.num_envs, self.cfg.robot_nums, self.cfg.history_len, self.cfg.per_robot_obs_dim),
            device=self.device,
        )
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.robot_nums, self.cfg.action_space), device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.cfg.robot_nums, self.cfg.action_space), device=self.device)
        self.static_obs_pos = torch.zeros((self.num_envs, self.cfg.static_obstacle_count, 2), device=self.device)
        self.dynamic_obs_pos = torch.zeros((self.num_envs, self.cfg.dynamic_obstacle_count, 2), device=self.device)
        self.dynamic_obs_vel = torch.zeros((self.num_envs, self.cfg.dynamic_obstacle_count, 2), device=self.device)
        self.success_window = deque(maxlen=self.cfg.log_window_size)
        self.collision_window = deque(maxlen=self.cfg.log_window_size)
        self.arrival_rate_window = deque(maxlen=self.cfg.log_window_size)

    # -------------------------------------------------------------------------
    # 场景创建与可视化
    # -------------------------------------------------------------------------

    def _setup_scene(self):
        """Isaac Lab 模板钩子：创建车辆、雷达、障碍物和场地。"""
        self.robots: dict[str, Articulation] = {}
        self.lidars = {}
        # 每辆车需要独立雷达：挂载位姿不同，且各雷达必须从目标网格中排除自身车体。
        for i, agent in enumerate(self.cfg.possible_agents):
            self.robots[agent] = Articulation(self.cfg.robot_cfgs[agent])
            self.scene.articulations[agent] = self.robots[agent]
            self.lidars[agent] = MultiMeshRayCaster(self.cfg.lidars[agent])
            self.scene.sensors[f"lidar_{i}"] = self.lidars[agent]

        self.goal_markers = VisualizationMarkers(self.cfg.goal_markers_cfg)
        spawn_ground_plane("/World/Ground", GroundPlaneCfg(color=(0.1, 0.1, 0.1)), translation=(0, 0, 0.0))
        self.scene.clone_environments(copy_from_source=False)
        self._spawn_walls()
        self._spawn_obstacles()
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_walls(self):
        """在场地边界创建四面静态墙。"""
        x0, x1 = self.cfg.arena_x_range
        y0, y1 = self.cfg.arena_y_range
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        t = self.cfg.wall_thickness
        h = self.cfg.wall_height
        specs = (
            ((x1 - x0, t, h), (center_x, y1 + t / 2.0, h / 2.0)),
            ((x1 - x0, t, h), (center_x, y0 - t / 2.0, h / 2.0)),
            ((t, y1 - y0, h), (x0 - t / 2.0, center_y, h / 2.0)),
            ((t, y1 - y0, h), (x1 + t / 2.0, center_y, h / 2.0)),
        )
        self.walls = {}
        for i, (size, pos) in enumerate(specs):
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Wall_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
            self.walls[f"wall_{i}"] = RigidObject(cfg)

    def _spawn_obstacles(self):
        """创建静态和动态圆柱障碍物。"""
        self.static_obstacles = {}
        for i in range(self.cfg.static_obstacle_count):
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/StaticObstacle_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=self.cfg.static_obstacle_radius,
                    height=self.cfg.static_obstacle_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.0)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(999.0, 999.0, -10.0)),
            )
            self.static_obstacles[f"static_{i}"] = RigidObject(cfg)

        self.dynamic_obstacles = {}
        for i in range(self.cfg.dynamic_obstacle_count):
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/DynamicObstacle_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=self.cfg.dynamic_obstacle_radius,
                    height=self.cfg.dynamic_obstacle_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.7, 1.0)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(999.0, 999.0, -10.0)),
            )
            self.dynamic_obstacles[f"dynamic_{i}"] = RigidObject(cfg)

    def _visualize_goals(self):
        """更新所有环境中的目标点可视化标记。"""
        translations = self.goal_pos.reshape(-1, 3)
        marker_indices = torch.arange(self.cfg.robot_nums).repeat(self.num_envs)
        self.goal_markers.visualize(
            translations=translations.detach().cpu().numpy(),
            marker_indices=marker_indices.numpy(),
        )

    # -------------------------------------------------------------------------
    # Isaac Lab 每步调用的模板钩子
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Isaac Lab 模板钩子：在物理步前积分动作，并更新动态障碍物。"""
        dt = self.cfg.dt * self.cfg.decimation
        self.prev_actions.copy_(self.last_actions)
        for i, agent in enumerate(self.cfg.possible_agents):
            action = actions[agent].clamp(-1.0, 1.0)
            self.last_actions[:, i] = action
            self.velocity[:, i] = (self.velocity[:, i] + action[:, 0] * self.cfg.max_acc * dt).clamp(
                -self.cfg.max_linear_vel, self.cfg.max_linear_vel
            )
            self.steering[:, i] = (
                self.steering[:, i] + action[:, 1] * self.cfg.max_steer_rate * dt
            ).clamp(-self.cfg.max_steering_angle, self.cfg.max_steering_angle)
            # 已到达车辆保持静止，避免在等待其他车辆时离开终点。
            self.velocity[:, i] = torch.where(self.reached_goal[:, i], 0.0, self.velocity[:, i])
            self.steering[:, i] = torch.where(self.reached_goal[:, i], 0.0, self.steering[:, i])
        self._move_dynamic_obstacles()

    def _apply_action(self) -> None:
        """Isaac Lab 模板钩子：把车辆速度和转角转换为关节控制指令。"""
        for i, agent in enumerate(self.cfg.possible_agents):
            steering, drive = action_to_wheel_commands(self.velocity[:, i], self.steering[:, i])
            self.robots[agent].set_joint_position_target(steering, self._steering_indices)
            self.robots[agent].set_joint_velocity_target(drive, self._drive_indices)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Isaac Lab 模板钩子：更新观测历史并返回各智能体局部观测。"""
        obs_now = self._compute_current_observations()
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=2)
        self.obs_history[:, :, -1] = obs_now
        return {
            agent: self.obs_history[:, i].reshape(self.num_envs, -1)
            for i, agent in enumerate(self.cfg.possible_agents)
        }

    def _get_states(self) -> torch.Tensor:
        """Isaac Lab 模板钩子：拼接全部车辆观测，作为 MAPPO 集中式状态。"""
        return self.obs_history.reshape(self.num_envs, -1)

    # -------------------------------------------------------------------------
    # 奖励、终止条件与训练指标
    # -------------------------------------------------------------------------

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Isaac Lab 模板钩子：计算进度、避障、平滑、到达和碰撞奖励。"""
        pos, yaw = self._robot_pose()
        _, _, dist = self._target_relative_pose(pos, yaw)
        lidar = self._lidar_scan()
        min_lidar = lidar.min(dim=-1).values
        progress = (self.prev_distance - dist) * self.cfg.progress_reward_scale
        obstacle = -self.cfg.obstacle_penalty_scale * (
            (self.cfg.obstacle_d_thresh - min_lidar).clamp_min(0.0) / self.cfg.obstacle_d_thresh
        ).square()
        action_delta = self.last_actions - self.prev_actions
        smooth = -self.cfg.smooth_action_weight * action_delta.square().sum(dim=-1)

        collision = self._collision_mask(pos, lidar)
        newly_reached = (~self.reached_goal) & (dist < self.cfg.reach_goal_threshold)
        self.reached_goal |= newly_reached
        done_reward = torch.where(newly_reached, torch.full_like(dist, self.cfg.arrive_reward), torch.zeros_like(dist))
        # 同一步既到达又碰撞时按失败处理，碰撞惩罚优先。
        done_reward = torch.where(collision, torch.full_like(dist, self.cfg.collision_penalty), done_reward)

        active = ~self.reached_goal
        reward = torch.where(
            active,
            progress + obstacle + smooth + self.cfg.time_penalty,
            torch.zeros_like(dist),
        ) + done_reward
        self.prev_distance = dist.detach()
        return {agent: reward[:, i] for i, agent in enumerate(self.cfg.possible_agents)}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Isaac Lab 模板钩子：判断碰撞、全部到达和超时，并记录 episode 指标。"""
        pos, yaw = self._robot_pose()
        _, _, dist = self._target_relative_pose(pos, yaw)
        lidar = self._lidar_scan()
        collision = self._collision_mask(pos, lidar)
        collision_any = collision.any(dim=1)
        self.reached_goal |= dist < self.cfg.reach_goal_threshold
        all_reached = self.reached_goal.all(dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = collision_any | all_reached
        finished = done | time_out
        success = all_reached & (~collision_any)
        arrival_rate = self.reached_goal.float().mean(dim=1)

        self._append_episode_log_window(
            finished=finished,
            success=success,
            collision=collision_any,
            arrival_rate=arrival_rate,
        )
        self.extras["log"] = {
            "success_rate": self._window_mean(self.success_window),
            "collision_rate": self._window_mean(self.collision_window),
            "arrival_rate": self._window_mean(self.arrival_rate_window),
        }
        self.extras["episode"] = {
            # play/eval 脚本只统计 finished=True 的环境，避免把未结束 episode 混进结果。
            "finished": finished,
            "success": success,
            "collision": collision_any,
            "arrival_rate": arrival_rate,
        }
        terminated = {agent: done for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _append_episode_log_window(
        self,
        *,
        finished: torch.Tensor,
        success: torch.Tensor,
        collision: torch.Tensor,
        arrival_rate: torch.Tensor,
    ):
        """只在 episode 结束时，把结果追加到滑动统计窗口。"""
        if not finished.any():
            return
        mask = finished.detach().cpu()
        self.success_window.extend(success.detach().cpu()[mask].float().tolist())
        self.collision_window.extend(collision.detach().cpu()[mask].float().tolist())
        self.arrival_rate_window.extend(arrival_rate.detach().cpu()[mask].float().tolist())

    def _window_mean(self, window: deque) -> torch.Tensor:
        """把 Python 滑动窗口转成 skrl 可记录的标量张量。"""
        if len(window) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.tensor(sum(window) / len(window), device=self.device)

    # -------------------------------------------------------------------------
    # 环境重置与位置采样
    # -------------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Isaac Lab 模板钩子：重置指定环境中的车辆、目标、障碍物和历史观测。"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        if n == 0:
            return

        self.velocity[env_ids] = 0.0
        self.steering[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.obs_history[env_ids] = 0.0
        self.reached_goal[env_ids] = False

        starts = self._sample_robot_positions(n)
        goals = self._sample_goal_positions(starts)
        self.goal_pos[env_ids, :, :2] = goals
        self.goal_pos[env_ids, :, 2] = 0.0

        for i, agent in enumerate(self.cfg.possible_agents):
            rpy = torch.zeros((n, 3), device=self.device)
            diff = goals[:, i] - starts[:, i]
            rpy[:, 2] = torch.atan2(diff[:, 1], diff[:, 0])
            root_position = torch.cat([starts[:, i], torch.full((n, 1), 0.4, device=self.device)], dim=-1)
            root_pose = torch.cat([root_position, euler_to_quaternion(rpy)], dim=-1)
            robot = self.robots[agent]
            robot.write_root_pose_to_sim(root_pose, env_ids)
            robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=self.device), env_ids)
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos[env_ids],
                robot.data.default_joint_vel[env_ids],
                None,
                env_ids,
            )

        self._reset_obstacles(env_ids, torch.cat([starts, goals], dim=1))
        self._visualize_goals()
        pos, yaw = self._robot_pose()
        _, _, dist = self._target_relative_pose(pos, yaw)
        self.prev_distance[env_ids] = dist[env_ids]
        obs = self._compute_current_observations()
        self.obs_history[env_ids] = obs[env_ids].unsqueeze(2).repeat(1, 1, self.cfg.history_len, 1)

    def _sample_goal_positions(self, starts: torch.Tensor) -> torch.Tensor:
        """整组采样终点，保证目标间距和每辆车的导航距离。"""
        n = starts.shape[0]
        goals = torch.zeros_like(starts)
        valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        for _ in range(self.cfg.position_sample_retries):
            candidate = self._sample_robot_positions(
                n,
                avoid=starts,
                position_clearance=self.cfg.min_goal_separation,
            )
            candidate_valid = (
                torch.norm(candidate - starts, dim=-1) >= self.cfg.min_start_goal_distance
            ).all(dim=1)
            use = (~valid) & candidate_valid
            goals[use] = candidate[use]
            valid |= use
            if valid.all():
                break
        if not valid.all():
            goals[~valid] = candidate[~valid]
        return goals

    def _sample_robot_positions(
        self,
        n: int,
        avoid: torch.Tensor | None = None,
        position_clearance: float | None = None,
    ) -> torch.Tensor:
        """在场地内部按车辆间距逐个采样位置。"""
        x0, x1 = self.cfg.arena_x_range
        y0, y1 = self.cfg.arena_y_range
        margin = self.cfg.spawn_wall_margin
        clearance = self.cfg.robot_spawn_clearance if position_clearance is None else position_clearance
        positions = torch.zeros((n, self.cfg.robot_nums, 2), device=self.device)
        for robot_id in range(self.cfg.robot_nums):
            candidate = torch.zeros((n, 2), device=self.device)
            valid = torch.zeros(n, dtype=torch.bool, device=self.device)
            for _ in range(self.cfg.position_sample_retries):
                proposal = torch.stack(
                    [
                        x0 + margin + (x1 - x0 - 2.0 * margin) * torch.rand(n, device=self.device),
                        y0 + margin + (y1 - y0 - 2.0 * margin) * torch.rand(n, device=self.device),
                    ],
                    dim=-1,
                )
                clear = torch.ones(n, dtype=torch.bool, device=self.device)
                if robot_id > 0:
                    clear &= (
                        torch.norm(proposal[:, None, :] - positions[:, :robot_id], dim=-1)
                        > clearance
                    ).all(dim=1)
                if avoid is not None:
                    clear &= (
                        torch.norm(proposal[:, None, :] - avoid, dim=-1) > self.cfg.robot_spawn_clearance
                    ).all(dim=1)
                use = (~valid) & clear
                candidate[use] = proposal[use]
                valid |= use
                if valid.all():
                    break
            if not valid.all():
                candidate[~valid] = proposal[~valid]
            positions[:, robot_id] = candidate
        return positions

    def _reset_obstacles(self, env_ids: torch.Tensor, protected_positions: torch.Tensor):
        """重新采样静态和动态障碍物，并把状态写入仿真。"""
        n = len(env_ids)
        if self.cfg.static_obstacle_count > 0:
            self.static_obs_pos[env_ids] = self._sample_non_overlapping_obstacles(
                env_ids=env_ids,
                count=self.cfg.static_obstacle_count,
                radius=self.cfg.static_obstacle_radius,
                robot_starts=protected_positions,
            )
            for i, obj in self.static_obstacles.items():
                idx = int(i.split("_")[-1])
                state = torch.zeros((n, 13), device=self.device)
                state[:, 3] = 1.0
                state[:, :2] = self.static_obs_pos[env_ids, idx]
                state[:, 2] = self.cfg.static_obstacle_height / 2.0
                obj.write_root_state_to_sim(state, env_ids)

        if self.cfg.dynamic_obstacle_count > 0:
            self.dynamic_obs_pos[env_ids] = self._sample_non_overlapping_obstacles(
                env_ids=env_ids,
                count=self.cfg.dynamic_obstacle_count,
                radius=self.cfg.dynamic_obstacle_radius,
                robot_starts=protected_positions,
                other_positions=self.static_obs_pos[env_ids],
                other_radius=self.cfg.static_obstacle_radius,
            )
            speed_rand = torch.rand((n, self.cfg.dynamic_obstacle_count), device=self.device)
            speed = self.cfg.dynamic_speed_min + (
                self.cfg.dynamic_speed_max - self.cfg.dynamic_speed_min
            ) * speed_rand
            heading = 2.0 * math.pi * torch.rand((n, self.cfg.dynamic_obstacle_count), device=self.device)
            self.dynamic_obs_vel[env_ids, :, 0] = torch.cos(heading) * speed
            self.dynamic_obs_vel[env_ids, :, 1] = torch.sin(heading) * speed
            self._write_dynamic_obstacles(env_ids)

    def _sample_non_overlapping_obstacles(
        self,
        *,
        env_ids: torch.Tensor,
        count: int,
        radius: float,
        robot_starts: torch.Tensor,
        other_positions: torch.Tensor | None = None,
        other_radius: float = 0.0,
    ) -> torch.Tensor:
        """采样障碍物位置，避免与机器人、已采样障碍物和已有障碍物重叠。"""
        n = len(env_ids)
        x0, x1 = self.cfg.arena_x_range
        y0, y1 = self.cfg.arena_y_range
        x0 += self.cfg.spawn_wall_margin
        x1 -= self.cfg.spawn_wall_margin
        y0 += self.cfg.spawn_wall_margin
        y1 -= self.cfg.spawn_wall_margin
        positions = torch.zeros((n, count, 2), device=self.device)
        margin = self.cfg.obstacle_clearance_margin
        robot_min_dist = self.cfg.collision_radius + radius + margin
        same_min_dist = 2.0 * radius + margin
        other_min_dist = radius + other_radius + margin

        for obstacle_id in range(count):
            candidate = torch.zeros((n, 2), device=self.device)
            valid = torch.zeros(n, dtype=torch.bool, device=self.device)
            for _ in range(self.cfg.obstacle_sample_retries):
                proposal = torch.stack(
                    [
                        x0 + (x1 - x0) * torch.rand(n, device=self.device),
                        y0 + (y1 - y0) * torch.rand(n, device=self.device),
                    ],
                    dim=-1,
                )
                robot_clear = (torch.norm(proposal[:, None, :] - robot_starts, dim=-1) > robot_min_dist).all(dim=1)
                if obstacle_id > 0:
                    same_clear = (
                        torch.norm(proposal[:, None, :] - positions[:, :obstacle_id, :], dim=-1) > same_min_dist
                    ).all(dim=1)
                else:
                    same_clear = torch.ones(n, dtype=torch.bool, device=self.device)
                if other_positions is not None and other_positions.shape[1] > 0:
                    other_clear = (
                        torch.norm(proposal[:, None, :] - other_positions, dim=-1) > other_min_dist
                    ).all(dim=1)
                else:
                    other_clear = torch.ones(n, dtype=torch.bool, device=self.device)
                use = (~valid) & robot_clear & same_clear & other_clear
                candidate[use] = proposal[use]
                valid |= use
                if valid.all():
                    break

            if not valid.all():
                fallback = torch.stack(
                    [
                        x0 + (x1 - x0) * torch.rand(n, device=self.device),
                        y0 + (y1 - y0) * torch.rand(n, device=self.device),
                    ],
                    dim=-1,
                )
                candidate[~valid] = fallback[~valid]
            positions[:, obstacle_id] = candidate
        return positions

    # -------------------------------------------------------------------------
    # 动态障碍物更新
    # -------------------------------------------------------------------------

    def _move_dynamic_obstacles(self):
        """按当前速度移动动态障碍物，并在墙体或障碍物干涉时反弹。"""
        if self.cfg.dynamic_obstacle_count <= 0:
            return
        dt = self.cfg.dt * self.cfg.decimation
        prev_pos = self.dynamic_obs_pos.clone()
        prev_vel = self.dynamic_obs_vel.clone()
        self.dynamic_obs_pos += self.dynamic_obs_vel * dt
        ox0, ox1 = self.cfg.arena_x_range
        oy0, oy1 = self.cfg.arena_y_range
        ox0 += self.cfg.spawn_wall_margin
        ox1 -= self.cfg.spawn_wall_margin
        oy0 += self.cfg.spawn_wall_margin
        oy1 -= self.cfg.spawn_wall_margin
        hit_x = (self.dynamic_obs_pos[:, :, 0] < ox0) | (self.dynamic_obs_pos[:, :, 0] > ox1)
        hit_y = (self.dynamic_obs_pos[:, :, 1] < oy0) | (self.dynamic_obs_pos[:, :, 1] > oy1)
        self.dynamic_obs_vel[:, :, 0] = torch.where(hit_x, -self.dynamic_obs_vel[:, :, 0], self.dynamic_obs_vel[:, :, 0])
        self.dynamic_obs_vel[:, :, 1] = torch.where(hit_y, -self.dynamic_obs_vel[:, :, 1], self.dynamic_obs_vel[:, :, 1])
        self.dynamic_obs_pos[:, :, 0] = self.dynamic_obs_pos[:, :, 0].clamp(ox0, ox1)
        self.dynamic_obs_pos[:, :, 1] = self.dynamic_obs_pos[:, :, 1].clamp(oy0, oy1)
        interference = self._dynamic_obstacle_interference()
        if interference.any():
            self.dynamic_obs_pos = torch.where(interference.unsqueeze(-1), prev_pos, self.dynamic_obs_pos)
            self.dynamic_obs_vel = torch.where(interference.unsqueeze(-1), -prev_vel, self.dynamic_obs_vel)
        self._write_dynamic_obstacles(torch.arange(self.num_envs, device=self.device))

    def _dynamic_obstacle_interference(self) -> torch.Tensor:
        """检测动态障碍之间、动态障碍与静态障碍之间的干涉。"""
        margin = self.cfg.obstacle_clearance_margin
        interference = torch.zeros(
            (self.num_envs, self.cfg.dynamic_obstacle_count),
            dtype=torch.bool,
            device=self.device,
        )
        if self.cfg.dynamic_obstacle_count > 1:
            pair_dist = torch.cdist(self.dynamic_obs_pos, self.dynamic_obs_pos)
            eye = torch.eye(self.cfg.dynamic_obstacle_count, dtype=torch.bool, device=self.device).view(
                1, self.cfg.dynamic_obstacle_count, self.cfg.dynamic_obstacle_count
            )
            pair_dist = pair_dist.masked_fill(eye, 1e6)
            interference |= pair_dist.min(dim=-1).values < (2.0 * self.cfg.dynamic_obstacle_radius + margin)

        if self.cfg.static_obstacle_count > 0:
            static_dist = torch.cdist(self.dynamic_obs_pos, self.static_obs_pos)
            interference |= static_dist.min(dim=-1).values < (
                self.cfg.dynamic_obstacle_radius + self.cfg.static_obstacle_radius + margin
            )
        return interference

    def _write_dynamic_obstacles(self, env_ids: torch.Tensor):
        """把指定环境中的动态障碍物位置和速度写入仿真。"""
        for i, obj in self.dynamic_obstacles.items():
            idx = int(i.split("_")[-1])
            state = torch.zeros((len(env_ids), 13), device=self.device)
            state[:, 3] = 1.0
            state[:, :2] = self.dynamic_obs_pos[env_ids, idx]
            state[:, 2] = self.cfg.dynamic_obstacle_height / 2.0
            state[:, 7:9] = self.dynamic_obs_vel[env_ids, idx]
            obj.write_root_state_to_sim(state, env_ids)

    # -------------------------------------------------------------------------
    # 观测与特征构造
    # -------------------------------------------------------------------------

    def _robot_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """读取全部环境中所有车辆的二维位置和航向角。"""
        pos = torch.stack([self.robots[a].data.root_state_w[:, :2] for a in self.cfg.possible_agents], dim=1)
        yaw = torch.stack(
            [quat_to_yaw(self.robots[a].data.root_state_w[:, 3:7]) for a in self.cfg.possible_agents],
            dim=1,
        )
        return pos, yaw

    def _target_relative_pose(
        self,
        pos: torch.Tensor,
        yaw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算目标在各车辆坐标系下的位置、方位角和距离。"""
        delta = self.goal_pos[:, :, :2] - pos
        flat_delta = delta.reshape(-1, 2)
        flat_yaw = yaw.reshape(-1)
        body = vec_to_body(flat_delta, flat_yaw).view(self.num_envs, self.cfg.robot_nums, 2)
        theta_rel = torch.atan2(body[:, :, 1], body[:, :, 0])
        dist = torch.norm(body, dim=-1)
        return body, theta_rel, dist

    def _lidar_scan(self) -> torch.Tensor:
        """读取各车辆独立雷达，并转换为固定长度的距离扫描。"""
        scans = []
        for agent in self.cfg.possible_agents:
            lidar = self.lidars[agent]
            lidar_pos = lidar.data.pos_w
            hits = lidar.data.ray_hits_w
            finite = torch.isfinite(hits).all(dim=-1)
            rel = hits - lidar_pos[:, None, :]
            dist = torch.norm(rel, dim=-1)
            dist = torch.where(
                finite,
                dist.clamp_max(self.cfg.lidar_max_range),
                torch.full_like(dist, self.cfg.lidar_max_range),
            )
            scans.append(dist[:, : self.cfg.lidar_rays])
        return torch.stack(scans, dim=1)

    def _compute_current_observations(self) -> torch.Tensor:
        """构造当前帧观测：雷达、目标、本车状态和邻车状态。"""
        pos, yaw = self._robot_pose()
        target_body, theta_rel, _ = self._target_relative_pose(pos, yaw)
        lidar = self._lidar_scan()
        lidar_norm = (lidar.clamp(0.0, self.cfg.lidar_max_range) / self.cfg.lidar_max_range) * 2.0 - 1.0
        target = torch.cat(
            [
                (target_body / self.cfg.goal_range).clamp(-1.0, 1.0),
                (theta_rel.unsqueeze(-1) / math.pi).clamp(-1.0, 1.0),
            ],
            dim=-1,
        )
        vehicle = torch.stack(
            [
                self.velocity / self.cfg.max_linear_vel,
                self.steering / self.cfg.max_steering_angle,
            ],
            dim=-1,
        )
        neighbors = self._neighbor_observation(pos, yaw)
        return torch.cat([lidar_norm, target, vehicle, neighbors], dim=-1).clamp(-1.0, 1.0)

    def _neighbor_observation(self, pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """为每辆车构造其他车辆的相对位姿和运动状态特征。"""
        features = []
        for i in range(self.cfg.robot_nums):
            ego_pos = pos[:, i]
            ego_yaw = yaw[:, i]
            items = []
            for j in range(self.cfg.robot_nums):
                if i == j:
                    continue
                rel = vec_to_body(pos[:, j] - ego_pos, ego_yaw)
                theta = wrap_to_pi(yaw[:, j] - ego_yaw)
                dist = torch.norm(rel, dim=-1)
                item = torch.stack(
                    [
                        (rel[:, 0] / self.cfg.goal_range).clamp(-1.0, 1.0),
                        (rel[:, 1] / self.cfg.goal_range).clamp(-1.0, 1.0),
                        (theta / math.pi).clamp(-1.0, 1.0),
                        (dist / self.cfg.goal_range).clamp(0.0, 1.0),
                        (self.velocity[:, j] / self.cfg.max_linear_vel).clamp(-1.0, 1.0),
                        (self.steering[:, j] / self.cfg.max_steering_angle).clamp(-1.0, 1.0),
                    ],
                    dim=-1,
                )
                items.append(item)
            features.append(torch.cat(items, dim=-1) if items else torch.zeros((self.num_envs, 0), device=self.device))
        return torch.stack(features, dim=1)

    # -------------------------------------------------------------------------
    # 碰撞检测
    # -------------------------------------------------------------------------

    def _collision_mask(self, pos: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        """合并车体、墙体、障碍物和车辆间碰撞结果。"""
        lidar_collision = self._lidar_rect_collision(lidar)
        obstacle_collision = lidar_collision | self._analytic_obstacle_collision(pos)
        pair_dist = torch.cdist(pos, pos)
        eye = torch.eye(self.cfg.robot_nums, dtype=torch.bool, device=self.device).view(
            1,
            self.cfg.robot_nums,
            self.cfg.robot_nums,
        )
        pair_dist = pair_dist.masked_fill(eye, 1e6)
        robot_collision = pair_dist.min(dim=-1).values < (2.0 * self.cfg.collision_radius)
        return obstacle_collision | robot_collision

    def _lidar_rect_collision(self, lidar: torch.Tensor) -> torch.Tensor:
        """按车体矩形边界判断 LiDAR 近距离碰撞，减少左右两侧外接圆保守量。"""
        angles = torch.linspace(-math.pi, math.pi, self.cfg.lidar_rays, device=self.device).view(1, 1, -1)
        cos_abs = torch.cos(angles).abs().clamp_min(1e-6)
        sin_abs = torch.sin(angles).abs().clamp_min(1e-6)
        length_limit = self.cfg.lidar_collision_half_length / cos_abs
        width_limit = self.cfg.lidar_collision_half_width / sin_abs
        rect_limit = torch.minimum(length_limit, width_limit) + self.cfg.lidar_collision_margin
        return (lidar < rect_limit).any(dim=-1)

    def _analytic_obstacle_collision(self, pos: torch.Tensor) -> torch.Tensor:
        """用几何距离补充物理碰撞判定，避免只依赖 LiDAR 漏检。"""
        robot_radius = self.cfg.collision_radius
        x0, x1 = self.cfg.arena_x_range
        y0, y1 = self.cfg.arena_y_range
        wall_collision = (
            (pos[:, :, 0] < x0 + robot_radius)
            | (pos[:, :, 0] > x1 - robot_radius)
            | (pos[:, :, 1] < y0 + robot_radius)
            | (pos[:, :, 1] > y1 - robot_radius)
        )

        obstacle_collision = wall_collision
        if self.cfg.static_obstacle_count > 0:
            static_dist = torch.norm(pos[:, :, None, :] - self.static_obs_pos[:, None, :, :], dim=-1)
            static_threshold = robot_radius + self.cfg.static_obstacle_radius
            obstacle_collision |= (static_dist < static_threshold).any(dim=-1)

        if self.cfg.dynamic_obstacle_count > 0:
            dynamic_dist = torch.norm(pos[:, :, None, :] - self.dynamic_obs_pos[:, None, :, :], dim=-1)
            dynamic_threshold = robot_radius + self.cfg.dynamic_obstacle_radius
            obstacle_collision |= (dynamic_dist < dynamic_threshold).any(dim=-1)

        return obstacle_collision
