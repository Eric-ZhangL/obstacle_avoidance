"""多车避障任务的几何和车辆控制工具。"""

from __future__ import annotations

import torch


def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """四元数转 yaw，Isaac Lab 根状态四元数顺序为 wxyz。"""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    """欧拉角转四元数，输入为 [..., roll, pitch, yaw]。"""
    roll, pitch, yaw = torch.unbind(euler, dim=-1)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    return torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=-1,
    )


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """把角度归一到 [-pi, pi]。"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def vec_to_body(vec: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """世界坐标向量转车体坐标。"""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack([c * vec[:, 0] + s * vec[:, 1], -s * vec[:, 0] + c * vec[:, 1]], dim=-1)


def action_to_wheel_commands(v_x: torch.Tensor, delta_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把车体线速度和等效转向角转换为四轮转角与轮速。"""
    wheelbase = 0.64
    track_width = 0.47
    wheel_radius = 0.13
    delta_safe = delta_z + 1e-8
    radius = (wheelbase / 2.0) / torch.abs(torch.tan(delta_safe))
    omega = v_x / radius

    delta_near = torch.atan((wheelbase / 2.0) / (radius - track_width / 2.0))
    delta_far = torch.atan((wheelbase / 2.0) / (radius + track_width / 2.0))
    dist_near = torch.sqrt((radius - track_width / 2.0).square() + (wheelbase / 2.0) ** 2)
    dist_far = torch.sqrt((radius + track_width / 2.0).square() + (wheelbase / 2.0) ** 2)

    v_fl = torch.where(delta_z > 0, dist_near * omega, dist_far * omega) / wheel_radius
    v_fr = torch.where(delta_z > 0, dist_far * omega, dist_near * omega) / wheel_radius
    delta_fl = torch.where(delta_z > 0, delta_near, -delta_far)
    delta_fr = torch.where(delta_z > 0, delta_far, -delta_near)

    straight = torch.abs(delta_z) < 1e-4
    straight_v = v_x / wheel_radius
    v_fl = torch.where(straight, straight_v, v_fl)
    v_fr = torch.where(straight, straight_v, v_fr)
    delta_fl = torch.where(straight, torch.zeros_like(delta_fl), delta_fl)
    delta_fr = torch.where(straight, torch.zeros_like(delta_fr), delta_fr)

    steering = torch.stack([delta_fl, delta_fr, -delta_fl, -delta_fr], dim=-1)
    drive = torch.stack([v_fl, v_fr, v_fl, v_fr], dim=-1)
    return steering, drive
