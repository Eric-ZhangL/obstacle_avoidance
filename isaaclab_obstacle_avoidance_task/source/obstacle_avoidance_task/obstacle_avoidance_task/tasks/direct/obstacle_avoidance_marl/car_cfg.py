"""4WS/4WD 小车 Articulation 配置。"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
CAR_URDF = PACKAGE_ROOT / "assets/car_4ws_4wd/urdf/car_4ws_4wd.urdf"

CAR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(CAR_URDF),
        activate_contact_sensors=True,
        fix_base=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            "fl_steer_joint": 0.0,
            "fr_steer_joint": 0.0,
            "rl_steer_joint": 0.0,
            "rr_steer_joint": 0.0,
            "fl_drive_joint": 0.0,
            "fr_drive_joint": 0.0,
            "rl_drive_joint": 0.0,
            "rr_drive_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "drive_wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_drive_joint"],
            velocity_limit_sim=200.0,
            effort_limit_sim=200.0,
            stiffness=0.0,
            damping=200.0,
        ),
        "steer_wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_steer_joint"],
            velocity_limit_sim=2.0,
            effort_limit_sim=120.0,
            stiffness=1000.0,
            damping=10.0,
        ),
    },
)
