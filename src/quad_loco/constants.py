from __future__ import annotations

JOINT_NAMES: tuple[str, ...] = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

FOOT_BODY_NAMES: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FOOT_GEOM_NAMES: tuple[str, ...] = ("FL_foot_col", "FR_foot_col", "RL_foot_col", "RR_foot_col")

# Static-friendly crouched stand found in MuJoCo (Isaac numbers fold this robot).
DEFAULT_JOINT_POS: dict[str, float] = {
    "FL_hip_joint": 0.08,
    "FR_hip_joint": 0.08,
    "RL_hip_joint": 0.08,
    "RR_hip_joint": 0.08,
    "FL_thigh_joint": -0.50,
    "FR_thigh_joint": -0.50,
    "RL_thigh_joint": -0.50,
    "RR_thigh_joint": -0.50,
    "FL_calf_joint": 0.90,
    "FR_calf_joint": 0.90,
    "RL_calf_joint": 0.90,
    "RR_calf_joint": 0.90,
}

HOME_HEIGHT = 0.62
PHYSICS_DT = 0.005
FRAME_SKIP = 4  # 50 Hz control, matching Isaac Lab decimation
CONTROL_DT = PHYSICS_DT * FRAME_SKIP
EPISODE_LENGTH_S = 20.0
ACTION_SCALE = 0.25

# PD / actuator. Stiffer than the Isaac dump (60/1.5) so this heavier
# robot can hold a stand; 45 N·m saturation is unchanged.
ACTUATOR_KP = 100.0
ACTUATOR_KV = 3.0
ACTUATOR_FORCE = 45.0

JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "hip": (-0.9, 0.9),
    "thigh": (-1.8, 1.5),
    "calf": (0.4, 2.7),
}
