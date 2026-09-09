from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from quad_loco.constants import (
    ACTION_SCALE,
    CONTROL_DT,
    DEFAULT_JOINT_POS,
    EPISODE_LENGTH_S,
    FOOT_GEOM_NAMES,
    FRAME_SKIP,
    HOME_HEIGHT,
    JOINT_NAMES,
)
from quad_loco.paths import scene_xml


@dataclass
class RewardWeights:
    track_lin_vel: float = 1.5
    track_ang_vel: float = 0.75
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -2.5
    torques: float = -2.0e-4
    action_rate: float = -0.01
    dof_acc: float = -2.5e-7
    dof_pos_limits: float = -10.0
    feet_air_time: float = 0.25
    alive: float = 0.5


class QuadrupedVelocityEnv(gym.Env):
    """Flat-terrain velocity tracking for the custom 12-DoF quadruped."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": int(1.0 / CONTROL_DT)}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        render_mode: str | None = None,
        easy: bool = False,
        command: tuple[float, float, float] | None = None,
        terminate_height: float = 0.22,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path) if xml_path is not None else scene_xml()
        if not self.xml_path.is_file():
            raise FileNotFoundError(
                f"Scene XML not found: {self.xml_path}. Run `python -m quad_loco.convert_urdf` first."
            )
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.easy = easy
        self.fixed_command = None if command is None else np.asarray(command, dtype=np.float64)
        self.terminate_height = terminate_height
        self.weights = RewardWeights()
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None

        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            missing = [n for n, i in zip(JOINT_NAMES, self.joint_ids) if i < 0]
            raise RuntimeError(f"Missing joints in MJCF: {missing}")
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.dof_adr = self.model.jnt_dofadr[self.joint_ids]
        self.act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self.default_qpos = np.array([DEFAULT_JOINT_POS[name] for name in JOINT_NAMES], dtype=np.float64)
        self.trunk_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.trunk_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "trunk_col")
        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geoms = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOM_NAMES],
            dtype=np.int32,
        )
        self.home_qpos = self.model.key("home").qpos.copy() if self.model.nkey else None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self._command = np.zeros(3, dtype=np.float64)
        self._last_action = np.zeros(12, dtype=np.float64)
        self._last_qvel = np.zeros(12, dtype=np.float64)
        self._air_time = np.zeros(4, dtype=np.float64)
        self._steps = 0
        self._max_steps = int(EPISODE_LENGTH_S / CONTROL_DT)
        self._command_resample_steps = int(5.0 / CONTROL_DT)

    def _sensor(self, name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        return self.data.sensordata[adr : adr + dim].copy()

    def _joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def _joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    def _projected_gravity(self) -> np.ndarray:
        rot = self.data.xmat[self.trunk_body].reshape(3, 3)
        return rot.T @ np.array([0.0, 0.0, -1.0])

    def _base_height(self) -> float:
        return float(self.data.qpos[2])

    def _resample_command(self) -> None:
        if self.fixed_command is not None:
            self._command = self.fixed_command.copy()
            return
        if self.easy:
            vx = self.np_random.uniform(0.2, 0.7)
            vy = self.np_random.uniform(-0.1, 0.1)
            yaw = self.np_random.uniform(-0.4, 0.4)
        else:
            vx = self.np_random.uniform(-0.4, 0.9)
            vy = self.np_random.uniform(-0.3, 0.3)
            yaw = self.np_random.uniform(-0.7, 0.7)
        self._command = np.array([vx, vy, yaw], dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        linvel = self._sensor("base_linvel")
        angvel = self._sensor("base_angvel")
        gravity = self._projected_gravity()
        qpos = self._joint_pos() - self.default_qpos
        qvel = self._joint_vel()
        obs = np.concatenate(
            [linvel, angvel, gravity, self._command, qpos, qvel, self._last_action],
            dtype=np.float32,
        )
        return np.clip(obs, -100.0, 100.0)

    def _feet_contact(self) -> np.ndarray:
        contact = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geoms = (con.geom1, con.geom2)
            for idx, foot in enumerate(self.foot_geoms):
                if foot in geoms:
                    contact[idx] = True
        return contact

    def _trunk_hit_floor(self) -> bool:
        if self.trunk_geom < 0 or self.floor_geom < 0:
            return False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geoms = (con.geom1, con.geom2)
            if self.trunk_geom in geoms and self.floor_geom in geoms:
                return True
        return False

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.default_qpos + ACTION_SCALE * action
        lo = self.model.actuator_ctrlrange[self.act_ids, 0]
        hi = self.model.actuator_ctrlrange[self.act_ids, 1]
        target = np.clip(target, lo, hi)
        self.data.ctrl[self.act_ids] = target
        return action

    def _reward(self, action: np.ndarray, prev_action: np.ndarray) -> tuple[float, dict[str, float]]:
        linvel = self._sensor("base_linvel")
        angvel = self._sensor("base_angvel")
        gravity = self._projected_gravity()
        qvel = self._joint_vel()
        qacc = (qvel - self._last_qvel) / CONTROL_DT
        torques = self.data.actuator_force[self.act_ids]

        lin_err = linvel[:2] - self._command[:2]
        yaw_err = angvel[2] - self._command[2]
        track_lin = np.exp(-np.sum(lin_err**2) / 0.25)
        track_ang = np.exp(-(yaw_err**2) / 0.25)

        qpos = self._joint_pos()
        lo = self.model.jnt_range[self.joint_ids, 0]
        hi = self.model.jnt_range[self.joint_ids, 1]
        limit_viol = np.sum(np.clip(lo - qpos, 0.0, None) + np.clip(qpos - hi, 0.0, None))

        contact = self._feet_contact()
        first_contact = contact & (self._air_time > 0)
        cmd_norm = np.linalg.norm(self._command[:2])
        air_rew = 0.0
        if cmd_norm > 0.1:
            air_rew = float(np.sum((self._air_time - 0.5) * first_contact))
        self._air_time = np.where(contact, 0.0, self._air_time + CONTROL_DT)

        terms = {
            "track_lin_vel": self.weights.track_lin_vel * float(track_lin),
            "track_ang_vel": self.weights.track_ang_vel * float(track_ang),
            "lin_vel_z": self.weights.lin_vel_z * float(linvel[2] ** 2),
            "ang_vel_xy": self.weights.ang_vel_xy * float(np.sum(angvel[:2] ** 2)),
            "orientation": self.weights.orientation * float(np.sum(gravity[:2] ** 2)),
            "torques": self.weights.torques * float(np.sum(torques**2)),
            "action_rate": self.weights.action_rate * float(np.sum((action - prev_action) ** 2)),
            "dof_acc": self.weights.dof_acc * float(np.sum(qacc**2)),
            "dof_pos_limits": self.weights.dof_pos_limits * float(limit_viol),
            "feet_air_time": self.weights.feet_air_time * air_rew,
            "alive": self.weights.alive,
        }
        return float(sum(terms.values())), terms

    def _terminated(self) -> bool:
        gravity = self._projected_gravity()
        if self._base_height() < self.terminate_height:
            return True
        if gravity[2] > -0.4:
            return True
        if self._trunk_hit_floor():
            return True
        return False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.home_qpos is not None:
            self.data.qpos[:] = self.home_qpos
        else:
            self.data.qpos[0:3] = (0.0, 0.0, HOME_HEIGHT)
            self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            self.data.qpos[self.qpos_adr] = self.default_qpos
        self.data.qvel[:] = 0.0
        self.data.qpos[0] += self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[1] += self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[self.qpos_adr] += self.np_random.uniform(-0.03, 0.03, size=12)
        self.data.ctrl[self.act_ids] = self.default_qpos
        mujoco.mj_forward(self.model, self.data)

        self._last_action[:] = 0.0
        self._last_qvel = self._joint_vel()
        self._air_time[:] = 0.0
        self._steps = 0
        self._resample_command()
        if options and "command" in options:
            self._command = np.asarray(options["command"], dtype=np.float64)
        obs = self._get_obs()
        return obs, {"command": self._command.copy()}

    def step(self, action: np.ndarray):
        prev_action = self._last_action.copy()
        applied = self._apply_action(action)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

        reward, terms = self._reward(applied, prev_action)
        self._last_action = applied
        self._last_qvel = self._joint_vel()
        self._steps += 1
        if self._steps % self._command_resample_steps == 0:
            self._resample_command()

        terminated = self._terminated()
        truncated = self._steps >= self._max_steps
        info = {
            "command": self._command.copy(),
            "base_height": self._base_height(),
            "lin_vel": self._sensor("base_linvel").copy(),
            "reward_terms": terms,
        }
        if terminated or truncated:
            info["episode_metrics"] = {
                "height": self._base_height(),
                "vx": float(self._sensor("base_linvel")[0]),
                "cmd_vx": float(self._command[0]),
            }
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
            else:
                self._viewer.sync()
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self._follow_camera())
        return self._renderer.render()

    def _follow_camera(self) -> mujoco.MjvCamera:
        """3/4 follow shot with the near-side front foot fully in frame."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = int(self.trunk_body)
        cam.distance = 2.05
        cam.azimuth = 140.0
        cam.elevation = -28.0
        return cam

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
