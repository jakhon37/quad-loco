from __future__ import annotations

import numpy as np
import pytest

from quad_loco.paths import scene_xml


@pytest.fixture(scope="session")
def scene():
    path = scene_xml()
    if not path.is_file():
        from quad_loco.convert_urdf import convert

        convert()
    return path


def test_model_loads(scene):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    assert model.nq == 19  # 7 freejoint + 12 hinges
    assert model.nu == 12


def test_reset_step_obs_shape(scene):
    from quad_loco.env import QuadrupedVelocityEnv

    env = QuadrupedVelocityEnv(xml_path=scene, easy=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (48,)
    assert np.isfinite(obs).all()
    assert "command" in info
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (48,)
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert terminated in (True, False)
    assert truncated in (True, False)
    env.close()


def test_pd_hold_does_not_explode(scene):
    from quad_loco.env import QuadrupedVelocityEnv

    env = QuadrupedVelocityEnv(xml_path=scene, easy=True, command=(0.0, 0.0, 0.0))
    env.reset(seed=1)
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        assert np.isfinite(obs).all()
        if terminated:
            break
    assert info["base_height"] > 0.2
    env.close()
