from gymnasium.envs.registration import register

from quad_loco.env import QuadrupedVelocityEnv

for _env_id, _easy in (("QuadVelocity-v0", False), ("QuadVelocityEasy-v0", True)):
    try:
        register(
            id=_env_id,
            entry_point="quad_loco.env:QuadrupedVelocityEnv",
            kwargs={"easy": _easy},
        )
    except Exception:
        pass

__all__ = ["QuadrupedVelocityEnv"]
