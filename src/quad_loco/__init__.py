from gymnasium.envs.registration import registry, register

from quad_loco.env import QuadrupedVelocityEnv

if "QuadVelocity-v0" not in registry:
    register(
        id="QuadVelocity-v0",
        entry_point="quad_loco.env:QuadrupedVelocityEnv",
        kwargs={"easy": False},
    )

if "QuadVelocityEasy-v0" not in registry:
    register(
        id="QuadVelocityEasy-v0",
        entry_point="quad_loco.env:QuadrupedVelocityEnv",
        kwargs={"easy": True},
    )

__all__ = ["QuadrupedVelocityEnv"]
