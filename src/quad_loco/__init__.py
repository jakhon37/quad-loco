from gymnasium.envs.registration import register

from quad_loco.env import QuadrupedVelocityEnv

register(
    id="QuadVelocity-v0",
    entry_point="quad_loco.env:QuadrupedVelocityEnv",
    kwargs={"easy": False},
)

register(
    id="QuadVelocityEasy-v0",
    entry_point="quad_loco.env:QuadrupedVelocityEnv",
    kwargs={"easy": True},
)

__all__ = ["QuadrupedVelocityEnv"]
