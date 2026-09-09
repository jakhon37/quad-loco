from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class ActorOnnx(nn.Module):
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.policy.get_distribution(obs)
        return dist.distribution.mean


# def export_sb3_onnx(model, path: Path, obs_dim: int = 48) -> Path:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
#     actor = ActorOnnx(model.policy).eval()
#     torch.onnx.export(
#         actor,
#         dummy,
#         str(path),
#         input_names=["obs"],
#         output_names=["actions"],
#         opset_version=17,
#         dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
#     )
#     return path
def export_sb3_onnx(model_path, path):
    import torch
    from stable_baselines3 import PPO

    # Load model on CPU
    model = PPO.load(model_path, device='cpu')
    # Define policy_net clearly for the export call
    policy_net = model.policy.to('cpu')
    policy_net.eval()

    # Create dummy input on CPU
    dummy_input = torch.randn(1, *model.observation_space.shape).to('cpu')

    torch.onnx.export(
        policy_net,      # Use the variable defined above
        dummy_input,     # Use the CPU dummy input
        path,
        opset_version=15,
        input_names=['observation'],
        output_names=['action']
    )

def verify_onnx(path: Path, model, obs: np.ndarray) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["actions"], {"obs": obs.astype(np.float32)[None, :]})[0][0]
    with torch.no_grad():
        torch_out = (
            ActorOnnx(model.policy)(torch.as_tensor(obs[None, :], dtype=torch.float32))
            .cpu()
            .numpy()[0]
        )
    return float(np.max(np.abs(onnx_out - torch_out)))
