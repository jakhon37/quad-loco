# What this project trains

`quad-loco` trains a neural network to make a **custom 12-DoF quadruped** walk by tracking a velocity command on flat ground, in simulation.

The policy does **not** output “walk forward” as a single button. At 50 Hz it outputs **12 joint position targets**. A PD controller in the simulator turns those targets into motor torques. If training works, the network learns a gait that realizes commanded `v_x`, `v_y`, and yaw rate `ω_z`.

NVIDIA GPU is **not** required. MuJoCo physics and the small MLP both run on CPU.

```
command (vx, vy, yaw)
        │
        ▼
  observation (48)  ──►  PPO policy (MLP)  ──►  12 actions in [-1, 1]
        ▲                         │
        │                         ▼
        │              q_target = q_stand + 0.25 · action
        │                         │
        │                         ▼
        └──────────  MuJoCo + PD motors (50 Hz)  ──────────┘
```

---

## Robot

Not a Unitree Go1/Go2. It is a custom quadruped (meshes/URDF from a public model) with three revolute joints per leg:

| Joint | Role |
|---|---|
| hip | abduction / adduction |
| thigh | hip flexion |
| calf | knee |

| | |
|---|---|
| Actuated DoF | 12 |
| Mass | ~22 kg (trunk 9.7 kg) |
| Default stand | ~0.62–0.65 m |
| Episode | 20 s or until fall (`height < 0.22 m` or large tilt) |

Motors are MuJoCo **position actuators**: `kp = 100`, `kv = 3`, torque limit `45 N·m`. Physics steps at `dt = 0.005 s`; the policy is queried every 4 steps (**50 Hz**).

---

## What is learned

**Algorithm:** PPO (Proximal Policy Optimization), on-policy actor–critic.

**Network:** MLP, hidden sizes `[256, 256]`, shared actor/critic (Stable-Baselines3 `MlpPolicy`).

**Observation (48 floats):**

| Block | Dim | Meaning |
|---|---|---|
| Base linear velocity | 3 | body frame (from IMU site) |
| Base angular velocity | 3 | gyro |
| Projected gravity | 3 | gravity in body frame (tilt) |
| Command | 3 | `(vx, vy, ω_z)` |
| Joint position | 12 | relative to the default stand pose |
| Joint velocity | 12 | |
| Last action | 12 | previous policy output |

**Action (12 floats)** in `[-1, 1]`:

```
q_target = q_default + 0.25 · action
```

`easy=True` (smoke / Colab default) samples mostly forward commands. Full mode samples a wider `vx, vy, yaw` box.

**Reward (weighted sum, Isaac-style velocity tracking):**

| Term | Intent |
|---|---|
| Exponential `vx, vy` tracking | follow the linear command |
| Exponential yaw-rate tracking | follow `ω_z` |
| Penalty on `vz`, roll/pitch rate, tilt | stay upright |
| Torque, action-rate, joint-acceleration penalties | smoother, cheaper motion |
| Joint limit penalty | stay in range |
| Feet air time | encourage stepping, not shuffling |
| Alive bonus | stay in the episode |

**Success to watch while training** (`progress.csv` and the one-line log):

| Metric | Healthy |
|---|---|
| `rew` | rising |
| `len` | toward **1000** (20 s at 50 Hz = no fall) |
| `ev` (explained variance) | toward **1** |
| `kl` | around **0.01** |
| `ent` | slowly falling, not collapsing to 0 |

An 8k-step smoke run typically **stands** (`len` short, `|vx − cmd|` still large). A usable walk needs on the order of **1.5×10^6** steps (tens of minutes on a laptop CPU).

A local CPU run at 1.5M steps (`logs/local_walk`) reached eval length **1000 / 1000**, mean return **~2400**, and `|v_x − 0.5| ≈ 0.045` m/s. The clip in the README is that policy.

---

## Tools and what each one does

This is **not** OpenAI Gym 0.21. The env API is **Gymnasium**. Colab sometimes still has the old `gym` package installed; ignore that warning or uninstall `gym`.

| Tool | Role |
|---|---|
| **URDF / STL** | Robot description and meshes (`robot/`) |
| **MuJoCo** | Rigid-body physics, contacts, PD actuators, rendering |
| **MJCF** | MuJoCo’s XML scene (`robot/mjcf/scene.xml`), compiled from the URDF |
| **Gymnasium** | Standard RL env API: `reset` / `step` / `observation_space` / `action_space` |
| **PyTorch** | Tensors and the policy network |
| **Stable-Baselines3** | PPO trainer, VecNormalize, checkpoints |
| **TensorBoard + `progress.csv`** | Full training metrics |
| **ONNX** | Exported actor for later use (e.g. a ROS 2 node) without SB3 |
| **imageio** | Eval video / PNG preview |

Gymnasium is only the **interface**. It does not simulate the robot. MuJoCo does.

Stable-Baselines3 is only the **learner**. It does not know about legs; it sees a 48-D box and a 12-D box.

**Not used at runtime:** Isaac Sim / Isaac Lab, NVIDIA GPU, the old `gym` package.

A GPU would matter for massively parallel GPU physics (Isaac Lab, MJX) or a vision policy. This MLP + CPU MuJoCo does not benefit from CUDA; SB3’s own guidance is to keep PPO-MLP on CPU.

---

## Pipeline on disk

```
robot/urdf/quadruped.urdf
        │  python -m quad_loco.convert_urdf
        ▼
robot/mjcf/scene.xml          # floor + robot + PD actuators + home keyframe
        │
        ▼
src/quad_loco/env.py          # Gymnasium QuadrupedVelocityEnv
        │
        ▼
scripts/train.py              # PPO, VecNormalize
        │
        ├─ logs/<run>/final_model.zip
        ├─ logs/<run>/vecnormalize.pkl    # keep with the zip
        ├─ logs/<run>/progress.csv
        └─ logs/<run>/policy.onnx         # scripts/export_onnx.py
```

`scripts/stand.py` does **not** use the network. It holds the default pose with PD (sanity check that the model stands).

`scripts/eval.py` loads the zip (+ vecnormalize), rolls out commands, optionally writes `videos/walk.mp4` and a PNG preview.

`scripts/check.py` times imports and steps the env **without** a trained policy (use this if Colab freezes).

---

## Where to run it

| | |
|---|---|
| **Local CPU (default)** | Python 3.10–3.12. Intel macOS: MuJoCo **3.10.x**. Train, eval, viewer. |
| **Google Colab** | Optional. Unattended job, extra CPU cores (8 envs). GPU runtime not needed. Colab **Python 3.13** can hang on `import stable_baselines3` — if that happens, use local 3.12. |

Typical local smoke:

```bash
export PYTHONPATH=src
python scripts/stand.py
python scripts/train.py --config configs/ppo_cpu.yaml --timesteps 8192 --run-name smoke
python scripts/eval.py --model logs/smoke/final_model.zip --easy --command 0.5 0 0 --video videos/walk.mp4
```

`--run-name smoke` writes `logs/smoke/final_model.zip`. After a full train, eval `logs/local_walk/final_model.zip` instead.
