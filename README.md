# quad-loco

![PD stand in MuJoCo](docs/stand.png)

Velocity-tracking locomotion for a custom 12-DoF quadruped in [MuJoCo](https://mujoco.org/), trained with PPO (Stable-Baselines3) and exported to ONNX.

No NVIDIA GPU is required. Physics and the MLP policy both run on CPU.

```
URDF → MJCF → Gymnasium env → PPO → eval / video → ONNX
```

## Robot

| | |
|---|---|
| DoF | 12 (hip, thigh, calf × 4) |
| Mass | ~22 kg (trunk 9.7 kg) |
| Stand height | ~0.65 m |
| Control | 50 Hz joint-position PD (`kp=100`, `kv=3`, `τ ≤ 45 N·m`) |
| Task | Track base velocity `(v_x, v_y, ω_z)` on flat ground |

**Observation** (48): body linear velocity (3), angular velocity (3), projected gravity (3), command (3), joint position relative to default (12), joint velocity (12), last action (12).

**Action** (12): `q_target = q_default + 0.25 · a`, `a ∈ [-1, 1]`.

Meshes and URDF are adapted from a public custom-quadruped model; this repository is the MuJoCo training and export stack.

## Setup

Python 3.10–3.12. On Intel macOS, MuJoCo must be **3.10.x** (later wheels dropped x86_64).

```bash
git clone https://github.com/jakhon37/quad-loco.git
cd quad-loco
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=src
```

The scene in `robot/mjcf/` is ready to use. Rebuild after changing the URDF:

```bash
python -m quad_loco.convert_urdf
python scripts/stand.py          # PD hold (no learned policy)
python -u scripts/check.py       # timed imports + env step (no trained policy)
python -u scripts/check.py --render   # also probe one off-screen frame
python -m pytest tests -q
```

Interactive view (plain `python`, not `mjpython`):

```bash
python scripts/stand.py --viewer
```

Close the window to exit. Headless `python scripts/stand.py` is the pass/fail check.

## Train

Training is **optional and local-first**. A 1.5M-step run is on the order of 30–40 minutes on a laptop CPU. Google Colab is not required.

**Local**

```bash
# short check
python scripts/train.py --config configs/ppo_cpu.yaml --timesteps 8192 --run-name smoke

# full run (edit total_timesteps in the yaml, or pass --timesteps)
python scripts/train.py --config configs/ppo_cpu.yaml --run-name local_walk
```

**Colab (optional)** — use it if you want the job off your laptop: longer unattended session, more CPU cores for 8 parallel envs (`configs/ppo_colab.yaml`). A GPU runtime is unnecessary for this MLP.

Colab’s default **Python 3.13** can hang for a long time on `import stable_baselines3`. If `scripts/check.py` sits there more than about a minute, stop the runtime and train locally with Python 3.12.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakhon37/quad-loco/blob/main/notebooks/colab_train.ipynb)

Opening the notebook from GitHub does not copy the repo; the first cell clones it. A CPU runtime is enough.

Console output is a progress bar plus one line every few rollouts (`rew`, `len`, `ev`, `kl`, `ent`). Full metrics are always written to disk:

```
logs/<run>/progress.csv
logs/<run>/events.out.tfevents*
logs/<run>/eval/evaluations.npz
```

```bash
tensorboard --logdir logs/<run>
```

`--verbose 1` prints the full Stable-Baselines3 tables in the terminal.

## Eval and export

Use the path printed at the end of training (`saved .../final_model.zip`). After the smoke run that is `logs/smoke/`:

```bash
python scripts/eval.py --model logs/smoke/final_model.zip --easy --command 0.5 0 0 --video videos/walk.mp4
python scripts/export_onnx.py --model logs/smoke/final_model.zip --out logs/smoke/policy.onnx
```

Eval writes a still (`*_preview.png`) and an MP4 when rendering is available (OSMesa on headless Linux). Keep `vecnormalize.pkl` next to the zip; observations were normalized during training.

## Layout

```
robot/          URDF, STL meshes, MJCF scene
src/quad_loco/  environment, converter, export
scripts/        stand, train, eval, export, smoke
configs/        PPO hyperparameters (local and Colab)
notebooks/      optional Colab notebook
tests/
```

## License

MIT. See [LICENSE](LICENSE).
