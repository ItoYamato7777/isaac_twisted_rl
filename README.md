# Isaac Twisted-RL — Rope Knot-Tying in Isaac Lab

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.1.0-76B900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-8A2BE2.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://docs.python.org/3/whatsnew/3.11.html)

This repository implements the **Twisted-RL** knot-tying task in Isaac Lab: a kinematic
parallel gripper picks and places a rope to tie a knot, the resulting knot is detected
as a topological invariant, and that invariant drives the reward.

## Installation

Install uv.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync dependencies.

```bash
uv sync
```

## Quickstart

List available environments.

```bash
uv run list_envs
```

Test with dummy agents.

```bash
uv run zero_agent --task Isaac-Twisted-RL-Twist-Play-v0    # send zero actions
uv run random_agent --task Isaac-Twisted-RL-Twist-Play-v0  # send random actions
```

## Knot Tying

Rope manipulation with topology (p-data) extraction, ported from the MuJoCo-based
TWISTED-RL implementation. A kinematic parallel gripper picks and places a rope to tie
a knot; the resulting knot is detected as a topological invariant and turned into reward.

The USD assets are **not committed** (see `**/*.usd` in `.gitignore`). Generate them once
before training or running any demo:

```bash
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py
```

The task is **versioned**: improvements are added as a new version rather than by
rewriting the existing one, so every earlier version stays runnable and comparable.

| Version | Goal |
|---|---|
| `v0` | make a single crossing that survives release |
| `v1` | tie an overhand knot |
| `v2` | overhand knot with a 3-step reward ladder |
| `v2-R2` | v2 with the ladder judged after Reidemeister-II reduction |
| `v3` | reset from an archive of reached states, terminate on success, BFS-distance potential |
| `v4` | one agent per stage (TWISTED-RL-C), one topological transition per episode, stage archive seeded from the scripted knot |
| `v5` | v4 + pick&place divergence fixes (release 5-10 cm above the floor, rope depenetration velocity 0.4 m/s) |
| `v6` | v5 + the next topological edge to follow (link masks) in the observation |
| `v7` | v6 with the rope depenetration velocity back to the v4 default (0.96 m/s) |

The current status of each version and the results are in [`docs/STATUS.md`](docs/STATUS.md) and
[`docs/experiments/`](docs/experiments/README.md) (Japanese).

```bash
uv run train --task Isaac-Twisted-RL-Twist-v0 --headless   # single crossing
uv run train --task Isaac-Twisted-RL-Knot-v3 --headless    # archive resets (uses the 28-link Knot rope)
uv run train --task Isaac-Twisted-RL-Knot-v4 --headless    # one agent per stage (see versions/v4/README.md for the seed archive)
uv run train --task Isaac-Twisted-RL-Knot-v6 --headless    # v4 + divergence fixes + next-edge observation
```

Evaluate a trained policy.

```bash
uv run play --task Isaac-Twisted-RL-Twist-Play-v0 --num_envs 1
uv run play --task Isaac-Twisted-RL-Knot-Play-v3 --num_envs 1
uv run play --task Isaac-Twisted-RL-Knot-Play-v4 --num_envs 16
```

For v4 and later, compare versions with the chain evaluation (straight rope to knot within 18 actions, counted per episode)
on at least two seeds. The definition and the standard command are in [`docs/system/evaluation.md`](docs/system/evaluation.md).

Grasp a rope with a parallel gripper:

```bash
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/rope_catch.py --rope stiff --headless
```

Manipulate a rope while logging every change of its topological state, writing
`events.log` and xy-projection plots to `outputs/topology_debug/`:

```bash
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/topology_debug.py --rope twist --headless
```

The topology layer imports no `isaaclab`/`omni`/`pxr`, so its unit tests run without Isaac Sim:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/
```

Documentation (Japanese): start from the map of documents, [`docs/README.md`](docs/README.md).
Development rules for contributors (including Claude Code) are in [`CLAUDE.md`](CLAUDE.md).

## Sim2Real Transfer

_Work in progress._

## Results

![rl-video-step-0](https://github.com/user-attachments/assets/890e3a9d-5cbd-46a5-9317-37d0f2511684)

## Acknowledgements

This project builds upon the excellent work of several open-source projects and communities:

- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** — The foundational robotics simulation framework that powers this project
- **[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)** — The underlying physics simulation platform
- **[RSL-RL](https://github.com/leggedrobotics/rsl_rl)** — Reinforcement learning library used for training policies
- **TWISTED-RL** (Freund et al.) — The hierarchical knot-tying approach this project ports to Isaac Lab (papers in `paper/`)

This project was originally scaffolded from the Isaac Lab SO-ARM100/SO-ARM101 project;
the SO-ARM task code has since been removed as this repository now focuses exclusively
on the Twisted-RL knot-tying task.

## License

See [LICENSE](LICENSE) for details.
