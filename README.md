# Isaac Twisted-RL — Rope Knot-Tying in Isaac Lab

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.1.0-76B900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-8A2BE2.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://docs.python.org/3/whatsnew/3.11.html)

This repository ports the **Twisted-RL** knot-tying task (Freund et al., originally MuJoCo) to Isaac Lab:
a kinematic parallel gripper picks and places a rope, the rope's topology (crossings, p-data, knot class) is
extracted every step, and that topology defines the stages, rewards and termination. The goal is to tie an
overhand knot starting from a straight rope.

In the current design (v4 and later), one RL step is one whole pick&place move; each topological stage
(`round(3Φ)`: 0 = trivial, 3 = knot) has its own policy head; an episode is one stage transition; and training
resets from an archive of reached states. See [`knot_tying/README.md`](src/isaac_twisted_rl/tasks/knot_tying/README.md).


## Train

<video src="https://github.com/ItoYamato7777/isaac_twisted_rl/raw/main/video/highlights/knot_training_mock.mp4" controls muted width="640"></video>

## Play

<video src="https://github.com/ItoYamato7777/isaac_twisted_rl/raw/main/video/highlights/knot_playing_mock.mp4" controls muted width="640"></video>

<!--

## Sim2Real Transfer

_Work in progress._



-->


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

The USD assets are **not committed** (see `**/*.usd` in `.gitignore`). Generate them once
before training or running any demo:

```bash
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py
```

The task is **versioned**: improvements are added as a new version rather than by
rewriting the existing one, so every earlier version stays runnable and comparable.
The table is abridged; the full one (with the episode length and the per-version README) is
"版の系譜" in [`knot_tying/README.md`](src/isaac_twisted_rl/tasks/knot_tying/README.md).
Which version is the baseline, and the results, are in [`docs/STATUS.md`](docs/STATUS.md) and
[`docs/experiments/`](docs/experiments/README.md) (Japanese).

| Version | Inherits | Change |
|---|---|---|
| `v0` | — | make a single crossing that survives release |
| `v1` | v0 | tie an overhand knot |
| `v2` | v1 | 3-step reward ladder; removed the divergence-prone `approach` velocity command |
| `v2-R2` | v2 | ladder judged after Reidemeister-II reduction |
| `v3` | v2-R2 | reset from an archive of reached states, terminate on success, BFS-distance potential |
| `v4` | v3 | one agent per stage (TWISTED-RL-C), one transition per episode, stage archive seeded from the scripted knot |
| `v5` | v4 | pick&place divergence fixes (release 5-10 cm above the floor, depenetration velocity 0.4 m/s) |
| `v6` | v5 | next topological edge to follow (link masks) in the observation |
| `v7` | v6 | depenetration velocity back to the v4 default (0.96 m/s) |
| `v8` | v7 | symmetry data augmentation in the PPO update (mirror / reverse) |
| `v9` | v7 | sample stage 2 twice as often when resetting |
| `v10` | v7 | stages restricted to the canonical path to the knot |
| `v11` | v7 | dead-end shapes terminate with -1 |
| `v12` | v7 | archive stores only the shape after a stage-up move |
| `v13` | v12 | penalty on policy means outside the action bounds (`BoundedPPO`) |
| `v14` | v13 | stage-0 reshape saved to the archive, rope-physics domain randomization, privileged critic observation |
| `v15` | v14 | no time-out on a move that changes the stage or diverges (fixes the time-out bootstrap bug) |

Rope presets: `Twist` / `Simple` / `Stiff` / `Fine` / `Knot`. Training since v3 uses `Knot` (28 links, 1.14 m).
Task IDs are `Isaac-Twisted-RL-<Rope>[-Play]-<version>`.

### Train and evaluate

v4 and later start from a seed archive of the scripted knot's intermediate states (`outputs/archive_seed/`,
untracked). Create it once, with a **new path** (never overwrite an existing archive; the default path is the
shared one):

```bash
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py \
    --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless \
    --dump_archive outputs/archive_seed/<new name>.pt
```

Train.

```bash
uv run train --task Isaac-Twisted-RL-Twist-v0 --headless   # single crossing
uv run train --task Isaac-Twisted-RL-Knot-v3 --headless    # archive resets (uses the 28-link Knot rope)
uv run train --task Isaac-Twisted-RL-Knot-v15 --headless   # one policy head per stage (latest version)
```

Evaluate a trained policy interactively (v4 and later: the `-Play-` task is the chain-evaluation environment).

```bash
uv run play --task Isaac-Twisted-RL-Twist-Play-v0 --num_envs 1
uv run play --task Isaac-Twisted-RL-Knot-Play-v3 --num_envs 1
uv run play --task Isaac-Twisted-RL-Knot-Play-v15 --num_envs 16
```

Compare versions with the **chain evaluation** (straight rope to knot within 18 moves, first episode of each
environment) on at least two training seeds. Pass the same Hydra overrides as in training, make `--device` and
`agent.device` agree, and check the last checkpoint number with `ls`. The definition and the standard
conditions are in [`docs/system/evaluation.md`](docs/system/evaluation.md).

```bash
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v15 \
    --num_envs 512 --headless --device cuda:0 --seed 7 \
    --checkpoint logs/rsl_rl/knot_tying_v15/<run>/model_<N>.pt agent.device=cuda:0 --out <run>/eval_chain_it<N>.json
```

Long Isaac Sim jobs are launched with `src/isaac_twisted_rl/scripts/run_job.sh` (timeout, log and record under
`outputs/jobs/`); see the rules in [`CLAUDE.md`](CLAUDE.md). Other scripts in `src/isaac_twisted_rl/scripts/rsl_rl/`
(`diag_stage_moves.py`, `eval_search.py`, `export_policy.py`, `view_archive.py`, ...) are described in
[`scripts/rsl_rl/README.md`](src/isaac_twisted_rl/scripts/rsl_rl/README.md).

### Demos and tests

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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q
```

### Documentation

Documentation is in Japanese.

| Document | Content |
|---|---|
| [`docs/README.md`](docs/README.md) | map of documents (start here) |
| [`docs/STATUS.md`](docs/STATUS.md) | current state, running jobs, next work |
| [`knot_tying/README.md`](src/isaac_twisted_rl/tasks/knot_tying/README.md) | version lineage, task IDs, folder layout |
| [`docs/decisions/`](docs/decisions/README.md) | design decisions across versions |
| [`docs/experiments/`](docs/experiments/README.md) | experiment records and results |
| [`CLAUDE.md`](CLAUDE.md) | development rules for contributors (including Claude Code) |

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
