# VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models

#### [[Project Page]](https://voxposer.github.io/) [[Paper]](https://voxposer.github.io/voxposer.pdf) [[Video]](https://www.youtube.com/watch?v=Yvn4eR05A3M)

[Wenlong Huang](https://wenlong.page)<sup>1</sup>, [Chen Wang](https://www.chenwangjeremy.net/)<sup>1</sup>, [Ruohan Zhang](https://ai.stanford.edu/~zharu/)<sup>1</sup>, [Yunzhu Li](https://yunzhuli.github.io/)<sup>1,2</sup>, [Jiajun Wu](https://jiajunwu.com/)<sup>1</sup>, [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)<sup>1</sup>

<sup>1</sup>Stanford University, <sup>2</sup>University of Illinois Urbana-Champaign

<img  src="media/teaser.gif" width="550">

This is the official demo code for [VoxPoser](https://voxposer.github.io/), a method that uses large language models and vision-language models to zero-shot synthesize trajectories for manipulation tasks.

In this repo, we provide the implementation of VoxPoser in [RLBench](https://sites.google.com/view/rlbench) as its task diversity best resembles our real-world setup. Note that VoxPoser is a zero-shot method that does not require any training data. Therefore, the main purpose of this repo is to provide a demo implementation rather than an evaluation benchmark.

> **This fork** extends the original VoxPoser codebase with:
> - **Headless-mode compatibility** for running RLBench tasks under a headless CoppeliaSim environment (physics-parameter tuning, joint-mode reset, and per-episode process isolation).
> - **Extended task support**: 22 RLBench task variants (up from the original demo set), including PressSwitch, StackCups, PlaceCups, BlockPyramid, PlaceShapeInShapeSorter, PutKnifeInKnifeBlock, PickAndLift, ReachTarget, EmptyContainer, StackBlocks, etc.
> - **Motion-planner plugin fix**: RLBench's `simExtOMPL` / `simExtIK` plugins fail to load when CoppeliaSim is launched in-process by PyRep, because they resolve the host library path from the process main module (`python.exe`). Launching via an interpreter located in the CoppeliaSim root fixes plugin loading.
> - **Genuine success criteria**: all teleport-based "fake success" fallbacks were removed; `success()` is now a pure wrapper around RLBench's native conditions, and every task runs with genuine random initialization.
> - **A 22-variant × 10-run benchmark** with separately tracked `env_success` / `planner_success`, value-map dumps, and a three-stage failure-attribution report.
> - **API-key-safe auto-push script** (`push_to_github.ps1`) that masks the DeepSeek API key before committing and restores it locally afterwards.

---

## Setup Instructions

Note that this codebase is best run with a display. For running in headless mode, refer to the [instructions in RLBench](https://github.com/stepjam/RLBench#running-headless).

- Create a conda environment:
```Shell
conda create -n voxposer-env python=3.9
conda activate voxposer-env
```

- See [Instructions](https://github.com/stepjam/RLBench#install) to install PyRep and RLBench (install these inside the created conda environment).

- Install other dependencies:
```Shell
pip install -r requirements.txt
```

### API Key Configuration

This fork uses **DeepSeek** as the LLM backend (OpenAI-compatible API). Set your API key via the environment variable **before** running:

```Shell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-你的DeepSeek密钥"

# Linux / macOS
export OPENAI_API_KEY="sk-你的DeepSeek密钥"
```

> The hardcoded fallback in `src/LMP.py` is a placeholder reminder — **replace it with your own key or rely on the environment variable**. Never commit real API keys.

---

## Running Demo

Demo code is at `src/playground.ipynb`. Instructions can be found in the notebook.

## Running the Benchmark

The benchmark harness lives outside this repo at `../experiment_results/` (relative to this project):

- `run_real10.py` — main driver: 22 task variants × 10 independent runs, with a fresh env per episode.
- `quick_eval_smoke_ext.py` — single-episode runner (also supports a `VOSPOSER_DIAG=1` read-only attribution mode).
- `coppelia_host_fix.py` — re-execs the script with the interpreter located in the CoppeliaSim root (motion-planner plugin fix).

```Shell
cd ../experiment_results
python run_real10.py
```

### Latest Benchmark Result (strict, 22 × 10 = 220 episodes)

Evaluation uses **genuine random initialization** (no fixed seed), **RLBench's native
`env.success()` with no success-criteria modification**, and a **fresh environment instance per episode**.

| Metric | Baseline (physics comp. off) | Physics comp. on (strict) |
|---|---|---|
| `planner_success` | 85.0% (187/220) | **90.9% (200/220)** |
| `env_success` (overall) | 8.2% (18/220) | **28.2% (62/220)** |
| joint / switch tasks (6 variants) | 13.3% (8/60) | **90.0% (54/60)** |

> Earlier "22/22 = 100%" figures were partly produced by teleport-based fallbacks
> (moving an object straight into the success sensor, or moving the end-effector marker
> onto the target pose and declaring success). Those fallbacks modified the success
> criteria and have been removed. The table above is the strict, fallback-free result.
> The OMPL / IK plugin fix also raised `planner_success` from 85.0% to 90.9%.

**Per-task breakdown (10 runs per variant):**

| Task | Var | Env | Planner |
|---|---|---|---|
| PushButton | 0 / 1 | 9/10 · 9/10 | 10/10 · 10/10 |
| LampOff | 0 / 1 | 9/10 · 8/10 | 9/10 · 8/10 |
| PressSwitch | 0 / 1 | 9/10 · 10/10 | 9/10 · 10/10 |
| SlideBlockToTarget | 0 | 2/10 | 9/10 |
| ReachTarget | 0 | 5/10 | 10/10 |
| StackCups | 0 / 1 / 2 | 0/10 · 0/10 · 0/10 | 10/10 · 10/10 · 7/10 |
| MeatOffGrill | 0 | 0/10 | 9/10 |
| PutRubbishInBin | 0 | 1/10 | 10/10 |
| TakeLidOffSaucepan | 0 | 0/10 | 10/10 |
| TakeUmbrellaOutOfUmbrellaStand | 0 | 0/10 | 10/10 |
| PlaceCups | 0 | 0/10 | 9/10 |
| PlaceShapeInShapeSorter | 0 | 0/10 | 7/10 |
| PutKnifeInKnifeBlock | 0 | 0/10 | 10/10 |
| PickAndLift | 0 | 0/10 | 10/10 |
| StackBlocks | 0 | 0/10 | 6/10 |
| EmptyContainer | 0 | 0/10 | 9/10 |
| BlockPyramid | 0 | 0/10 | 8/10 |

**Current bottleneck (post-fix):** non-joint tasks fail at the **grasp execution** stage —
the gripper closes near the object but does not actually grasp it
(`NothingGrasped=True` / `GraspedCondition=False`), while the motion planner reaches the
target with ~0.10 m residual error and then falls back to heuristics. See the
failure-attribution report for the three-stage (perception / code generation / planning-execution) analysis.

---

## Code Structure

Core to VoxPoser:

- **`playground.ipynb`**: Playground for VoxPoser.
- **`LMP.py`**: Implementation of Language Model Programs (LMPs) that recursively generates code to decompose instructions and compose value maps for each sub-task.
- **`interfaces.py`**: Interface that provides necessary APIs for language models (i.e., LMPs) to operate in voxel space and to invoke motion planner.
- **`planners.py`**: Implementation of a greedy planner that plans a trajectory (represented as a series of waypoints) for an entity/movable given a value map.
- **`controllers.py`**: Given a waypoint for an entity/movable, the controller applies (a series of) robot actions to achieve the waypoint.
- **`dynamics_models.py`**: Environment dynamics model for the case where entity/movable is an object or object part. This is used in `controllers.py` to perform MPC.
- **`prompts/rlbench`**: Prompts used by the different Language Model Programs (LMPs) in VoxPoser.

Environment and utilities:

- **`envs`**:
  - **`rlbench_env.py`**: Wrapper of RLBench env to expose useful functions for VoxPoser. **This fork adds headless compatibility, physics-layer compensation for joint/switch tasks, the motion-planner plugin fix, and a strict (fallback-free) `success()` that mirrors RLBench's native conditions.**
  - **`task_object_names.json`**: Mapping of object names exposed to VoxPoser and their corresponding scene object names for each individual task.
- **`configs/rlbench_config.yaml`**: Config file for all the involved modules in RLBench environment.
- **`arguments.py`**: Argument parser for the config file.
- **`LLM_cache.py`**: Caching of language model outputs that writes to disk to save cost and time.
- **`utils.py`**: Utility functions.
- **`visualizers.py`**: A Plotly-based visualizer for value maps and planned trajectories.

### Key Modifications in This Fork

| Module | Change |
|---|---|
| `LMP.py` | Switched backend from OpenAI to DeepSeek (`base_url=https://api.deepseek.com`). |
| `envs/rlbench_env.py` | `success()`: pure wrapper around RLBench's native success conditions — all teleport-based success overrides removed. |
| `envs/rlbench_env.py` | `_patch_task_for_headless()`: no-op — no `is_static_workspace` / `validate` / `SpawnBoundary` patches, so every task keeps genuine random initialization. |
| `envs/rlbench_env.py` | Physics-layer compensation for joint / switch tasks: physics tuning (dt=2ms / substeps=10), joint scan, FORCE / KINEMATIC dual strategy, plus a fallback compensation pass (toggle with `VOSPOSER_DISABLE_PHYSICS_COMP=1`). |
| `envs/rlbench_env.py` | Closed-loop IK fallback: 2 cm small-step approach from the *measured* EE pose; the tip-Dummy transform is guarded by `_capture_tip_nominal()` / `_restore_tip()`. |
| `envs/task_object_names.json` | Object-name mappings extended for the 22 task variants. |
| `push_to_github.ps1` | Auto-masks the API key in tracked files before commit/push and restores it locally afterward. |

---

## Headless-Mode Notes

Running RLBench in headless mode can trigger CoppeliaSim physics crashes (ACCESS_VIOLATION)
for tasks that use `SpawnBoundary.sample()`, `ForceSensor`, or complex furniture `.ttm` models.
This fork mitigates these **without touching task initialization or success criteria**:

1. **Snapshot isolation** — each episode runs in its own subprocess with a fresh CoppeliaSim instance; a crashed episode counts as a genuine failure.
2. **Physics-parameter tuning** — `dt=2ms` / `substeps=10`, plus a joint-mode / joint-position reset before each reset, to stabilise contact handling in headless mode.
3. **Motion-planner plugin host path** — launching via the interpreter located in the CoppeliaSim root so `simExtOMPL` / `simExtIK` actually load (otherwise all `arm.get_path` calls fail with `The call failed on the V-REP side. Return value: -1`).
4. **No success overrides** — `success()` reports only RLBench's native conditions; headless crashes and unreachable targets are genuine failures.

---

## Real-World Deployment

To adapt the code to deploy on a real robot, most changes should only happen in the environment file (e.g., you can consider making a copy of `rlbench_env.py` and implementing the same APIs based on your perception and controller modules).

Our perception pipeline consists of the following modules: [OWL-ViT](https://huggingface.co/docs/transformers/en/model_doc/owlvit) for open-vocabulary detection in the first frame, [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#segment-anything) for converting the produced bounding boxes to masks in the first frame, and [XMEM](https://github.com/hkchengrex/XMem) for tracking the masks over time for the subsequent frames. Now you may consider simplifying the pipeline using only an open-vocabulary detector and [SAM 2](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#latest-updates----sam-2-segment-anything-in-images-and-videos) for segmentation and tracking. Our controller is based on the OSC implementation from [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control). More details can be found in the [paper](https://voxposer.github.io/voxposer.pdf).

To avoid compounded latency introduced by different modules (especially the perception pipeline), you may also consider running a concurrent process that only performs tracking.

---

## Acknowledgments

- Environment is based on [RLBench](https://sites.google.com/view/rlbench).
- Implementation of Language Model Programs (LMPs) is based on [Code as Policies](https://code-as-policies.github.io/).
- Some code snippets are from [Where2Act](https://cs.stanford.edu/~kaichun/where2act/).

If you find this work useful in your research, please cite using the following BibTeX:

```bibtex
@article{huang2023voxposer,
      title={VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models},
      author={Huang, Wenlong and Wang, Chen and Zhang, Ruohan and Li, Yunzhu and Wu, Jiajun and Fei-Fei, Li},
      journal={arXiv preprint arXiv:2307.05973},
      year={2023}
    }
```
