# ABI: Actor-Free Behavior-Induced Representations for Zero-Shot Test-Time Guidance

Project page and interactive playground. `index.html` is the project page (abstract, video, figures).
`playground/` is a fully local demo of ABI's critic guiding frozen policies in OGBench `cube-single-play`
and `antmaze-medium-navigate` and in the DeepMind Control quadruped, cheetah and walker. Physics (MuJoCo,
WebAssembly) and all networks (ONNX Runtime Web) run in the browser. Nothing is sent over the network.

## Run

```
python3 serve.py          # then open http://localhost:8000 (the playground is at /playground/)
```

Any recent Chrome, Edge, Firefox or Safari works. `serve.py` only adds the two headers that enable
multi-threaded WebAssembly. Any static file server (GitHub Pages included) also works, with one thread.
Opening `index.html` directly from disk does not work, because browsers block module workers on
`file://`. The environment is chosen at the top right (or with `?env=<name>`), and each environment
loads only its own files.

## What you can do

The page starts an episode right away. The orange agent is a frozen policy guided by ABI's critic. The
blue agent is the same policy without ABI, started from the same state with the same random draws.

- **Goal.** Cube: click inside the marked area on the table to place the cube's target. AntMaze: click
  the floor or the map to place a goal, shift+click (or right-click on the map) to set the start. The
  buttons *Task 1–5* load the five OGBench evaluation tasks. DMC: the buttons load the four tasks of each
  domain (for example walker stand, walk, run, spin). Changing the goal or the task while the agent moves
  continues from its current state.
- **Policy guided by ABI.** Flow BC, the behavior-cloning policy that ABI guides in the paper (4×512 on
  OGBench, 2×256 on DMC, and a smaller 2×256 policy on AntMaze). On Cube, ABI can also guide the trained
  actor of TD-JEPA. The same critic guides every policy without retraining. ABI itself trains no actor.
- **Guidance** (*Advanced settings*). *Best-of-N*: the policy proposes N chunks of 8 actions and the
  critic picks one. *QGF*: critic-gradient steps inside the flow sampler, strength α. *QGF, best of 4+1*:
  four QGF samples and one unguided sample, the critic picks one. For the TD-JEPA actor the critic ranks
  16 candidate actions (see below).
- **Execution** (*Advanced settings*). Replan at every step (execute the first action of the chosen
  chunk, the paper's main protocol) or execute the whole 8-step chunk.
- **Blue run** (*Advanced settings*). Instead of the same policy without ABI, the blue run can be the
  unguided Flow BC or, on Cube and AntMaze, the trained actor of TD-JEPA or of one-step FB. These actors
  read the goal through their own zero-shot task inference.

## What happens when you place a goal

The critic is never retrained. A goal only sets the reward labels of the critic's reference states: a
reference state is labelled when it satisfies the environment's own success test for that goal (Cube:
the cube within 4 cm, in 3-D; AntMaze: the ant within 0.5 m). In DMC the label of a reference state is
the task's reward there (dm_control's reward functions). In OGBench the critic's score of a chunk is a
softmax-weighted average of these labels over the reference states, so a goal far from every reference
state gives no signal. In DMC a ridge readout of the reference states turns the labels into the task
latent the critic scores against. The *Critic scores* card shows the scores of the last candidates and
the value of the executed chunk over time.

## ABI guiding the TD-JEPA actor

The actor proposes 16 actions at every step: its own action and 15 further samples. The frozen Flow BC
completes each action into 8-step chunks (4 completions, shared by all 16 actions), ABI's critic scores
them, and the action with the best mean score is executed. The actor reads the goal through its own task
inference (least squares on its inference samples, labelled with the same success test).

## The video

`assets/abi_video.mp4` is a narrated overview (about 3.5 minutes, captions burned in). Its episodes were recorded
from this playground in its capture mode (`?capture&cspeed=<speed>`): each video frame shows a fixed amount of
simulated time, and every clip that does not play in real time shows its speed. The episodes are examples at fixed start seeds. The success
rates below are the measured averages. The method and result shots are animations of the paper's figures and tables.

## Contents

| path | content |
|---|---|
| `index.html`, `assets/` | the project page, its video and figures |
| `playground/models/<env>/` | ABI critic (training seed 3917), Flow BC policies, reference states and labels |
| `playground/models/<env>/baselines/` | the TD-JEPA and one-step FB actors (training seed 3917) and their inference samples |
| `playground/env/<env>/` | the scene and its task and reset parameters (Cube: robot meshes, licences included) |
| `playground/js/` | simulation (`js/envs/`), guidance, task inference of the actors, rendering and UI |
| `playground/vendor/` | MuJoCo 3.17.0 (Apache-2.0), ONNX Runtime Web 1.31.1 (MIT), three.js 0.186.1 (MIT) |

## Faithfulness

The networks are the paper's checkpoints for training seed 3917, exported to ONNX in fp32. The model pads
each 8-step chunk to 32 action steps and masks the padding. The critic graphs here feed only the 8 chunk
steps, which halves their cost and leaves the scores unchanged (AntMaze: bit-identical; Cube: within
1.5e-5 relative, the same choice in every checked window). On recorded paper episodes the best-of-16
choice matches the original PyTorch modules at every step.

The browser runs MuJoCo 3.17, the paper's evaluations ran MuJoCo 3.3.7, and contact dynamics differ
between the two versions, so single trajectories differ from the paper's. Success rates of this demo's
own code, measured headless on the five evaluation tasks (MuJoCo 3.17, same start distribution):

**Cube (`cube-single-play`), 20 starts per task, every step**

| policy | guidance | this demo | paper (seed 3917) |
|---|---|---|---|
| Flow BC 4×512 | unguided | 12% | 9% |
| Flow BC 4×512 | best-of-16 | 75% | 73% |
| Flow BC 4×512 | QGF, best of 4+1 (per-task α and gripper rule) | 50% | 57% |
| TD-JEPA actor | unguided | 22% | 25% |
| TD-JEPA actor | best of 16 actions | 76% | 74% |

In the Cube QGF modes the critic's push on the gripper dimension is masked on the tasks where the paper
masks it. On tasks 4 and 5 the QGF mode succeeds less often under MuJoCo 3.17 than in the paper. The same
graphs reproduce the paper under MuJoCo 3.3.7.

**AntMaze (`antmaze-medium-navigate`), 20 starts per task**

| policy | guidance | execution | this demo | paper (seed 3917) |
|---|---|---|---|---|
| Flow BC 4×512 | unguided | every step | 2% | 6% |
| Flow BC 4×512 | best-of-16 | every step | 77% | 67% |
| Flow BC 4×512 | best-of-16 | whole chunk | 63% | 56% |
| Flow BC 4×512 | QGF, best of 4+1 (per-task α) | whole chunk | 55% | 57% |
| Flow BC 2×256 | unguided | every step | 0% | – |
| Flow BC 2×256 | best-of-16 | every step | 64% | – |

**DMC, 20 starts per task (QGF, best of 4+1: 10), every step, mean episode return**

| domain | guidance | this demo | paper (seed 3917) |
|---|---|---|---|
| quadruped | unguided | 45.3 | 53.2 |
| quadruped | best-of-16 | 691.3 | 695.7 |
| quadruped | QGF, best of 4+1 (per-task α) | 735.8 | 743.2 |
| cheetah | unguided | 110.4 | 106.9 |
| cheetah | best-of-16 | 579.3 | 568.1 |
| cheetah | QGF, best of 4+1 (per-task α) | 689.1 | 683.6 |
| walker | unguided | 158.4 | 157.9 |
| walker | best-of-16 | 633.2 | 630.4 |
| walker | QGF, best of 4+1 (per-task α) | 800.7 | 797.3 |

The DMC critic reads a task through a ridge readout of its reference states. The demo bakes the readout
into the critic graph, so a task only sets the 10,000 reference labels. Its latent matches the paper's
to about 1e-7 relative.

**Zero-shot RL actors without ABI (blue run), 100 and 500 episodes over the five tasks**

| environment | actor | this demo (100 / 500) | paper evaluation (100) |
|---|---|---|---|
| Cube | TD-JEPA | 22% / 25% | 25% |
| Cube | one-step FB | 3% / 3% | 2% |
| AntMaze | TD-JEPA | 75% / 74% | 76–79% |
| AntMaze | one-step FB | 36% / 39% | 45% |

The QGF strengths of the five tasks are the ones selected on development episodes in the paper. For a
goal you place yourself the slider value is used (Cube: the strength and gripper rule of the nearest
task).
