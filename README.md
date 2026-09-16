# Table for Two — bimanual SO-101 table setting on OpenVINO

![Both SO-101 arms carrying the plate to the placemat between them](docs/media/cover.png)

Two simulated SO-101 arms set a dinner table from a spoken or typed instruction:
carry the plate to the placemat with both hands, open the drawer while placing
the mug, lay the fork, pour water from the bottle into the mug, put the bottle
back, and hand the spoon from one arm to the other. Built for the **Intel
Physical AI Online Challenge — Bimanual VLA Manipulation** at the AI Infra
Summit Hackathon (lablab.ai, September 2026).

Everything runs in MuJoCo; every learned model runs through **OpenVINO**.

**No object is ever attached to a gripper.** Every object moves only because
the finger pads squeeze it and simulated friction holds it, the drawer opens
because a finger pulls on its handle, and the water is 24 small beads that
pour like lentils. The gripper is modelled on the stock 7.4 V servo at the
torque it can sustain (0.8 N·m, about 10 N at the fingertip, not the 2.94 N·m
stall figure), with rubber pads of μ = 0.8. Object mass, pad friction and
object friction are randomised per scene.

## Why it matters

Table setting is a stand-in for the everyday two-handed jobs a home or
care-home assistant robot would do: fetch from a drawer, carry something too
wide for one hand, pour, pass an object from one hand to the other. The SO-101
costs about $100 an arm and its whole inference stack here — planner, policy
and stage detection — runs on an Intel laptop's CPU and integrated GPU through
OpenVINO, with no cloud and no discrete GPU at run time. Everything the arms do
is held by simulated friction, so what works here has a chance of working on
the printed fingers of a real SO-101.

## How it works

```mermaid
flowchart LR
    V["Speech (Speechmatics)<br/>or typed instruction"] --> P
    P["Language planner<br/>Qwen2.5-1.5B INT4 · OpenVINO GenAI"] -->|"validated subtask plan"| E
    E["Stage executor<br/>subtask one-hot"] --> A
    C["4 cameras 128×128 (2 scene + 2 wrist)<br/>+ 12 joint states"] --> A
    A["ACT visuomotor policy<br/>OpenVINO IR (FP32 / FP16 / INT8)"] -->|"12 joint targets @ 10 Hz"| S
    S["MuJoCo contact physics: 2 × SO-101<br/>drawer, plate, mug, bottle + bead water, utensils"] --> C
```

1. **Planner** — a 1.5 B instruction model writes a compact plan language, one
   stage per line (`bimanual_place plate`, `open_drawer left ; pick_place right mug`,
   `handoff spoon left right`, …).
   The plan is parsed, then checked twice: syntax (known skills, arms, objects)
   and a small simulation of what each hand holds (no pouring without the bottle
   in hand and the mug set down, no placing what is not held, utensils only after
   the drawer opens). Errors go back to the model for one retry; leftover bad
   steps are dropped; a keyword planner is the last resort.
2. **Executor** — runs the plan stage by stage; subtasks in one stage run on both
   arms at the same time. Each stage gives the policy a one-hot subtask token —
   this token is how language reaches the policy (a planner-conditioned ACT, not
   an end-to-end VLA).
3. **Policy** — LeRobot ACT trained on scripted demonstrations, exported to
   OpenVINO and quantised to INT8 with NNCF. It sees four 128×128 cameras and
   the joint positions, nothing else.
4. **Scripted skills** — generate the demonstrations and act as the fallback in
   hybrid mode (see below). They use ground-truth object poses and contact
   events from the simulator; the policy has to learn the same behaviour from
   pixels.

## Grasping by contact

The scripted skills position the open jaws around an object, close past contact
so the servo keeps squeezing, and move. Getting there took measuring the robot:

| Skill | How the SO-101 does it | Why |
|---|---|---|
| Bottle | side grip on the body, open hand lowered around it from above; pours by tilting about the grip while the lowest point of the lip stays over the mug | a top-down grip covers the mouth, so the water runs onto the wrist (0/24 beads in the mug) |
| Drawer | top-down pinch on the bar pull, fixed finger on the arm side, moving jaw dropped into the gap at a set opening | closing the other way round needs more wrist roll than the arm has; opening wider lands the finger on the drawer front |
| Utensils | top-down pinch on a 10 mm-square handle, jaws stopping 1 mm above the surface | the jaws' collision bodies end 8 mm below the tool point; a flat 5 mm handle leaves the pads 1–2 mm to squeeze |
| Hand-over | both wrist-camera mounts face away from the other hand, spoon held 30° off crosswise, grips 5.6 cm apart | found by a search over angle, grip points and height, measuring the distance between the two arms' collision bodies; straight crosswise they touch |
| Plate | carried by **both arms**, each pinching an opposite 5 mm wall (thin fixed finger inside, moving jaw outside), moving in lockstep | one hand on one wall leaves the plate hanging 6–8° (its centre of mass is 4.5 cm from the pinch); with two hands it stays within 2.5°. On a 3 mm wall the jaws closed to their joint limit and the plate slid 1–2 cm through the pads at set-down once the servo torque was made realistic |
| Mug | top-down across the body just below the rim, handle beside the jaws, tipped level while carried | — |
| Utensils (one arm) | gripped over their centre of mass | gripped at the handle end the fork sagged 12–17° and slid |
| Every set-down | lowered in 1.5 mm steps until the object touches a support, then the fingers open and slide off the object before lifting | releasing at a computed height dropped the fork 2.8 cm at 0.7 m/s; lifting straight after opening let two hands inside the plate lift it and fling it |

Reach shaped the layout: a top-down hand reaches at most 9 cm above the table
(18–24 cm from the base), a jaws-horizontal hand reaches the table only 30 cm
or more out, so the bottle stands at the far edge of the right arm's workspace.
`tools/grasp_lab.py`, `tools/pour_lab.py` and `tools/trace_stage.py` are the
single-object labs and the per-stage tracer used to find each of these.

## Scene

| Item | Detail |
|---|---|
| Robots | 2 × SO-101 (MuJoCo Menagerie `robotstudio_so101`), 6 position actuators each, Menagerie's grasp contact settings (elliptic cones, `impratio` 10) |
| Gripper | torque capped at 0.8 N·m (stock 7.4 V STS3215: 1.62 N·m stall, overload protection trips within seconds at stall); finger pads μ = 0.8; measured pad force while holding 20–27 N |
| Task objects (child-tableware scale, sized for a 5 cm jaw) | cabinet with a full-extension drawer and bar pull; spoon and fork with chunky 10 mm handles (9.6 cm long, ~23 g); a 9 cm ramekin-sized deep dish (120 g); a 4.8 cm espresso mug (130 g); a 3.2 × 7 cm glass vial (40 g) with 24 water beads (6.5 ml); placemat |
| Cameras | overhead, operator (behind the arms), front-high, one on each wrist |
| Control | 20 Hz joint targets; policy at 10 Hz |
| Randomisation per seed | object position ±1.2 cm (utensils ±4 mm, ±0.08 rad yaw), object mass ×0.8–1.2, pad friction ×0.7–1.3 (this is what changes grasps: the pads' friction governs every finger–object contact), object friction ×0.7–1.3 (sliding on the table), light ×0.6–1.2, table colour |
| Success (scored 1 s after the last motion) | drawer open > 4.5 cm *now*; plate, fork, spoon and mug within 3 cm of their places, resting on the table only (not on each other), upright (< 12°) and released; ≥ 60 % of the beads at rest inside the mug with ≤ 2 spilled; bottle lifted, then standing upright within 3 cm of where it started |

## Results

**Scripted contact pipeline.** Seeds 0–9 are the evaluation seeds the brief
asks for (seed 0 is the nominal layout, 1–9 randomised) and also the seeds the
skills were debugged on, so they are reported next to a block of 30 seeds
(1000–1029) that were never looked at:

| Sub-goal | Seeds 0–9 | Unseen seeds 1000–1029 |
|---|---|---|
| Drawer opened (pulled by its handle) | 10/10 | 30/30 |
| Mug placed | 10/10 | 30/30 |
| Plate on placemat (two-handed) | 10/10 | 30/30 |
| Fork placed | 10/10 | 30/30 |
| Spoon handed over and placed | 10/10 | 30/30 |
| Poured (≥ 60 % of the beads in the mug, ≤ 2 spilled) | 10/10 | 30/30 |
| Bottle put back | 10/10 | 30/30 |
| **Full task** | **10/10** | **30/30** |

`tools/audit_contact.py` checks every hold on seeds 0–9 (`out/audit_contact_final.txt`):
no object is lost while the fingers squeeze; every set-down touches the table
at ≤ 0.031 m/s before the fingers open (one exception: on seed 1 the fork was
let go 0.4 mm up, landing at 0.05 m/s). In the hand, the plate tilts up to
10° and shifts up to 13 mm when one arm carries more of it; the spoon sags
8–17° during the hand-over; the mug and the fork stay within 4°.

**Language planner.** On 20 paraphrased instructions (`tools/planner_eval.py`,
different wording, order, synonyms, partial requests, named arms) every plan
passes the syntax and hand-state checks, and 14/20 contain exactly the steps
asked for; 13 of the 20 plans came from the language model (the rest from the
keyword fallback after the model's plan failed the checks). The commonest
mistake is adding an unrequested step. Latency on Intel hardware
(`bench/benchmark.py`, 5 instructions, current prompt; on those five the
model's own plan was accepted as written 2/5 times, the other three went
through precondition completion or the keyword fallback):

| Device | Plan latency mean / p95 (s) | Time to first token (ms) | Time per token (ms) | Tokens/s |
|---|---|---|---|---|
| CPU (i9-14900HX) | 2.19 / 6.14 | 704 | 24.0 | 41.7 |
| Intel iGPU (Raptor Lake UHD) | 2.37 / 4.73 | 774 | 34.9 | 28.7 |

**Learned policy (v1: 150 demonstrations, 40k steps, four cameras).** Stage
success on the ten evaluation seeds (`policy/rollout.py`; a stage counts only
if the policy finished it inside 40 s; `home` stages are not scored). Measured
in the earlier scene (prop friction 1.0, 4x anti-aliased cameras): there, the
wrist-camera renders were not bit-exact, so the same seed could give a
different rollout, and with 10 seeds a rate's 95% interval is about ±0.25 -
read single cells with care. `policy/rollout.py` now scores 30 unseen seeds
(1000–1029) with Wilson intervals in the current scene:

| Stage | Chained, action queue | Chained, temporal ensemble | Each stage from a scripted start | Chained, INT8 |
|---|---|---|---|---|
| Two-handed plate carry | 9/10 | 10/10 | 9/10 | 9/10 |
| Drawer + mug | 9/10 | 10/10 | 10/10 | 10/10 |
| Bottle lift + fork | 8/10 | 10/10 | 10/10 | 8/10 |
| Pour | 3/10 | 6/10 | 7/10 | 5/10 |
| Bottle back | 5/10 | 6/10 | 10/10 | 5/10 |
| Spoon hand-over | 2/10 | 0/10 | 9/10 | 1/10 |
| **Whole table in one run** | **0/10** | **0/10** | — | **0/10** |

Hybrid mode (a scripted skill finishes any stage the policy timed out on,
counted as assisted, never as a success) sets no full table either: 1.9
assisted stages per episode, most often the hand-over (8/10) and the pour
(7/10).

How to read this: every skill is learned — started from a clean scripted
state, each stage succeeds 7–10 times in 10 — but the chain breaks because the
states the policy leaves behind (where it set the bottle down, how the mug
ended up) drift away from the demonstrations, so the later stages see inputs
they were never trained on. Temporal ensembling (`--ensemble 0.01`: one
inference per step, overlapping chunks blended with the ACT paper's weights)
doubles the pour rate without retraining but has not yet made the hand-over
work. A second policy trained on 300 demonstrations is in progress and this
table will be extended with it.

OpenVINO latency of the four-camera network, isolated single call (zero
inputs, 200 calls; `bench/benchmark.py` writes `out/benchmark.md`):

| Model | CPU mean / p95 (ms) | Intel iGPU mean / p95 (ms) | Size (MB) | Action error vs FP32, mean / max (rad) |
|---|---|---|---|---|
| FP32 | 19.6 / 27.7 | 14.2 / 16.3 | 136.8 | — |
| FP16 | 17.1 / 20.1 | 13.8 / 14.3 | 68.4 | 0.00006 / 0.0007 |
| INT8 (NNCF) | 7.0 / 7.9 | 14.8 / 15.6 | 38.6 | 0.0037 / 0.104 |

INT8 on CPU is the fastest configuration (142 calls/s); its worst-case action
error of 0.10 rad (6°) is the likely reason the INT8 chained column sits a
little below FP32. Every rollout report also records the in-the-loop cost
(render + preprocess + infer + postprocess) of its own run.

## Honest notes

- **The mug stands on the table while it is poured into; the other hand does not
  steady it.** The brief's scene has one hand hold the cup while the other pours.
  With two SO-101s a full pour is only reachable by rolling the bottle sideways
  toward the other arm, which puts the steadying hand beside the mug's rim. We
  searched 12 mug positions × pour directions × steadying grips (body side-grip,
  handle pinch; 6 cm and 9 cm mugs), measuring the distance between the arms'
  collision bodies over every pour pose: the best layout left 3 mm, most
  overlapped, and with a 9 cm mug no position allowed a full pour at all. A
  second search (`tools/steady_search.py`, `out/steady_search.json`) tried the
  grips the first one had not: a top-down pinch on the mug rim from 8
  directions, with the mug on the table or lifted 4 or 8 cm towards the bottle,
  at 20 mug positions, against the real pour planner. None of the 431
  combinations kept 10 mm between the arms: wherever the left hand reaches the
  rim, the arms overlap by at least 21 mm during the pour, and a lifted mug is
  out of reach of a top-down hand (it tops out ~9 cm above the table). So the
  bimanual coordination is shown instead by the two-handed plate carry and the
  spoon hand-over.
- **The hand-over holds the spoon off its balance point** (each hand 2.8 cm from
  the handle centre, the only collision-free layout found), so the spoon sags
  8–17° in the hands; it is still set down by touch.
- **During the pour the bottle can graze the mug rim** (seed 9: an 8 N bump,
  the mug moved 0.9 mm, all 24 beads still landed inside).
- **Water is 24 beads**, not a fluid: each is a sphere of 4 mm radius and
  0.27 g (6.5 ml in all), a granular proxy that pours like lentils. Container
  bases are 8 mm thick; with 5 mm bases the beads tunnelled through the disc.
- **The gripper model is our choice, not Menagerie's.** The Menagerie file
  models the optional 12 V servo at stall (2.94 N·m, ~37 N at the tip, 75–160 N
  of pad force in our earlier runs — 340× a spoon's weight) with hard fingers
  at μ = 1. We cap the torque at 0.8 N·m and use μ = 0.8 rubber pads. Pad force
  while holding is now 20–27 N. Real pad friction on a wet ceramic mug may be
  lower.
- **The props are child-tableware scale** (the jaws open 5 cm): chunky 10 mm
  handles, a ramekin-sized dish, a vial for a bottle. A flat 2 mm steel fork is
  out of this gripper's reach.
- **The demonstrations are privileged.** Scripted skills read exact object
  poses, centres of mass and contact events (set-downs stop at the first
  contact). The policy sees only cameras and joints, with no gripper load
  signal, so it must learn timing from pixels.
- **Stage completion: two modes, both reported.** With `--switch oracle` the
  evaluator reads the physical state to decide when a stage is done and
  advances the policy's subtask token. With `--switch head` a small
  stage-completion network (`policy/stage_head.py`, same cameras and joints as
  the policy, trained on the same demonstrations, run through OpenVINO) makes
  that call and the oracle only scores it. `home` stages are never scored, and
  a stage's success is also reported conditional on every earlier stage having
  been solved by the policy. On v1 the learned switch is not usable yet: it
  ends stages early, and the chained run with `--switch head` scores 7/10,
  10/10, 10/10 and then 0/10 for the pour, the bottle return and the hand-over
  (`out/rollout_policy_fp32_head.json`), so every number in the results table
  uses the oracle switch.
- **The planner fills in preconditions.** If the language model writes
  "pour" without picking up the bottle, or asks for a fork with the drawer
  shut, `planner.complete()` inserts the missing steps and logs it; on 20
  paraphrased instructions the rate of fully correct plans is reported in
  `out/planner_eval.json`.
- **Hardware.** The challenge targets Intel Core Ultra Series 2/3. This build
  was developed and benchmarked on an Intel Core i9-14900HX with its Raptor Lake
  integrated GPU; the NVIDIA RTX 4060 in the same laptop was used only to train
  ACT — every inference number is Intel CPU or iGPU. `bench/benchmark.py`
  reproduces every number on a Core Ultra machine in one command.
- **Hybrid mode** is reported separately from pure-policy results: a stage the
  scripted skill had to finish is counted as assisted, never as a policy success.
- **Runs are deterministic given the seed** (IK restarts are seeded per
  episode); policy results on OpenVINO iGPU/INT8 may not be bit-stable, and are
  single runs unless stated.
- An earlier version of this project (git history) attached objects to the
  gripper with weld constraints; it was replaced because that is not how a real
  gripper holds anything.

## Run it

Requirements: Python 3.12 (Ubuntu 24.04's default), git, ~10 GB of disk.
On Ubuntu: `sudo apt install python3.12-venv libegl1 libgl1 fonts-dejavu-core git`.
A headless Linux machine (no display) also needs `export MUJOCO_GL=egl` before
anything that renders (recording, rollouts, videos); the scripted evaluation
renders nothing and runs anywhere.

### 1. Install (10–20 min, mostly PyTorch)

```bash
bash scripts/install.sh                                         # Linux; add --cuda for GPU training
powershell -ExecutionPolicy Bypass -File scripts\install.ps1    # Windows; add -Cuda for GPU training
```

This creates `.venv`, installs the pinned `requirements.txt`, clones only
`robotstudio_so101` from MuJoCo Menagerie into `third_party/`, generates
`scene/bimanual_table.xml` and runs the 21 unit tests. Below, `python` means
`.venv/bin/python` (Linux) or `.venv\Scripts\python` (Windows; set
`$env:PYTHONUTF8 = "1"` first).

### 2. Scripted contact pipeline — no downloads, CPU only

```bash
python -m sim.task                                   # seeds 0-9, ~30 s wall each; add --video out/scripted.mp4
python -m sim.task --seeds $(seq 1000 1029)          # the unseen block (PowerShell: --seeds (1000..1029))
python -m tools.audit_contact --seeds 0 1 2          # balance / set-down / drop audit, ~30 s per seed
python -m tools.trace_stage --stage 5 --object spoon # one stage (0-based; 5 = hand-over) with close-ups
python -m tools.robustness_envelope --seeds 1 2 3     # push friction / mass / position past the randomised ranges (~25 min)
python -m tools.render_media                         # cover stills and the 10-seed grid video (~8 min)
```

### 3. Language planner — one 0.9 GB download

```bash
python -m planner.download_model                     # Qwen2.5-1.5B-Instruct INT4 IR from Hugging Face (not gated)
python -m planner.planner "Set the table and pour a drink"    # ~2 s per plan on CPU after loading
python demo.py --seed 3                              # instruction -> plan -> scripted execution -> out/demo.mp4
python -m bench.benchmark --skip-planner             # policy IRs only; drop the flag to time the planner too
python -m tools.planner_eval                         # 20 paraphrased instructions -> valid / correct plan rate
```

Spoken input: copy `.env.example` to `.env`, add a Speechmatics key, then
`python demo.py --audio your.wav`.

### 4. Learned policy — optional; needs an NVIDIA GPU (or the released checkpoint)

Download the trained checkpoint and OpenVINO IRs from the GitHub release
(link to be added) and unpack them to `outputs/act_contact/` and
`models/policy/`, or retrain:

```bash
# Every command takes its data / model paths explicitly (no silent defaults to an older model).
python -m tools.record_parallel --root data/contact_v3 --episodes 300   # 8 recorder processes, ~1-1.5 h -> data/contact_v3/merged
python -m policy.train --dataset-root data/contact_v3/merged --output-dir outputs/act_contact_v3 --steps 40000
                                                     # ~3.5 h on an RTX 4060 (install with --cuda); --device cpu works but takes days
python -m policy.export_openvino --checkpoint outputs/act_contact_v3/checkpoints/040000/pretrained_model \
    --dataset-root data/contact_v3/merged --out-dir models/policy_v3   # FP32 / FP16 / INT8 IR + accuracy report
P="--policy models/policy_v3/act_fp32.xml --checkpoint outputs/act_contact_v3/checkpoints/040000/pretrained_model"
python -m policy.rollout $P --mode policy            # learned policy alone on 30 unseen scenes (seeds 1000-1029),
                                                     # every rate with a 95% Wilson interval; 40 s per stage
python -m policy.rollout $P --mode hybrid            # a scripted skill finishes any timed-out stage (counted as assisted)
python -m policy.rollout $P --stagewise              # each stage from a scripted start state
python -m policy.stage_head train --dataset-root data/contact_v3/merged   # stage-completion head (~10 min GPU)
python -m policy.stage_head export --dataset-root data/contact_v3/merged
python -m policy.rollout $P --mode policy --switch head   # the policy side ends each stage; the oracle only scores
python -m policy.rollout $P --mode policy --ensemble 0.01 # temporal ensembling: infer every step, blend overlapping chunks
python -m bench.benchmark                            # planner + every IR in models/policy on CPU / iGPU / NPU
```

### 5. Media — optional, not needed for any result

```bash
python demo.py --executor policy --seed 8 --out out/demo_policy_seed8.mp4
python -m sim.task --video out/grid_10seeds.mp4
python -m piper.download_voices en_US-lessac-medium --download-dir models/tts
python -m tools.voiceover && python -m tools.make_video --voice-dir out/voice
cd docs/slides && npm ci && node build_deck.js       # Node 18+; the built table_for_two.pptx is tracked
```

## Layout

| Path | What it is |
|---|---|
| `scene/` | MJCF builder (`build_scene.py`) |
| `sim/` | environment, IK, grasp geometry, scripted skills (`skills.py`, `pour.py`), plan executor |
| `planner/` | OpenVINO GenAI planner with plan validation |
| `data/` | LeRobot dataset recorder |
| `policy/` | ACT training wrapper, OpenVINO export, OpenVINO runtime, rollout evaluation |
| `bench/` | Intel inference benchmark |
| `tools/` | grasp and pour labs, per-stage tracer, contact audit, video and results helpers |
| `docs/` | slides (`docs/slides`, pptxgenjs), narration, media |
| `scripts/` | one-command installers |

Generated and ignored: `scene/bimanual_table.xml`, `third_party/`, `models/`,
`data/*/`, `outputs/`, `out/`. Versions are pinned in `requirements.txt`
(MuJoCo 3.13, LeRobot 0.6.1, OpenVINO 2026.3, NNCF 3.3, PyTorch 2.11).

## Credits and licences

This repository is MIT. It uses, unmodified and not vendored:

- SO-101 model: [MuJoCo Menagerie `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0), derived from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100).
- Physics: MuJoCo (Apache-2.0). Policy: Hugging Face LeRobot, ACT (Apache-2.0). Inference: Intel OpenVINO, OpenVINO GenAI, NNCF (Apache-2.0).
- Planner model: Qwen2.5-1.5B-Instruct (Apache-2.0), INT4 OpenVINO conversion from the OpenVINO Hugging Face organisation.
- Voice-over: Piper TTS (MIT) with the `en_US-lessac-medium` voice (see its model card for the voice licence). Speech input: Speechmatics (commercial API, key required, optional).
