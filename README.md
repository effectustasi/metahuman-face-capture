# facecap

Webcam, video file or iPhone → ARKit blendshape scores → MetaHuman face board → RigLogic.

Drives a MetaHuman face rig in Blender from an ordinary camera, and contains a
**differentiable reimplementation of RigLogic in PyTorch** that makes the inverse
problem — *which control combination produced these landmarks?* — tractable.

> **Status:** working, actively developed, rough edges documented honestly below.
> The solver is experimental and its limits are measured, not hidden.

---

## Why this exists

Epic's RigLogic is a compiled CPU library. It is fast to evaluate forward, but it
is a black box: no gradients, no GPU, no batching. A numeric Jacobian needs N+1
**sequential** RigLogic calls per iteration — at 174 parameters that was 341 ms
per iteration, which makes landmark-driven solving impractical.

Rewriting the evaluation chain as differentiable tensor ops removes the N+1 factor
entirely. Autograd returns the full gradient in one backward pass:

| | numeric Jacobian | differentiable rig |
|---|---|---|
| cost per iteration | O(N) RigLogic calls | **O(1)** |
| measured gradient time | 341 ms | **4.5 ms** |
| accuracy vs real RigLogic | — | **3.3e-07** max error |

That is not a 2x speedup, it is a change of complexity class. Accuracy is verified
against the real library in `tests/test_riglogic_torch.py`.

The reconstructed chain (recovered by probing the API, see
`scripts/05_extract_behavior.py`):

```
GUI (174)
  |  piecewise linear:  raw = slope * gui + cut,  gui in [from, to]
raw (263)
  |  PSD: weighted PRODUCTS of raw controls, appended after raw
control (808 = 263 + 545)
  |  122 joint groups, each a dense (out x in) matrix
joint deltas (7830 = 870 x 9)
  |  direct gather
blendshape weights (782)
```

---

## Architecture — two processes

MediaPipe **cannot be installed into Blender's Python**; the protobuf/numpy
conflict breaks Blender. So the capture side and the rig side are separate
processes that talk over UDP:

```
  [detector/.venv  Python 3.11]          [Blender 5.0  Python 3.11]
   webcam / video                         meta_human_dna 0.5.4
        |                                          ^
   mediapipe FaceLandmarker                        |
        |  51 ARKit scores + head matrix           |
        |                                   blender_addon/charface
        +-- offline: takes/*.jsonl  ------->  charface.bake_take
        +-- live:    UDP :11111    -------->  charface.live
                                                   |
                    core/  (mapping, filters, take schema — stdlib, shared)
```

`core/` is imported by both sides, so the mapping logic lives in exactly one place.

### How the chain is driven

```
ARKit pose --(Epic posemap)--> raw CTRL_expressions.* --(inverse GUI->raw)--> GUI axis
                                                                               |
                                       face board pose_bone.location ----------+
                                                                               |
                                           RigLogic --> bones + shape keys + masks
```

Raw controls are **never written directly** — the addon calls
`mapGUIToRawControls()` on every evaluation and regenerates raw values from GUI,
so anything you write to a raw control is overwritten in the same frame. Details
with file:line references: [docs/api-notes.md](docs/api-notes.md).

---

## Install

### 1. Detector

Python 3.11 is required: MediaPipe publishes no wheels for 3.13/3.14, and the
addon's DNA bindings are compiled for py311 too.

```bash
py -3.11 -m venv detector/.venv
detector/.venv/Scripts/python -m pip install -r detector/requirements.txt
curl -L -o detector/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 2. Test / solver environment

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

**On Blackwell (RTX 50, sm_120) the cu128 wheel is mandatory.** Torch from the
default index fails on these cards with `no kernel image is available for
execution on the device`. Verify with a real kernel, not just `is_available()`:

```bash
.venv/Scripts/python -c "import torch; x=torch.randn(512,512,device='cuda'); print((x@x).sum())"
```

Torch is installed here deliberately and not borrowed from another environment:
`core/riglogic_torch.py` is a real dependency of facecap.

### 3. Blender addon

Install `blender_addon/charface` as an addon (or symlink it into `scripts/addons`).
It sits **on top of** Character DNA Pro / `meta_human_dna`, it does not replace them.

```bash
python scripts/install_addon.py --version 5.0
```

Add `--uninstall` to remove it.

### 4. MetaHuman-derived data — you must generate this yourself

The mapping table and the solver's forward model are extracted from a MetaHuman
DNA file and from Epic's ARKit posemap asset. **That content is licensed by Epic
and is not redistributed here.** Generate it from your own character:

```bash
blender --background --python scripts/01_dump_dna_controls.py -- <head.dna> mapping/_generated/dna_controls.json
python scripts/02_build_mapping.py --posemap <posemap.json> --controls mapping/_generated/dna_controls.json
```

For the solver, additionally:

```bash
blender --background --python scripts/05_extract_behavior.py -- <head.dna> mapping/_generated/behavior.npz
blender --background --python scripts/04_extract_face_model.py -- <head.dna> mapping/_generated/face_model_landmarks.npz --vertices mapping/_generated/correspondence/correspondence.json
blender --background --python scripts/06_dump_reference.py -- <head.dna> mapping/_generated/riglogic_reference.npz 32
```

Coverage: **51/51** MediaPipe channels plus `tongueOut` for iPhone. `mouthClose`
could not be derived from Epic's asset and is hand-written — source and confidence
bounds in [docs/api-notes.md](docs/api-notes.md) §8 and §11.

---

## Usage

### GUI

Run `facecap.bat`. Camera selection, sending to Blender, recording and live
channel bars in one window.

```
+-------------------------+----------------------+
|                         |  Camera  [0] [Scan]  |
|        preview          |  UDP 127.0.0.1:11111 |
|                         |  [x] Preview         |
|                         |  [x] Filter          |
|                         |  [ ] Raw landmarks   |
|                         |  [ ] Write to file   |
|                         |      [ START ]       |
|                         |  face YES | 28 fps   |
+-------------------------+----------------------+
| Mouth / jaw                                    |
|  mouthSmileRight ##########      0.96 (0.96)   |
|  jawOpen         ###             0.27 (0.28)   |
| Eye                                            |
|  eyeBlinkRight   ######          0.64 (0.77)   |
| Brow / nose                                    |
|  browDownRight   ###             0.37 (0.76)   |
+------------------------------------------------+
```

**The channel bars are the heart of calibration.** Without seeing which channel
fires how much when you make an expression, finding the dead one is guesswork.

The bars show **deviation from neutral**, not the raw value, and they are grouped
by region. A single "top 8" list was useless, because MediaPipe reports high eye
and brow values even on a neutral face. Measured on a real capture, subject sitting
expressionless:

```
eyeSquintLeft 0.473   browDownRight 0.395   browDownLeft 0.305
eyeLookUpLeft 0.301   eyeBlinkLeft  0.290   eyeSquintRight 0.267
```

Eight channels permanently above 0.2, while mouth channels — genuinely 0.00 at
neutral — never entered the list at all. Eyes mislead too: `eyeBlinkLeft` starts at
0.290, so a raw 0.87 is really a 0.58 blink. The raw value in parentheses is kept
so you can see whether a channel is saturating.

Neutral is learned from the first 30 frames after start — **hold a neutral face**
then. If lighting or distance changes, hit `Re-take Neutral`.

Preview is capped at 15 fps (drawing every frame into Tk costs more than capture).
`cv2.imshow` is deliberately not used: HighGUI runs its own event loop and can
deadlock next to the Tk mainloop on Windows.

### Command line

Produce a take from a video file:

```bash
detector/.venv/Scripts/python detector/detect.py video --input take.mp4 --out takes/take.jsonl
```

Live:

```bash
detector/.venv/Scripts/python detector/detect.py live --udp 127.0.0.1:11111 --preview
```

Then *Live Face Capture* in Blender (ESC to exit).

---

## Writing keyframes

Two routes. Live recording is the normal one; baking is for re-trying a take.

### 1. Live recording, straight to the timeline

1. `facecap.bat` → START
2. Blender: **Live Face Capture**
3. Put the playhead on the frame you want
4. Press **`Record to Timeline`** (turns red), play, press again

Keyframes are written while you play and the playhead advances. The panel shows
the live count.

Timing comes from the **wall clock, not the packet counter**: packets arrive at
30 fps while the scene is usually 24 fps, so one keyframe per packet would slow the
animation by 25%. When several packets land on the same scene frame, the axis value
with the **largest absolute value** wins — taking the last one loses an 89 ms blink
peak (verified headless: 30 packets → 25 keyframes, peak 0.9 preserved).

The scene range is only ever **extended**, never narrowed, so it cannot clip your
existing animation. The real range is printed in the panel.

### 2. Bake from a recording

Useful for re-running the same take with a different calibration.

1. In the GUI tick **Write to file**, START, perform, STOP
2. Blender: N-panel → **charface** → *Bake Take to Face Board* → pick the .jsonl

| Option | Meaning |
|---|---|
| `Start Frame` | scene frame the animation starts on |
| `Neutral Calibration` | treats the first 30 frames as neutral and subtracts it. **Leave on** |
| `Single Channel` | only this ARKit channel (validate with `jawOpen` first) |
| `One-Euro Filter` | enable here if you did not filter while recording |
| `Mirror Left/Right` | the user's left is the character's right in a camera image |

**The take is fitted to the scene fps by timestamp**, not 1:1 frame to frame.
Measured: a webcam delivers 30 fps, Blender's default scene is 24 fps — 1:1 writing
turns a 37 s take into 46 s. Frame intervals are not constant either (12–50 ms
measured). Logic in `core/take.py:plan_scene_frames`, tests in `tests/test_take.py`.

**On your first attempt put `jawOpen` in `Single Channel`.** If the jaw opens, the
chain is sound — clear the field and bake again. Wiring everything at once hides
which layer is broken.

---

## Calibration

### Head

N-panel → charface → **Head Calibration**. Settings live on the *scene*, so they can
be changed while the live stream runs and take effect immediately — as an operator
property, every experiment would have meant stop/start.

- **Live readout**: yaw / pitch / roll in degrees, plus packet counter and face
  YES/NO. Without seeing how far each angle travels, finding the wrong axis is
  guesswork.
- **Re-take Neutral**: neutral is built from the first 30 frames; if you were not
  sitting straight then, the head stays permanently skewed. Look straight, press.
- **Per-angle mapping**: target bone axis + `Invert` + `Gain` for each angle.
  Gain 0 disables the angle, 2 doubles it.
- **Axis test**: X / Y / Z keys rotate the head on that axis so you can see what
  each one does. `Reset` returns to neutral.

Order that works: axis test first, then set the angle mapping, then fix direction
with `Invert`, and only then tune amount with `Gain`.

### Channel range (ROM capture)

Channels do not reach 1.0. Measured on a real iPhone recording: `eyeBlink`
saturates at **0.917**, `mouthSmile` at 0.865. Since the rig is built to open fully
at 1.0, **the eye never fully closes** without this step.

In the Face Calibration panel:

1. Press `Learning Range`
2. Perform every expression **to its limit** (blink, open jaw, smile, raise brows…)
3. Turn it off
4. **`Save Ranges`** — otherwise it is lost when Blender closes

The profile is saved to `//charface_profile.json` (next to the scene, or in the
project root if the scene is unsaved) and loaded automatically when live capture
starts. Saving does not clobber the existing profile: gains and dead zones you set
by hand are preserved, only `input_max` is updated.

---

## Sources

The operator only listens on UDP; it does not know who is sending.

| Source | Status |
|---|---|
| `scripts/replay_take.py` | ✅ works — replays a recorded take in real time, fake detector |
| `detector/detect.py live` | ✅ webcam — verified on real photographs |
| iPhone Live Link Face | ✅ **supported** — parser verified against 546 real packets |

### iPhone (Live Link Face)

The operator **auto-detects the packet**: JSON means webcam/replay, binary means
Live Link Face. No setting, same button, same port.

| | webcam (MediaPipe) | iPhone (ARKit) |
|---|---|---|
| channels | 51 | 52 |
| `tongueOut` | no | **yes** |
| `mouthClose` | no (hand-written) | **real measurement** |
| stability | RGB estimate | TrueDepth sensor |
| eye gaze | from blendshapes | 6 separate channels |

Setup:

1. App Store → **Live Link Face** (Epic Games, free). iPhone X or newer.
2. Top-left gear → **Live Link** → **Add Target** → your PC's IP, port **11111**
3. Phone and PC on the same network
4. Verify: `iphone_test.bat` (or `python scripts/llf_probe.py`)
5. *Live Face Capture* in Blender

**The packet layout is not officially documented by Epic** and community sources
contradict each other. The parser was run against **546 packets** from a real
iPhone recording: all parsed, the fallback path was never taken. Details in
[docs/api-notes.md](docs/api-notes.md) §12. `llf_probe.py` still reports which path
was used — if it says `scan`, the layout differs on your iOS version.

---

## Scene performance panel

N-panel → charface → **Scene Performance**. Unrelated to live capture, always usable.

### Hair particles making the scene crawl

The hair emitter is bound to a mesh that RigLogic deforms continuously, so every
deformation recomputes every strand. The real cost is not visible in the `count`
field — it is multiplied by child particles:

```
2000 strands x 25 children = 50,000 strands, each with 2^3 segments
```

The panel computes and shows this product. **`Lighten Hair`** lowers the viewport
values (`display_percentage` 10, `child_percent` 0, `display_step` 1) — measured:
50,000 strands → 200. **Rendering is unaffected**; `render_step` and
`rendered_child_count` are separate fields and are left alone. The backup is stored
in the particle settings datablock, so `Undo` restores the exact previous values
even across a save/reload.

### RigLogic auto-evaluation

With the addon's `evaluate_dependency_graph` flag on, RigLogic runs on **every
depsgraph update** (`meta_human_dna/rig_logic.py:37`) — selecting an object,
changing frame, editing particles all re-evaluate an 800+ bone rig. The addon only
disables this programmatically during bake/import and exposes it in no UI.

The panel lets you turn it off; the face then stops updating on its own and you
trigger it with `Evaluate Now`. Worth keeping off while doing hair or modeling work.

### During live capture

**Preview Hz** throttles the viewport update. It **does not affect recording** —
keyframes come from the incoming values, not from what is on screen, so the
animation records at full rate even at 15 Hz viewport. The panel shows measured
numbers: rig ms, apply ms, and how many frames per second that allows.

---

## Solver (experimental)

The ARKit mapping works channel by channel: "I saw jawOpen 0.42, write 0.42 to the
jaw control". The solver does the inverse: *which control combination produces this
landmark configuration?* This is what MetaHuman Animator does.

### The forward model must come from the character's own DNA

With the default `head.dna`, the fit scale to Blender world came out as 0.010565
(should be 0.01 — a 5.7% error) with 2.75 mm residual. The same measurement with the
character's own DNA gives 0.009994 and 0.47 mm — **5.8x better**. Modeling a
different head is the failure that happens silently and ruins everything downstream.

```bash
detector/.venv/Scripts/python detector/detect.py live --out takes/take.jsonl --landmarks --preview
python scripts/10_solve_take.py --take takes/take.jsonl
```

`--landmarks` also writes all 478 raw landmarks (about 30x larger frames, hence not
the default). The **first 30 frames of the take must be expressionless**; neutral is
learned there and identity difference is cancelled against it.

### Correspondence must be built from a NEUTRAL, FRONT render

The first correspondence was built with the head turned ~17°. MediaPipe found more
landmarks (473 → 478), but landmarks on the far cheek were bound by a front-cast ray
to vertices on the **near** side. Measured cost:

| | turned head | neutral + tangent filter |
|---|---|---|
| landmarks | 478 | 447 |
| DNA → Blender fit residual | 0.471 mm | **0.028 mm** |
| left / right landmark balance | 176 / 260 | **202 / 207** |

`Render Head` now zeroes head rotation and the whole facial expression for the
duration of the render and restores them afterwards, whatever pose the character is
in. Landmarks whose ray hits the surface at more than ~70° are also culled
(`facing < 0.35`).

**Landmark count is not the goal.** Few and correctly bound beats many and wrong.

### Measured against a real face (1114-frame take)

Error in mm, lower is better:

| frame | expression | baseline (no controls) | ARKit mapping | solver |
|---|---|---|---|---|
| 143 | smile | 5.360 | 7.139 | **3.335** |
| 187 | jaw open | 1.488 | 3.893 | **0.670** |
| 246 | blink | 1.291 | 1.887 | **1.100** |
| 775 | lip pucker | 1.847 | 2.917 | **0.838** |

The solver beats the ARKit mapping by a wide margin on every frame. ARKit comes out
worse than the baseline because it finds the mouth correctly but injects fake motion
across the rest of the face: 57 landmarks that move 0.7 mm in the observation are
driven 4.9 mm, at cosine −0.19.

**But the solver still explains only 32–51% of the motion.** This is *not* an
optimization problem: with regularization fully off, 2000 iterations and 136
controls, the result gets **worse** (3.438 vs 3.365 mm). The ceiling is in the model.

The open problem: transferring a real human's landmark motion onto a particular
MetaHuman rig needs more than rigid alignment plus delta transfer. MetaHuman Animator
solves it by first extracting an actor-specific DNA (identity solve), solving the
expression on that rig, then retargeting. That step does not exist here.

**Tried and did not work** (recorded so it is not retried): solving pose jointly with
expression (`solve(joint_pose=True)`) — 3.475 → 4.026 mm on the smile frame; the pose
and expression gradients eat each other.

---

## Layout

| | |
|---|---|
| `core/` | Shared logic: mapping, one-euro filter, calibration, take schema. Stdlib-only except `facemodel.py` (numpy) and `riglogic_torch.py` (torch). |
| `detector/` | MediaPipe side. Separate venv. |
| `blender_addon/charface/` | Bake and live operators + panels. |
| `mapping/arkit_to_mh.json` | Generated mapping table. Not hand-edited. Not redistributed — see Install §4. |
| `mapping/_generated/` | DNA dump, build report, introspection output. Not redistributed. |
| `gui.py` | Tk interface. Runs under `detector/.venv`; launched by `facecap.bat`. |
| `scripts/` | Introspection, DNA dump, table generation, fixtures, UDP replay/monitor, addon install. |
| `tests/` | Unit tests that do not require Blender. |

```bash
.venv/Scripts/python -m pytest tests/
```

---

## Phase status

| Phase | Status |
|---|---|
| 0 — Discovery | ✅ `docs/api-notes.md`, with file:line references |
| 1 — Detector | ✅ verified — 51 scores + head quaternion from a real photograph |
| 2 — Mapping | ✅ 51/51 channels, 88 axes; `mouthClose` hand-written (§11) |
| 3 — Blender applier | ✅ live path verified in a real MetaHuman scene; bake operator not yet field-tested |
| 4 — Calibration / filter | ✅ head + face calibration tunable live, group gains, three-band one-euro |
| 5 — Export | uses the addon's own bake/export, no extra code |
| 6 — Derived controls | ✅ eyelid/gaze coupling, lid pressure, micro-saccade, channel range (§15) |
| 7 — Differentiable rig | ✅ RigLogic rewritten in PyTorch, 3.3e-07 error, 4.5 ms gradient (§14) |
| 8 — Landmark correspondence | ✅ 477/478 landmarks, barycentric, from the character's own head (§15) |
| 9 — Solver | ✅ two-stage (L1 + debias), 0.002 control error on synthetic tests (§16) |
| 10 — Identity separation | ✅ `core/observation.py` — observation/model neutral difference cancelled |
| 11 — Real-face trial | ⚠️ works, explains only 32–51% of motion (see above) |

---

## Third-party notices

This project interoperates with, but does not redistribute, the following:

- **MetaHuman**, **RigLogic**, the ARKit posemap asset (`PA_MetaHuman_ARKit_Mapping`)
  and DNA files are © Epic Games and licensed under Epic's terms. No MetaHuman
  content is included in this repository; `scripts/01`–`06` regenerate what is
  needed from your own licensed assets.
- **Character DNA Pro / `meta_human_dna`** is a third-party Blender addon that must
  be installed separately.
- **MediaPipe** (`face_landmarker.task`) and **Live Link Face** are downloaded or
  installed separately under their own licenses.

## License

[Apache License 2.0](LICENSE).

The Turkish original of this document is kept at [docs/README.tr.md](docs/README.tr.md).

## More tools by effectustasi

- [agent-receipts](https://github.com/effectustasi/agent-receipts): Skills that make AI coding agents prove "done" with real test output
- [blender-dlss5-neural-rendering](https://github.com/effectustasi/blender-dlss5-neural-rendering): Blender viewport and renders through DLSS 5 neural rendering
- [autodesk-inventor-mcp](https://github.com/effectustasi/autodesk-inventor-mcp): Connect AI agents to a live Autodesk Inventor session
- [unreal-groom-alembic-exporter](https://github.com/effectustasi/unreal-groom-alembic-exporter): Export UE Groom assets (MetaHuman hair) to Alembic
