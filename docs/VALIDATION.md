# Current validation — 22 October 2026

## M2 software checks

| Check | Result |
| --- | --- |
| Full color app JavaScript suite | 226 passed |
| Full QR transport/authentication JavaScript suite | 223 passed |
| Neural corrector Python suites | 16 passed; 1 native ONNX Runtime check explicitly skipped |
| Canonical dataset and capture checks | Physical phase/reference/parity labels, failed-payload retention, stable-header guards, bounded records, full-header joins, distinct IDs, exact CRC/RS recovery, and path checks passed |
| Python training/evaluation checks | Session/payload split rejection, masked loss and inactive-class gradients, correlated corruption, optimizer updates, checkpoint validation, and canonical decoder integration passed |
| Complete CPU pipeline smoke | Clean OBM5 examples → correlated corruption → current baseline regeneration → 10 optimizer updates → validation selection → separate held-out evaluation → checkpoint export completed |
| Trained checkpoint Web/WASM parity in Node | Both G64/G128 exports passed all 10 measured runs; maximum absolute error 4.84×10⁻⁸ / 5.96×10⁻⁸ |
| Prepared model assets | Trained status, checkpoint provenance, both graphs, fixtures, and pinned runtime validated and copied locally |

The 10-update checkpoint is **only a wiring smoke test**, not a useful decoder:
it recovered 0/4 held-out synthetic frames while the classical baseline
recovered 4/4. The test therefore establishes neither a recovery improvement
nor real-device generalization. The checkpoint and generated data are ignored
local outputs, not shipped weights. The actual pilot and real-data fine-tuning
remain for the user's GPU and collected sessions.

Native Python ONNX Runtime still cannot initialize in this environment, so
the smoke export explicitly records native parity as `not_run`. Actual ONNX
Runtime Web/WASM numerical checks ran in Node. A headless browser binary could
not be obtained from its download endpoint; the new capture UI has automated
module/lifecycle tests but still requires a real-device collection run.
The supplied M1 browser reports below are separate evidence for the inference
path. Follow [the M2 guide](NEURAL_CORRECTOR_TRAINING.md) for device work.

## M1 device reports — 22 October

The two supplied reports establish a useful **64×64 starting profile** for M2.
They do not contain 128×128 measurements or a successful PC camera crop.

| Device / test | Backend | Result |
| --- | --- | --- |
| Android Chrome, G64 fixture, camera off | WebGPU | All 10 runs passed full-logit parity; warm p50 26.8 ms, p95 33.7 ms |
| Linux Chrome, G64 fixture, camera off | WASM, one thread | All 10 runs passed full-logit parity; warm p50 10.4 ms, p95 11.5 ms |
| Phone camera, two G64/16-color crops | WebGPU | Both inferred finite logits; preprocessing 191.8 / 133.2 ms, inference 37.4 / 30.2 ms |
| PC camera, three G64 crop attempts | Not reached | One unreadable header, two incomplete marker detections; no inference ran |

Both fixture reports have maximum absolute logit error 2.98×10⁻⁸ and no
failures. All recorded operations had zero animation-frame gaps above 50 ms.
The phone used a 720×1280, 60 fps rear-camera feed, with about six observed
pixels per cell. The PC used 1920×1080 at 30 fps. WebGPU was exposed by the PC
browser but adapter initialization failed; automatic WASM fallback worked.
Device model names and power settings were not supplied, and reported JS heap
values do not measure total runtime/GPU memory.

Proceed with the existing network and G64 for initial training. Keep G128
available as an experimental profile. PC marker/header acquisition remains an
optical alignment/detection issue; learning payload colors cannot repair a
missing crop. Phone preprocessing is already more expensive than inference,
so capture and later receive scheduling must remain bounded. Additional G128,
explicit phone WASM, camera-active fixture, and PC crop checks remain useful,
but do not block building M2 capture/training tools.

Source reports (retained by the user; summarized here rather than copied into
the repository):

- `decoder-m1-2026-10-22T14-22-55-695Z.json` — SHA-256
  `5011dc809e74ad84e780dd5e297f643782b5a187c61a8ae6694145324ffd4749`.
- `decoder-m1-2026-10-22T14-31-23-287Z.json` — SHA-256
  `77ac46205aab1db70d6ca905558dc33fef3d247beeba7a77cd490b6f58ccb90f`.

## Neural corrector M1

The untrained model/export, rectified input and score adapter, and separate
Corrector Lab were implemented and checked on 20 October. No physical device
test or trained-model recovery result is claimed.

| Check | Result |
| --- | --- |
| Full color optical bench Node suite | 215 passed, including 22 new decoder/input/lab tests |
| Full QR transport/authentication Node suite | 223 passed |
| Original Python experiment suite with CPU PyTorch | 10 passed; the four previously skipped dynamics checks now executed |
| New model/export suite | 7 passed; 1 native ONNX Runtime check explicitly skipped |
| PyTorch model and ONNX graph checks | 121,792 parameters; both G64/G128 shapes, identical initialized weights, input sensitivity, manifest and fixture byte checks passed |
| ONNX Runtime Web 1.23.2 WASM in Node | Both sizes passed all 10 measured runs after 2 warm-ups, against every PyTorch output logit |
| Node WASM maximum absolute logit error | 2.98×10⁻⁸ at both sizes; no non-finite outputs or tolerance failures |
| Model/runtime preparation | Pinned runtime and both fixture bundles prepared locally with length/SHA-256 checks |

The Web runtime check used the same `ort.webgpu.min.mjs` bundle and local JSEP
WASM assets as the lab, with the WASM provider and one thread, in Node 24.19.0.
It checked 65,536 logits per G64 run and 262,144 per G128 run, passing both the
native fixture tolerance (2e-5 absolute + 2e-4 relative) and browser tolerance
(1e-4 absolute + 1e-3 relative). In this development environment, warm Node
p50/p95 was about 11.7/13.6 ms at G64 and 46.3/47.2 ms at G128. These are
**Node WASM measurements**, not phone/browser performance estimates.

Native Python ONNX Runtime aborts during CPU initialization in this restricted
environment, so its check runs in an isolated subprocess and is reported as
unavailable. The local smoke bundle was explicitly exported with
`--skip-ort-check`, recording `not_run`. PyTorch execution, ONNX graph checks,
and Web WASM parity ran successfully. A headless browser binary was unavailable;
its installation was blocked, so browser UI/worker execution and WebGPU parity
have not been verified here. Both real-device reports remain the M1 completion
gate; follow [Corrector Lab](CORRECTOR_LAB.md).

New input tests cover actual rendered markers, perspective/rotation, current
palette correction, stale/invalid headers, unsupported profiles, dense-crop
normalization, palette masking and phase inversion, reference exclusion,
finite-output rejection, and downstream CRC acceptance/rejection. Lab tests
cover manifest/asset bounds and hashes, sticky parity failures, timing summaries,
busy/cancel/worker lifecycle, and navigation. They do not establish optical
recovery quality from the untrained network.

Reproduce the M1 checks after the optional setup:

```sh
npm run test:corrector
python -m unittest discover -s experiments/neural-corrector/tests -v
npm run test:corrector:wasm
```

## Repository and cryptography baseline — 19 October

The following checks were run from a fresh copy containing only tracked source.
No previously generated app modules or private experiment results were present.
The npm hooks successfully prepared both independently servable apps from the
canonical shared modules and fixtures.

| Check | Result |
| --- | --- |
| Color optical bench Node suite | 193 passed, 0 failed |
| QR transport/authentication Node suite | 223 passed, 0 failed |
| Python experiment suite | 6 passed; 4 PyTorch-only checks explicitly skipped because PyTorch is not installed in this environment |
| Python syntax | All restored experiment, core, utility, and test files compile |
| Python fixture regeneration | 7 fixture files reproduced |
| Deterministic Python/JavaScript proof parity | 5 exact proof-byte matches |
| Python-generated proof verification in JavaScript | 5 accepted |
| Fresh JavaScript proof verification in Python | 28 cases: 5 accepted, 23 intentionally invalid cases rejected |
| Source selection | Both attachment manifests/checksum sets and all 14 nested ZIPs passed integrity checks |
| Repository layout | One browser proof core and one fixture set; no source ZIPs, copied notebooks, or old report directories tracked |

The NumPy tests include keygen/challenge/prove commands, compact proof size and
repetition policy, public-only verification, nonce consumption/replay rejection,
three circuit sizes, and Python optical framing/capture invariants. The restored
floating-point experiment retains its original numerical implementation; only
its shared utility import location changed. Its four PyTorch runtime tests were
subsequently run successfully in the 20 October checks above.

The current browser core, worker modules, fixtures, and vendor bytes were moved
without algorithm changes. The restored ZKBoo CLI delegates to the current core
instead of retaining the archived prover/serializer. The Python-only NPM1
simulator uses its own domain label for sequence identity, independent of the
cryptographic protocol version.

These checks do not repeat physical phone/webcam tests. Device observations are
recorded separately in [ROADMAP.md](ROADMAP.md). Fresh benchmark output belongs
in ignored `local-results/`; old run logs remain in Git history.

Reproduce from the repository root:

```sh
npm test
npm run test:python
npm run test:parity
# Include all Python runtime checks:
python -m pip install -r experiments/neural-dynamics/requirements.txt
npm run test:python
```

## Capture collector iteration — 2026-10-29

Automatic alignment scanning and filled overlays replace the manual check button.
Collection uses separate frame/phase pools (32 by default), uniform random
thinning (keep 16 by default), and worker-computed sampled pairwise RGB RMS
distance. The importer accepts the larger bounded exports (128 records / 160 MiB).

`npm run test:corrector`: **39 tests passed**. Checks include real rendered-header
probes, capture signatures, worker metric dispatch, 32-sample pools, thinning and
refill, phase isolation, byte-budget recovery, a 24-record offline join, duplicate
and stale-frame protections, and the existing learned-input/decoder parity
contracts. A deterministic camera/worker/DOM harness exercises the actual page
controller through automatic scan, fill, thin, refill, phase switch, pause, camera
stop, and hidden-page cleanup. No browser binary or physical camera was available
for a new visual/device run. Device acquisition and perceived overlay visibility
remain user checks; no claim of improved marker detection or trained decoding.

## Continuous collection and larger training runs — 2026-10-30

The submitted compact-model log selected a synthetic checkpoint at step 19,400:
117/128 exact frames (91.40625%) versus classical 124/128 (96.875%). Its real
validation selected step 30,600 by loss (1.387155), with 48.32963% cell error and
zero exact frames for both methods; the run ends at 32,000 steps. These are
128-sample checkpoint-selection measurements, not full-corpus test results.
No user's recorded tensors or trained checkpoint were attached, so alignment,
split coverage, and real-device predictions could not be independently replayed.
The source log SHA-256 is `802a0956bca7988195fc5a6ef0e4491b002eb2c759840c68a703f45e3724dcc4`.

This iteration preserves recording versions/keys/tensor layouts, moves capture
controls and counters above the camera, and continuously refines full phase pools
using a randomized distance heuristic. Offline corpus count caps are removed or
raised as documented, with 256 MiB manifests and a configurable synthetic output
budget. Pilot/fine-tuning defaults follow the completed 20k/32k runs. Compact
checkpoints remain usable; the new wide model has 487,104 parameters and requires
new training to establish any recovery gain. Evaluation now reports per-session
coverage and selects bounded subsets by session/profile round-robin.

Validation on this host:

- **42 JavaScript tests passed** (`npm run test:corrector`), including continuous
  replacement, stale results, phase/memory bounds, recording export shape,
  offline joins, generation argument limits, and model-profile validation.
- **25 Python tests passed; one native ONNX Runtime test skipped** because its
  native CPU initialization is unavailable on this host. Tests include an actual
  50,001-record corpus manifest, compact/wide checkpoint loading, optimizer
  updates, initialized defaults/profile checks, session-safe selection, gradients,
  color sensitivity, and ONNX graph/fixture checks.
- Wide G64/G128 graphs exported and passed **ONNX Runtime Web 1.23.2 WASM parity**
  in Node, 10 runs per size, with maximum absolute error 1.85e-6/3.10e-6. These are
  execution-agreement tests, not browser/phone performance or recovery results.
- Decoder asset preparation passed with the wide bundle. No site was deployed.

The controller is exercised with a deterministic DOM/camera/worker harness; a
physical-device layout/capture check remains necessary. No trained wide weights
are shipped and no improved real optical recovery is claimed.

## Broader synthetic lighting — 2026-10-30

The generator now includes burst-correlated residual white balance and signed
exposure, plus a translucent black or white rectangle covering 20–30% of affected
crops. Each effect has a CLI control. A separate photometric random stream keeps
moiré, motion, and sensor-noise draws fixed when toggling the new effects.
Algorithm/settings hashes distinguish synthetic recipes without changing the
recording or dataset formats, supervision, or session/payload split ownership.

- **14 affected Python tests passed**: six photometric tests and eight existing
  training/data tests. Checks cover both crop sizes, deterministic generation,
  exposure signs and clipping, WB bounds, quarter-area shade blending, temporal
  correlation, disabled controls, invalid options, unchanged labels/masks/bytes,
  recipe identity, corrupted-input baselines, and the existing optimizer/split
  checks.
- A CLI smoke generated and loaded **eight 16-color crops across all four palette
  phases** with custom lighting settings, verifying hashes, tensors, and recorded
  controls through the current dataset reader and canonical baseline bridge.

These checks establish generator correctness, not improved learned recovery.
No full training or held-out real-device evaluation was performed for this change.

## Training-run analysis and preview mirroring — 2026-10-31

The submitted wide-model log contains 100 pilot evaluations (20k updates) and
160 real fine-tuning evaluations (32k). The pilot selects step 16,000 with 125/128
exact frames versus classical 68/128. Real fine-tuning selects step 5,200 with
7/128 exact frames, 40.50% cell error and CE 1.14159; the final step has 1/128,
46.75% error and CE 2.20347. These fixed-subset validation measurements support
overfitting after early adaptation, not convergence at the final update. No full
real validation/test report or capture tensors were attached. The attachment
SHA-256 is `4b989acb551924588d277931eeb4733dcd9c1fded20c7109385b153c43aac906`.

Both fresh and initialized runs now default to 32k updates; learning rates stay
1e-4/5e-5. The training guide records the dataset counts, sampling implications,
and proposed controlled experiments. New losses, sampling strategies, automatic
early stopping and confidence fitting are proposals, not enabled behavior.

The capture page adds a corner mirror toggle for video and alignment overlay.
Detection, worker pixels, stored tensors, originals and recording formats are
unchanged. The preview preference persists through camera restarts within a page.

- **42 JavaScript tests passed** (`npm run test:corrector`), including asymmetric
  source-pixel, drawImage, overlay CSS, toggle, restart and recording checks in
  the page-controller harness.
- **Eight affected Python training/data tests passed**, including defaults and
  explicit overrides, optimizer/checkpoint wiring and split safeguards.

No physical-device mirror/layout check, full training, or deployment was performed.

## Target-matched synthesis and training — 2026-10-31

The submitted adaptation run recovered 62/768 real validation frames versus
69/768 for the earlier `wide-real` checkpoint, on the same captures. All 62 were
already recovered by the earlier model; phone→PC remained at zero. The
unsuccessful adaptation recipe, online augmentation, hard-cell objective and
plateau controller have been removed. The current training path uses ordinary
masked cross-entropy; source/session balancing, explicit direction metrics,
initial-checkpoint protection and split checks remain useful and are retained.

The new `targets` dataset command reconstructs canonical clean matrices from
verified sender truth in joined recordings. It deduplicates captures by physical
session, payload, frame and phase, preserves split identities and verifies
regenerated headers, labels, masks and payload bytes. It never treats a noisy
camera image as clean. Recording formats and existing joined corpora are unchanged.

Synthetic generation now adds independent bursts per target, drifting rolling-
shutter exposure/color bands, bounded scanline skew, and rotated soft rectangles
and ellipses with gradients and tints. Shadows attenuate light; reflections add
light. These are calibrated-crop approximations, not a full sensor simulation.
The paired sampler places equal numbers of real captures and synthetic variants
of the same exact target in each update. Matching content does not establish
matching camera distributions or improved generalization.

Validation on this host:

- **62 Python tests passed; one native ONNX Runtime initialization check was
  explicitly skipped** on this restricted host (63 tests total).
- 32 affected JavaScript tests passed, covering target reconstruction, capture
  controls, learned-input conversion and corrector-lab checks.
- End-to-end target→corruption→pairing tests verify independently seeded bursts,
  unchanged label hashes and phase identities, reproducibility, split isolation
  and the scaled output budget. Actual paired CPU optimizer/CLI tests preserve
  the initialized baseline and reject wrong accepted payloads.
- Deterministic corruption tests cover rolling-shutter row/color variation and
  temporal drift, bounded displacement, rotated/feathered shapes, multiplicative
  shade, additive reflection, zero-effect controls and both input sizes.

Canonical fixtures marked real exercise data/training branches only; they are
not device measurements. No new full GPU training, optical recovery improvement or browser
performance result is claimed, and no site is deployed by this change.


## Four-frame decoder and integrated motion blur — 2026-10-31

The submitted paired single-frame run selected step 22k: 68/768 exact captures,
43.07% cell error and CE 2.1346. This improves on the discarded adaptation run's
62 but does not beat the retained earlier 69. Phone→PC remained 0/256, with
validation CE rising 1.847→4.934→6.495 at 3k/22k/32k while unaugmented training
CE fell 1.745→.349→.182. Its last 32k model recovered 55/768. Those are correlated
single-exposure validation measurements, not live transport results.

The initial wide four-frame model has 488,544 parameters (+1,440 over one frame).
It stacks four independently corrected RGB crops into 12 channels, retaining the
same backbone and output grid. Explicit warm start divides repeated RGB-facing
weights by four; identical repeated inputs preserve the prior function. The
strict checkpoint loader supports exact one/four-frame variants; export does
not silently reshape models. Recording files remain unchanged.

Training selects distinct same-burst/same-header/phase exposures at random and
shuffles frame slots; it rejects undersized pools. Synthetic stacks never cross
independent corruption bursts. Evaluation uses nonoverlapping chronological
four-capture groups and classical mean-score fusion over those same captures.
Reports expose framesPerInput, candidate counts, consumed captures and leftovers.
Expected labels/bytes do not select observations or repair decoder output.

Synthetic exposure integration replaces the tiny horizontal two-sample smear
with arbitrary-direction linear, rotational and radial-zoom blur. Seven centered
samples are averaged by default; a shared trajectory drifts across each burst.
Gaussian blur, moiré, rolling-shutter bands/skew and photometric effects remain.

The browser Corrector Lab validates 12-channel exports and can collect a bounded
four-exposure same-phase camera stack, verifying media timestamps and
presented-frame counters through video-frame callbacks. The live transport
remains classical.
A fresh untrained wide four-frame export passes ONNX graph checks and actual ONNX Runtime
Web/WASM parity in Node at both G64/G128. Maximum absolute errors were 1.43e-6 and
1.34e-6 respectively. Native Python ORT could not initialize on this restricted
host; the export explicitly records that check as `not_run`. Node timings are not
phone/browser latency measurements. No site was deployed, no full GPU training
was performed here and no four-frame recovery improvement is claimed.

Automated verification: **79 Python tests passed with one native ORT skip**
(80 total). The suite includes actual four-frame optimization from a single-frame
checkpoint, strict export/load checks, same-burst membership, distinct inputs,
disjoint evaluation groups, equal-budget canonical recovery and integrated
linear/rotational/radial-zoom artifact tests. **55 JavaScript tests passed**,
including actual controller callback/cancellation behavior and same-budget
classical fusion. Four-frame model/runtime preparation also completed locally.


## Configurable burst width, faster capture, and concise training — 2026-11-01

The current model, checkpoint/export contract, evaluator and browser lab accept
any integer frame count from 1 through 64. Wide keeps its 32/64/128 feature
widths and has 489,024 parameters at five frames, 501,984 at 32, and 517,344
at 64. Compact remains below 500k throughout the supported range. Only the
RGB-facing kernels expand; explicit single-frame→T initialization preserves the
repeated-input function. Loading/exporting remains strict. Source bursts must
have at least T distinct members; recording formats and synthetic generation
controls are unchanged.

Capture collection defaults to 100 ms start-to-start, with the existing 500 ms
stable-header guard. The classical live reader polls at 100 ms with its existing
busy/new-frame guards. The Corrector Lab targets 100 ms between distinct presented
exposures and scales its retry budget with T. These cadences do not guarantee
0.5-second complete inference: settling, processing, phase changes and inference
add latency. Camera stack assembly uses one preallocated tensor; at G128/T64 its
192 MiB is additional to fixtures and runtime working memory.

Training uses a throttled tqdm bar and concise evaluation lines. Full evaluation
records and interval source/group summaries remain in history.json at the
existing evaluation cadence; interrupted/failed runs preserve completed records.
Tests cover terminal/nonterminal output and interruption handling.

Verification: **84 Python tests passed, one native ORT check skipped** (85 total),
and **59 JavaScript tests passed**. This includes an actual five-frame optimizer
run from a single-frame checkpoint, equal-budget canonical recovery, five-frame
ONNX export, 32/64 parameter/checkpoint shapes, metadata boundaries, and capture
cadence/cancellation. The native Python ONNX Runtime remains unavailable on this
host. A fresh untrained wide five-frame export passed actual ONNX Runtime Web
WASM parity in Node at G64 and G128, with maximum absolute errors **2.20e-6** and
**3.01e-6**. These are numerical agreement checks, not recovery or device speed
measurements. No full GPU training or site deployment was performed.


## Fixed random evaluation and sampling audit — 2026-11-02

Evaluation now uses one seed-7 permutation per burst before making disjoint
T-frame groups. The previous chronological partition is removed. Repeated
validation and standalone evaluation reuse the same draw; relocation, input
ordering, timestamp changes and unrelated bursts do not alter a group's member
order. Classical and learned recovery use identical groups. Reports and training
settings record the evaluation method and seed. Existing checkpoints can be
reevaluated without retraining, but earlier chronological reports are a different
observation grouping and should not be compared as an architecture change.

The sampling audit found one narrow conditional bias: after a capture had already
been drawn from a burst within an optimizer update, a random start followed by a
forward scan favored the next unused array entry. It now chooses a uniform rank
among unused captures. A deterministic regression checks each possible rank for
several used-capture patterns. The submitted run did not reuse bursts within an
update, so this particular bug does not explain its loss curve.

A controlled 1,000-update audit expanded 16,000 five-frame stacks. Source and
acquisition-direction quotas, target pairing and distinct same-burst members
were correct. All 32 monitored real captures, all 160 capture/slot combinations,
and all 16 synthetic corruption variants were sampled. No RNG reset or missing
training shuffle was found. The rare bounded fallback for choosing eligible
bursts remains an approximation to hierarchical sampling, not a claim of exact
conditional uniformity across all hierarchies.

The submitted five-frame run's validation CE rises .917→3.160 between 3k and 32k
updates while real-training CE falls .946→.071. This remains evidence of a
substantial generalization gap under the previous evaluation policy. Its cause
cannot be assigned to checkpoint ranking. Corrected evaluation needs the user's
checkpoint and captured tensors, which are not available here.

The design document now diagrams both implemented models and a **proposed**
125,600-parameter five-frame model: a 16→16 normalized inlet, 16/32/64 features,
normalized feature skips and no RGB readout. Its module count was verified
locally; the proposed profile is not implemented or trained. Another concrete
source mismatch is documented: corrupted synthetics are limited to [-1,1], while
calibrated real inputs permit [-1.5,1.5]; actual prevalence in the recorded data
has not been measured. No synthetic or recording format changes were made.

Verification: **86 Python tests passed, one native ONNX Runtime check skipped**
(87 total), and **59 JavaScript tests passed**. Coverage includes fixed random
membership, group/path/order independence, disjoint equal-budget canonical
recovery, uniform unused-capture draws, and actual five-frame training that
records the evaluation policy. The smaller architecture was counted only; no
new model training, checkpoint recovery measurement or site deployment occurred.


## Normalized compact and one final neural input range — 2026-11-02

The accepted narrower architecture is now implemented as `normalized-compact`,
the fresh-training/export default. Its inlet is 3T→16→16, its backbone 16/32/64,
and all 12 hidden convolutions have channel LayerNorm followed by ReLU. There is
no direct RGB readout; learned-feature skips remain, and the head is linear.
The five-frame model has **125,600 parameters**; 1–64 exposures remain supported.
Strict model/checkpoint inference distinguishes all implemented profiles and
rejects mixed weights, inconsistent metadata and incompatible initializations.

Capture normalization remains `c/127.5 - 1` with the existing [-1.5,1.5] storage
bounds. Affine calibration can exceed camera-byte bounds, which explained the
extra range retained in real recordings. Every model now begins with
**clip(x,-1,1)** before any learned convolution, including wide's RGB branch.
This preserves recorded/joined files while making real and synthetic learned
input ranges identical. The ONNX graph contains the same first operation;
export manifests require `networkInputClip:[-1,1]` and explicit capture
normalization metadata. Existing prepared bundles must be regenerated together.
The synthetic generator and recording format are unchanged.

**92 Python tests passed, one native ONNX Runtime check skipped** (93 total), and
**61 JavaScript tests passed**. Tests include actual fresh five-frame training,
strict checkpoint round trips/rejections, both grid shapes, graph-input clipping,
unchanged overshooting recordings and matching first-layer input bounds.
An untrained normalized five-frame export passed actual ONNX Runtime Web WASM
checks in Node at G64/G128: maximum absolute errors against PyTorch were
**1.34e-6/1.46e-6**. Raw overshoot fixtures and their explicitly clipped copies
produced exactly identical WASM logits, including 74,979/297,887 clipped input
values. Both model bundles and local runtime assets were prepared successfully.

These checks establish implementation and numerical agreement, not improved
recovery. No full GPU training or physical-device accuracy run was performed.
No site was deployed. The next run starts fresh with the existing real and
matched synthetic datasets; wide-burst5 can be archived independently.

## Harder paired synthesis and mild real augmentation — 2026-11-02

The submitted normalized five-frame run selected step 31,000: **19/144 exact
validation groups (13.19%)**, **32.43% cell error**, CE **1.3507**, and zero wrong
acceptances. Classical recovery on identical observations was 0/144. All learned
successes were PC→phone (19/96); phone→PC remained 0/48 with 65.60% cell error.
Final step 32,000 recovered 18/144. At the selected step synthetic training CE
was .053 versus .510 for real inputs. These are user-run measurements on fixed
random groups, not a controlled comparison against earlier wide architectures.

The current generator adds a coherently moving elliptical glare core and broad
tinted veiling halo, plus optional per-burst severity ranges. Independent RNG
streams keep glare and severity selection separate from other nuisance draws.
The recommended corpus has 32 independent bursts × 12 frames per recorded
training target, with severity .45–.9 and stronger photometric limits. Targets,
labels/masks, phase identities and split ownership are unchanged.

Training now applies mild real-only noise (.008 RGB01 standard deviation), tint
(.06 stops), and exposure (.08 stops) after random burst assembly. Gains are
shared within a stack; noise is independent across exposures/pixels. A dedicated
RNG leaves sampling behavior unchanged. Original files and synthetic inputs
are not mutated; validation receives no augmentation. The existing model input
clip remains the final common [-1,1] boundary. Settings/history/checkpoints
record the new controls, with all-zero strengths disabling the augmentation.

**101 Python tests passed, one native ONNX Runtime check skipped** (102 total),
and **61 JavaScript tests passed**. Checks include deterministic glare motion,
local contrast loss/saturation, independent severity draws, same-target pairing,
unchanged label/mask/expected-byte hashes and split ownership, real-only
augmentation, non-mutation, and actual normalized five-frame CLI continuation.
The continuation reproduced the parent checkpoint's unaugmented validation
baseline, retained its profile/frame count, and used a fresh optimizer while
preserving the original checkpoint and data files. Whitespace checks passed.

The recommended next run initializes `normalized-burst5/best.pt` at `3e-5` for
32,000 additional updates. No full GPU training, new recovery gain, physical
device throughput measurement or site deployment is claimed for this change.

## Effective capture resolution simulation — 2026-11-02

The user reported little improvement from the glare refinement and proposed
simulating the reduced pixel footprint of the phone screen in a PC webcam.
That geometry is phone→PC in the dataset's direction labels. No new numerical
evaluation report was supplied for this change.

Synthetic crops now optionally pass through area downsampling, a small Gaussian
blur at the reduced resolution, and bilinear enlargement back to the model's
input dimensions. Defaults select 75% of bursts and draw one integer native side
from .25–1 times the crop side: 64–256 for G64, or 128–512 for G128. Blur sigma
is sampled between .175 and .35 native pixels. Both settings stay fixed within
a burst, while existing corruption trajectories still evolve across captures.
Resolution uses an independent RNG and is not scaled by photometric severity;
zero severity retains exact clean identity. Configuration and actual per-burst
settings are recorded in the current recipe metadata. The stored dataset format,
real recordings, pairing targets, model dimensions and evaluation are unchanged.

**106 Python tests passed, one native ONNX Runtime check skipped** (107 total).
Checks cover both minimum capture sizes, blur before enlargement, preserved
constant colors, lost unresolvable detail, deterministic isolated randomness,
zero/disabled controls, and unchanged label/mask/expected-byte hashes and splits.
An actual CLI generation produced **40 synthetic captures**, each forced through
64×64 with blur before returning to 256×256; canonical baseline preparation and
recorded per-burst metadata succeeded. Whitespace checks passed.

The next recipe keeps the earlier paired targets, seed, photometric controls and
five-frame architecture, adding the capture-resolution mixture. It initializes
the original `normalized-burst5/best.pt` at `3e-5` for 32,000 updates in a new run
directory. One native pixel per cell is a deliberately difficult endpoint, not
a reliable-recovery guarantee. No new training gain or device result is claimed.

## Direction-specific generalization and the tubular profile — 2026-11-02

The available new history is a fresh normalized-compact five-frame run through
20,000 updates at `1e-4`. A second same-named upload was unavailable, so the
reported initialized attempt was not analyzed numerically. The history records
training settings but not the synthetic generator recipe; the resolution corpus
is identified by the user's description, not independently verified in this log.

At 20k the new run has validation CE **1.0514**, cell error **32.759%** and **8/144
exact groups**, compared with **1.1657**, **32.995%** and **8/144** for the original
normalized run at the same update count. Overall CE bottoms at **.96655 at 14k**;
cell error bottoms at **32.416% at 19k**. Phone→PC diverges earlier: from 7k to
20k, its validation CE rises **1.6223→2.1400** and cell error **60.10%→63.42%**,
while real training CE falls **1.4106→.8879**. It recovers no groups. PC→phone
continues improving and recovers 8/96 at 20k. This supports direction-specific
generalization failure; aggregate progress partly hides it. These logs do not
establish whether data diversity, information loss or residual domain differences
are the primary cause.

The optional **U-tube** profile implements the proposed constant-width
architecture: both inlet outputs and all hidden backbone outputs are 32 channels,
with ChannelLayerNorm/ReLU after all twelve hidden convolutions. Skip
concatenations have 64 channels before reduction to 32; the final head maps
32→16 logits. There is no direct RGB branch. The existing input clip, spatial
strides, learned-feature skips, frame-count range and default profile remain.

The five-frame model has **125,808 parameters**, versus **125,600** for normalized
compact. Its formula is **121488 + 864T**, reaching 176,784 at 64 frames. This
redistributes capacity toward finer features; it is not parameter-count
regularization. Convolution MACs at G64/T5 are **351,272,960 versus 168,558,592**,
about **2.08×**, excluding normalization/interpolation and memory work. These are
operation counts, not phone latency measurements. The inlet still downsamples
256→128→64 and introduces no additional full-resolution skip.

The new profile is supported by strict checkpoint loading, CLI training/export
and browser manifest validation. An untrained five-frame export passed ONNX
Runtime Web WASM checks at G64 and G128, with maximum absolute errors against
PyTorch **1.51e-6 and 1.37e-6**. Overshoot and explicitly clipped fixtures produced
exactly identical logits. No full GPU training, real recovery gain or site
deployment is claimed. The next run trains tube from scratch on the existing
datasets; normalized-compact weights cannot initialize the different internal
shapes.

**107 Python tests passed, one native ONNX Runtime check skipped** (108 total),
and **61 JavaScript tests passed**. Coverage includes both grid shapes, channel
widths/normalization, input clipping, gradients, strict profile identity, frame
expansion through 64 inputs, ONNX fixtures and exact browser parameter-count
validation. An actual fresh five-frame tube CLI training run updated weights,
saved a correctly identified checkpoint and evaluated through canonical recovery.
Model preparation and whitespace checks passed.

## Full-resolution tube and output reduction — 2026-11-02

The tube experiment now follows the requested full-resolution U-Net layout.
Both inlet convolutions use stride 1. At G64 the learned feature path is
**256→128→64→128→256**, with a full-resolution normalized skip. Only after that
decoder do two new normalized stride-2 outlet convolutions reduce
**256→128→64**, followed by the 16-logit head. All fourteen hidden convolutions
output 32 channels. G128 follows the corresponding 512-pixel input path. Other
profiles and the stored input/output contracts remain unchanged.

The current tube has **144,432 parameters at five frames**, with formula
**140112 + 864T** (195,408 at 64 frames). At G64/T5, convolution MACs are
**4,928,307,200**, about **14.03×** the earlier inlet-reduced tube and **29.24×**
normalized-compact. Activation memory also increases. These operation counts
are not measured GPU/phone runtimes. The recommended fresh run uses batch 2
and accumulation 8, retaining effective batch 16, and reuses the existing
datasets in a new `normalized-tube5-fullres` output directory.

This replaces the previous tube definition under the same experimental profile.
Checkpoint validation requires the learned outlet and rejects the earlier tube
state, because convolution strides alone are absent from a state dictionary.
Browser manifests require the new exact parameter count. No compatibility
reader or weight migration was introduced. Spatial-hook tests verify both
inlet layers, encoder/decoder stages, outlet layers and final logits at both
supported grid sizes.

**107 Python tests passed, one native ONNX Runtime check skipped** (108 total),
and **61 JavaScript tests passed**. Tests include actual fresh five-frame CLI
training, gradients, unchanged source data, profile/frame-count round trips,
old tube checkpoint rejection, ONNX graph/fixture checks and browser rejection
of the earlier tube parameter count. Both grid exports and model preparation
completed. No full GPU training, new real recovery measurement or deployment
was performed.

The untrained full-resolution five-frame export also passed actual ONNX Runtime
Web WASM checks at G64/G128. Maximum absolute errors against PyTorch were
**1.61e-6/1.55e-6**; original overshooting fixtures and explicitly clipped copies
produced exactly identical logits. These establish numerical agreement, not
recognition accuracy or phone performance.

## Full-resolution result, U-tube-2x, and RGB motion offsets — 2026-11-03

The submitted full-resolution tube history selects step **29,000**: validation
CE **.46612**, cell error **17.3069%**, and **68/144 exact groups (47.22%)**.
PC→phone recovers **68/96 (70.83%)**, with 2.4919% cell error; phone→PC remains
**0/48**, with 46.9368% cell error. Final step 32k also recovers 68/144, with CE
.47093 and 17.3421% cell error. Classical recovery remains zero. This is a
substantial measured improvement, but the history also records batch 16 and
accumulation 2 (effective batch **32**, twice the earlier recommendation), so it
is not a controlled architecture-only comparison. The user reports stopping a
low-rate initialized refinement; no history for that separate run was supplied.

The full-resolution **U-tube is now the default**, including the model
constructor, training and export. Its state keys and seeded parameter tensors
are unchanged, retaining the working trained model. Compact, wide and
normalized-compact implementations and active recipes have been removed. Only
U-tube and **U-tube-2x** are supported by Python and browser manifests.

U-tube-2x keeps the 32-channel, 4G→2G→G→2G→4G U and output reduction. E0, E1,
D1 and D0 each contain **four 3×3 convolutions in two residual pairs**; the
bottleneck contains **eight in four residual pairs**. Each resolution block has
one channel LayerNorm after all its pairs. The inlet retains two full-resolution
convolutions with one final normalization; each of the two outlet reduction
stages has one normalization. This gives 28 main hidden 3×3 convolutions, four
1×1 shortcut projections, and eight normalizations, followed by the logit head.
There is no direct RGB shortcut. Five-frame parameter count is **279,792**
(`275472 + 864T`, 330,768 at T=64); G64/T5 convolution work is **8,363,442,176
MACs**, about 1.697× the default tube. These counts are not device timings.

The design rule is that the U's outer resolution matches the rectified crop and
its bottleneck matches the target grid: **256/64 for the current G64 task**.
Existing G128 shape checks use **512/128**; they do not establish cross-size
recognition. Any future G128 recognition experiment should train separate
weights. Conditional-depth multi-size routing remains deferred. Random symbols
motivate preserving cell-level evidence, but do not prove that every coarse
feature must overfit: illumination, references and optical artifacts can have
larger-scale correlations.

Synthetic recipe v7 adds **RGB-offset motion blur**. In selected motion-blurred
bursts, each color integrates a slightly shifted exposure window along the same
linear/rotational/zoom trajectory. Relative displacements are centered across
RGB and bounded in dense-crop pixels. Defaults are probability **.6** conditional
on motion and maximum offset **.5 pixels** before severity scaling. No-motion
and zero-strength cases retain common-blur behavior. A dedicated RNG preserves
the other corruption draws; labels, masks, target identities and recording
formats remain unchanged. This approximates color-fringed motion without
asserting that all cameras acquire colors at different times.

The next experiment generates fresh paired synthetics from the existing clean
training targets and trains U-tube-2x from scratch at `1e-4`. Recommended batch
8 with accumulation 4 preserves the user's effective batch 32 while allowing
for the deeper network's additional activations. Tube weights are not silently
expanded into U-tube-2x. No new full GPU run or real recovery result is claimed.

**110 Python tests passed, one native ONNX Runtime check skipped** (111 total),
and **61 JavaScript tests passed**. Coverage includes the new default, removed
profile rejection, exact retained tube states, both current checkpoint/export
identities, full-resolution/target-grid spatial hooks, residual identity and
projection gradients, one normalization per residual block, frame expansion,
and actual fresh U-tube-2x training. Synthetic checks verify color-centroid
separation, centered bounded offsets, no-motion/disabled identity, independent
RNG streams, target hashes and split ownership. An actual CLI run generated
**40 captures** with motion and RGB offsets forced on; canonical baselines and
recipe metadata were produced successfully. Model preparation and whitespace
checks passed. No site was deployed.

The untrained five-frame U-tube-2x export also passed ONNX Runtime Web's WASM
provider in Node for G64 and G128, including all ten measured parity runs and
exact clipping parity. Maximum absolute differences from PyTorch were
**3.13e-6** and **2.62e-6**, respectively. This checks exported computation,
not trained recognition or phone performance. Single-thread Node median
inference times were approximately **763 ms** (G64) and **3,171 ms** (G128).

## 2026-11-03 — recording names and repeatable dataset rebuild

Capture downloads share the first logical matrix's payload CRC32 as an
eight-digit filename prefix. Palette phases retain the same prefix; header
CRCs are not used. The new filename-only migration scans existing device/split
folders recursively and replaces only the leading manual/checksum prefix.
Original filename suffixes, timestamps inside names, JSON bytes and folder
placement are preserved. Diagnostics-only bundles use `unpaired`; full header
and payload checks, rather than filename prefixes, remain the join authority.

The rebuild CLI reads `local-data/{phone,pc}/{train,val}`. It validates current
recordings, tensors, stable-header evidence, complete sender/receiver matching,
device direction, duplicates and whole-session/payload split isolation before
removing only six fixed generated directories and `session-groups.json`.
Python dependency/configuration failures also precede removal. Raw files are
read one bundle at a time; large capture tensors are not retained across files.
It rebuilds real burst identities, canonical paired targets, direction groups
and separate training/validation synthetics. Defaults use 32/4 independent
bursts per target, 12 frames, seeds 41/42 and the current harder RGB-offset
recipe. Existing differently named datasets and checkpoints remain untouched.
Rebuild must run after any training/evaluation job using those generated paths.

**76 affected JavaScript tests passed, with no skips**, including the actual
Python synthetic-generation integration. The small fixture produced two real
captures, one clean target and two synthetic captures per split; target hashes,
burst identities, direction groups and split membership were checked. Paths
containing commas exercised array-based joins/target reconstruction. Raw-file
hashes, an unrelated dataset and a checkpoint sentinel were preserved, while
the previous managed dataset was removed. Negative tests cover unmatched and
duplicate exports, wrong-side captures, malformed tensors, split leakage,
symlink paths, unavailable Python and insufficient configured output budget.
Filename checks cover exact suffix preservation, JSON/date independence,
phase agreement, overwrite refusal, dry runs and idempotent reruns. Whitespace
checks passed. No actual user recordings were migrated here and no site was
deployed or reactivated.

## 2026-11-03 — learned global anchor, window attention and variable bursts

Added `U-tube-2x-attn` as a combined G64 experiment. It retains U-tube-2x's
full-resolution 32-channel U and residual depth, adds a trainable
`[1,32,64,64]` positional tensor immediately after the inlet (bilinearly expanded
to 256×256), and adds two four-head attention blocks at the 64×64 bottleneck.
Each uses 8×8 windows, a 32→64→32 feed-forward block and residual connections;
the second shifts windows by four cells with an explicit nonwrapping boundary
mask. The learned anchor contributes 131,072 parameters; the complete model
has **432,848**. Existing tube and U-tube-2x seeded state keys and values were
checked to remain identical. This is a new profile, not a checkpoint conversion.

The fixed input is `[N,32,256,256]`: eight RGB slots followed by eight binary
presence planes. Training chooses 5–8 distinct same-burst/phase observations
uniformly, assigns them to random slots, and pads only after real augmentation.
Real/synthetic pairs share the observation count even after microbatch shuffling.
Absent RGB slots are gated in the model. Stored recording and dataset formats
do not change. Training bursts need at least eight members; current 12-frame
synthetic bursts are reusable.

Validation uses nested 5/6/7/8 subsets of the same fixed shuffled eight-capture
blocks. Classical decoding receives only the actual observation IDs for each
preset. Reports distinguish unique candidates from repeated observation cases,
give per-count/per-direction metrics, and select checkpoints by equal-weight
frame-count and real-direction scores. `phoneToPCGoal` records the requested
50% exact-recovery target for each count; it is not evidence that the target
has been achieved. The reported ongoing U-tube-2x result of 72/144 was provided
by the user, not reproduced by this implementation test.

Corrector Lab now accepts the new packed layout, offers 5/6/7/8 camera exposures,
and reports actual observations separately from eight-slot capacity. The
attention profile exports G64 only because its learned global anchor is tied
to this geometry. Fixed profiles retain their existing size support. Current
exports include explicit input-layout/frame-policy fields and must be prepared
with the matching current browser runtime.

The final untrained ONNX graph is **2,836,028 bytes**, SHA-256
`ea243a21db0dca92b7e7af4d46acb8b7d5680a62db0c2fdcb188ffec2de57fa6`.
ONNX graph checking passed. ONNX Runtime Web 1.23.2 in single-thread Node WASM
matched the PyTorch five-observation fixture in all ten measured runs, with
maximum absolute error **2.6673e-6**. RGB clipping and changing absent-slot RGB
were exactly invariant. Additional 5/6/7/8-count runtime checks produced finite
logits and preserved missing-slot invariance. Median inference was about
**1,210 ms** in this Node environment; this is not a phone/WebGPU measurement.
Preparation of local assets passed; no site was deployed. Native Python ONNX
Runtime parity remains explicitly skipped in the restricted environment.

The full affected suites completed with **126 Python tests passed and one
native-runtime skip** (127 total), plus **80 JavaScript tests passed** with no
skips. Coverage includes an actual U-tube-2x-attn optimizer update and strict
checkpoint reload, learned-anchor/attention gradients, shifted-window boundary
isolation, paired observation counts across shuffled accumulated microbatches,
padding after augmentation, deterministic nested evaluation and equal actual-K
classical evidence. Existing fixed-frame training/export and the recording
rebuild checks also passed. These CPU/synthetic fixtures verify implementation;
no new trained device-recovery result or achieved 50% phone→PC rate is claimed.

## 2026-11-03 — phase-grouped attention with four-to-eight captures

U-tube-2x-attn now assigns its eight slots to four known palette-phase pairs.
Every input has one or two captures of each phase, so four observations suffice
for inference. Training draws K uniformly from 4–8, canonicalizes data labels
to phase zero, and applies exact XOR phase-group permutations plus joint
horizontal/vertical flips (default probability .25 per axis). Reference colors
are not XOR-transformed. Augmentation is training-only; evaluation and CRC/RS
recovery use original coordinates and canonical logical scores. The unchanged
full-resolution architecture has **432,848 parameters**.

Synthetic generation now supports coherent four-phase trajectories, with shared
slow optical conditions and evolving motion, moiré, exposure and sensor noise.
The default rebuild validates complete phase groups before replacing derived
datasets. Recorded JSON files retain their existing format and contents.
Evaluation reports nested K=4/5/6/7/8 results on identical logical candidates,
with the classical comparator receiving the same actual observations and each
capture's own palette mapping. Browser collection retains earlier phases across
transient detection failures, bounded by eight captures, 60-second evidence age
and 512 collection attempts; a different logical header or run resets the group.

Validation covered **134 Python tests**: after correcting two stale test-fixture
expectations and rerunning their affected modules, **133 passed**, with the
existing native ONNX Runtime check explicitly skipped in this environment.
All **88 affected JavaScript tests passed**, without skips. Checks include an
actual CPU attention optimizer update, strict checkpoint reload, all five
evaluation presets, physical-label equivalence under phase permutations,
invertible joint flips, rejection of mixed/incomplete groups, coherent synthetic
generation, a small full rebuild, and failure-before-deletion behavior.

ONNX graph checking and local asset preparation passed. The untrained graph is
**2,836,028 bytes**. ONNX Runtime Web 1.23.2 in single-thread Node WASM matched
the PyTorch four-observation fixture across ten measured runs, with maximum
absolute error **2.0415e-6**. Input clipping and absent-slot perturbations were
exactly invariant. All five capture-count configurations returned finite logits
and preserved missing-slot invariance. These are runtime checks, not trained
phone recovery or browser/WebGPU performance measurements. No site was deployed
or reactivated; the real-device recovery target remains to be tested.

## 2026-11-04 — neural-corrector naming, compact metrics and serving guide

Renamed the learned component to `neural-corrector`, its Python class to
`NeuralCorrector`, and the three profiles to `U-tube`, `U-tube-2x` and
`U-tube-2x-attn`. Source directories, scripts, browser lab routes, npm commands
and documentation links use the new names. Mathematical operations, weight
tensor keys and input/output semantics are unchanged. Stored recording,
dataset, session-group and checkpoint format identifiers remain stable, so
existing data needs no JSON conversion. There are no old-name model aliases.
The serving guide documents a manual metadata-only checkpoint relabel; loading
that result was checked to preserve every model tensor exactly.

Evaluation now returns compact reports by default. A shared idempotent
projection retains group/source/capture-count aggregates and exact selection
scores, while omitting session rows, capture identities, per-candidate recovery
and repeated confusion/calibration arrays. `--detailed-report` writes optional
diagnostics separately. A stress fixture shrank from about 25 MB to 7.2 KB,
without changing aggregate metrics or selection values. Training history uses
these summaries. The live progress bar contains only current training CE/CER,
ordered source/direction statistics and learning rate. Validation prints
separate consolidated Train/Val lines, ordered K summaries, raw recovery and
the balanced selection score. The phone-to-PC goal line is opt-in via
`--show-goal`.

The user-reported completed run reached 529/560 observation cases (94.46%) and
93.54% direction-balanced selection at step 32,000. These are 112 unique
candidate groups evaluated at five capture counts. Phone-to-PC recovered
34/48, 37/48, 42/48, 48/48 and 48/48 for K=4/5/6/7/8; PC-to-phone recovered
64/64 at every K. Overall cell error was 0.64%. These figures come from the
submitted training log, not a new run performed here, and do not yet measure
live browser goodput or generalization to new recording sessions.

The affected suite passed **139 Python tests with one native-runtime skip**
(140 total), and **88 JavaScript tests without skips**. This includes actual
optimizer/checkpoint/evaluation flows, compact report size and metric
preservation, stable console ordering, all phase presets, recording rebuilds,
renamed imports and strict model contracts. JavaScript syntax and whitespace
checks passed. The renamed untrained `U-tube-2x-attn` export remains 432,848
parameters and 2,836,028 ONNX bytes. Static asset preparation and ONNX Runtime
Web/WASM parity passed; maximum absolute error against the PyTorch fixture was
1.4604e-6, and K=4–8 input/missing-slot checks passed. Native Python ONNX Runtime
still cannot initialize in this restricted environment and is explicitly
skipped. No trained user checkpoint was available for browser testing here.

Architecture documentation now has separate diagrams and sections for every
profile, the common full-resolution design, input variants and all four mask
mechanisms. The serving guide describes local `.pt` to ONNX export, prepared
same-origin static assets, trusted HTTPS, backend selection and the remaining
live corrected-score/CRC integration. No site was deployed or reactivated.

## 2026-11-04 — Neural-ZKBoo color authentication integration

Implemented the separate `neural-zkboo.html` interface with the shared
Neural-ZKBoo authentication session, a stop-and-wait OLM1 channel, four-phase
trained-corrector data reception, and small classical ACK matrices. The default
is 32 repetitions, 95 ms phase display and 100 ms requested camera sampling.
Public identities, pairing confirmation, a fresh challenge, proof verification,
result delivery and the final ACK are exercised as one authentication attempt.

Checks completed:

- All **36 new demo tests** passed: bounded progressive phase collection,
  canonical logical-score/CRC recovery, rendered ACKs, packet padding, turn
  scheduling, cancellation, timeout and complete real-proof exchanges.
- The end-to-end tests use the actual 32-repetition proof engine and rendered
  DATA/ACK matrices. Deliberately lost final ACKs are retried without duplicate
  application delivery. Tampered proofs and replayed messages are rejected.
  Another case uses the actual `TurnChannel` with explicit manual handoffs while
  the next application response is queued behind its sticky final ACK.
- **303 color-app regression tests** passed; two unrelated dataset rebuild
  checks skipped because their Python dependency probe was unavailable. Ten
  additional content tests passed. The two JPEG-fixture cases were not run: this
  workspace could not materialize the existing binary JPEG through the text-only
  repository reader. The repository's JPEG remains unchanged.
- **223 QR regression tests** passed after moving authentication to canonical
  shared source. The public circuit and compact proof format are unchanged.
- The documented metadata-only checkpoint relabel was tested on a matching
  fixture checkpoint; weights remain bitwise identical and repeating the command
  safely reports that the profile is already current.
- JavaScript syntax, local DOM/asset bindings and whitespace checks passed.
  A browser preview could not start because this environment's managed preview
  service was unavailable; no visual/device-camera validation is claimed.

No trained user checkpoint or physical cameras were available here. The clean
render tests use the classical image decoder as a test harness; separate
corrector tests inject ideal logits only at the score/codec boundary. Neither
result establishes live trained-model recovery. The production data path
requires a trained `U-tube-2x-attn` bundle and never accepts sender truth or ideal
logits. Trained device performance, capture timing and a full physical exchange
remain the next check in [the demo guide](NEURAL_ZKBOO_DEMO.md).

Model assets are fetched during explicit initialization and then retained in the
worker. Repeated transport scans do not fetch the manifest or model. No hosted
site was deployed or reactivated.
