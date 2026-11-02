# Current validation — 22 October 2026

## M2 software checks

| Check | Result |
| --- | --- |
| Full color app JavaScript suite | 226 passed |
| Full QR transport/authentication JavaScript suite | 223 passed |
| Optical decoder Python suites | 16 passed; 1 native ONNX Runtime check explicitly skipped |
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
path. Follow [the M2 guide](OPTICAL_DECODER_TRAINING.md) for device work.

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

## Optical decoder M1

The untrained model/export, rectified input and score adapter, and separate
Decoder Lab were implemented and checked on 20 October. No physical device
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
gate; follow [Decoder Lab](DECODER_LAB.md).

New input tests cover actual rendered markers, perspective/rotation, current
palette correction, stale/invalid headers, unsupported profiles, dense-crop
normalization, palette masking and phase inversion, reference exclusion,
finite-output rejection, and downstream CRC acceptance/rejection. Lab tests
cover manifest/asset bounds and hashes, sticky parity failures, timing summaries,
busy/cancel/worker lifecycle, and navigation. They do not establish optical
recovery quality from the untrained network.

Reproduce the M1 checks after the optional setup:

```sh
npm run test:decoder
python -m unittest discover -s experiments/optical-decoder/tests -v
npm run test:decoder:wasm
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

`npm run test:decoder`: **39 tests passed**. Checks include real rendered-header
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

- **42 JavaScript tests passed** (`npm run test:decoder`), including continuous
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

- **42 JavaScript tests passed** (`npm run test:decoder`), including asymmetric
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
  controls, learned-input conversion and decoder-lab checks.
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

The browser Decoder Lab validates 12-channel exports and can collect a bounded
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
busy/new-frame guards. The Decoder Lab targets 100 ms between distinct presented
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

The optional **normalized-tube** profile implements the proposed constant-width
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
