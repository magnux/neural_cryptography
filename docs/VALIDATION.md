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
