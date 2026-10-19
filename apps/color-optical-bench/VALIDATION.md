# Validation

This file records validation performed during earlier experiment iterations.
The standalone Python suite and CLI mentioned below were not present in the
recovered sources; their earlier results are not a current reproducible gate.
The retained NumPy core and the runnable browser parity gate are documented in
[`experiments/python-zkboo/`](../../experiments/python-zkboo/README.md).

## Compact proof v2 and per-challenge repetitions · 2026-10-17

The complete JavaScript suite passed **193 tests**, with no failures or skipped
tests, in 4.04 seconds. This includes the existing optical and multipart
transport checks, the new compact proof tests, worker checks, and Crypto Lab
controller/report tests. Controller tests use a DOM harness; rendered-browser
layout and new phone performance measurements remain unverified here.

The standalone Python suite passed **16 tests** in 1.58 seconds, including the
base neural experiment checks, compact proof/policy cases, and a CLI round trip
where omitting `--repetitions` correctly honors a 220-repetition challenge.

Python/JavaScript parity checks reproduced the manifest, five proof fixtures,
and the browser reference fixture. All five proofs matched byte for byte with
the same test entropy. Five Python proofs verified in JavaScript, and Python
correctly handled all 28 cross-verification cases: five valid and 23 rejected.
The independent review also confirmed identical ASTs for the 22 shared Python
functions/classes in the standalone experiment and crypto-only reference.

`node scripts/crypto-size-benchmark.mjs` measured fresh proofs with one key and
challenge requirements of 32, 64, 110, 160, and 219. Every proof verified, and
each was rejected against a higher expected challenge requirement. Results are
in `validation/crypto-compact-v2.json`. These are local engine timings, not
camera/phone throughput or security measurements.

| Proof repetitions | Compact bytes | Previous encoding bytes |
|---:|---:|---:|
| 32 | 17,474 | 24,122 |
| 64 | 34,890 | 48,186 |
| 110 | 59,926 | 82,778 |
| 160 | 87,138 | 120,378 |
| 219 | 119,249 | 164,746 |

The default proof is 27.62% smaller. Circuit dimensions, circuit-round count,
and the three proof lanes are retained. The new protocol domain and wire format
are v2 only; old public contexts and proofs are rejected. Tests cover invalid
trits/padding, every transmitted component, extra/truncated bytes, a structurally
valid 33-to-32 repetition truncation, altered policies and challenge requirements,
fresh randomness, input snapshots, and requests above the enrolled floor.

The UI preserves its key when the requested count changes and invalidates the
old challenge/proof. Counts below the enrolled floor are rejected visibly.
Automatic recency tracking, persistent enrollment, and authenticated sessions
are not implemented. Both hosted test sites remain intentionally empty; the
updated source is available for local testing.

## Step 3.3a · reliable multipart messages (v0.8)

**181 automated tests pass**, including 43 link tests, the existing optical
suite, and browser crypto fixture/worker checks. The link tests cover strict
packets, selective retry, lost completion acknowledgments, conflicting inputs,
absolute deadlines, cancellation races, bounded state, streaming optical
decoding, and the actual simulation worker.

`node scripts/link-benchmark.mjs` records four reproducible channel scenarios
in `validation/link-step33a.json`. The three 165,000-byte cases recovered exact
bytes and a completion acknowledgment: clean (0 retries), default faults
(7 retries), and heavier faults (42 total retries across windows). The blackhole
case reached its configured deadline without delivering content. Virtual optical
times were 16.8, 21.3, and 32.2 minutes respectively; these count display holds
and simulated channel events, not measured camera throughput. Execution time
was under one second per case in this environment and is not a phone benchmark.

The simulator uses the real OBM5 header, FEC, symbol codec, and link state
machines. It does not simulate camera color measurements. Browser visual QA and
two-device camera testing were unavailable here; the physical acceptance tests
in `LINK_TRANSPORT.md` remain necessary. Live Demo currently transfers bytes,
without pairing or authentication.

## Optical decoder baseline (v0.6)

This release uses only OBM5 and performs decoding in browser workers. It adds
progressive 8/16-phase reception, bounded per-cell robust selection, and RS
errors-and-erasures recovery. The optical page is focused on transport. The
separate Crypto Lab and its verification scope are described in `CRYPTO_LAB.md`;
crypto results are recorded separately under `validation/crypto-step*.json`.

## Automated verification status

**88 automated tests passed in 5.55 seconds** in the implementation environment.


Run the current suite with:

```sh
npm test
node scripts/transport-benchmark.mjs
```

The suite covers:

- OBM5 header validation, phase counts through sixteen, advertised hold timing,
  public mapping coverage, capacity, and multi-frame assembly.
- RS errors and erasures at the `2e + s ≤ parity` bound, shortened codewords,
  correct erased bytes, parity-byte damage, and invalid erasure positions.
- Reliability propagation across cell/byte boundaries, bounded erasure trials,
  ordinary-RS-first behavior, and mandatory CRC acceptance.
- Per-cell medoid/nearest-subset selection, normalized score vectors, deterministic
  ties, and recovery after incorrect early observations leave the FIFO pool.
- Progressive fusion before every phase has arrived, equal eligible-phase weight,
  missing-phase isolation, continued refinement, age expiry, and memory reductions.
- Independent live/lab workers, reset behavior, photo decoding, report bounds,
  first-success measurements, and HTML/module bindings.
- Existing byte/text/image/audio, geometry/orientation, reference estimation,
  diagnostic error measurements, and synthetic camera stressors.

## Reproducible synthetic benchmark

The current `scripts/transport-benchmark.mjs` exercises 64×64 and 128×128 at eight
and sixteen phases, with sixteen colors, grid 8, RS32, nine captures per phase,
robust 5-of-7 selection, first attempt after four phases, and erasure-assisted
recovery. Its ordinary mean control sees the same evidence. Captures use 960×760,
perspective/color cast, additive noise σ12, moiré amplitude 44, spacing 16, curved
color-dependent bands, drift 1.1 radians per exposure, and seed 1234. Full results
are in `validation/transport-v06.json`.

| Grid | Phases | Captures | First verified capture | Mean result | Robust result |
|---|---:|---:|---:|---|---|
| 64×64 | 8 | 72/72 headers | 32 | 41/41 raw CRC | 41/41 raw CRC |
| 64×64 | 16 | 144/144 headers | 32 | 113/113 raw CRC | 113/113 raw CRC |
| 128×128 | 8 | 72/72 headers | 32 | 41/41 raw CRC | 41/41 raw CRC |
| 128×128 | 16 | 144/144 headers | 32 | 113/113 raw CRC | 113/113 raw CRC |

Both estimators recover these synthetic cases, with zero raw symbol errors in
every fused candidate. This benchmark therefore does not establish an advantage
for robust selection or sixteen phases. All four runs first verify after capture
32: seven retained samples in each of the first three phases and five in the
fourth. The receiver continues updating through the remaining phases.

At 128×128/eight phases, the single-exposure local model verifies 47 of 72
captures with ordinary RS and ten additional captures with erasures. This shows
the erasure path operating on synthetic optical evidence; the fused candidates
need neither correction method in this case. Device benefit remains unmeasured.

## What device observations establish so far

The user's earlier qualitative device tests found 64×64 with sixteen colors usable
with four palette phases, RS32, and eight-cell reference blocks. They ranked palette
phases as the largest apparent contributor, followed by error recovery, reference
grid, five-point sampling, and camera resolution. Those tests were not controlled
and did not retain JSON reports, so they establish neither comparative effect sizes
nor measured error rates.

Earlier uniform startup calibration became invalid after small phone movements
or autofocus changes. The receiver therefore uses current-exposure references and
rolling logical-symbol evidence rather than a frozen calibration map. This does
not imply that spatial references always improve a given capture.

## Limits of the evidence

Quantitative physical camera comparisons remain outstanding. Browser visual QA and real camera
access are unavailable in the implementation environment. The user subsequently
reported successful reception at 128×128 with sixteen colors; no quantitative
reports for those runs were retained here. Automated module/worker
checks do not replace those tests. Follow `TESTING.md` before concluding that
sixteen phases, robust selection, or erasure ranking improves device performance.

A robust subset can agree on the wrong color when most available observations
share a bias. FIFO replacement permits that consensus to change; it does not make
consensus correct. Ordinary mean remains a separate candidate for this reason.
Additional phases can improve diversity while also adding transmission time.

RS correction limits apply per codeword. Confidence-guided erasures help only when
they identify enough actual errors; erasing correct bytes consumes capacity. CRC
checks remain mandatory, but CRC is neither sender authentication nor proof of
cryptographic security. Large numbers of correlated retry attempts are not
independent statistical trials.

The 64 MiB limit covers retained score evidence, not camera images, intermediate
arrays, or browser overhead. Effective sample/phase limits can be reduced on dense
profiles. The 512×512 format limit is not a demonstrated handheld resolution.
The monochrome header has CRC but no RS protection. Transition guards reduce, but
do not eliminate, mixed rolling-shutter exposures.

Synthetic moiré, blur, and noise are controlled stressors, not a complete physical
screen/camera model. Lab first-verification time measures processing, not optical
link speed. Device comparisons need fixed acquisition budgets and fresh-run
success counts as well as time to complete content.
