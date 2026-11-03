# Decoder Lab — model parity and device checks

Decoder Lab tests the small optical model, reproducible exports, and worker
inference. Export without a checkpoint uses **untrained** weights; M2 also
supports a selected trained checkpoint. The status appears after loading the
prepared model. See [capture and training](OPTICAL_DECODER_TRAINING.md) to collect
data, train, evaluate recovery, and prepare checkpoint exports.
See [the architecture](OPTICAL_DECODER.md) and [the POC roadmap](ROADMAP.md).

## Prepare once

From the repository root, with Node 22+ and a Python virtual environment:

```sh
python -m pip install -r experiments/optical-decoder/requirements.txt
python experiments/optical-decoder/export.py --model attention-tube --frames 8
npm ci --prefix experiments/optical-decoder/browser-runtime
npm run prepare:apps
npm run prepare:decoder
python apps/color-optical-bench/serve.py --port 8000
```

Open `http://localhost:8000/decoder.html` on the computer. For the phone, serve
the prepared color app on **trusted HTTPS**, using the existing
[local serving instructions](DEVELOPMENT.md#local-serving). No site is deployed
by these commands. Export runs on CPU; the 4090 is not needed for M1.

The optional runtime is ONNX Runtime Web **1.23.2**, installed using its lockfile.
Preparation checks model/fixture hashes and copies runtime JS/WASM, notices,
models, and fixtures into ignored app-local directories. Inference uses those
local assets with no CDN, model service, or uploads. The normal transport and
crypto setup remains independent of these additional dependencies.

The exporter normally checks native Python ONNX Runtime against PyTorch. If
that native runtime cannot initialize, `--skip-ort-check` creates the fixtures
with a visible `not_run` validation status. It does not claim that parity passed.
The browser's full-output parity test must still pass. Current development
validation and environment limitations are recorded in [VALIDATION.md](VALIDATION.md).

## What to test on the phone and PC

Keep the page visible, and leave **Measured runs** at 10 initially. Each
benchmark performs two warm-ups, then compares every measured output against
the Python fixture. The browser requires finite logits and elementwise
`abs(actual - expected) <= 1e-4 + 1e-3 * abs(expected)`.

1. With the camera off, select **WASM** and run the fixture at **64**.
   For fixed-frame tube/longtube exports, also test **128**. Attention-tube
   supports G64 only; requesting 128 produces an explicit unsupported-size error.
2. Select **Auto** and repeat. The report identifies the actual backend and any
   fallback. If WebGPU is available, also select it explicitly to make a backend
   failure visible rather than falling back.
3. Start the camera at **720p** and repeat the supported sizes with the working backend.
   Keep the camera preview running during the numerical test. Compare 1080p only
   if relevant to the device's actual receiving configuration.
4. On the other device, display the **same 64×64/16-color logical matrix in all
   four permuted phases**, with grid-8 references. Cycle the phases while the
   attention-tube probe runs; a static single phase cannot fill its input.
   Select size 64, align the entire marker, and press **Test matrix burst**.
   Use **Camera exposures** to test **4, 5, 6, 7 and 8** separately; **Model
   default** collects eight. Four means one capture of every phase. Use a
   fixed-frame tube/longtube export separately for the 4-color control and G128.
   Reverse devices so both PC→phone and phone→PC captures are exercised.
5. Try **Cancel**, then run again. Stop the camera when finished. Hidden pages
   stop camera capture and terminate pending worker work.
6. Add device/browser/power-mode notes and **Download JSON report** from both
   endpoints, including failures. Save before leaving the page; reports are
   kept only in memory, capped at 50 tests.

The camera probe validates **Camera exposures** against the model's declared
`frameCounts`. Fixed-frame tube/longtube exports accept only their T and collect
distinct exposures of one complete header/phase. Attention-tube accepts K=4–8,
requires G64/16 colors and four permuted phases, and collects one or two exposures
per phase. A phase change preserves a matching logical matrix; changing any
other identity field clears the partial group. Geometry failure, cancel, restart
or a new run discards it. `requestVideoFrameCallback` media timestamps and
presented-frame counters verify new exposures; playback time alone is not enough.

Attention RGB slots have fixed meanings **P0a, P0b, P1a, P1b, P2a, P2b, P3a,
P3b**, followed by eight spatially constant binary presence planes. Each phase
has at most two retained exposures, and completion requires at least one per
phase. Remaining RGB slots and their presence planes are zero. The worker
validates this layout and phase coverage; the model gates absent RGB internally.
Repeated captures of a held phase cannot substitute for a missing phase. The
report includes per-phase counts. Fixed-frame models retain RGB-only stacks.

Collection targets 100 ms start-to-start sampling and is bounded to 512 sample
attempts for phase-grouped input, with at most eight retained crops. Fixed-frame
probes retain their smaller count-based retry budget. Slow processing lowers the
actual scan rate. If a phase is missed, retry while cycling the same matrix;
never duplicate captures to fill slots. Four-phase acquisition, settling and
inference add latency: four observations do not imply a 400 ms end-to-end result.
K=4 saves acquisition time but still executes the fixed eight-slot graph.

The camera probe revalidates geometry/header and samples each exposure's
palette, even if payload CRC would fail. It reports preprocessing and inference
cost; numerical checks alone cannot establish symbol accuracy. A wrong grid selection,
unsupported palette, or invalid header produces a failure record. The lab does
not feed these outputs into live recovery or report decoded content.

Report fields include model and fixture hashes, package versions, requested and
effective backend, fallback reasons, full-output parity, warm p50/p95, load and
initialization time, camera resolution, `framesPerInput` capacity, `inputCaptures`/`requiredCaptures`
actual count, `inputLayout`, allowed `frameCounts`, per-phase counts, exposure timestamps and UI
animation-frame gaps. Known
buffer sizes and optional main-page JS heap readings are **partial memory
accounting**, not total process/GPU/worker memory. Use browser/OS diagnostics if
available to investigate memory pressure. No camera frames, payload bytes, or
keys are exported.

The initial targets remain warm p95 ≤100 ms at G64 and ≤200 ms at G128, with a
responsive UI and bounded memory. Treat these as engineering targets. Device
results determine whether to keep this graph, halve its widths, or start with
G64. Training begins after we establish a usable receiving profile on each side.

## Code boundaries

| Component | Implementation |
| --- | --- |
| Full-resolution tube, longtube and anchored attention-tube ONNX graphs | [`experiments/optical-decoder/`](../experiments/optical-decoder/README.md) |
| Validated dense physical crop, including references | `vision.js` → `extractDensePayload()` |
| Per-exposure correction and `[1,3,4G,4G]` crop | `learned-input.js` → `extractLearnedInput()` |
| Fixed RGB stacks or masked eight-slot `[1,32,256,256]` input assembly | `js/decoder-lab/burst.js` → `CameraBurst` |
| Palette mask, physical-output phase inversion or canonical-output pass-through, reference exclusion | `learned-input.js` → `logitsToLogicalScores()` |
| Bounded inference, asset hashing, backend selection | `js/decoder-lab/worker.js` and `runtime.js` |
| Worker lifecycle and independent lab UI | `js/decoder-lab/client.js`, `lab.js`, `decoder.html` |
| Local model/runtime preparation | [`scripts/prepare-decoder.mjs`](../scripts/prepare-decoder.mjs) |

The score adapter returns logical scores in the existing 0–255 convention.
Tests exercise the downstream RS/CRC boundary, but M1 does not add a learned
evidence pool to `TemporalDecoder`. That connection waits for a trained model.
The original proof engine and OBM5/OLM1 wire formats are unchanged.

## Automated checks

```sh
npm run test:decoder
python -m unittest discover -s experiments/optical-decoder/tests -v
npm run test:decoder:wasm
```

Current exports declare the exact input layout and counts. Fixed-frame profiles
use `inputLayout: "stacked-rgb"`, `frameCounts: [T]`, capacity/minimum T and 3T
channels and physical output labels. Attention-tube uses
`inputLayout: "phase-paired-rgb-presence-planes"`, capacity 8, minimum 4,
`frameCounts: [4,5,6,7,8]`, `phaseMasks: [0,15,7,8]`,
`outputLabelSpace: "logical"`, 32 channels and one G64 case. Its positional
anchor is tied to G64; the lab never reshapes it into a G128 decoder. The
canonical logical output must not be phase-inverted a second time.
Regenerate local bundles after updating the current contract.

A batch-one attention input takes 8 MiB of float32 storage before runtime
buffers. A fixed-five-frame G128 input takes 15 MiB (192 MiB at T=64). The default
full-resolution tube has 144,432 parameters at T=5; longtube has 279,792;
attention-tube has **432,848**, including 131,072 learned positional parameters.
The attention model keeps the full-resolution U and adds two local attention
blocks at the 64-cell bottleneck. The two fixed-frame profiles retain 64/128
structural exports, which do not establish cross-size recognition.

The last command requires the exported/prepared assets and runs the same ONNX
Runtime Web bundle's WASM backend in **Node** against the Python fixtures. It is
a reproducible runtime check, not a browser or phone benchmark. Actual browser
WebGPU support, camera contention, and physical crop quality still need the
device checklist above.


Current exports require `networkInputClip: [-1,1]` in addition to capture
normalization `c/127.5 - 1` with storage bounds [-1.5,1.5]. Camera calibration and
recordings preserve those wider values; the first ONNX operation clips every
RGB input before learned convolutions. Presence planes remain binary and are
validated separately; missing RGB slots are masked inside attention-tube. Capture adds no extra [-1,1] clamp;
its existing wider storage bounds are preserved.
Regenerate and prepare the export together after updating; previous bundles
without the explicit contract are rejected. The Node WASM check also compares an
overshoot fixture against its explicitly clipped copy and requires identical
outputs. Attention-tube also checks that arbitrary values in absent RGB slots
cannot affect the output. Supported profiles are `normalized-tube`, `longtube`
and `attention-tube`; removed architecture profiles are rejected. The default
profile remains `normalized-tube` when no model is requested.
