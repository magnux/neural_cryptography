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
4. On the other device, display a **64×64/16-color** matrix in Optical Bench,
   with grid-8 references. Select size 64 in Decoder Lab, align the entire marker,
   and press **Test matrix burst**. With attention-tube, use **Camera exposures**
   to test **5, 6, 7 and 8** separately; **Model default** collects eight. Repeat
   with the 4-color control. Only use 128 with a profile that exports that case.
   Reverse devices so both PC→phone and phone→PC captures are exercised.
5. Try **Cancel**, then run again. Stop the camera when finished. Hidden pages
   stop camera capture and terminate pending worker work.
6. Add device/browser/power-mode notes and **Download JSON report** from both
   endpoints, including failures. Save before leaving the page; reports are
   kept only in memory, capped at 50 tests.

The camera probe validates **Camera exposures** against the model's declared
`frameCounts`. Fixed-frame tube/longtube exports allow only their T; attention-tube
allows 5, 6, 7 or 8 with a fixed capacity of eight. **Model default** uses capacity.
It collects K distinct video exposures of the same complete header/phase, verified
by `requestVideoFrameCallback` media timestamps and presented-frame counters,
then rectifies and color-corrects each separately. A changed identity clears the
partial stack; geometry failure, cancel, restart or a new run discards it.
The playback clock alone is not accepted as evidence of a new frame.

For attention-tube, the K collected RGB crops occupy the first K of eight slots;
remaining slots are zero. Eight spatially constant binary presence planes follow
the 24 RGB planes. Training randomizes slot placement, while this deterministic
browser packing makes diagnostics repeatable. Missing slots never count as real
observations. The worker validates the input layout and masks, then the model
gates absent RGB internally. Fixed-frame models retain their RGB-only stack.

Collection targets 100 ms start-to-start sampling, bounds retries in proportion
to the chosen count, and never duplicates exposures to fill slots. Processing
and inference add latency; this is a cadence target rather than an end-to-end
rate. Choosing five observations saves acquisition time but still runs the
fixed eight-slot attention graph.

The camera probe revalidates geometry/header and samples each exposure's
palette, even if payload CRC would fail. It reports preprocessing and inference
cost; numerical checks alone cannot establish symbol accuracy. A wrong grid selection,
unsupported palette, or invalid header produces a failure record. The lab does
not feed these outputs into live recovery or report decoded content.

Report fields include model and fixture hashes, package versions, requested and
effective backend, fallback reasons, full-output parity, warm p50/p95, load and
initialization time, camera resolution, `framesPerInput` capacity, `inputCaptures`/`requiredCaptures`
actual count, `inputLayout`, allowed `frameCounts`, exposure timestamps and UI
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
| Palette mask, phase inversion, reference exclusion | `learned-input.js` → `logitsToLogicalScores()` |
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
channels. Attention-tube uses `inputLayout: "rgb-presence-planes"`, capacity 8,
minimum 5, `frameCounts: [5,6,7,8]`, 32 channels and one G64 case. Its positional
anchor is tied to the G64 model; the lab never reshapes it into a G128 decoder.
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
