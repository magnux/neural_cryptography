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
python experiments/optical-decoder/export.py
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

1. With the camera off, select **WASM** and run the fixture at **64**, then **128**.
2. Select **Auto** and repeat. The report identifies the actual backend and any
   fallback. If WebGPU is available, also select it explicitly to make a backend
   failure visible rather than falling back.
3. Start the camera at **720p** and repeat the two sizes with the working backend.
   Keep the camera preview running during the numerical test. Compare 1080p only
   if relevant to the device's actual receiving configuration.
4. On the other device, display a **64×64/16-color** matrix in Optical Bench,
   with grid-8 references. Select size 64 in Decoder Lab, align the entire marker,
   and press **Test one matrix crop**. Repeat at 128 and with the 4-color control.
   Reverse devices so both PC→phone and phone→PC captures are exercised.
5. Try **Cancel**, then run again. Stop the camera when finished. Hidden pages
   stop camera capture and terminate pending worker work.
6. Add device/browser/power-mode notes and **Download JSON report** from both
   endpoints, including failures. Save before leaving the page; reports are
   kept only in memory, capped at 50 tests.

The camera probe revalidates geometry/header and samples the current exposure's
palette, even if payload CRC would fail. It reports preprocessing and inference
cost; numerical checks alone cannot establish symbol accuracy. A wrong grid selection,
unsupported palette, or invalid header produces a failure record. The lab does
not feed these outputs into live recovery or report decoded content.

Report fields include model and fixture hashes, package versions, requested and
effective backend, fallback reasons, full-output parity, warm p50/p95, load and
initialization time, camera resolution, and UI animation-frame gaps. Known
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
| 121,792-parameter U-Net and fixed-shape ONNX graphs | [`experiments/optical-decoder/`](../experiments/optical-decoder/README.md) |
| Validated dense physical crop, including references | `vision.js` → `extractDensePayload()` |
| Current color correction and `[1,3,4G,4G]` float32 tensor | `learned-input.js` → `extractLearnedInput()` |
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

The last command requires the exported/prepared assets and runs the same ONNX
Runtime Web bundle's WASM backend in **Node** against the Python fixtures. It is
a reproducible runtime check, not a browser or phone benchmark. Actual browser
WebGPU support, camera contention, and physical crop quality still need the
device checklist above.
