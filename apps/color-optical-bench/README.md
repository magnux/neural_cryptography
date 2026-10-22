# Neural Key · Optical Bench, Crypto Lab and Live Demo

Four browser-local workspaces share top-level navigation:

- **Optical Bench** (`index.html`) tests optical transport of bytes, UTF-8,
  PNG/JPEG, and PCM WAV, with Live link and Image lab workspaces.
- **Crypto Lab** (`crypto.html`) tests the neural-key proof engine in a dedicated
  worker, including round trips, tampering, public-only verification, and device
  performance reports.
- **Live Demo** (`link.html`) tests reliable multipart optical messages, with a
  local fault simulation and a two-device camera transfer. Step 3.3a transfers
  bytes; pairing and authentication are not implemented in this tab yet.
- **Decoder Lab** (`decoder.html`) tests the prepared optical model or checkpoint in a
  dedicated worker. It compares full outputs against Python fixtures and measures
  local inference, with an optional live camera crop. It does not decode payloads.
  Follow the [separate setup and device checklist](../../docs/DECODER_LAB.md).

**Capture Dataset** (`capture.html`), linked from Decoder Lab, adds the M2
sender/receiver data collector. Display a static random matrix, collect bounded
camera bursts, and export the sender manifest and receiver captures separately.
No model installation is needed to collect data. Follow the
[capture and training guide](../../docs/OPTICAL_DECODER_TRAINING.md) for offline
label joining, train/validation/test splits, and checkpoint evaluation.

The reusable crypto engine and worker have one canonical source in
[`shared/browser-crypto/`](../../shared/browser-crypto/). `npm run prepare:app`
copies them into ignored app-local paths for serving. The lab UI and reports
remain in `dist/js/crypto-lab/`; message framing, transfer state, and optical
adaptation live in `dist/js/link/`. Each workspace has independent state.

The hosted test sites are intentionally blank. The source remains available for
local use and reference. Crypto Lab now uses the compact v2 proof format; the
color transport experiments remain available without a new deployment.

The next planned increment is [Neural ZKBoo authentication over color
matrices](../../docs/ROADMAP.md), reusing the QR experiment's authentication flow
and this app's multipart channel. The [optical decoder design and training
plan](../../docs/OPTICAL_DECODER.md) specifies a small U-Net receiver and labeled
capture tooling. M1 supports a G64 profile on the tested phone/PC; M2 collection,
synthetic generation, training, evaluation, and checkpoint export are implemented.
Real-data recovery validation remains open. The current Live Demo still
transfers bytes only. [Research connections](../../docs/RESEARCH_CONNECTIONS.md)
describe the exploratory background and later cryptanalytic work.

The optical transport supports 8/16 palette phases, progressive reception, per-cell robust
sample selection, and reliability-guided Reed–Solomon recovery. **OBM5 is the only
accepted format.** Reload both devices together. There is no old-format reader or
migration path.

## Reliable optical messages · step 3.3a

Open **Live Demo → Local simulation** and run the default 165,000-byte transfer.
The test sends real OBM5-encoded packets through a simulated channel with loss,
duplication, and corruption, then compares the recovered bytes exactly. Its
optical time is virtual; wall-clock execution time is reported separately.

For physical testing, open **Optical transfer** on both devices. Start the
receiver first, then the sender. Both cameras must see the other device's screen
because acknowledgments travel optically too. Begin with 1,024 bytes before
attempting 165,000 bytes. At the default 128×128/16-color, eight-phase profile,
a complete no-loss display of the proof-sized payload takes about 17 minutes;
allow additional time for camera acquisition and retries.

See [LINK_TRANSPORT.md](LINK_TRANSPORT.md) for the protocol, API, test sequence,
and reports to collect, including failed transfers. The full-message SHA-256
check detects corruption; it does not authenticate the sender.

```sh
npm run test:link
```

## Run

From this app directory, prepare the shared modules with Node.js 22+ before
serving `dist/` on any static HTTPS host. Preparation copies files without
bundling or downloading dependencies. The browser needs no Flask, enrollment
backend, or package installation. Python is an optional local serving convenience
and the reference-test language:

```sh
npm run prepare:app
python serve.py --port 8000
```

Open `http://localhost:8000`, or `http://localhost:8000/crypto.html` for the proof
lab. A phone needs an HTTPS origin for camera access.
Use the server's `--cert` and `--key` options with a certificate
trusted by the phone. Run `python serve.py --help` for the server options.
Serving the app needs only Python's standard library. Browser cryptography uses
native Web Crypto and a bundled SHAKE implementation; no CDN or npm install is
needed. Running the Python parity reference additionally requires NumPy. The
canonical Python core and a standalone API example are in
[`experiments/python-zkboo/`](../../experiments/python-zkboo/README.md).

## Browser cryptography · steps 3.1 and 3.2

Open **Crypto Lab** and choose **Run round trip** with the default profile. See
[CRYPTO_LAB.md](CRYPTO_LAB.md) for device tests, report fields, file exchange, and
the remaining integration goals. [CRYPTO_PARITY.md](CRYPTO_PARITY.md) describes
the module API, proof layout, and reference fixtures. The port matches the current
Python binary rotating-convolution Feistel circuit. It runs on bytes and Boolean
operations; it does not require PyTorch, WebGPU, or a neural inference runtime.

The compact proof reconstructs omitted commitments and output shares. The
default side-8, 16-circuit-round, 219-repetition proof contains **119,249 bytes**.
It retains every circuit lane and repetition. Python and JavaScript
use one current format, `NeuralPicnic-POC-MPC3-v2` / `NPZK0002`; earlier proof and
public-context files are not accepted. See the
[Python implementation](../../experiments/python-zkboo/README.md) for the matching
reference and standalone experiment CLI.

Crypto Lab separates the **key minimum repetitions** from **requested proof
repetitions**. Requests can change for each fresh challenge without generating a
new key, provided they respect its enrolled floor. For development, generate a
key with floor 32, then compare requests of 32, 64, 110, 160, and 219. A default
219-floor key cannot accept a lower-count proof. The repetition estimate is an
idealized soundness term, not a certified security level. There is no automatic
recency policy or persistent trust history in this iteration.

```sh
npm run test:crypto   # JS tests, including complete Python fixture parity
npm run test:parity   # regenerate Python fixtures and cross-verify fresh JS proofs
npm run benchmark:crypto # same-key proof size and engine timing at five repetition counts
```

Use Node 22 or newer. The second command requires `python3` with NumPy; set
`PYTHON=/path/to/python` to select another interpreter. The fixture keys and
deterministic entropy are public test data. The served engine always uses secure
randomness. This is a custom, unaudited research protocol, with no claimed
security level.

## Use

1. On the transmitting device, choose **Display**, configure content, and press
   **Play** when more than one phase/frame is present.
2. On the receiver, choose **Receive**, select a camera and resolution, then start
   the camera. Configure transmission is hidden in Receive mode.
3. Start with **64×64, 16 colors, grid 8, RS32, eight permuted phases**, and
   **0.25 phases/s**. Use **Robust subset + mean control**, five minimum samples,
   two overhead samples, first attempt after four phases, and erasure-assisted RS.
4. Keep all corner markers and palette strips visible. Reset the receiver after
   generating a new sequence. Decoded content appears after all logical data
   frames pass CRC. The receiver keeps gathering evidence when an attempt fails.
5. **Both** retains the large outgoing matrix and small camera alignment window.
6. **Image lab** is independent of the camera. Its default A/B test uses identical
   repetitions and palette permutations with equal exposure budgets and the same
   disturbance trajectory. It continues after success so both variants receive
   their full budget. A photograph is decoded as one self-contained phase.

## Transport controls

| Control | Values | Meaning |
|---|---|---|
| Grid size | 8–512 | Physical matrix cells per side, including references |
| Palette | 2, 4, 8, 16 | Colors per cell |
| Content frames | 1–8 | Distinct content chunks |
| Reference blocks | Off, 4, 8, 16 | One-cell reference lines around data blocks |
| RS parity | Off, 16, 32 | Parity bytes per codeword |
| Phases | 1, 2, 4, 8, 16 | Repeated transmissions of each chunk |
| Phase mapping | Repeated / permuted | Identical colors or public XOR label mappings |
| Phase hold | 0.25–10 seconds | Advertised in the header for retention timing |
| First attempt after | 1, 2, 4, 8, 16 phases | Minimum eligible phases, clamped to transmitted count |

At 16 colors, 16 permuted phases visit each physical color once for every logical
symbol. Smaller palettes repeat mappings when phase count exceeds color count.
Phases add optical observations, not payload capacity. References and parity
reduce capacity: at 16 colors, grid 8, and RS32, a logical frame carries 1,344 bytes
at 64×64, 5,472 at 128×128, or 90,098 at 512×512. Small grids may be too small for
the selected parity setting; the UI rejects those profiles.

## Continuous refinement

Each exposure is rectified and color-corrected independently. Its palette mapping
is undone before evidence is stored. The default robust mode keeps seven recent
observations per cell and phase. It finds the observation with the smallest total
L1 distance to the others, selects the five closest to it, and averages those
normalized logical-symbol scores. Each cell can select a different subset.

Decoding starts as soon as four phases each have five observations; it does not
wait for all eight or sixteen. Eligible phases have equal weight. New observations
replace the oldest ones, including observations previously selected as good. This
allows a wrong early consensus to leave the pool. Later phases join the estimate
when they have enough samples. The ordinary mean of the same available pool runs
as a separate control, with both border and local color models. Conflicting
CRC-valid candidates are rejected.

Recovery first tries the received bytes, then ordinary RS. Optional erasure
recovery ranks uncertain bytes and tries a bounded set of erasure budgets. With P
parity bytes, a codeword can correct e unknown errors and s known erasures when
`2e + s ≤ P`. Erasing a correct byte consumes capacity too. Every accepted result
must pass the content CRC. The comparison report separates raw passes, ordinary RS
passes, and additional erasure-assisted passes; CRC itself does not locate errors.

Retained score evidence is capped at **64 MiB** and eight logical frame groups.
The receiver reduces pool/sample targets when necessary, and may retain a rolling
subset of phases if even one capture per phase cannot fit. Effective limits are
reported. Sample lifetime is at least 60 seconds or two advertised full sequence
cycles, whichever is longer. Camera images and temporary decoder arrays require
additional memory. 512 is a format/test limit, not demonstrated handheld camera
capability.

## Code map

- `dist/js/format.js`: current wire format, framing, interleaving, CRC, bounded
  erasure trials, sequence assembly, and pure RGBA rendering.
- `grid.js`: reference geometry and reversible public palette mappings.
- `fec.js`: dependency-free shortened RS errors-and-erasures encoder/decoder.
- `vision.js`: finder detection, homography, cell sampling, color classification,
  byte reliability inputs, and synthetic camera stressors.
- `references.js`: current-exposure local color prototypes; no frozen RGB map.
- `temporal.js`, `sampling.js`: progressive phase fusion, per-cell selection,
  rolling pools, and memory/age limits.
- `decoder-worker.js`, `decoder-client.js`: worker execution and transferable buffers.
- `diagnostics.js`, `transport-view.js`: post-classification error measurements,
  reports, and comparison UI. Known payloads never help decode or repair data.
- `content*.js`, `receive-session.js`: byte/content handling and independent sessions.
- `app.js`, `index.html`, `style.css`: live views, lab comparisons, and device controls.
- `link/protocol.js`, `link/session.js`: OLM1 packet framing and bounded multipart
  sender/receiver state machines, independent of camera and UI.
- `link/optical.js`, `link/simulation.js`: OBM5 packet adaptation and the virtual
  fault channel; `link.html` is the Live Demo entry point.

## Verification

```sh
npm test
node scripts/transport-benchmark.mjs
```

See `FORMAT.md` for the wire format, `TESTING.md` for device tests, and
`VALIDATION.md` for automated results and their limits. Camera images and received
content are processed locally. CRC checks accidental corruption; it is not sender
authentication or a cryptographic proof.
