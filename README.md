# Neural Cryptography

Experiments in neural dynamical systems, a custom ZKBoo-style proof of knowledge,
and camera-to-screen optical communication. The browser applications perform
computation locally. The demo server serves static assets and can optionally
store recording JSONs explicitly uploaded from the data collector.

The goal is **fast, reliable and secure optical authentication between a phone
and computer**. A complete Neural-ZKBoo device exchange has now succeeded using
the trained neural corrector for color data and standard QR ACKs. The new acquisition format
is **Neural-QR**: our custom optical matrices built specifically for neural
correction, with asymmetric geometric markers and phase cues instead of the
current dense monochrome header. Neural-QR is not a standard QR code.

The [Neural-QR design](docs/NEURAL_QR.md) and [roadmap](docs/ROADMAP.md) define the
remaining model and packet-integration steps. Version-2 header-free acquisition is implemented
in `capture.html`, including optical size identification, G16/G32/G64/G128
recordings at 4×4 pixels per cell and server uploads. The [W-tube data pipeline](docs/W_TUBE_DATA.md)
imports eligible earlier captures and generates harder paired synthetics. **W-tube
training and evaluation are implemented** for this four-pixel corpus; model export
and browser integration remain next steps. The `multires` model retains its
separate 2×2-cell experiment contract. The existing `neural-zkboo.html` remains the working
baseline while that format is developed. Functional success does not establish
cryptographic security or a controlled throughput advantage over QR; those are
separate acceptance criteria. Earlier experiments remain available as research
foundations.

The repository keeps one current implementation of each experiment; earlier
revisions belong in Git history. The hosted test sites are intentionally blank.
See the [source inventory](docs/SOURCE_INVENTORY.md) for the source layout and
the [Neural Waves notebook on Colab](https://colab.research.google.com/drive/19kV6VOT5zi6daMn6RaNWz_lDCxeAZgci#scrollTo=irKCCLQhd0_t)
for the original dynamical-systems exploration.

## Experiments

| Experiment | Source and entry point | Purpose |
| --- | --- | --- |
| Neural dynamics / symmetric neural key | [Python experiment](experiments/neural-dynamics/README.md) | Recurrent CNNs, rotating kernels, diffusion, bias, numerical agreement, and surrogate attacks. |
| Python ZKBoo | [Python experiment and core](experiments/python-zkboo/README.md) | Exact binary circuit, compact prover/verifier, command-line provisioning and optical simulation, and Python↔JavaScript parity. |
| Browser Crypto Lab | [Color app](apps/color-optical-bench/README.md), `crypto.html` | Generate keys, prove, verify, reject tampering, and measure local performance in a worker. |
| Neural-QR transport research | [Design](docs/NEURAL_QR.md), [current Optical Bench](apps/color-optical-bench/README.md), `index.html` and `link.html` | Custom neural-readable matrices. Header-free rendering/acquisition runs in `capture.html`; OBM5 remains the working authentication baseline until multires model and packet integration. |
| QR transport and authentication | [QR app](apps/qr-neural-key/README.md), `index.html` and `auth.html` | Standard QR byte transport and complete device-to-device proof exchange with independent outgoing profiles. |
| Neural-ZKBoo authentication | [Device guide](docs/NEURAL_ZKBOO_DEMO.md), `neural-zkboo.html` | A 32-repetition phone/PC exchange completed with trained four-phase data reception and static QR ACKs. Baseline for Neural-QR integration. |
| Neural corrector | [Model and training](experiments/neural-corrector/README.md), [collection guide](docs/MULTIRES_COLLECTION.md), `corrector.html` and `capture.html` | W-tube training/evaluation on version-2 G16–G128 four-pixel data; separate U-tube and two-pixel multires experiments remain available. Earlier phase-grouped G64 training powers the separate working authentication baseline. |

Read [use cases](docs/USE_CASES.md), [format boundaries](docs/FORMATS.md), and
[development instructions](docs/DEVELOPMENT.md) before changing a protocol.
The [current validation record](docs/VALIDATION.md) lists the checks performed.
Optical matrix dimensions and cryptographic circuit dimensions are separate
parameters: transport carries the serialized proof bytes.

## Plan and research context

- [POC roadmap](docs/ROADMAP.md): browser feasibility, capture/training,
  completed authentication, Neural-QR development, and repeatable device tests.
- [Neural-QR specification](docs/NEURAL_QR.md): geometric anchors, phase
  identification, header-free neural acquisition, payload integrity and ACKs.
- [Neural corrector design](docs/NEURAL_CORRECTOR.md): architecture, input/output
  contract, dataset collection, RTX 4090 training, and browser inference.
- [W-tube collection](docs/MULTIRES_COLLECTION.md): G16/G32/G64/G128, 4×4 pixels
  per cell, repeated optical size codes, server uploads and device/camera metadata.
- [W-tube datasets](docs/W_TUBE_DATA.md): native and auxiliary recording imports,
  split preservation, 4:1 paired synthetic generation and repeatable rebuilds.
- [Connections to previous research](docs/RESEARCH_CONNECTIONS.md): neural
  synchronization, adversarial learning, neural cryptanalysis, SIMON, LowMC,
  and Picnic, with primary references and future experiments.

## Quick start

Use Node.js 22+ for preparation/tests and Python 3.10+ for the reference/optional
static server. Preparation copies shared source and bundled dependencies into
the two apps; it needs no npm install or network access.

```sh
# From the repository root:
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
npm run prepare:apps
npm test
npm run test:python
npm run test:parity

# Serve the color bench, including Crypto Lab:
python apps/color-optical-bench/serve.py --port 8000

# Or serve the QR app for a same-computer check:
python -m http.server 8001 --bind 127.0.0.1 --directory apps/qr-neural-key/dist
```

Open `http://localhost:8000/crypto.html` for Crypto Lab or
`http://localhost:8001/auth.html` for QR authentication. A phone accessing another
computer requires **trusted HTTPS** for camera/Web Crypto access; ordinary LAN
HTTP is not equivalent to localhost. Serve either app's `dist/` directory through
a trusted HTTPS static host. The color app's optional server also accepts a
trusted certificate and key; see [development instructions](docs/DEVELOPMENT.md).

For the PyTorch dynamics experiment, install its additional dependency:

```sh
python -m pip install -r experiments/neural-dynamics/requirements.txt
python experiments/neural-dynamics/neural_key_experiment.py --compare
python experiments/python-zkboo/neural_zkboo_experiment.py --camera
```

For **Corrector Lab**, follow [its optional setup](docs/CORRECTOR_LAB.md).
It adds pinned PyTorch/ONNX build dependencies and a local browser runtime.
[M2 capture and training](docs/NEURAL_CORRECTOR_TRAINING.md) is now implemented;
the capture page itself needs only the existing static app. To test the whole
stack, export and prepare the trained checkpoint, then open **Neural-ZKBoo** on
both devices using [the setup and turn sequence](docs/NEURAL_ZKBOO_DEMO.md).
The original Optical Bench/Live Demo retain their classical receiver; the new
authentication page requires the trained corrector for DATA and uses static QR ACKs.

## Status and scope

On **4 November 2026**, the owner completed a 32-repetition Neural-ZKBoo exchange
using learned G64 data recovery and static QR ACKs. Verification and final result
delivery completed in approximately **6 minutes 17 seconds**. The 17,474-byte
proof took approximately **160 seconds** to transfer and acknowledge, or **109 B/s**
including turns and alignment. This validates the integrated path, not a matched
speed comparison with QR. No private report contents or recordings are published here.

Real-device tests in this project reached 128×128 matrices with 16 colors using
the color decoder's recovery mechanisms. QR transport was more reliable in the
tested setup and reached roughly 1 kB/s in favorable transfer tests. A separate
32-repetition QR authentication completed, but took about 15 minutes 31 seconds;
local verification took about 31 ms. These are reported observations from
particular devices, not comparable standardized benchmarks or guarantees.

The neural primitive and ZKBoo-style protocol are research prototypes, not an
audited authentication system or a claimed post-quantum security construction.
CRC, Reed–Solomon correction, and transport digests detect/correct channel errors;
they do not authenticate a sender. See each experiment's documentation for its
actual checks and limits.

## License and attribution

Original Neural Cryptography code and documentation, including **Neural-QR**,
**Neural-Corrector** and **Neural-ZKBoo**, are licensed under
[Apache-2.0](LICENSE). Copyright 2026 **Alejandro Hernandez Ruiz (magnux)**.
Commercial use is permitted; the license disclaims warranties and requires
preserving applicable license and attribution notices on redistribution.
Keep the project's [NOTICE](NOTICE), and use [CITATION.cff](CITATION.cff) when
citing the research. See [licensing scope](docs/LICENSING.md) for model releases
and method descriptions. Dependencies retain their own terms in
[THIRD_PARTY.md](THIRD_PARTY.md).
