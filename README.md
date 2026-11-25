# Neural Cryptography

Can a neural network compute a useful cryptographic key? This project started
with a small experiment: hold a network’s weights fixed, flip one input bit,
and measure how many output bits change. Our first application, **Neural Key**,
asks whether an offline device can prove it holds a secret using only its
screen and camera.

Three components make the demonstration possible:

| Component | Its job |
| --- | --- |
| [Neural-ZKBoo](experiments/python-zkboo/README.md) | Proves knowledge of a secret for a fresh challenge using an exact Boolean circuit. The secret stays on the prover. |
| [Neural-QR](docs/NEURAL_QR.md) | Carries the serialized messages as color matrices across a screen-to-camera link, including the return acknowledgments. |
| [Neural-Corrector](docs/NEURAL_CORRECTOR.md) | A trained network estimates the transmitted symbols from several camera exposures. Its weights are public; the proof verifier makes the authentication decision. |

**What works:** two phones completed a [32-repetition exchange in about
130 seconds](docs/NEURAL_AUTH_DEMO.md#acknowledgments) with G64 data matrices
and G16 receipts. That is one measured setup. It demonstrates a working optical
link and proof exchange; it does not establish a cryptographic security level.
[Try the mobile demo locally](docs/NEURAL_AUTH_DEMO.md) or read the
[device results and limitations](#status-and-scope).

**Project talk:** [Download the editable PowerPoint](docs/presentation/neural-cryptography.pptx)
for a guided tour of the neural-key experiments, proof, optical link and two-phone result.
The [presentation guide](docs/presentation/README.md) includes the slide map and editing notes.

## Join the research

Researchers, engineers and coding agents are welcome. The
[open research guide](CONTRIBUTING.md) gives each direction a starting point,
a baseline and a result someone else can check:

- **Integration:** turn the existing JavaScript and Python proof cores into
  importable libraries that verify each other’s proofs.
- **Optical communication:** find out when the Neural-QR color-reference grid
  earns the space it occupies, using verified payload bytes per second.
- **Neural cryptography:** explore exact discrete circuits, learned keyed
  transformations, continuous dynamics and adversarial training as distinct
  experiments.

A useful first contribution can be a reproduction, a negative result or a test
fixture. To start on a laptop with **Node.js 22+**, run `npm run test:demo`
from the repository root. No camera, GPU, trained checkpoint or npm install is
needed for this software check. For the original motivation, see the
[Neural Waves notebook](https://colab.research.google.com/drive/19kV6VOT5zi6daMn6RaNWz_lDCxeAZgci#scrollTo=irKCCLQhd0_t)
and [connections to earlier neural cryptography research](docs/RESEARCH_CONNECTIONS.md).

## Experiments

| Experiment | Source and entry point | Purpose |
| --- | --- | --- |
| Neural dynamics / symmetric neural key | [Python experiment](experiments/neural-dynamics/README.md) | Recurrent CNNs, rotating kernels, diffusion, bias, numerical agreement, and surrogate attacks. |
| Python ZKBoo | [Python experiment and core](experiments/python-zkboo/README.md) | Exact binary circuit, compact prover/verifier, command-line provisioning and optical simulation, and Python↔JavaScript parity. |
| Browser Crypto Lab | [Color app](apps/color-optical-bench/README.md), `crypto.html` | Generate keys, prove, verify, reject tampering, and measure local performance in a worker. |
| Neural-QR transport research | [Design](docs/NEURAL_QR.md), [current Optical Bench](apps/color-optical-bench/README.md), `index.html` and `link.html` | Custom neural-readable matrices. Header-free acquisition runs in `capture.html`; Neural-ZKBoo uses the trained corrector and in-band RS+CRC packets. |
| QR transport and authentication | [QR app](apps/qr-neural-key/README.md), `index.html` and `auth.html` | Standard QR byte transport and complete device-to-device proof exchange with independent outgoing profiles. |
| Neural-ZKBoo authentication | [Device guide](docs/NEURAL_ZKBOO_DEMO.md), `neural-zkboo.html` | Header-free Neural-QR DATA and G32 ACK/REQUEST controls with trained VW/W-tube inference and independent directional geometry limits. |
| Mobile neural authentication demo | [Standalone app](apps/neural-auth-demo/README.md), [setup and tests](docs/NEURAL_AUTH_DEMO.md) | Five-step camera authentication with single-exposure G16 digest ACKs (or trained G32 ACKs), two-device single-phase echo and trained four-phase timing tests, and a local ideal-output simulation. |
| Neural corrector | [Model and training](experiments/neural-corrector/README.md), [collection guide](docs/MULTIRES_COLLECTION.md), `corrector.html` and `capture.html` | W-tube and VW training, evaluation and static ONNX export on version-2 G16–G128 four-pixel data; U-tube and two-pixel multires remain separate experiments. |

Read [use cases](docs/USE_CASES.md), [format boundaries](docs/FORMATS.md), and
[development instructions](docs/DEVELOPMENT.md) before changing a protocol.
The [current validation record](docs/VALIDATION.md) lists the checks performed.
Optical matrix dimensions and cryptographic circuit dimensions are separate
parameters: transport carries the serialized proof bytes.

## Plan and research context

- [Research and engineering roadmap](docs/ROADMAP.md): the demonstrated baseline,
  priorities for integration and transport, and how to compare results.
- [Source inventory](docs/SOURCE_INVENTORY.md): where the maintained code,
  fixtures and experiments live.
- [Neural-QR specification](docs/NEURAL_QR.md): geometric anchors, phase
  identification, header-free neural acquisition, payload integrity and ACKs.
- [Neural corrector design](docs/NEURAL_CORRECTOR.md): architecture, input/output
  contract, dataset collection, RTX 4090 training, and browser inference.
- [W-tube collection](docs/MULTIRES_COLLECTION.md): G16/G32/G64/G128, 4×4 pixels
  per cell, repeated optical size codes, server uploads and device/camera metadata.
- [W-tube datasets](docs/W_TUBE_DATA.md): native and auxiliary recording imports,
  split preservation, 4:1 paired synthetic generation and repeatable rebuilds.
- [Neural cryptography: origins and connections](docs/RESEARCH_CONNECTIONS.md):
  Pointcheval's identification scheme, key agreement by synchronization, learned
  encryption and cryptanalysis, ZKBoo/Picnic, and the learned optical receiver.

## Quick start

Use Node.js 22+ for preparation/tests and Python 3.10+ for the reference/optional
static server. Preparation copies shared source and bundled dependencies into
the three apps; it needs no npm install or network access. The new mobile demo
is an independent `dist/` directory and does not change the bench's entry points.

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

# Or serve the standalone mobile demo (prepare trained models for live use):
python apps/neural-auth-demo/serve.py --port 8002
```

Open `http://localhost:8000/crypto.html` for Crypto Lab or
`http://localhost:8001/auth.html` for QR authentication. A phone accessing another
computer requires **trusted HTTPS** for camera/Web Crypto access; ordinary LAN
HTTP is not equivalent to localhost. Serve an app's `dist/` directory through
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
authentication page requires the trained corrector for Neural-QR DATA and G32 ACK/REQUEST controls.

## Status and scope

In a November 2026 device test of the [standalone mobile demo](docs/NEURAL_AUTH_DEMO.md),
the owner reported around **two minutes for an end-to-end exchange** and about
**600 ms for a single-phase echo** in the separate timing lab. The live settings
that worked well in that setup, **350 ms per phase and 250 ms between capture
attempts**, remain available under fixed display timing. The current live demo
starts each packet at 200 ms per phase and lengthens the dwell on repeats, and
can trigger capture on a visible phase change after lock. The echo measures both camera
legs. The proof repetition count for this test was not reported.

On **12 November 2026**, the owner completed all five Neural-ZKBoo messages using
the trained VW corrector with header-free Neural-QR data and ACKs. The paired
reports confirm verifier `resultDelivered` and prover `peerReported`, using G128
computer→phone and G64 phone→computer. The 17,731-byte proof message (including its
17,474-byte proof) took about **268 seconds / 66 B/s** through its final ACK;
local proof verification took **41.5 ms**. The current app automates POC pairing and uses explicit G32 REQUEST controls
for recoverable turn handoffs. It records individual packet/ACK round trips and
separate REQUEST events. The November 12 reports predate that handoff change.

On **4 November 2026**, the owner completed a 32-repetition Neural-ZKBoo exchange
using the earlier learned G64 transport and static QR ACKs. This historical result
predates the current Neural-QR integration. Verification and final result
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

