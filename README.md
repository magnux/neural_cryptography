# Neural Cryptography

Experiments in neural dynamical systems, a custom ZKBoo-style proof of knowledge,
and camera-to-screen optical communication. The browser applications perform
computation locally; their servers only serve static files.

The immediate goal is **Neural ZKBoo authentication between a phone and computer
using color-matrix optical transport**. The existing experiments are the
exploratory foundation. The next increment adds a small learned optical decoder
and reuses the working authentication flow with the color multipart channel.
Cryptanalytic comparisons and adversarial architecture experiments are future
research, outside the shortest path to this functional POC.

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
| Color optical transport | [Optical Bench](apps/color-optical-bench/README.md), `index.html` and `link.html` | Color matrices, image simulations, temporal fusion, error correction, and multipart transfers. |
| QR transport and authentication | [QR app](apps/qr-neural-key/README.md), `index.html` and `auth.html` | Standard QR byte transport and complete device-to-device proof exchange with independent outgoing profiles. |
| Learned optical decoder | [Model and training design](docs/OPTICAL_DECODER.md) | Planned small U-Net symbol classifier, optional temporal fusion, synthetic training, and real-device capture. No trained model or implementation yet. |

Read [use cases](docs/USE_CASES.md), [format boundaries](docs/FORMATS.md), and
[development instructions](docs/DEVELOPMENT.md) before changing a protocol.
The [current validation record](docs/VALIDATION.md) lists the checks performed.
Optical matrix dimensions and cryptographic circuit dimensions are separate
parameters: transport carries the serialized proof bytes.

## Plan and research context

- [POC roadmap](docs/ROADMAP.md): browser feasibility, capture/training,
  authentication integration, first color exchange, and repeatable device tests.
- [Optical decoder design](docs/OPTICAL_DECODER.md): architecture, input/output
  contract, dataset collection, RTX 4090 training, and browser inference.
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

## Status and scope

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

The project has no selected top-level license yet. Existing third-party code
retains its own licenses and provenance in [THIRD_PARTY.md](THIRD_PARTY.md).
