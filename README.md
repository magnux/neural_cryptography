# Neural Cryptography

Experiments in neural dynamical systems, a custom ZKBoo-style proof of knowledge,
and camera-to-screen optical communication. The browser applications perform
computation locally; their servers only serve static files.

This repository collects the latest recoverable implementations as of
**19 October 2026**. The hosted test sites are intentionally blank. See the
[source inventory](docs/SOURCE_INVENTORY.md) for provenance and the original
Python scripts/notebook that still need to be restored.

## Experiments

| Experiment | Source and entry point | Purpose |
| --- | --- | --- |
| Neural dynamics / symmetric neural key | [Experiment record](experiments/neural-dynamics/README.md) | Recurrent CNNs, rotating kernels, diffusion, bias, numerical agreement, and surrogate attacks. Original script pending recovery. |
| Python ZKBoo core | [Python reference](experiments/python-zkboo/README.md) | Exact binary circuit, compact prover/verifier, and Python↔JavaScript reference behavior. |
| Browser Crypto Lab | [Color app](apps/color-optical-bench/README.md), `crypto.html` | Generate keys, prove, verify, reject tampering, and measure local performance in a worker. |
| Color optical transport | [Optical Bench](apps/color-optical-bench/README.md), `index.html` and `link.html` | Color matrices, image simulations, temporal fusion, error correction, and multipart transfers. |
| QR transport and authentication | [QR app](apps/qr-neural-key/README.md), `index.html` and `auth.html` | Standard QR byte transport and complete device-to-device proof exchange with independent outgoing profiles. |
| Learned optical decoder | [Roadmap](docs/ROADMAP.md) | Planned U-Net classifier, temporal fusion, synthetic training, and real-device capture. No trained model or implementation yet. |

Read [use cases](docs/USE_CASES.md), [format boundaries](docs/FORMATS.md), and
[development instructions](docs/DEVELOPMENT.md) before changing a protocol.
The [import validation record](docs/VALIDATION.md) lists the checks performed.
Optical matrix dimensions and cryptographic circuit dimensions are separate
parameters: transport carries the serialized proof bytes.

## Quick start

Use Node.js 22+ for JavaScript tests and Python 3.10+ for the reference/optional
static server. The browser apps bundle their dependencies and need no npm install.

```sh
# From the repository root:
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
npm test
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
