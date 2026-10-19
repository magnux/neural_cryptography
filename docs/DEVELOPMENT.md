# Development and local execution

## Repository layout

`apps/color-optical-bench/` and `apps/qr-neural-key/` are independently runnable
static browser apps. Their `dist/` directories contain the actual authored source,
not disposable generated bundles. Keep their vendored JavaScript and licenses.
Tests, deterministic public fixtures, benchmarks, and validation records live
outside the served directory.

`experiments/python-zkboo/` contains the canonical recovered Python core. The
color app's fixture scripts import it from this location. `experiments/neural-dynamics/`
and `notebooks/` currently document source-recovery gaps. No training code is
implied by the learned-decoder roadmap.

The two apps retain copies of the browser crypto modules so they can be served
as isolated static directories. This import preserves that behavior. Protocol
changes must keep both copies and the Python implementation aligned.

## Requirements and checks

- Node.js 22+ for the built-in test runner and Web Crypto.
- Python 3.10+ and NumPy for Python↔JavaScript parity.
- A browser with camera access and a secure context for device testing.

```sh
python -m pip install -r requirements.txt
npm test
npm run test:parity
npm run benchmark:crypto
npm run benchmark:qr
```

No npm dependency installation or browser build is needed. The root package
delegates to the existing app packages. For a custom Python interpreter:

```sh
PYTHON=/absolute/path/to/python npm run test:parity
```

The checked-in fixture keys and entropy seeds are public deterministic test data.
Runtime keys are generated with secure randomness. Keep real capture exports,
private keys, and device reports in ignored `local-data/` unless deliberately
reviewed for publication.

## Local serving

```sh
python apps/color-optical-bench/serve.py --port 8000
python -m http.server 8001 --bind 127.0.0.1 --directory apps/qr-neural-key/dist
```

| URL on the serving computer | Workspace |
| --- | --- |
| `http://localhost:8000/` | Color optical bench and image lab |
| `http://localhost:8000/crypto.html` | Browser Crypto Lab |
| `http://localhost:8000/link.html` | Color multipart transport |
| `http://localhost:8001/` | QR transport, scan check, and simulation |
| `http://localhost:8001/auth.html` | QR neural-key authentication |

For a phone, use trusted HTTPS. An example for the color app, using certificates
already issued for and trusted by the test devices:

```sh
python apps/color-optical-bench/serve.py --host 0.0.0.0 --port 8443 \
  --https --cert certs/dev.pem --key certs/dev-key.pem
```

Use the hostname/IP covered by that certificate. An untrusted self-signed
certificate or ordinary `http://LAN-IP` does not provide the same camera-access
conditions as localhost. Either `dist/` can also be served by an HTTPS static
server you already use. Serving requires no enrollment backend or uploads.

The former hosted demo URLs intentionally serve blank pages. This repository
contains no hosting identity or automatic deployment configuration.

## What a passing test establishes

The Node suites exercise codecs, state machines, malformed input rejection,
simulated faults, and crypto behavior. The parity harness regenerates Python
fixtures and cross-verifies fresh JavaScript proofs. Neither predicts physical
camera throughput. Device testing must measure both optical directions and
include alignment, dropped captures, acknowledgment recovery, and expiry.

See [USE_CASES.md](USE_CASES.md) for the app to use for each experiment and
[FORMATS.md](FORMATS.md) before changing serialization.
