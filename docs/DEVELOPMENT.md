# Development and local execution

## Source layout

Each experiment has one maintained implementation. Browser cryptography and its
worker modules live in `shared/browser-crypto/`, and current public proof fixtures
live in `shared/fixtures/zk/`. The NumPy proof core and CLI are in
`experiments/python-zkboo/`; PyTorch dynamics are in `experiments/neural-dynamics/`.
Both Python experiments use `shared/python/neural_optical.py`.

The two browser apps remain distinct: the color optical bench and QR transport
have different channel implementations and experiments. Neither is an archived
version of the other. Their authored HTML/CSS/JS remains under `apps/*/dist/`.

## Prepare and test

Requirements: Node.js 22+, Python 3.10+, and NumPy. The dynamics experiment also
requires PyTorch. The existing transport/crypto apps need no JavaScript dependency
install, bundler, or network fetch during preparation. The optional learned
decoder lab has a separate pinned dependency install described below.

```sh
python -m pip install -r requirements.txt
npm run prepare:apps
npm test
npm run test:python
npm run test:parity
```

`prepare:apps` copies canonical crypto, worker modules, and public fixtures into
ignored app-local directories. It replaces stale generated files so each `dist/`
can be served independently. Edit canonical shared source, then prepare again;
never edit the generated copies. App npm tests/benchmarks automatically prepare
before execution. The generated directories are not committed.

For the dynamics experiment and its tests:

```sh
python -m pip install -r experiments/neural-dynamics/requirements.txt
python experiments/neural-dynamics/neural_key_experiment.py --compare
npm run test:python
```

Without PyTorch, Python test output explicitly skips the four dynamics checks;
the NumPy/CLI checks still run. `PYTHON=/absolute/path/to/python npm run test:parity`
selects a different Python for browser cross-verification.

The current Python ZKBoo commands are documented in
[the experiment guide](../experiments/python-zkboo/README.md). Both CLIs work from
any working directory. Default results go into `neural_key_results/` and
`neural_zkboo_results/`; these ignored directories may contain private keys.
Never commit real private keys or camera exports. Fixture secrets are public,
deterministic test material and must not be reused for enrollment.

## Local serving

### Optional learned-decoder lab

Follow [Decoder Lab setup and device tests](DECODER_LAB.md) to install its pinned
Python and browser-runtime dependencies, export the untrained model, and run
`npm run prepare:decoder`. This is separate from `prepare:apps`; the original
transport/crypto workspaces keep their lightweight setup. Model export and asset
preparation are local build steps, not a Python inference service. Generated
assets and datasets are ignored by Git.

### Serve a prepared app

Prepare first, including after updating shared modules:

```sh
npm run prepare:apps
python apps/color-optical-bench/serve.py --port 8000
# In another terminal, or instead of the color server:
python -m http.server 8001 --bind 127.0.0.1 --directory apps/qr-neural-key/dist
```

| URL on the serving computer | Workspace |
| --- | --- |
| `http://localhost:8000/` | Color optical bench and image lab |
| `http://localhost:8000/crypto.html` | Browser Crypto Lab |
| `http://localhost:8000/link.html` | Color multipart transport |
| `http://localhost:8000/decoder.html` | Optional M1 Decoder Lab, after `prepare:decoder` |
| `http://localhost:8000/capture.html` | M2 dataset sender/receiver; model assets not required |
| `http://localhost:8001/` | QR transport, scan check, and simulation |
| `http://localhost:8001/auth.html` | QR neural-key authentication |

A phone requires trusted HTTPS. For the color app, use a certificate already
issued for and trusted by the test devices:

```sh
python apps/color-optical-bench/serve.py --host 0.0.0.0 --port 8443 \
  --https --cert certs/dev.pem --key certs/dev-key.pem
```

Use the hostname/IP covered by the certificate. Ordinary `http://LAN-IP` is not
the localhost camera exception. Either prepared `dist/` can be served by a
trusted HTTPS static host. Keep tests, fixtures, and private results outside the
served directory. No enrollment backend, server-side inference, or upload is
required. The hosted test sites remain blank; repository work does not deploy.

## Benchmarks and formats

The [M2 capture/training guide](OPTICAL_DECODER_TRAINING.md) covers canonical
dataset generation, offline joining, CPU smoke checks, RTX 4090 training,
held-out recovery evaluation, and browser checkpoint preparation. These tools
use the current OBM5 renderer/decoder through Node; Python does not maintain a
second optical wire implementation. Keep downloaded camera files and model
checkpoints in the ignored `local-data/` and `checkpoints/` directories.

```sh
npm run benchmark:crypto
npm run benchmark:qr
```

Benchmark files are written to ignored app-local `local-results/` directories
when the benchmark has file output; crypto-size results are printed as JSON.
Keep current validation conclusions in [VALIDATION.md](VALIDATION.md), not
versioned run-log copies. Git history provides previous source states.

The Node suites exercise codecs, state machines, malformed input, fault
simulation, and crypto behavior. Python parity regenerates fixtures and
cross-verifies fresh proofs. None predicts physical camera throughput. Device
tests must measure both optical directions, alignment, recovery, and expiry.

See [USE_CASES.md](USE_CASES.md) and [FORMATS.md](FORMATS.md) before changing a
protocol. Each layer accepts only its current format; changing the Python-only
simulator is separate from changing browser optical framing.
