# Neural authentication demo

A separate mobile-first, static application for Neural-QR authentication.
The front camera and outgoing matrix share the screen. Human-readable device
names travel with the authenticated session; cryptographic identifiers remain
internal. Statistics are optional and include per-stage and per-matrix timings.

From the repository root:

```sh
npm run prepare:apps -- --app neural-auth-demo
npm run test:demo
python apps/neural-auth-demo/serve.py --port 8002
```

Open `http://localhost:8002/` for a desktop check. Physical phones require a
trusted HTTPS static origin. Live camera reception additionally needs the
trained G32/G64 corrector export (G128 for the larger receive setting) and local runtime described in the
[setup, simulation and device test guide](../../docs/NEURAL_AUTH_DEMO.md).
With an existing certificate trusted by both devices, the optional static helper
can serve the app directly:

```sh
python apps/neural-auth-demo/serve.py --host 0.0.0.0 --port 8002 --https \
  --cert certs/dev.pem --key certs/dev-key.pem
```

Open the certificate's hostname on port 8002. The helper serves only this app's
`dist/`, with no directory listings, uploads or application APIs.

Only the prepared `dist/` directory is needed on the static host. Its JavaScript
dependencies are copied transitively from their canonical bench/shared source;
no sibling app, server computation, CDN or runtime package installation is
required. Preparation does not publish a site. Private keys and camera frames
stay in the browser.

The local simulation exercises cryptography, packet handling, acquisition,
packing and real inference, with an explicitly assumed correct neural output.
Its untrained model is not a live decoder or evidence of trained accuracy.
Simulation models and trained live models have separate preparation commands
and directories. See the guide before interpreting its timing results.
