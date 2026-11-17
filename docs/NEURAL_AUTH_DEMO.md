# Standalone neural authentication demo

`apps/neural-auth-demo/dist/index.html` is a separate mobile-first app. Its
outgoing Neural-QR matrix and front-camera view share the screen so two devices
can face one another throughout the exchange. The existing Optical Bench and
its Neural-ZKBoo page remain independent entry points.

The browser performs camera acquisition, phase packing, trained W-tube/VW
inference, RS+CRC recovery, packet handling and cryptographic proof work locally.
The live receive path never reads target payloads. Human-readable names travel
inside the current authentication envelope and are bound into the proof challenge.
Names are self-selected labels, not an independently established real identity.

## Prepare a standalone directory

Use Node.js 22+ for preparation and tests. From the repository root:

```sh
npm run prepare:apps
npm run test:demo
```

Preparation copies the demo's JavaScript dependency closure from canonical bench
and shared sources, including worker imports. It copies project and vendor
notices and records source paths in `dist/dependency-manifest.json`. The generated
`dist/js/` tree must not be edited. Preparation needs no download or npm install.
To prepare only this app, use `npm run prepare:apps -- --app neural-auth-demo`.

Live use additionally needs your trained export. G32/G64 are enough for the
initial G64 receive limit; include G128 to enable the larger incoming grid:

```sh
python experiments/neural-corrector/export.py \
  --checkpoint checkpoints/neural-corrector/vw-distill/best.pt \
  --sizes 32 64 128 \
  --out experiments/neural-corrector/artifacts/selected
npm ci --prefix experiments/neural-corrector/browser-runtime
npm run prepare:corrector -- --artifacts experiments/neural-corrector/artifacts/selected
npm run test:corrector:wasm -- --artifacts experiments/neural-corrector/artifacts/selected
```

Replace the checkpoint path with your actual trained checkpoint. Corrector
preparation validates the manifest, sizes, SHA-256 hashes and pinned local
runtime before writing. By default it prepares both the bench and standalone
demo. Append `--app neural-auth-demo` or `--app color-optical-bench` to select one.
G32 is required for control packets; include G64 for the default incoming data
and G128 when selecting the larger receive limit.
The live worker rejects untrained exports. Models and prepared runtime files
remain ignored by Git.

Only the resulting `apps/neural-auth-demo/dist/` directory is needed on a static
HTTPS host. It has no runtime references to a sibling app or CDN, and requires
no inference server, enrollment service or upload endpoint. The existing
optional bench recording server is not needed. Preparation does not publish or
reactivate a hosted site.

For a computer-only localhost check:

```sh
python apps/neural-auth-demo/serve.py --port 8002
```

Open `http://localhost:8002/`. A phone opening a computer's LAN address requires
trusted HTTPS, with a certificate trusted by that phone and matching the
hostname. Reuse your existing trusted development certificate from the repository
root with the optional standard-library static helper:

```sh
python apps/neural-auth-demo/serve.py \
  --host 0.0.0.0 --port 8002 --https \
  --cert certs/dev.pem --key certs/dev-key.pem
```

Open `https://YOUR-CERTIFICATE-HOSTNAME:8002/` on both devices. Replace the cert/key
paths if yours differ; use the hostname covered by the certificate, and ensure
both devices trust it. The helper serves only this app's `dist/`, has no APIs or
uploads, disables directory listings and caching, rejects symlink/path escapes,
and supplies JavaScript/WASM MIME types. Any trusted HTTPS static host can serve
the same directory instead. Plain LAN HTTP does not provide camera/Web Crypto
access. A phone's front camera is the initial preference; explicit camera selection
takes precedence when available.

## Names, roles and completion

Choose a readable local name before starting. The shared current envelope is
`neural-zkboo-auth-v2` / `NZA2`; HELLO and IDENTITY carry the normalized name,
device UUID and public key. The pairing binding covers both ordered identities,
including names, and is included in the proof challenge. Reload/update both
endpoints together; earlier envelope versions are rejected.

Names are normalized to NFC with whitespace collapsed and trimmed. They must
contain 1–64 Unicode code points and at most 256 UTF-8 bytes; control and hidden
formatting characters are rejected. The internal UUID and key fingerprint still
identify the cryptographic session; a friendly name does not replace them.

One device acts as verifier and the other as key holder. The five application
messages are HELLO, IDENTITY, CHALLENGE, PROOF and RESULT, with optical START/DATA
and G32 ACK exchanges. This POC advances turns and confirms pairing automatically;
it does not establish identity through manual code comparison. A verified proof
on the verifier authenticates possession of the pinned key for this attempt.
The key holder's success text is a peer-reported result, not a second proof.

Wait for final result delivery: verifier `resultDelivered` and key holder
`peerReported` must be true. A locally accepted proof with its result still in
transit is incomplete. Keep both pages visible; cancel, hiding the page, or
restarting stops the active attempt and releases camera/worker resources.

## Optional timing statistics

Statistics are on initially. Keep them enabled before an attempt to collect stage
count, minimum, mean and maximum times, optical packet timings, and the final
report. Disabled statistics must skip timing-sample collection and detailed
report construction; functional completion and necessary protocol state remain.
Compare the same grid, role, browser backend and physical arrangement. A minimum
is the best observed sample, not a guaranteed speed or a timing security claim.

Efficient scanning is on initially. Capture scheduling skips work when the
worker is busy. Small frame/ROI checks may avoid repeated acquisition only after
RS+CRC recovery and transport acceptance of a burst; each observed phase has its
own template. A one-second refresh checks current geometry/orientation/phase,
and a four-second acceptance expiry forces full capture/inference even if an
image fingerprint collides. New attempts, mode changes and clock rollback reset
the gate. Packet identity, protected payload recovery and session checks remain
authoritative. Target bytes never guide live reception.

## Explicit ideal-output simulation

The simulation is separate from live camera operation. It runs both endpoints
with actual 32-repetition cryptography, Neural-QR drawing/acquisition, phase
packing, an actual **untrained VW ONNX inference**, RS+CRC recovery, multipart
packets, all five authentication messages, and final acknowledgment handling.
After measuring actual inference, the simulator explicitly substitutes ideal
logits constructed from its known rendered target. This assumption lets the
rest of the pipeline complete; it does not measure learned correction accuracy,
real camera reliability, or physical throughput.

Prepare separate untrained simulation artifacts once:

```sh
python experiments/neural-corrector/export.py \
  --model VW --frames 8 --sizes 32 64 \
  --out experiments/neural-corrector/artifacts/demo-simulation
npm ci --prefix experiments/neural-corrector/browser-runtime
npm run prepare:corrector -- \
  --simulation-artifacts experiments/neural-corrector/artifacts/demo-simulation
```

Simulation preparation accepts untrained VW G32/G64 only and writes
`models/simulation-corrector/` in this app. It does not overwrite
`models/neural-corrector/`, which is reserved for live models. Initialization
verifies artifact hashes and numerical parity before the comparison. Warmup and
initialization are separate from handshake timing. Simulation defaults to G64
data in both directions and G32 control packets, using the single-thread WASM
runtime. All processing stays in the local worker.

Run the in-page simulation or the command-line benchmark:

```sh
npm run benchmark:demo
# Save a local report; the path is relative to the app directory for npm --prefix:
npm run benchmark:demo -- --out ../../local-results/neural-auth-demo-simulation.json
# Explicit protocol-only check when model assets are unavailable:
npm run benchmark:demo -- --protocol-only
# Measure the optimized pipeline without detailed stage statistics:
npm run benchmark:demo -- --optimized-only --no-statistics
```

The baseline and optimized runs use the same scan schedule and independent fresh
keys/challenges/proofs. Both use the live automatic pairing and retained-ACK/START
handoff logic, including the three-second turn timer. Compare completed outcomes, scan and accepted packet
counts, inference counts, stage totals and whole-run wall time. Wall time also
includes rendering, scheduling and cryptography; stage times can overlap.
`--protocol-only` skips image acquisition and inference and is labeled accordingly;
it cannot substitute for the full benchmark. Missing models/runtime cause an
explicit error instead of silently falling back to fabricated inference timing.

## Device and packaging test plan

The initial settings are verifier role, G64 outgoing and incoming maximum, G32
controls, 32 proof repetitions, front camera, requested 1080p, 250 ms phases,
100 ms captures, Auto backend, efficient scanning enabled and statistics enabled.
Choose an incoming maximum of G128 to preload/test that larger receiving grid.

1. Prepare and copy **only** `dist/` to an empty static serving directory. Load
   the page, workers and all local dependencies. Confirm no request escapes that
   directory, no request goes to a sibling app/CDN, and missing runtime/model
   files produce an actionable failure. `npm run test:demo` checks the standalone
   dependency graph and required static assets without needing model downloads.
2. On two phones or a computer/phone pair over trusted HTTPS, select names and
   roles. Confirm the front camera is usable, preview orientation is clear, and
   the outgoing matrix remains fully visible in portrait and landscape. Permission
   denial or an unavailable selected camera must not silently select another.
3. Start with G64 in both directions, 32 repetitions, and statistics disabled.
   Confirm trained parity for the selected incoming grids, complete all five messages and final ACK,
   and check the displayed peer name and role-specific completion wording.
4. Repeat with statistics enabled in the same arrangement. Check finite stage
   min/mean/max and packet timing values, then disable statistics for a new
   attempt and confirm its detailed counters/report are absent. Do not infer
   performance differences from a single run.
5. Hold a matrix steady, briefly obstruct a camera, then restore it. Confirm
   periodic acquisition recovers, repeated ACKs remain usable, no duplicate
   application delivery occurs, and changing a phase or packet resumes work.
6. Test G128 in one direction, name edits before a fresh attempt, a role swap,
   cancellation, page backgrounding, and restarting. Active workers and camera
   tracks must stop; stale callbacks must not resume the prior attempt.
7. Run the full local simulation with statistics on and off. Both modes must
   reach proof verification and final delivery. Record backend, runtime/model
   hashes, initialization, warmup, baseline/optimized timings and the explicit
   ideal-output assumption with every performance comparison.

Automated tests and ideal-output simulation do not establish physical device
performance or cryptographic security. This remains a custom, unaudited research
protocol. CRC, RS and transport digests detect channel errors; the proof and
challenge binding perform the separate authentication check.

The optional static helper's configuration/path/MIME checks run without network
or certificate access:

```sh
python3 -m unittest discover -s tests/python -p test_demo_server.py -v
```

## Local validation record · 17 November 2026

One sequential baseline/optimized run used Node.js 24.22.1 on Linux 6.18.57
x86_64, ONNX Runtime Web 1.23.2, single-thread WASM, and an untrained 696,400-
parameter VW model. The CPU model was not exposed. Both runs used G64 data,
G32 controls, 32 proof repetitions, two held phase cycles, automatic pairing,
and the live three-second retained-ACK/START handoff.

Both runs completed all five messages, verified the 17,474-byte proof, exchanged
the readable names, and reached `resultDelivered`, `peerReported` and
`finalResultAckAccepted`. Each processed 368 scans and 46 displayed packet
windows, including four retained-ACK windows, with no manual handoff.

| Measured work | Baseline | Efficient scanning |
| --- | ---: | ---: |
| Complete run wall time | 132.427 s | 68.007 s |
| Actual ONNX inference calls | 92 | 51 |
| Actual ONNX inference total | 123.031 s | 63.342 s |
| Marker detection calls / total | 183 / 2.715 s | 102 / 1.353 s |
| Rectification calls / total | 368 / 2.382 s | 204 / 1.200 s |
| RS+CRC calls / total | 92 / 154.497 ms | 51 / 71.922 ms |
| Skipped scans | 0 | 164 |

Actual inference minimum/mean/maximum was 491.005/1337.296/2501.193 ms for the
baseline and 486.893/1241.993/2373.652 ms for the optimized run, pooled across
G32 and G64. Initialization took a separate 4.147 s, including G32/G64 parity
warmups of 648.321/2166.415 ms. Both numerical comparisons passed at absolute
`1e-4` and relative `1e-3` tolerance. The exported model SHA-256 values were:

- G32: `7f425b64b61439ed2a46328e7872c80d15f65f9de8a5239ac4ef640646725b56`
- G64: `701bbf7bf656440974a25655b0402237e6e310d1e57bf253a2493dc920edb594`

The optimized run made 44.6% fewer inference calls under this synthetic scan
schedule. Actual untrained inference was followed by the explicit assumed-correct
logit substitution in both runs. These are local computation measurements from
one comparison, not trained accuracy, camera throughput, phone performance or
cryptographic security evidence. Per-stage timings can overlap and should not
be summed as independent elapsed time. Raw local reports and model files are
not part of the tracked validation record.
