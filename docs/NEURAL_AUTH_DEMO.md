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

## Screen layout and alignment

With Settings closed, the exchange fits one viewport: the outgoing matrix stays
square, the camera preview and controls remain visible, and the page does not
scroll. The initial matrix position is at the bottom, which helps align it with
the other device's front camera. **QR: bottom / top** changes its position; on a
wide landscape screen, the matrix uses a separate column and aligns to the
selected edge. A desktop browser presents the same interface inside a bounded
phone frame. The outgoing square has an extra 10–16 CSS pixels of clearance
on every side. The viewfinder has a neutral background. Look for a green outline
when a matrix is detected; this is alignment feedback, not confirmation that
the packet passed its checks. Neither the clearance nor the outline changes
the transmitted pixels.

The preview starts **Mirrored**. Its toggle reflects the visible video and
alignment overlay together; camera samples sent to acquisition/inference and
the outgoing matrix are unchanged. The mirror and matrix-position preferences
are saved in this browser. They can be changed during an exchange.

Open **Settings** to scroll through setup, statistics, simulation comparisons
and report controls. Closing it restores the fixed exchange view. Keep Settings
closed while aligning the devices so the complete matrix remains visible.

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
messages are HELLO, IDENTITY, CHALLENGE, PROOF and RESULT, with optical START/DATA,
G32 ACK and G32 REQUEST exchanges. This POC advances turns and confirms pairing automatically;
it does not establish identity through manual code comparison. A verified proof
on the verifier authenticates possession of the pinned key for this attempt.
The key holder's success text is a peer-reported result, not a second proof.

Both screens show **Step 1/5 · Hello**, then **Identity**, **Challenge**,
**Proof**, and **Result**. The chunk counter tells you how far you are *within*
a step. Both screens therefore stay at 4/5 while a multipart proof is in
flight. The sender advances when it reads the final ACK and displays REQUEST;
the receiver advances when it sees that REQUEST. The brief difference between
the counters shows which handoff is still crossing the optical link. Step 5/5
means the result is in flight or delivered; the completion text distinguishes them.

Wait for final result delivery: verifier `resultDelivered` and key holder
`peerReported` must be true. A locally accepted proof with its result still in
transit is incomplete. Keep both pages visible; cancel, hiding the page, or
restarting stops the active attempt and releases camera/worker resources.

## Finish and share an exchange

Leave the key holder's final ACK on screen until the verifier confirms delivery.
Then tap **Finish exchange** on each device. A summary replaces the outgoing
matrix. With statistics enabled it shows **Setup** (Start through model and
camera readiness) separately from the five authentication steps. The total
runs from Start until that device's completion signal: result ACK received on
the verifier, result received on the key holder. Download the short JSON report
from that panel; the detailed report remains available in Settings. With
statistics off, the short report still records the outcome and configuration,
without timings or packet statistics. For a timing comparison, save the short
report from **both** devices with statistics enabled.

Each device uses its own clock and sees the handoffs at slightly different
times. For example, the sender must read the last ACK before the receiver can
see the next REQUEST. Compare the two **Proof** rows to see how long that step
occupied each device; adding their durations would count the same optical
exchange twice. Waiting to press **Finish exchange** after local completion
does not increase the recorded exchange time.

## Request-driven handoff

Each START or DATA packet stays displayed until its matching ACK arrives. At a
message boundary, the receiver keeps the final ACK visible while preparing its
reply. The sender must read that ACK before it displays a **G32 REQUEST** asking
for the next message. Only a matching REQUEST releases the held ACK and permits
the reply's START. A request received before proof work finishes remains granted;
the prepared reply starts when it is ready.

The REQUEST is one fixed 44-byte OLM1 `poll` packet per handoff, bound to the
completed message's session ID, message ID, message type and final chunk index
(`base = chunkCount - 1`). It has no changing nonce or sequence counter. It stays
visible until the reply's START is accepted. Lost ACKs and REQUESTs are recovered
by continuing to display the same control packet; elapsed time never switches
between a final ACK and START. Duplicate or unrelated controls cannot advance a
different transfer.

All four palette-phase images are rendered once and cached for each displayed
packet, including its random padding. Cycling phases reuses those images instead
of generating new pixels for an unchanged packet. New logical packets get their
own phase images. RESULT is the final application message and does not request
another reply; its receiver keeps the final ACK visible for delivery confirmation.

## Optional timing statistics

Statistics are on initially. Keep them enabled before an attempt to collect stage
count, minimum, mean and maximum times, optical packet timings, and the final
report. Disabled statistics must skip timing-sample collection and detailed
report construction; functional completion and necessary protocol state remain.
Compare the same grid, role, browser backend and physical arrangement. A minimum
is the best observed sample, not a guaranteed speed or a timing security claim.

START/DATA round trips end at their matching accepted ACK. REQUEST controls have
separate bounded records for display and acceptance, linked to the preceding
message; they do not contribute to START/DATA RTT summaries. Statistics and
report controls are available inside Settings.

Efficient scanning is on initially. Capture scheduling skips work when the
worker is busy. Small frame/ROI checks may avoid repeated acquisition only after
RS+CRC recovery and transport acceptance of a burst; all four phase templates
from that accepted packet are retained, including phases collected before the
successful inference. A one-second refresh checks current geometry/orientation/phase,
and a four-second acceptance expiry forces full capture/inference even if an
image fingerprint collides. New attempts, mode changes and clock rollback reset
the gate. Packet identity, protected payload recovery and session checks remain
authoritative. Target bytes never guide live reception.

The live receiver keeps at most two distinct exposures **per phase**. It tries
the newest from each phase first (K4). If more are needed, K8 uses both slots
per phase. Once a phase is full, each new exposure replaces one of its two
slots at random; captures from the other phases stay put. The fingerprint
travels with the exposure it describes. A failed K8 is retried after four
fresh captures; missed markers and duplicate video callbacks do not fill slots.

## Matrix roundtrip timing

Use **Matrix timing** on both devices and select the same experiment:

| Experiment | The receiver displays | What the sender measures |
| --- | --- | --- |
| Single-phase echo | The colors it read from one exposure, including any wrong cells | One probe → echo RTT, matched fraction and cell error rate; no trained model required |
| Trained four-phase sweep | A G32 ACK after recovering a protected DATA packet from camera frames | One DATA → matching ACK RTT at each pair of display and capture intervals; trained corrector on both devices |

Neither experiment generates a proof or performs a full authentication. Use the
sweep to choose four-phase settings; use the echo to isolate single-exposure
optical behavior.

### Single-phase echo

This experiment holds one displayed phase per trial.
There are no authentication keys or trained models to set up. The receiver
uses the same Neural-QR markers, geometry and border palette as the live reader,
then classifies each test cell from that single exposure. It displays an echo
of the colors it **observed**. The sender scans the echo and compares the
returned cells with its original probe. A wrong color counts as a cell error;
it does not disqualify a timing sample. Two copies of a CRC-checked trial ID
distinguish an echo from the outgoing probe, an old echo or a different run.
The ID areas and fixed grid references are excluded from the error rate.

1. On both devices select **Matrix timing** and the same grid. A good starting
   point for two front cameras is G64, 100 ms capture, no dummy inference.
   The timing lab's capture setting is independent of the live exchange's
   250 ms capture setting.
   Start **Receiver** first, align the cameras, then start **Sender**.
2. Begin with **Probe exposure = 0**. The sender holds one phase until it
   gets the matching echo or times out. Each new trial advances to the next
   of the four phase cues. The receiver holds its echo until a newer trial
   arrives. Stop and restart the receiver before starting a new sender run.
3. For a phase-duration test, repeat in the same placement with finite
   exposures, for example **250, 150, then 100 ms**. The sender hides the probe
   at the exposure limit but keeps looking for its echo until the same timeout.
   Compare the matched fraction, timeout count and cell error rate (CER), as
   well as the minimum, mean and maximum RTT. Repeat with the roles swapped.
   A short RTT among a handful of successful trials is poor evidence for a
   reliable interval. Use a promising exposure in a real four-phase exchange
   before changing the authentication default.
4. Optionally enable **dummy inference on the receiver** at G32 or G64 after
   preparing the untrained simulation assets described below. This runs one
   actual model pass on a fixed tensor before displaying each new echo. Its
   output is discarded. Compare runs with it off and on to isolate the added
   cost; model loading, parity checks and warmup precede the measurements.

The sender times from submitting the probe to its canvas to receiving the
worker's decode of the matching echo, using one local clock. That includes
both cameras and screens, the return leg and scheduling. It is neither a
photon-level timestamp nor the minimum one-way phase duration. CER is end to
end on **matched** trials; an unreadable ID leads to a timeout, not a low CER.
There is no payload RS+CRC recovery in this diagnostic. In Settings, **Save
measurements** on each device to see local stage timings for camera readback,
detection and decoding. The receiver additionally measures capture-to-echo
latency and optional inference; the sender records individual trial outcomes.

### Trained four-phase sweep

Use the same [trained corrector export as live reception](#prepare-a-standalone-directory)
on both devices. A receiver reads the DATA grid and sends a G32 ACK; the sender
must **also** run the corrector, on the ACK. The untrained dummy from the
single-phase test cannot substitute on either leg. The first inference uses
one distinct exposure from each phase (K4); if recovery fails, the live reader
can retry with two per phase (K8). A corner lock or a completed inference alone
does not finish a step: the receiver must recover the protected packet and the
sender must recognize the ACK for that particular probe. The ACK carries a
digest of the DATA bytes actually recovered. A stale ACK or a correct-looking
matrix from a different step cannot count as a successful roundtrip.

1. Face the two devices toward one another. In **Matrix timing**, select the
   trained sweep and opposite roles. Start with **G64**, **500 ms per displayed
   phase** and **450 ms between captures** on both devices. These are test
   settings; the live authentication defaults remain 350/250 ms. Keep camera,
   grid, backend, lighting and placement unchanged during the sweep. Tap
   **Start camera** on both devices and wait for model and camera setup.
2. Tap **Start step 1** on the receiver, then on the sender. The sender
   cycles through four phases of one DATA packet. The receiver's step ends
   when it displays the first phase of its ACK. The sender's step ends when
   it recovers that matching ACK. Keep the devices aligned through both events.
3. Tap **Next: step N** on each device when you are ready. The next pairs
   are **450/400 ms** and **400/350 ms**; the final default pair is **100/50 ms**.
   A timeout (30 s by default) closes only the local step. There is no automatic
   advance. Wait for **ACK read** on the sender (or for that step to time out).
   Then press **Next on the receiver first**: it clears the old ACK and stays
   blank until it decodes new DATA. Press **Next on the sender** to show that
   DATA. Advancing the receiver prematurely withdraws an unread ACK.
4. Open **Settings & measurements → Download sweep report** on each
   device. Stop and restart both cameras to repeat a pace. Keep both reports.
   Repeat in the same placement, then swap roles: one success at a faster
   setting says little about its reliability. Count timeouts as misses.
   To distinguish a display limit from a capture limit, repeat at the same
   display interval but change the starting capture interval. Try the best
   pair in **Camera exchange**. To include the live default of **350/250 ms**
   in the sweep, begin at **500/400 ms** (or start at 350/250 directly). Reducing
   both intervals together preserves their initial difference.

Read the sender's RTT as **first DATA draw → matching ACK recovery on that
sender**. It includes two camera paths, display refresh and waiting for all
four phases; it is not twice the model inference time. On the receiver, look
at DATA acquisition; on the sender, look at ACK acquisition. Compare the time
to first lock, the time to see all four phases, packet recovery, and the first
ACK draw. Distinguish the requested intervals from the intervals actually
achieved between capture attempts and between phase draws. Both are local
measurements: do not subtract timestamps from different phones. In particular,
a receiver armed before DATA appears can spend time simply looking at a blank
screen. The JSON keeps a short capture trace (first 32 and last 32 scans per
step, with omitted scans counted), phase counts, and whole-step aggregates.
Reported display cadence is based on the actual outgoing draws; a receiver can
keep cycling its last ACK after the step finishes until Next or Stop, without
extending its recorded arm-to-ACK time.

If one phase never enters the burst, inspect the four counts on the **sender**
for the ACK leg. The November 19 two-phone sweep recovered G64 DATA at all nine
paces, but admitted only **110 / 238 / 52 / 0** G32 ACK exposures on the sender.
No ACK inference ran. Phase rejections in new reports are broken down by weak
reference colors, ambiguous or distant cue colors, and disagreement between
the top and bottom markers. The nearest-color votes on rejected scans are
diagnostic only; they never enter the burst.

To distinguish a sampling problem from a persistent cue-reading problem:

1. Try a long display interval, e.g. **1500 ms display / 250 ms capture**, with fixed
   spacing. This gives the camera several chances to see the middle of each phase.
2. At a faster setting (e.g. **250/200 ms**), repeat with **Vary capture spacing
   (±20%)** off, then on **only on the sender**. This varies when the sender
   samples the G32 ACK without changing either display or the DATA receiver.
   Compare successful roundtrips, admitted phase counts, rejection reasons and
   actual exposure spacing. Worker time can keep the camera from meeting the
   requested spacing. Keep the phones in place for the comparison.

| Reading stage | What to look for when a step slows down |
| --- | --- |
| Corners and geometry | Marker search, perspective transform, orientation, geometry and phase cues. A reused geometry can skip a new corner search. |
| Rectification | Color calibration and sampling the camera image into the rectified grid, then normalization. Compare the same grid size. |
| Burst and model | Phase collection and packing, trained inference, and RS+CRC recovery. Multiple attempts can dominate the step even if one inference is fast. |
| Overall | Compare local capture/readback and worker dispatch-to-return time with the optical RTT. Worker time includes the stages inside it; do not add it to their sum. A stage with no samples did not run. |

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
keys/challenges/proofs. Both use the live automatic pairing and request-driven
handoff. Each completed nonterminal message has a final ACK followed by a stable
G32 REQUEST and the next START. Compare completed outcomes, scan and accepted packet
counts, inference counts, stage totals and whole-run wall time. Wall time also
includes rendering, scheduling and cryptography; stage times can overlap.
`--protocol-only` skips image acquisition and inference and is labeled accordingly;
it cannot substitute for the full benchmark. Missing models/runtime cause an
explicit error instead of silently falling back to fabricated inference timing.

The simulator runs cryptography and inference on one event loop; live operation
uses separate workers. After REQUEST is accepted, if both senders are idle and
real cryptographic work is still pending, the simulator pauses virtual camera
time until that work finishes. This prevents repeated WASM scans of an unchanged
REQUEST from starving sequential WebCrypto operations. It never skips an unread
ACK/REQUEST or an outgoing packet. Reports expose these pauses as
`cryptoIdleWaits`; they are a simulation scheduling assumption, not a measured
camera delay. Repeated display windows reuse a bounded cache of each endpoint's
four phase images and random padding.

## Device and packaging test plan

The initial settings for **Camera exchange** are verifier role, G64 outgoing
and incoming maximum, G32 controls, 32 proof repetitions, front camera,
requested 1080p, **350 ms per displayed phase** and **250 ms between capture
attempts**, Auto backend, efficient scanning enabled and statistics enabled.
The initial display preferences are a mirrored preview and the matrix at the bottom.
Choose an incoming maximum of G128 to preload/test that larger receiving grid.

1. Prepare and copy **only** `dist/` to an empty static serving directory. Load
   the page, workers and all local dependencies. Confirm no request escapes that
   directory, no request goes to a sibling app/CDN, and missing runtime/model
   files produce an actionable failure. `npm run test:demo` checks the standalone
   dependency graph and required static assets without needing model downloads.
2. On two phones or a computer/phone pair over trusted HTTPS, select names and
   roles. Confirm the front camera is usable, the preview and overlay mirror
   together, and both matrix positions remain fully visible in portrait and
   landscape. Check short phone and desktop viewports with Settings closed: the
   matrix stays square and the page does not scroll. Open Settings to inspect
   statistics, then close it and confirm the fixed view returns. Permission
   denial or an unavailable selected camera must not silently select another.
3. Start with G64 in both directions, 32 repetitions, and statistics disabled.
   Confirm trained parity for the selected incoming grids, complete all five messages and final ACK,
   and check the displayed peer name and role-specific completion wording.
4. Repeat with statistics enabled in the same arrangement. Wait a few seconds
   after confirmation, then finish on both devices and save their short reports.
   Check the five local step durations, compare the reported roles and optical
   settings, and check the separate detailed packet timings in Settings. Start
   a new attempt and confirm the matrix returns. Do not infer performance
   differences from a single run.
5. Hold a matrix steady, briefly obstruct a camera, then restore it. Confirm
   periodic acquisition recovers, all four accepted phase templates remain
   usable, and a new packet resumes work. Miss a final ACK and then a REQUEST:
   each must remain visible until its peer advances, with no duplicate application
   delivery and no timed ACK/START alternation.
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

## Local validation scope · 17 November 2026

Automated integration checks exercise the actual authentication session and
32-repetition proof engine through request-driven turns. The manual scheduler
check completes HELLO, IDENTITY, CHALLENGE, PROOF and RESULT after dropping the
first final ACK for all five messages and the first REQUEST for all four reply
handoffs. It confirms that a queued reply cannot start before the matching
REQUEST, and that final RESULT delivery creates no further request.

Capture tests retain every phase of an accepted packet, detect changed packets,
and exercise periodic geometry refresh and acceptance expiry. Separate telemetry
checks keep REQUEST records bounded and outside START/DATA round-trip summaries.
These checks do not establish live camera recovery, physical-device layout,
trained accuracy, throughput or cryptographic security. The current handoff and
layout still need the device checks above; no current benchmark timings are
reported here.
