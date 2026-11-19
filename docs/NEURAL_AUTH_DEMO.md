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
G32 is required for trained REQUESTs even with static G16 ACKs; include G64 for the default incoming data
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

### Demonstrate an exchange in flight mode

1. Open **Camera exchange** on both phones while connected. Select the roles,
   incoming grids and backends. The page prepares the trained corrector and
   the cryptography worker immediately; it does not open the camera or create
   an authentication key during preparation. Changing the incoming grid or
   backend prepares that new configuration.
2. Wait for **Ready for flight mode** on **both** phones. This means the selected
   receiving profiles (including G32 controls), runtime and parity checks are
   ready in the open page, and the crypto worker passed its self-test. A
   missing model, runtime or failed check must not show the ready message.
3. Turn on flight mode, confirm Wi-Fi is off too, return to the **same open
   pages**, then tap **Start authentication**. Start reuses the prepared workers,
   and only now requests the camera and starts the session. If you tap Start
   during preparation, it waits for the same work to finish.

Readiness applies to this camera exchange on this open page. A reload, page
discard, switch to a lab, or a different model configuration can need the
server again. Finish/cancel releases the workers. For a second demonstration,
reconnect and prepare again. The recorded setup time starts at **Start**, so
it includes only any remaining wait for preparation.

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
G16 digest ACKs by default and four-phase G32 REQUESTs. This POC advances turns and confirms pairing automatically;
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

## Acknowledgments

In **Camera exchange → Settings → Acknowledgments**, choose the same option on
both devices before starting. The choice applies in both directions; it is not
negotiated over the link.

| ACK profile | What the receiving device displays | What the sending device reads |
| --- | --- | --- |
| **G16 · single-exposure digest (default)** | One static receipt, held until the next packet or turn handoff | A calibrated single exposure, with RS32 and CRC. The resulting 32-byte fingerprint must match the exact outstanding ACK. No model inference or four-phase wait on this receipt. |
| **G32 · full ACK with neural corrector** | Four phases of the full 84-byte OLM1 ACK | The trained reader collects phases and recovers the ACK through RS32 and CRC. |

DATA and the four G32 REQUEST handoffs keep their trained reader in either
mode; the proof, repetitions and terminal RESULT confirmation are identical.
There is no need to export or load a G16 model. The G16 camera reader has no
access to the expected digest; only the transport can match a decoded receipt.
An old readable receipt leaves the current packet on screen. See the
[G16 layout and binding](NEURAL_QR.md#acks-and-control-messages).

## Request-driven handoff

The first matrix of each message carries its metadata **and chunk zero**. If
the message fits, the receiver checks the whole-message SHA-256 and issues its
final ACK immediately. Otherwise it returns a progress ACK for chunk zero and
the sender continues at DATA index 1. Losing the first ACK only repeats the
same matrix; it cannot restart the message or deliver it twice.

For the short (32-repetition) G64 exchange, the first matrix has **1,244 bytes**
of content space and subsequent matrices have **1,288**. HELLO, IDENTITY,
CHALLENGE and RESULT each take one confirmed matrix. The proof envelope takes
14, including its first matrix. That is **18 DATA/ACK round trips instead of
23**, plus the same four one-way REQUEST handoffs (40 distinct matrices instead
of 50 before retries). Proof bytes, repetitions and final delivery confirmation
are unchanged. At other sizes or proof lengths the first matrix's extra
metadata can move the chunk boundary, so use the measured packet count.

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

For a four-phase packet, the app renders and caches all four images, including
the packet’s random padding. Cycling phases reuses those images instead
of generating new pixels for an unchanged packet. New logical packets get their
own phase images. RESULT is the final application message and does not request
another reply; its receiver keeps the final ACK visible for delivery confirmation.

### Show the outgoing phases before reading

**Outgoing phase timing** controls how long each of a packet's four phases
stays visible. In **Progressive** mode, a new packet starts at **200 ms per
phase**. If still on screen, its second pass uses 250 ms per phase, then 300,
350, 400, 450 and finally 500 ms on every further pass. You can change the
starting duration, the increment and the cap. The timing resets at each new
four-phase packet, including a G32 ACK or REQUEST. Static G16 receipts have no phase
timer. A cycling display slows down based on how long its packet is needed; the receiver doesn't send a signal to change the
rate. **Fixed interval** instead uses one duration on every pass (initially
350 ms). Neither setting changes the camera's search interval.

In **Camera exchange**, **Show all four phases before reading** is on by default.
On a new four-phase packet (START, DATA, G32 ACK or REQUEST), this phone
draws phases 0, 1, 2 and 3 in order. Until the fourth draw has had a refresh
opportunity, it keeps the camera preview live but starts no phase probes,
full-frame readbacks or corrector scans. From the first draw, that wait is
nominally **three starting phase intervals plus one refresh**: just over 0.60 s
at 200 ms, or 1.05 s at 350 ms. A busy browser can take longer.

A phone with nothing to display listens immediately. A phone displaying a G16
receipt can listen for the next packet as soon as the static image is drawn.
The sender checks fresh exposures at the configured camera search interval;
it does not wait for a phase change on a static receipt. After the first pass,
the phone can read the reply while its own matrix continues cycling; another
cycle of the same packet never starts a new pause. An early reply stays on the
peer's screen under the normal handoff rules. In particular, the final ACK
continues to display until Finish. Turning the option off restores simultaneous
reading with the **same ordered display cadence**. The setting applies only
to Camera exchange; the Matrix timing lab has its own controls.

To compare on two phones, use G64 in both directions, 32 repetitions,
**Fixed interval** at **350 ms display / 300 ms search**, and leave **Follow
phase changes** and statistics on. Run a complete authentication with **Show
all four phases before reading** off on both; repeat with it on, keeping roles
and placement fixed. For a third run, keep the first-pass wait on and select
**Progressive** (200 / +50 / 500 ms). This separates the effect of the wait
from the effect of changing the dwell. Save the **detailed
report from Settings on each device** for each run; the Finish summary gives
the local proof step time. Compare completion and that step time, as well as
START/DATA-to-ACK round trips and capture attempts in the detailed report. The
`firstPresentation` statistics describe local presentation; that time overlaps
the peer's reading and must not be counted as time saved from the round trip.
In that report, `fullReadbacksBeforeFourth` should be zero with the first-pass
wait on. `ackReadByRound` counts matching ACKs the sender read during passes
1 through 7 and 8-or-later. It tells you how long the sender kept displaying
a packet; it cannot tell which pass the other camera decoded.

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
report controls are available inside Settings. The short live report includes
`settings.ackSize` and, with G16 statistics enabled, `receiptReads` (scans,
decoded receipts, matched and unmatched receipts). A scan before the peer
displays its ACK need not decode anything. In detailed packet rows
`ack.packetBytes` counts the bytes inside the optical frame: 32 for a digest
and 84 for a full ACK.

Efficient scanning is on initially. Capture scheduling skips work when the
worker is busy. Small frame/ROI checks may avoid repeated acquisition only after
RS+CRC recovery and transport acceptance of a burst; all four phase templates
from that accepted packet are retained, including phases collected before the
successful inference. A one-second refresh checks current geometry/orientation/phase,
and a four-second acceptance expiry forces full capture/inference even if an
image fingerprint collides. New attempts, mode changes and clock rollback reset
the gate. Packet identity, protected payload recovery and session checks remain
authoritative. Target bytes never guide live reception.

### Capture on a phase change

With both phones resting face to face, the receiver can use the other screen
as its clock. Until it has a valid matrix lock, it searches at the configured
capture interval. After lock, it reads a **12×12 RGB thumbnail of the matrix
center** on fresh camera callbacks while the worker is free. Its footprint is
about six cells across, so it samples comparable content on G32 and G128.
The reference comes from the **same unmirrored capture that established the
lock**, not whatever the camera shows when the worker finishes. A change is a cue to take a full
capture on a **new camera frame at least 10 ms later**. An unchanged patch does
not need a full-frame readback on every callback. The 10 ms is a minimum
settling delay after observing a change, not a guaranteed exposure time:
camera delivery and a busy worker can delay the next usable frame.

The patch never decides the matrix's phase or whether to accept a packet.
The opposite-border phase cues, corrector, RS+CRC and packet checks still do
that work. Camera motion or a reflection can trigger an unnecessary scan; a
weak phase transition can be missed. A watchdog requests another scan at twice
the search interval (bounded to 250–1000 ms); loss of lock returns the camera
to timed search. This also supplies additional exposures for K8 recovery.
The sender's phase interval and wire format do not change. Turn off the
phase-change option to use the timed reader for comparison.

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
| Trained four-phase sweep | After recovering DATA: a four-phase G32 ACK (default), or an experimental static G16 receipt | One DATA → matching receipt RTT; DATA always uses the trained corrector. G16 needs only one readable exposure on the return leg. |

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
on the DATA receiver, and also on the sender for the default G32 ACK. For
**G16**, both devices must select the experimental ACK option. The receiver
still decodes exactly the same four-phase DATA, then displays one static G16
receipt until Next. The sender reads it with the camera's measured palette,
RS and CRC, without a G16 checkpoint or a phase burst. There is no automatic
switch of ACK format. For the trained legs, the first inference uses
one distinct exposure from each phase (K4); if recovery fails, the live reader
can retry with two per phase (K8). A corner lock or a completed inference alone
does not finish a step: the receiver must recover the protected packet and the
sender must recognize the ACK for that particular probe. The ACK carries a
digest of the DATA bytes actually recovered. A stale ACK or a correct-looking
matrix from a different step cannot count as a successful roundtrip.

1. Face the two devices toward one another. In **Matrix timing**, select the
   trained sweep and opposite roles. Start with **G64**, **500 ms per displayed
   phase** and **450 ms between captures** on both devices. These are test
   settings; **350/250 ms** is the fixed-cadence live comparison. Keep camera,
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
   pair in **Camera exchange**. To include the live fixed baseline of **350/250 ms**
   in the sweep, begin at **500/400 ms** (or start at 350/250 directly). Reducing
   both intervals together preserves their initial difference.

To test **Progressive transmit timing**, first leave it **off** on both devices
for a fixed-dwell baseline. Next, with both devices still at the same initial
phase/search (for example **150/100 ms**), enable it **only on the sender** to
change DATA timing while G32 ACK stays fixed. On a separate run, enable it
only on the receiver to change ACK timing. Each new packet starts at the
current step’s phase interval: with **+50 ms per pass** and a **500 ms cap** its
phase dwell goes 150, 200, 250 … 500 ms; all four phases get the same dwell
within a pass. Receiver search still starts when armed and sender ACK search
still starts at its first DATA draw, regardless of this local display setting.
Progression changes neither the bytes on the wire nor the test-step settings
that must match between devices. Compare RTT, valid phases and inference
attempts. The report records each device’s local settings,
`display.completeCycles` (passes that actually drew all four phases),
`peakDwellMs` (maximum requested dwell drawn) and `spacingMs`
(observed time between draws). For a receiver, this includes holding the ACK
until Next, so waiting before advancing can increase its reported peak.

Read the sender's RTT as **first DATA draw → matching ACK recovery on that
sender**. It includes both cameras, display refresh, and phase collection on
each trained leg; it is not twice the model inference time. On the receiver, look
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

To compare ACK formats, fix both phones in the same positions at G64 and keep
the matrix display area, capture cadence, lighting and backend unchanged. Run
the G32 baseline, then run the same sweep with G16 selected on **both** phones.
Swap sender and receiver and repeat. Save both reports even if a step times out.
Compare **completed / attempted** first, then sender RTT and the number of
camera reads needed for a matching ACK. A quick success averaged only over
readable steps can hide timeouts. The G16 ACK leg should show **zero corrector
inferences**. Live authentication defaults to this same G16 receipt; the timing
lab starts with the G32 comparison profile.

If one phase never enters the burst, inspect the four counts on the **sender**
for the ACK leg. The November 19 two-phone sweep recovered G64 DATA at all nine
paces, but admitted only **110 / 238 / 52 / 0** G32 ACK exposures on the sender.
No ACK inference ran. Phase rejections in new reports are broken down by weak
reference colors, ambiguous or distant cue colors, and disagreement between
the top and bottom markers. The nearest-color votes on rejected scans are
diagnostic only; they never enter the burst.

To measure the change-triggered reader, run back-to-back sweeps with the
phones fixed in place. Start at **350 ms display / 300 ms search**, a setting
already known to work. Use timed capture with variation off for the first
run, then enable **Follow phase changes** on both phones and repeat. Keep all
other settings the same. Only then shorten the displayed phase (e.g. 250 ms,
then 200 ms). Compare completion and K4/K8 attempts as well as round trips;
a faster minimum with more timeouts is not an improvement. Each report
records how many full reads were triggered by change or by the watchdog,
the cost of probing, and the actual change-to-capture delay. The sender
reports these for G32 ACK; the receiver reports them for DATA. A change
detected by the thumbnail is not itself a recovered phase.

To distinguish a sampling problem from a persistent cue-reading problem
using **timed capture** (phase-change capture off):

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
data in both directions, static G16 ACKs and G32 REQUESTs. It reads the G16
receipt from rendered pixels; only trained-packet outputs are simulated. The
neural inference benchmark uses the single-thread WASM runtime. All processing stays in the local worker.

Run the in-page simulation or the command-line benchmark:

```sh
npm run benchmark:demo
# Compare against the trained four-phase ACK path:
npm run benchmark:demo -- --ack-size 32
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
rendered phases, or its one static receipt image.

## Device and packaging test plan

The initial settings for **Camera exchange** are verifier role, G64 outgoing
and incoming maximum, G16 ACKs, G32 REQUESTs, 32 proof repetitions, front camera,
requested 1080p, **progressive 200 / +50 / 500 ms** and **250 ms between searches**,
Auto backend, first-cycle display priority, phase-change capture after lock,
efficient scanning and statistics enabled.
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
3. Start with G64 DATA in both directions, the default G16 ACKs, 32 repetitions, and statistics disabled.
   Confirm trained parity for the selected incoming grids, complete all five messages and final ACK,
   and check the displayed peer name and role-specific completion wording.
4. Repeat with statistics enabled in the same arrangement. Compare a second run
   with G32 ACKs selected on both devices and all other settings unchanged.
   Finish on both devices and save their short reports. In the default G64,
   32-repetition G16 run, expect **15 matched receipts** on the key holder
   and **3** on the verifier (18 total).
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
