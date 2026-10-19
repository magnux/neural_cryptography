# QR authentication and transport device tests

## Authentication acceptance test

Use `auth.html` for this test. Each device's outgoing settings control the QRs
on **its own screen**, regardless of its authentication role. Use **Phone** on
the phone and **Desktop (Fast)** on the computer, with requested **1280 × 720** capture.
The preset selected initially by viewport size is only a convenience; confirm it
on each device. Leave **Adapt outgoing speed to ACK progress** enabled and control correction at Q.
Use the phone's front camera if that permits both directions to see each other.

| Device | DATA / correction | Chunks before ACK | QR interval | ACK hold | Width |
| --- | --- | ---: | ---: | ---: | ---: |
| Phone | 128 bytes / Q | 1 | 350 ms | 750 ms | 100% |
| Computer | 1,024 bytes / M | 4 | 200 ms | 500 ms | 65% |

1. Arrange both cameras before starting. Keep the phone's entire code visible to
   the webcam. Reduce the PC's physical QR width if necessary so the phone can
   move closer to the webcam without losing the PC code from its own field of
   view. Width is adjustable from 40% to 100%; keep all white borders visible.
   Record both physical code widths and approximate distances.
2. Select **32 proof repetitions on both devices** and the default **20-minute**
   deadline. Each key uses the count as its minimum, and the verifier requests
   the higher count, so a 219 selection on either side prevents a short run.
   Select and start each camera. Select **Key holder** on the phone and click
   **Generate key & wait for verifier** first. Select **Verifier** on the computer
   and click **Generate key & start verifier**.
3. Wait for public identity exchange, compare the entire 24-hex-character pairing
   fingerprint, and confirm it on both devices using
   **The codes match · confirm pairing**. If it differs, cancel the attempt.
4. Leave both cameras running until the verifier shows its local decision and
   the result transfer is acknowledged. Bytes received alone do not mean
   authentication succeeded. If dense DATA stops progress, try the local smaller
   chunk and **Apply outgoing settings** without resetting either device.
5. Reset both devices and repeat at **219 repetitions** once the short run works.
   Raw proof sizes should be 17,474 and 119,249 bytes respectively; framing adds
   bytes. A phone sending at 128 bytes per chunk may need several minutes.
6. Reset and swap roles, keeping the **physical-device profiles**: Phone stays on
   the phone and Desktop (Fast) stays on the computer. This tests the large proof in the
   opposite direction. A result on the key-holder screen is a peer report, not
   proof of the verifier's identity.

### Focused directional and adaptation comparisons

Do these at 32 repetitions and export both reports after each attempt. Change
one variable at a time; avoid spending a full-size proof on an unreadable setup.

| Check | Action | What to observe |
| --- | --- | --- |
| Conservative baseline | Use 128-byte chunks on both devices if either direction remains uncertain; turn Adaptive off for a fixed-profile comparison. | Which application stage completes, and whether ACKs are read. Small DATA does not make the fixed 28-byte ACK smaller. |
| Independent settings | Restore Desktop (Fast) on the PC and Phone on the phone. | The phone's outgoing chunk remains 128 while its incoming PC messages may use 1,024. Both public identities must transfer. |
| Apply a smaller chunk | With Adaptive off, start a sender at 1,024 bytes. If DATA stalls, select 128 on that device and apply during the message. | Partial transfer restarts with smaller chunks. The same key and already-generated proof remain in use; pairing and the deadline do not restart. |
| Automatic fallback | In a fresh run, enable Adaptive and deliberately try 1,024-byte outgoing DATA on the weaker direction. | After three retry intervals without newly ACKed bytes, the effective chunk decreases and timing relaxes. Record the event, final chunk, and whether it recovers. |
| Unreadable START | Briefly misalign before START can be acknowledged. | Timing may relax, but reducing DATA cannot solve a control-code visibility problem. Restore alignment and check recovery. |
| Ceiling and recovery | Allow a successful reduced-profile transfer to continue through clean windows. | Any increase occurs at a later message and stays at or below that device's configured ceiling. Phone never exceeds a 128-byte ceiling. |

For the automatic fallback check, missing ACKs do not identify which camera
failed. Include the displayed packet kind, each screen's stage, both read
counters, and the adaptation events. Stop the test if neither camera can see a
complete QR; adaptation cannot repair that geometry.

After a clean short run, test recovery with **32 repetitions**:

| Check | Action | Expected behavior |
| --- | --- | --- |
| Missing proof data | Cover the verifier's camera for 2–3 seconds during proof transfer. | Transfer resumes through retries and verifies once, or ends with a clear transport failure if its limits are exceeded. |
| Missing ACKs | Briefly cover the key holder's camera near the end of proof transfer, then uncover it. | A repeated completion ACK allows handoff to the result; no second proof or authentication decision is created. |
| Cancel | Cancel during proof generation or transfer. | The local attempt ends and cannot later become accepted. Reset is local; the peer needs its own reset or must reach its deadline. |
| Fresh retry | Reset both devices after cancellation and start again. | A new identity/session/challenge flow completes; old visible codes do not complete the new attempt. |
| No pairing confirmation | Leave one device unconfirmed briefly. | Proof generation does not proceed until the key holder confirms; the verifier does not issue its challenge before its own confirmation. |

Choose a 10-, 20-, or 30-minute deadline before starting; the default is 20.
The chosen deadline is absolute and does not extend when settings change or a
message restarts. A one-sided reset is not automatic re-enrollment: use
**Reset / discard key** on both sides before a fresh attempt.
Do not change repetition, role, or deadline settings during an active run; reset
first. Outgoing transport settings can be applied during the run.
Moving a page into the background cancels its attempt and stops its camera.

To check operation without a server connection, open both pages, start both
cameras and attempts, and wait for local key generation and camera decoding to
start. Then disconnect only networking while keeping both pages in the foreground
and cameras running. Workers load lazily, so disconnecting before the first start
can prevent them from loading. This version has no installable offline cache; a
page reload requires the HTTPS server again.

For each attempt, report:

| Field | What to report |
| --- | --- |
| Setup | Phone/browser, webcam, roles, actual camera resolution, physical QR width on each screen, approximate distance and screen angle |
| Settings | Requested proof repetitions, key minimum, raw proof bytes, deadline, both local outgoing profiles and ceilings, effective chunk/window/QR interval/ACK hold, and incoming peer settings |
| Outcome | Verifier's local verdict, key holder's received report, whether the final result ACK completed, and exact failure text |
| Timing | Key/proof generation, proof transfer, verification, and total attempt duration where available |
| Recovery | Which camera was covered, stage and duration, whether progress resumed, retries, manual Apply actions, and automatic adaptation/restart events |

Use **Export JSON report** on both devices before resetting, and keep them paired by
attempt. Report successes and failures; a screenshot of the status is useful
when a failure occurs. Do the 32-repetition test first so alignment failures do
not require waiting for a full proof transfer. Physical testing should establish
completion and recovery, not infer cryptographic strength from a successful scan.

## Transport lab diagnostics

The goal is to find the fastest setting that **finishes repeatedly on your
devices**, including the final acknowledgment. Keep the physical arrangement
fixed while comparing presets. A larger QR or faster cycling setting is useful
only if the cameras still decode it reliably.

If your current alignment already transfers reliably, start with the short
speed comparison in section 2. Use section 1 when a denser code stops decoding
or when the phone-to-computer acknowledgment path becomes unreliable.

For the reported multi-chunk hiccup, first choose **Faster**, then set
**Chunks before waiting for ACK = 1** and **Milliseconds between QRs = 500**.
This retains the 2,048-byte/M payload setting while allowing each QR to be read
and acknowledged individually. Compare 1 versus 2 chunks with the same 500 ms
interval, then try 300 ms if both complete reliably. Use the same 16 KiB payload
and save both reports for each attempt. If the dense static QR itself will not
decode, reduce the payload size rather than only extending the interval.

## 1. Check the return path, then reverse it

Open **Scan check** on both devices. On the phone choose **28 bytes · ACK-sized**;
on the computer start its webcam. Keep the white border fully visible.

Try each setting for approximately 15 seconds, changing one thing at a time.
**Stop camera**, then **Start camera** for each comparison to reset its counters
and first-read timer; changing the displayed sample alone does not reset them.

| Variable | Initial test | Comparison |
| --- | --- | --- |
| QR correction | Q | H |
| Display size | Normal | Expanded display |
| Webcam request | 1280 × 720 | 1920 × 1080, if supported |
| Distance | Comfortable focus, whole QR visible | Modestly farther away |

Record actual distance and approximate physical QR width; an expanded display
does not always change its size on every screen. Record the camera's actual
resolution where shown, since a request may fall back. If pixels shimmer, try
more distance or a small change in angle while preserving the complete border.

Choose the sender preset in Transfer, then return to Scan check and choose
**Current sender chunk + header** with matching correction. Start at Medium:
512 useful bytes plus 18 transport bytes, with Q correction. Try **Maximum size
at selected correction** to check the camera's density limit independently of
multipart timing. It fills one version 40 QR at the selected correction level;
it is deliberately a demanding test.

Reverse devices and repeat. The binary diagnostic samples are not active
transfer packets; use this app, not a generic phone QR scanner, to read them.

Report the displayed **percentage of frames decoded** and **time to first read**,
plus whether reads remain continuous or intermittent. The alignment dock shows
reads/attempts and decode time. Export the JSON report after each comparison;
its `cameraRun` records actual resolution, attempts, successful reads, decode
rate, and timings. First-read timing begins when camera playback starts, excluding
the permission prompt. Include camera model, phone/browser, physical display
size, distance, and correction level. If no read succeeds, report that result too.

## 2. Quick speed comparison

Use the best scan settings. Select **Receive** on the phone and start its receiver;
then select **Send** on the computer. Both cameras must see the opposite screen.
For facing displays, try the phone's front camera; check that both cameras can
focus in this arrangement.

The sender's preset defines the DATA settings. Leave **control correction at Q**
on both devices initially. The receiver reads the sender's transfer parameters
from START and does not need to select a matching preset. Keep the deadline
the same on both sides.

Use **4 KiB** for a quick sweep, then **16 KiB** to compare the best settings with
less influence from startup. Use the same payload size for every run in a sweep.

| Order | Sender preset | DATA bytes / correction | Chunks before ACK wait / minimum interval | Purpose |
| --- | --- | --- | --- | --- |
| 1 | Medium | 512 / Q | 1 / 250 ms | Baseline with a larger payload per QR |
| 2 | Fast | 1,024 / M | 4 / 200 ms | Larger QRs and fewer ACK waits |
| 3 | Faster | 2,048 / M | 8 / 150 ms | Dense QRs and quicker cycling |
| 4 | Fastest | 2,935 / L | 8 / 100 ms | Maximum binary DATA capacity |
| Fallback | Slow | 128 / Q | 1 / 350 ms | Check alignment when larger settings fail |

Run each preset **three times**. Keep the receiver's final ACK visible until the
sender confirms. Export both reports before starting the next attempt, including
failures. If a preset is unreadable, stop that attempt; do not spend a full
165,000-byte transfer testing it. Check its static sample, then try a less dense
setting or a longer hold time.

Record this compact set for each attempt:

| Field | What to report |
| --- | --- |
| Configuration | Preset, actual chunk bytes, DATA/control correction, window, hold, retry interval |
| Outcome | Receiver digest passed? Sender received final ACK? Any error or timeout? |
| Speed | Sender's end-to-end goodput and elapsed time, with the same payload size |
| Recovery | Retry count and whether progress paused repeatedly |
| Camera | Actual resolution, percentage of frames decoded, mean decode time, time to first read |
| Setup | Sending direction, phone/browser/webcam, approximate distance and QR width, light/angle changes |

Use the median goodput of the completed runs and also report the number completed
out of three. A setting with one fast success and two failures is not equivalent
to three consistent successes. Keep sender and receiver reports together; the
receiver's completion time alone omits the final return ACK.

### Refine the best two presets

Change one variable at a time using **Custom**:

- Try a smaller chunk at the same correction and window to reduce density.
- Try Q or H correction with a chunk that fits. The app displays any capacity
  reduction when the previous chunk is too large for the new correction level.
- Type a longer **Milliseconds between QRs** interval if the camera needs more
  time. Values from 50 to 2,000 ms are accepted, including values between presets.
- Compare **Chunks before waiting for ACK** values 1, 2, and 4 at the same chunk
  size and interval. A group is confirmed before the next begins. With 1, each
  QR stays visible until both its ACK and minimum display interval are satisfied;
  an ACK arriving after that interval allows immediate progress.

L correction permits the largest QR byte payload, but that does not make it the
fastest reliable setting. The live loop targets one camera attempt every 50 ms;
actual processing can take longer and skip the nominal cadence. Do not infer
physical throughput from hold time alone.

## 3. Small payload and recovery checks

For a new arrangement, start with **Slow**, then:

1. Send the default text. Confirm that received UTF-8 text is displayed correctly.
2. Send text containing accents and an emoji, for example `Mañana: café ☕`.
3. Send **1,024 test bytes**. Keep the receiver's final ACK visible until the
   sender reports confirmation.
4. Reverse sender and receiver and repeat the 1,024-byte test.

For each attempt export reports from **both** devices before starting another.
Distinguish these outcomes: no START acknowledgment; chunks stop progressing;
receiver succeeds but sender never confirms; both confirm. These isolate
different failures. Reports do not include your transmitted text or filename.

Once a speed configuration works repeatedly:

1. Transfer the built-in **165,000 test bytes** using the best repeatable setting.
   Increase the deadline if required; do not assume synthetic timings predict
   live throughput. Downloading the bytes permits an independent hash comparison.
2. During a small transfer, cover the receiving camera for 2–3 seconds, uncover
   it, and check recovery. Repeat while covering the **sender's** camera to lose
   ACKs instead. Stay below the retry/deadline limits for this recovery test.
3. Test **Cancel transfer** during an attempt. A peer only learns cancellation
   if its camera sees that packet; otherwise it must eventually time out.
4. Reverse sender and receiver. The two directions may prefer different DATA
   presets; a return-path control QR remains small in either direction.

Save JSON reports for successes **and failures**, paired by device and attempt.
Include total bytes, settings, which device sent, whether both sides confirmed,
elapsed time, goodput, retries, any error text, and changes to position or lighting.

## 4. Local regression checks

Run **Simulation** with the same preset used for the physical comparison and its
seeded faults. Start with 4 or 16 KiB; use 165,000 bytes once that setting passes.
Success requires both exact payload recovery and sender confirmation. Export its
report separately from physical tests. Simulation uses the same presets and
capacity checks, and reports the effective settings as well as the preset name.

From the project directory, run `npm test` for automated checks and
`npm run benchmark:speed` for a seeded 16 KiB simulation across the five presets.
The original `npm run benchmark` remains available for the 165,000-byte run.
Simulation uses rendered QR pixels, not a mock decoder, but it does not
model moiré, glare, autofocus, motion blur, or screen/camera interactions. A local
pass establishes a useful baseline, not a claim about device performance.
