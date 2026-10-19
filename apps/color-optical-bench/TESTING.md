# Device tests for v0.6

Reload both devices. Confirm **OBM5 · v0.6** in the footer. Saved matrices from
another release are unsupported. There is no startup calibration sweep. Reset the
receiver before each measured run, including after a new sequence, playback rate,
or receiver setting change.

## Start with this profile

| Setting | Value |
|---|---|
| Content | Diagnostic → Pseudorandom bytes, seed 1234 |
| Matrix / palette | 64×64 / 16 colors |
| Logical content frames | 1 |
| Reference grid | Every 8 data cells |
| Error correction | RS · 32 parity bytes / block |
| Phases / colors | 8 / Palette permutations |
| Playback rate | 0.25 phases/s, 4 seconds each |
| Requested camera | 1080p, rear camera where available |
| Cell sampling | Five-point average |
| Combine evidence | Robust subset + mean control |
| Minimum samples / overhead | 5 / +2 |
| First attempt after | 4 sampled phases |
| Recovery | RS + uncertain-byte erasures |

Press **Play** on the transmitter. Keep the entire matrix, four markers, and both
palette strips visible. Check actual camera resolution and projected pixels per
cell, since the device can negotiate different capture settings. Try to have at
least four camera pixels per cell; this is a starting condition, not a guarantee.

Watch the phase counts. A phase becomes eligible at five samples; the first fused
attempt starts when four distinct phases are eligible. Once the seven-slot pool
fills, new captures replace old ones. Failure should keep reception in a refining
state, rather than stop acquisition. If too few samples arrive during a phase,
reduce processing density or lengthen the phase hold, then reset for the new run.

## A small, ordered test set

Use **five fresh runs per condition**, or ten if practical. Give each run the same
**90-second acquisition budget** and record success/failure plus time to first
verified content. Do not count repeated CRC passes during one run as independent
successful trials. Alternate condition order where practical so focus, lighting,
and phone temperature do not consistently favor one setting.

1. **Baseline, supported phone:** run the profile above with the phone supported
   and the matrix filling the same region of the camera image. Export a report
   from at least one success and one failure, if both occur.
2. **Normal handheld movement:** repeat without changing the profile. Small natural
   movement and autofocus changes are useful stressors; keep geometry visible.
3. **Four / eight / sixteen phases:** compare all three with the same 90-second
   budget, phase hold, sample target, and first-attempt threshold of four. A full
   sixteen-phase cycle is 64 seconds at this rate. More phases do not need to
   delay the first attempt, and they do not add content bytes.
4. **Ordinary RS / erasure-assisted RS:** use the better phase count from step 3.
   The report already separates passes obtainable with ordinary RS from extra
   erasure-assisted passes. A separate ordinary-RS-only run checks whether the
   added computation affects acquisition rate on the phone.
5. **128×128:** repeat the best 64×64 condition, keeping 16 colors and other settings
   fixed. Return to 64×64 if headers or geometry become the limiting factor. Only
   then try higher camera resolution as a separate change.

The robust mode reports ordinary mean and robust selection on **the same retained
observations**. Use those rows to compare estimators without running a second,
different camera trajectory. Mean-only mode is useful for measuring processing
cost; its five-sample pool is not the same estimator as the seven-observation mean
control inside the default robust mode.

For an additional recovery test, begin with an awkward but detectable angle or
slight blur, then improve alignment without resetting. Note whether old incorrect
evidence clears and decoding recovers. Keep it separate from the fixed-condition
runs: this tests adaptation, not steady-state success probability.

## What to report

Export the **live JSON report** after each useful run. It excludes camera images
and received content. Add phone model, browser, screen type/curvature, distance,
display brightness, room lighting, and supported/handheld movement to Test notes.

If exporting every run is inconvenient, send this compact record:

| Field | What to record |
|---|---|
| Profile | Size, colors, grid, parity, phases/mapping, phase hold, logical frames |
| Receiver | Requested minimum, overhead, first-attempt threshold, recovery, sampling |
| Effective limits | Actual target/pool, retained phase limit, memory use/reductions |
| Capture | Actual camera resolution, projected pixels per cell, phone/browser |
| Success | Complete messages / fresh runs, and time budget per run |
| First verification | Seconds, capture number, and phase sample counts at first success |
| Error measurements | Mean and robust border/local raw symbol-error rates |
| Recovery | Raw CRC passes, ordinary RS passes, extra erasure passes, attempts |
| Failure behavior | Missing headers/phases, persistent color errors, slow processing, or resets |
| Conditions | Screen/brightness, distance, movement, and visible autofocus adjustments |

For multi-frame messages, first verified timing refers to the first recovered
logical frame. Also record seconds to **complete content** and recovered/expected
frame count. Run one → four → eight content frames only after a one-frame profile
works reliably. Use unique verified bytes per elapsed time when comparing useful
throughput; parity, reference spacing, and extra phases all have costs.

The report stores first verified capture/time, phase counts, confidence/selection
summaries, bounded recent history, and effective retention limits. Ordinary RS
passes include raw-valid candidates. Extra erasure passes count successful
candidates that ordinary RS could not verify. The latest state is not necessarily
the state at first success; use the first-success fields or report history.

## Image lab

Use the baseline optical profile and these stressors:

- Additive noise **12**, moiré strength **44**, band spacing **16**.
- **Curved** bands and **Color-dependent amplitude**.
- **Robust**, minimum **5**, overhead **+2**, first attempt after **4**, erasure recovery.
- **4 extra captures / phase**, **1 complete cycle**, camera **Auto**, seed **1234**.
- Leave the identical-repetitions/permutations A/B checkbox enabled.

Five minimum plus four extra captures gives nine observations per phase. A
seven-slot pool fills, then replaces its two oldest samples. Try two complete
cycles to exercise revisiting phases. The test continues after successful recovery
so repeated and permuted variants receive equal requested exposure budgets.

Compare **Drift between exposures** with **Keep interference fixed**. Then add
**0.6 px** blur and **±10%** gain swing separately. Curvature bends the moiré bands,
not matrix geometry. Drift changes interference phase and the selected blur
strength. Noise seeds and disturbance trajectories match between the two mapping
variants. Mean and robust comparisons share the same captures within each variant.

For four/eight/sixteen-phase lab comparisons, equal captures **per phase** means
unequal total work. Compare at a shared total capture budget or inspect first
verification at matched capture numbers. A single end-of-run pass percentage is
not a fair efficiency comparison. Lab elapsed time is processing time, not an
optical transmission duration.

The simulation does not reproduce autofocus, rolling shutter, display PWM, HDR,
or physical sensor/display aliasing. Synthetic success cannot establish a physical
128×128 capability. Device results decide whether these changes help.
