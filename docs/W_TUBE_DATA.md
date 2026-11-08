# W-tube data: reuse recordings and generate paired synthetics

This pipeline prepares the next **4×4-pixel-cell, G16/G32/G64/G128** corpus.
Collection, data preparation and **W-tube training/evaluation are implemented**.
Use [the training commands](NEURAL_CORRECTOR_TRAINING.md#train-w-tube) after rebuilding.
Origin-aware sampling and lean training/evaluation reports are implemented;
browser export remains a separate step. Existing `multires` checkpoints use two
pixels per cell and cannot consume this corpus.

## Three recording roots, one dataset contract

Keep the recording directories as they are. The importer reads each round using
its explicit source contract and writes `neural-qr-dataset`, version 2:

| Default root | Origin tag | Eligible recordings | Input conversion |
| --- | --- | --- | --- |
| `local-data/recordings/` | `native` | Current Neural-QR v2, G16/G32/G64/G128 | Preserve native `[3,4G,4G]` camera crops. |
| `local-data/recordings-1/` | `resampled` | Neural-QR v1, G16/G32/G64/G128 | Bicubic enlargement from `[3,2G,2G]` to `[3,4G,4G]`. G8/G256 are skipped. |
| `local-data/recordings-0/` | `header-g64` | Original OBM G64, 16 colors, four phases and grid-8 references | Preserve the existing `[3,256,256]` payload crop; the old monochrome header is outside it. |

The first round retains its `pc/train`, `pc/val`, `phone/train` and `phone/val`
folders. The importer maps `val` to `validation`; it does not move recordings.
The later rounds use `<train|validation>/g<size>/<sender|receiver>/`.
Completed JSON files are considered; unfinished `.json.tmp` senders and active
collection descriptors are excluded. Missing roots are allowed, so a build can
start with the available rounds; an empty collection root does not require a
dummy session. Record new sessions only in `local-data/recordings/`.

Bicubic enlargement makes tensor dimensions compatible; it does **not** restore
camera detail. `resampled` and `header-g64` remain auxiliary origins. W-tube evaluation uses fresh native real validation as its primary result when
available and reports auxiliary origins separately. Without native candidates,
it explicitly labels the result auxiliary-only. Do not pool them into a single headline
score that could hide native failures.

The importer preserves sender and receiver physical-session identities, split,
phase, capture grouping and device/camera provenance. Reference cells are input
evidence but excluded from the supervised data-cell loss. Existing per-session
sample counts can differ; there is no requirement to fill every collection pool.
Samples from one physical session or one logical target cannot cross splits.
A rebuild needs at least one capture of every phase per target. The plan flags
targets that lack two per phase with `someTargetsLackK8`; they remain usable at
lower observation counts, but cannot supply a full K8 example. Captures are
never duplicated to fill a missing observation.

The current W-tube trainer samples all K4…K8 budgets and therefore requires
**at least two captures of every phase per target**, in both training and
validation. A one-per-phase target can be retained by the importer, but is not
ready for this training command. Collect more captures or leave that incomplete
target out of the training corpus. Ordinary completed 8-, 16- and 32-per-phase
pools exceed this minimum.

Original OBM message bytes are not treated as Neural-QR symbols. The offline
import reconstructs the original frame from the sender manifest, resolves its
phase and frame identity, then stores the corresponding logical matrix target.
Only offline supervision sees sender truth. No compatibility reader is added to
the collector or live authentication receiver.

## Build and rebuild

From the repository root, activate the training environment and install its
existing dependencies if needed:

```sh
python -m pip install -r experiments/neural-corrector/requirements.txt
```

Inspect the plan, then run the same build:

```sh
node scripts/rebuild-corrector-data.mjs --plan
node scripts/rebuild-corrector-data.mjs --python python
```

The default roots are `local-data/recordings`, `local-data/recordings-1` and
`local-data/recordings-0`; output goes to `local-data/w-tube`. No path flags are
needed for that layout. To override locations, use explicit roots, for example:

```sh
node scripts/rebuild-corrector-data.mjs \
  --recordings-root /data/recordings \
  --resampled-root /data/recordings-1 \
  --header-root /data/recordings-0 \
  --out local-data/w-tube --python python --plan
```

Remove `--plan` to build. The plan reports eligible source counts, skipped
recordings, synthetic counts and estimated disk requirements. Retain the same
roots and output name for later rebuilds after collecting more sessions.

Progress goes to stderr: source scanning/checking, capture import, clean-target
rendering, synthetic generation, baseline preparation, and final installation.
Long sample loops print counts periodically, roughly every five seconds, and at
completion. The final build summary stays JSON on stdout, so it can be redirected
without swallowing progress messages.

Each component has `<train|validation>/manifest.json`, containing the supported
grids present in that split. The result has these components:

| Component | Contents |
| --- | --- |
| `real/` | Imported native and auxiliary captured evidence, with explicit origin and provenance. |
| `clean/` | Four exact palette renders per target, one per phase, used only to generate synthetic inputs and labels. |
| `synthetic/` | Corrupted renders of those same targets, preserving paired target/session/split identities. |
| `build.json` | Build settings, source summary, generation counts and the managed-output marker. |

Binary tensors are stored separately from manifests. Recordings and dataset
manifests are read/written incrementally; the JSON format is unchanged. Large
manifest files no longer hit the former 256 MiB limit during baseline preparation
or training. Metadata indexes still consume memory, but the build does not
assemble the whole image corpus or a giant serialized JSON string in RAM. The
rebuild reads the generator's small completion summary instead of reopening the
synthetic manifest merely to count samples. This remains an offline disk workload
and can be large, especially G128.

A rebuild stages a new dataset and replaces the previous managed output only
after the build succeeds. It refuses to replace an unrelated or unmarked output
directory. Recordings and checkpoints are never deleted. Allow space for old and
new outputs simultaneously, and stop training before replacing a dataset it is
reading. The three recording roots must be separate from the output directory. The
`build.json` marker is `neural-corrector-w-tube-build`, version 1; do not hand-edit
an unrelated dataset to acquire that marker.

The synthetic validation set is generated for diagnostics. It does not replace
held-out native real validation or become part of the training split.

### Individual commands

For an imported real split, the clean and synthetic stages can also be run
separately into new or empty output directories:

```sh
node scripts/corrector-dataset.mjs targets \
  --dataset local-data/w-tube/real/train --out local-data/w-tube-clean-train
python experiments/neural-corrector/synthetic.py \
  --clean local-data/w-tube-clean-train \
  --paired-real local-data/w-tube/real/train \
  --recipe w-tube --synthetic-ratio 4 --frames 8 \
  --seed 41 --out local-data/w-tube-synthetic-train --plan
```

Remove `--plan` from the Python command to generate. Its plan includes requested
and achieved per-target counts and ratios. The managed rebuild is the simpler
option when combining all three recording rounds.

## Four synthetic captures per real capture

The ratio is based on **individual stored captures**, not file size, matrix
area, model examples or optimizer steps. With the default `--frames 8`, one
synthetic trajectory supplies eight captures of each of the four phases: 32
captures total. For a target with R real captures, generation uses:

\[
B = \left\lceil\frac{4R}{4F}\right\rceil,\qquad
S = 4FB,\qquad F=8.
\]

Here B is the number of trajectories and S is the synthetic capture count. The
rounding preserves complete four-phase bursts; the reported actual ratio may be
slightly above 4:1. For example, 128 real captures produce 512 synthetic captures
in 16 trajectories. With 64 real captures, generation produces 256 in eight.
`--frames` still means captures **per phase**, not the number of model input
slots. K4–K8 model examples will draw from these captures during training.

The intended training mix remains **50% real and 50% synthetic** with paired
targets. A larger synthetic corpus provides more corruptions of each target;
it does not make the training sampler use 80% synthetic inputs. W-tube defaults
to `--native-real-fraction .75` within each grid/direction real quota when both
native and auxiliary recordings are available. The auxiliary share is divided
equally between available `resampled` and `header-g64` origins. Missing origins
redistribute their share among those present; a direction without native data
therefore lowers the overall native fraction. Native-first validation remains
independent of this training policy. See [sampling details](NEURAL_CORRECTOR_TRAINING.md#w-tube-sampling-policy)
for exact quotas, coverage and reported counts.

## Harder, coherent corruptions

The `w-tube` recipe combines camera-resolution loss, curved moiré, noise,
white-balance and exposure changes, shadows/reflections/glare, rolling shutter,
linear and rotational/radial motion blur, and small RGB-channel blur offsets.
All four phase renders follow a shared camera/environment trajectory, with
frame variation. This preserves the relationship a phase-grouped model needs.
The default severity is sampled uniformly from **0.5 to 1.0** per trajectory.
This keeps a range of examples rather than forcing every corruption to its
maximum. Recipe settings can be overridden explicitly; the rebuild defaults are:

| Option | Default | Meaning |
| --- | --- | --- |
| `--synthetic-ratio` | `4` | Requested synthetic captures per real capture, per target. |
| `--frames` | `8` | Captures per phase in one coherent trajectory. |
| `--severity-min` / `--severity` | `.5` / `1` | Range of trajectory severities. |
| `--seed` | `41` | Reproducible generation seed. |
| `--max-output-mib` | `1048576` | Aggregate estimated output budget; `0` disables this budget check. |

The capture-resolution stage can reach one camera pixel per logical cell before
returning to four pixels per cell. Upsampling that degraded image reproduces
information loss, not recovered detail. The aim is broader camera variation,
not making every synthetic unreadable.

Clean inputs come from the actual sender targets, including targets recovered
from auxiliary recordings. Synthetic corruption never starts from a denoised
real capture. Classical scores are recomputed from each corrupted image without
access to its clean target. Output metadata records the recipe, random seed,
paired target and source origin so data generation is repeatable and auditable.

See [the collection guide](MULTIRES_COLLECTION.md) for more native recordings,
[the W-tube architecture](NEURAL_CORRECTOR.md#w-tube), and
[the training guide](NEURAL_CORRECTOR_TRAINING.md#train-w-tube) for training and
evaluation commands. Separate prepared U-tube and two-pixel multires experiments
remain documented there.

