# Source inventory and recovery

This GitHub import was assembled on **19 October 2026** after automated workspace
maintenance removed the earlier local files. Both app repositories were recovered
from their saved Git source; their sites remain blank. The import records what
was recovered instead of recreating missing files and calling them originals.

## Recovered source

| Repository location | Latest recovered source | Notes |
| --- | --- | --- |
| `apps/color-optical-bench/` | Optical app v0.9.0, source commit `ad7ec85423a023c207d4bf0c5f7a1ef962260b52` | Includes color/image labs, multipart transport, compact v2 browser Crypto Lab, tests, fixtures, and validation records. |
| `apps/qr-neural-key/` | QR app v0.3.0, source commit `b9ee8f9b43b5e95806dbba0598888758439a4d62` | Includes latest directional profiles, adaptive transport, QR authentication, tests, fixtures, and validation records. Last source commit only parks hosting; executable app source is retained. |
| `experiments/python-zkboo/neural_zkboo_reference.py` | Python core from the optical repository above | Moved byte-for-byte; fixture importer updated for the monorepo. This is the compact v2 crypto extraction, not the complete original CLI. |

Deployment identifiers, blank `out/` pages, Git metadata, and credentials are not
part of the import. Application source remains in each app's `dist/` folder.
Project-wide docs and package commands were added; app documentation was adjusted
for local paths and current status. The crypto and transport implementations are
preserved. Public deterministic test fixtures and third-party licenses are kept.

## Original files still needed

Neither recovered repository's tracked history contained these originals:

- `neural_key_experiment.py` — floating-point CNN/wave and Feistel comparisons,
  rotating convolution, architecture probes, camera simulation, configurable
  palettes/sizes/frame counts.
- `neural_zkboo_experiment.py` — full Python ZKBoo command-line experiment.
- `neural_optical.py` — shared Python optical utilities.
- `test_neural_pocs.py` — original Python POC tests.
- `neural_waves.ipynb` — the original dynamical-systems notebook.

The earlier project archives may contain them, particularly
`neural_key_project_reference_2026-10-17.zip` or `neural_key_compact_v2.zip`.
Those archives were not available in the recovered workspace. Reattach the
latest originals or an archive to finish this part of the import. The directory
guides in `experiments/neural-dynamics/` and `notebooks/` are explicit placeholders,
not replacement implementations.

The NumPy crypto core is sufficient for the current Python↔JavaScript proof
parity tests, but cannot replace the original floating-point experiment or its
camera/command-line tooling.

## Device evidence

Historical app validation artifacts are retained under each app's `validation/`.
The 17 October 2026 successful QR authentication is described from the verifier
report already inspected in the project conversation: 32 repetitions, 17,474
proof bytes, acceptance at approximately 930.546 seconds, and 30.7 ms local
verification. Its original JSON attachment was removed from this workspace by
maintenance and is not included in this import. Do not treat the summary as a
new measurement or as a replacement raw report.
