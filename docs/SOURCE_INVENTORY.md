# Current source inventory

One current implementation is maintained for each component. Source archives,
superseded protocol implementations, and old benchmark reports are not retained
in the working tree; prior repository states remain in Git history.

| Location | Current component |
| --- | --- |
| `experiments/neural-dynamics/` | Restored PyTorch recurrent CNN/rotating-convolution/Feistel experiment. |
| `experiments/python-zkboo/` | Restored command-line experiment using the current NumPy compact ZKBoo core; no archived proof implementation alongside it. |
| `shared/python/neural_optical.py` | Shared Python palette, packing, and simulated-camera utilities. |
| `shared/browser-crypto/crypto/` | Single browser binary circuit, compact prover/verifier, byte helpers, and licensed SHAKE dependency. |
| `shared/browser-crypto/worker/` | Single reusable crypto worker/client/operations and public reference fixture. |
| `shared/fixtures/zk/` | Current deterministic Python↔JavaScript proof fixtures. |
| `apps/color-optical-bench/` | Current OBM5 optical/image bench, Crypto Lab UI, and color multipart transport. |
| `apps/qr-neural-key/` | Current QR transport and authentication, including independent outgoing settings and adaptive fallback. |
| `tests/python/` | Python experiment tests. |
| `notebooks/README.md` | Link to the existing Colab notebook; no duplicate `.ipynb`. |

Run `npm run prepare:apps` to materialize ignored shared-code copies in each
app's served/test directories. Those files are build output, not separately
maintained implementations. Neither preparation nor tests deploy a site.

## Source selection

The two supplied archives were checked against their manifests and SHA256SUMS;
all 22 checksum entries and all 14 nested ZIP integrity checks passed. Both
archives contain the same six Python experiment files. These restored the
missing command-line experiments, utilities, and tests.

Their browser snapshots predate the current repository's compact proof engine
and QR authentication/directional adaptation. The existing browser source was
therefore retained. The archived Python ZKBoo command-line shell was adapted to
the current `NeuralPicnic-POC-MPC3-v2` / `NPZK0002` core; the earlier serializer,
prover, and verifier were not imported.

For provenance, the recovered current browser sources came from optical source
commit `ad7ec85423a023c207d4bf0c5f7a1ef962260b52` and QR source commit
`b9ee8f9b43b5e95806dbba0598888758439a4d62`. These are source-origin references,
not alternate runnable versions included here. The compact NumPy core was
already present before restoring the older CLI shell.

See [FORMATS.md](FORMATS.md) for the current layer identifiers,
[VALIDATION.md](VALIDATION.md) for checks, and [ROADMAP.md](ROADMAP.md) for device
observations and future work.
