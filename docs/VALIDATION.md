# Current validation — 19 October 2026

The following checks were run from a fresh copy containing only tracked source.
No previously generated app modules or private experiment results were present.
The npm hooks successfully prepared both independently servable apps from the
canonical shared modules and fixtures.

| Check | Result |
| --- | --- |
| Color optical bench Node suite | 193 passed, 0 failed |
| QR transport/authentication Node suite | 223 passed, 0 failed |
| Python experiment suite | 6 passed; 4 PyTorch-only checks explicitly skipped because PyTorch is not installed in this environment |
| Python syntax | All restored experiment, core, utility, and test files compile |
| Python fixture regeneration | 7 fixture files reproduced |
| Deterministic Python/JavaScript proof parity | 5 exact proof-byte matches |
| Python-generated proof verification in JavaScript | 5 accepted |
| Fresh JavaScript proof verification in Python | 28 cases: 5 accepted, 23 intentionally invalid cases rejected |
| Source selection | Both attachment manifests/checksum sets and all 14 nested ZIPs passed integrity checks |
| Repository layout | One browser proof core and one fixture set; no source ZIPs, copied notebooks, or old report directories tracked |

The NumPy tests include keygen/challenge/prove commands, compact proof size and
repetition policy, public-only verification, nonce consumption/replay rejection,
three circuit sizes, and Python optical framing/capture invariants. The restored
floating-point experiment retains its original numerical implementation; only
its shared utility import location changed. Its four PyTorch runtime tests still
need to be run in an environment with PyTorch installed.

The current browser core, worker modules, fixtures, and vendor bytes were moved
without algorithm changes. The restored ZKBoo CLI delegates to the current core
instead of retaining the archived prover/serializer. The Python-only NPM1
simulator uses its own domain label for sequence identity, independent of the
cryptographic protocol version.

These checks do not repeat physical phone/webcam tests. Device observations are
recorded separately in [ROADMAP.md](ROADMAP.md). Fresh benchmark output belongs
in ignored `local-results/`; old run logs remain in Git history.

Reproduce from the repository root:

```sh
npm test
npm run test:python
npm run test:parity
# Include all Python runtime checks:
python -m pip install -r experiments/neural-dynamics/requirements.txt
npm run test:python
```
