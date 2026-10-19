# Repository import validation — 19 October 2026

Validation after organizing the recovered sources:

| Check | Result |
| --- | --- |
| Color optical bench Node suite | 193 passed, 0 failed |
| QR transport/authentication Node suite | 223 passed, 0 failed |
| Python fixture regeneration | 7 fixture files reproduced |
| Deterministic Python/JavaScript proof parity | 5 exact proof-byte matches |
| Python-generated proof verification in JavaScript | 5 accepted |
| Fresh JavaScript proof verification in Python | 28 cases: 5 accepted, 23 intentionally invalid cases rejected |
| Python README example | Generated and verified a 17,474-byte, 32-repetition proof |
| Relative Markdown links | All targets present |
| QR and SHAKE dependency provenance | Bundled checksums and licenses retained |

Environment: Node.js 24.19.0, Python 3.12.14, NumPy available. The Python core
was moved without changing its contents; its fixture importer was updated.
Application crypto/transport implementations were not refactored in this import.
Existing upstream vendor whitespace and source checksums were preserved.

These checks do not repeat physical phone/webcam tests. Earlier simulation
records remain under the individual apps' `validation/` directories, and device
observations are distinguished from automated checks in [ROADMAP.md](ROADMAP.md).

Reproduce the main checks from the repository root:

```sh
npm test
npm run test:parity
```
