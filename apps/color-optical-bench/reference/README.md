# Python reference location

The canonical NumPy implementation now lives in
[`experiments/python-zkboo/`](../../../experiments/python-zkboo/README.md).
Its implementation was moved byte for byte; no duplicate copy is kept here.
See that README for a round-trip example, dependency setup, source provenance,
and parity-fixture instructions.

The browser app continues to use `dist/js/crypto/`. Its
[`scripts/zk-fixtures.py`](../scripts/zk-fixtures.py) imports the canonical Python
module; `npm run test:parity` runs the bidirectional compatibility checks.
