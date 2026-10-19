# Parked color-matrix POC

The working color-matrix, browser crypto, and multipart POC at commit
`b08289cb7a823daa62828831cd276cc3635535bd` was preserved on 2026-10-16 before
retiring the hosted demo. The source remains in `dist/`; its tests and neural
proof fixtures remain in this repository. `optical_matrix_bench_parked_step33a.zip`
is an export of that exact commit.

The hosting manifest now serves only `out/`. The former application assets
are not part of the served static directory. There is no unpublish operation
exposed by the hosting connector, so the old origin shows a retirement notice.

The separate QR transport experiment is developed in the `qr_neural_key`
project. No color-matrix compatibility path is added to it. The neural circuit,
ZKBoo implementation, and parity fixtures are preserved here for later reuse.
