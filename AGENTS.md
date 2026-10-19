# Working on these experiments

Read the README and the applicable app's AGENTS.md before editing.
Keep each layer on one current format; do not add compatibility readers or
migrations for earlier POC formats. Update both endpoints and fixtures together.

Keep computation in the browser, bounded queues/buffers, and heavy work in
workers. Diagnostic ground truth must not enter decoding or error correction.
Never replace secure runtime randomness with the public deterministic test RNG.

Keep one canonical implementation per component. Browser crypto and worker
logic live in shared/browser-crypto/, and proof fixtures in shared/fixtures/zk/.
Run npm run prepare:apps to materialize ignored app-local copies before serving.
Edit shared source, never generated copies. The two distinct optical transports
remain separate experiments; neither is a historical version of the other.

Run the affected existing tests. Root `npm test` runs both apps; root
`npm run test:parity` requires Python with NumPy. Run npm run test:python for
the Python experiments; neural-dynamics checks additionally require PyTorch.
Do not deploy or reactivate
the parked sites as a side effect of repository maintenance.

The original Python scripts have been restored and the ZKBoo CLI uses the current
compact proof core. Do not import archived protocol implementations alongside it.
The neural-waves notebook is linked to Colab, not copied into this repository.
Git history retains superseded source; do not add source ZIPs, versioned backup
directories, historical run logs, or compatibility parsers to the working tree.
