# Working on these experiments

Read the README and the applicable app's AGENTS.md before editing.
Keep each layer on one current format; do not add compatibility readers or
migrations for earlier POC formats. Update both endpoints and fixtures together.

Keep computation in the browser, bounded queues/buffers, and heavy work in
workers. Diagnostic ground truth must not enter decoding or error correction.
Never replace secure runtime randomness with the public deterministic test RNG.

The color and QR apps deliberately remain independently runnable snapshots.
Their copied browser crypto modules currently share an implementation. When
changing crypto, update both copies and run the Python parity tests; avoid
introducing an unrelated module-layout refactor during transport experiments.

Run the affected existing tests. Root `npm test` runs both apps; root
`npm run test:parity` requires Python with NumPy. Do not deploy or reactivate
the parked sites as a side effect of repository maintenance.

Original standalone Python experiments and the neural-waves notebook are missing
from this import. Do not reconstruct them and label them as recovered originals.
See docs/SOURCE_INVENTORY.md.
