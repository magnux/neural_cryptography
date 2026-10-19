# Project status and next experiments

The original five-step plan was a guideline. Static browser apps replaced the
proposed Flask enrollment service, and local key generation replaced server-side
key distribution. Cryptography, optical transport, and their integration remain
separate experiments so each can be measured independently.

## Current state

| Area | Implemented or observed | Remaining question |
| --- | --- | --- |
| Neural dynamics | Recurrent CNN, rotating-kernel, initialization, and Feistel experiments were run during development. Their original script and notebook are missing from the recovered snapshot. | Recover the original files before extending or reproducing those experiments. |
| Python and browser cryptography | Exact binary circuit, compact v2 ZKBoo-style proof, configurable repetitions, worker-based browser lab, reference fixtures, and parity/tamper tests. | Continue independent analysis; passing tests is not a security proof. |
| Color optical bench | Camera geometry, palette references, phased transmission, progressive sample selection, error recovery, content decoding, and image-lab simulations. | Achieve reliable bidirectional throughput without excessive repeated observations. |
| Color multipart transport | Local fault simulator and a two-device multipart transfer page. | The successful single-matrix observations do not establish a complete proof-sized physical transfer. |
| QR transport | Multipart transfer, acknowledgments, configurable per-device outgoing settings, and adaptive fallback. | Screen/camera geometry and slow return traffic still dominate some sessions. |
| QR authentication | Full pairing, challenge, proof transfer, verification, and result acknowledgment completed in device testing with 32 proof repetitions. | Reduce end-to-end latency and repeat the test reliably across device pairs. |
| Learned optical decoding | Architecture and training approach proposed. | No model, dataset generator, training script, or dedicated collection app has been implemented. |

The recovered source versions and omissions are listed in
[SOURCE_INVENTORY.md](SOURCE_INVENTORY.md). Historical hosted demos were blanked;
this repository preserves their source for local use and further development.

## What the device observations establish

These are reported development observations, not a controlled benchmark suite:

- Color matrices were successfully decoded at **128×128 cells with 16 colors**
  using the accumulated transport improvements. This does not establish a
  sustained application-data rate, and 256×256 remained unreliable on the tested
  setup.
- Favorable QR transfer tests reached roughly **1 kB/s**. This is a useful
  comparison target, not the throughput obtained in every geometry or direction.
- The reported complete 32-repetition authentication used a **17,474-byte raw
  proof** and took approximately **15 minutes 31 seconds**. Local verification
  took about **30.7 ms**. The verifier report showed a final incoming profile of
  **64-byte chunks with one chunk per acknowledgment**.

The slow authentication run and the earlier favorable QR transfer tests measure
different conditions. Its proof-waiting interval includes challenge delivery,
proof generation, and proof delivery; it is not a pure channel-throughput
measurement. The PC report alone also cannot reconstruct the phone's full
sequence of adaptive profile changes.

## Next: learned decoding of color matrices

Start with a small fully convolutional U-Net that predicts a distribution over
palette symbols for every logical cell. Its main objective is symbol accuracy,
not visual smoothness. The existing CRC and error-correction layers remain the
checks on recovered bytes.

Distinguish the input raster from the logical matrix. At four rectified pixels
per cell, a 128×128 matrix supplies a 512×512 image. A 512×512 matrix would supply
a 2048×2048 image and substantially more computation. Shared convolutional weights
can support multiple spatial sizes; separate inlet branches or tiled processing
should be added only when their benefit is measured.

| Milestone | Deliverable | Evidence required before proceeding |
| --- | --- | --- |
| 1. Browser feasibility | Export a small untrained 2D model and benchmark it in a worker on the actual phone and PC. | Successful inference at intended input sizes, numerical comparison with Python, latency, and responsive UI. |
| 2. Capture tooling | Collect labeled bursts using known transmitted payloads, frame/phase identifiers, and existing geometry detection. | Failed payload decodes still produce correctly labeled examples; exports preserve camera images and rectification metadata. |
| 3. Single-frame baseline | Synthetic generator, plain PyTorch training, real-data fine-tuning, and model export. | Better held-out symbol/frame recovery than the existing decoder on session-disjoint captures. |
| 4. Temporal fusion | Compare one image with five images stacked as 15 channels; evaluate feature fusion if useful. | Improved verified throughput after accounting for the extra capture and computation time. |
| 5. Transport integration | Feed probabilities or erasure confidence into existing recovery and repeat complete message transfers. | End-to-end application goodput, failure rate, and transfer latency compared with QR on the same devices. |

The first target is **64×64 and 128×128 matrices with 16 colors**. Train on the
project owner's RTX 4090; browser devices perform inference. The intended runtime
path is PyTorch to ONNX Runtime Web, testing WebGPU and CPU fallback early.
Five-frame channel stacking uses ordinary 2D convolution. Genuine 3D convolution
and transformer variants are optional later comparisons, subject to actual
export/backend support and measured device cost.

### Data requirements

Synthetic examples should include spatially varying moiré, screen subpixels,
camera sampling, defocus, motion blur, color/exposure changes, noise, and clipping.
Curved moiré does not require a geometrically bent matrix. A burst must contain
correlated captures with changing focus, motion, and moiré phase, rather than five
independent noise overlays.

Real-device captures should retain lossless camera images or crops, per-frame
rectification, timestamps, palette/phase information, and reproducible labels.
Browser camera RGB is already processed by the camera; it is not sensor RAW.
Payload CRC success must not be a condition for retaining an example, otherwise
the dataset would omit the failures the model needs to learn from.

Use complete capture sessions for train/validation/test splits. Do not split
neighboring frames between training and evaluation. Keep any payload-generation
seed outside the model input, including masking diagnostic metadata if it exposes
that seed. Temporal fusion must group the same payload and handle palette phases
explicitly rather than averaging incompatible symbol colors.

### Success criteria

Measure cell errors and exact frame recovery, but judge the complete system by
**CRC-verified application bytes per second**, including capture, repeated phases,
inference, error correction, acknowledgments, and retries. Record both transport
directions, tail latency, and failed attempts. A high average cell accuracy can
still leave many errors in a large matrix.

Use the earlier approximate **1 kB/s QR result** as the initial comparison target.
Neither a synthetic accuracy improvement nor beating only the unusually slow
authentication run is sufficient evidence that the new transport is better.

## Ideas deliberately deferred

- Persistent enrollment and mutual authentication within a single session.
- Temporary session credentials or OTPs for previously authenticated devices.
- Verifier-owned repetition policies based on prior successful authentication.
- Chained challenges or proofs, including the proposed XOR construction.
- Hidden-inverse asymmetric architectures, dynamical layers, or LoRA mixtures.
- A transformer or hybrid decoder if the simpler convolutional baseline leaves
  a measured limitation.

These are research directions, not implemented features or established security
properties. In particular, a previous long proof does not automatically strengthen
a subsequent short proof, and optical proximity does not replace cryptographic
analysis.
