# Experiments and use cases

This repository brings together several related proofs of concept. The neural
circuit, the proof protocol, and the optical transport are separate layers:
changing the camera transport does not require changing the proof relation.
These experiments do not establish a production cryptographic construction or
a claimed post-quantum security level.

The immediate delivery goal is the [color authentication POC](ROADMAP.md): reuse
the current Neural ZKBoo engine and authentication flow over color multipart
transport, with the [learned optical decoder](OPTICAL_DECODER.md) improving
camera reception. The existing experiments are its exploratory foundation.
[Cryptanalytic comparisons](RESEARCH_CONNECTIONS.md) remain future research.

## Choose an experiment

| Question | Entry point | Current scope |
| --- | --- | --- |
| Can a recurrent CNN preserve diversity and diffuse small input changes? | [Neural dynamics experiment](../experiments/neural-dynamics/README.md) | Runnable PyTorch comparison, avalanche/bias/surrogate probes, and simulated camera input. |
| How does the exact binary circuit and its ZKBoo-style proof work? | [Python reference](../experiments/python-zkboo/neural_zkboo_reference.py) and [crypto format/API](../apps/color-optical-bench/CRYPTO_PARITY.md) | Current NumPy compact proof core plus the command-line provisioning, proof, verification, and camera simulation experiment. |
| Can the browser generate and verify proofs without a Python backend? | [Crypto Lab](../apps/color-optical-bench/CRYPTO_LAB.md), page `crypto.html` | Local key generation, proof round trips, tamper checks, public-only verification, and performance reports. |
| Can colored matrices carry arbitrary bytes between cameras and screens? | [Optical Bench](../apps/color-optical-bench/README.md), page `index.html` | Rendering, capture, rectification, palette phases, temporal sampling, recovery, and synthetic image tests. |
| Can color matrices carry a complete multipart message? | [Color multipart transport](../apps/color-optical-bench/LINK_TRANSPORT.md), page `link.html` | Local fault simulation and bidirectional camera transfer. This page does not perform authentication. |
| How much data can standard QR codes carry reliably on these devices? | [QR transport lab](../apps/qr-neural-key/README.md), page `index.html` | Multipart bytes, outgoing profiles, pacing, acknowledgments, retries, and measurements. |
| Can a device authenticate over an optical link? | [QR authentication](../apps/qr-neural-key/AUTHENTICATION.md), page `auth.html` | Local ephemeral keys, human pairing confirmation, a fresh challenge, optical proof transfer, and local verification. |
| Can the planned decoder run on these devices? | [Decoder Lab](DECODER_LAB.md), page `decoder.html` | Untrained U-Net, full-logit Python parity, worker latency/UI reports, and validated camera crop inference. Learned recovery and training are still planned. |

Paths to HTML pages are relative to the selected app's `dist/` directory. Both
apps can be served as static files after `npm run prepare:apps`; Decoder Lab additionally needs its optional
model/runtime preparation. Camera access on a phone requires a secure
origin; see each app's serving instructions. Historical test deployments were
blanked, so the repository is the starting point for running these experiments.

## Neural dynamics versus the proof circuit

The original neural dynamics work explored recurrent convolution, normalization,
activation functions, orthogonal initialization, rotating kernels, and a Feistel
construction. Its probes measured effects such as avalanche behavior, output
bias, covariance dimension, numerical agreement, and learned surrogate accuracy.
Good results on these probes are interesting experimental observations; they
do not demonstrate resistance to cryptanalysis.

The current proof engine instead uses an exact binary rotating-convolution
Feistel circuit. It runs with byte and Boolean operations in Python and
JavaScript, without PyTorch or a browser neural-inference runtime. Its neural
terminology should not be confused with the planned image-decoding network.
The original [Neural Waves notebook is published on Colab](../notebooks/README.md).

## Local proof experiments

Use Crypto Lab when the question concerns proof behavior rather than camera
performance. It can generate a key, create a challenge, prove, verify, export
a binary proof and public context, and verify those files on another device.
Secrets remain in the crypto worker's temporary memory.

Keep three controls separate:

- **Circuit rounds** change the relation being evaluated.
- **Proof repetitions** repeat the MPC proof experiment and increase proof size.
- **Transport settings** change how bytes reach the other device.

For the default side-8, 16-circuit-round profile, 32 proof repetitions produce
17,474 raw proof bytes, while 219 produce 119,249. Envelopes and optical framing
add overhead. Low repetition counts are useful for development, but are not
certified security presets. The verifier's requested count and the public
identity's minimum policy constrain acceptance.

## Optical authentication

The QR authentication demo is an offline-capable experiment once its static
assets have loaded. The server does not generate or distribute keys. Each side
generates an ephemeral key locally, the users confirm a matching pairing
fingerprint, and the verifier requests a proof tied to the current attempt.

This implements **one-direction authentication per attempt**. The key holder's
displayed result is a report from its peer, not a proof of the verifier's
identity. Reverse roles in a new attempt to exercise the opposite direction.
Persistent enrollment, remembered trust, encrypted payloads, temporary session
credentials, and automatic recency-based repetition policies are not implemented.

Optical proximity does not make messages secret. CRCs, Reed–Solomon recovery,
and transport digests address damage or missing data; the cryptographic protocol
is responsible for authentication. See the [authentication protocol](../apps/qr-neural-key/AUTHENTICATION.md)
for challenge binding, expiry, duplicate handling, and transport adaptation.

## Payload and wire formats

The color bench's content demonstrations cover raw bytes, UTF-8 text, PNG/JPEG
images, and PCM WAV audio. These content types sit above its optical framing;
they do not require a different cryptographic circuit.

| Layer | Current format documentation |
| --- | --- |
| Colored optical matrix | [OBM5 format](../apps/color-optical-bench/FORMAT.md) |
| Multipart color transport | [Link transport](../apps/color-optical-bench/LINK_TRANSPORT.md) |
| Binary proof and public context | [Compact v2 proof format](../apps/color-optical-bench/CRYPTO_PARITY.md) |
| QR packets and transport configuration | [QR transport lab](../apps/qr-neural-key/README.md) |
| Authentication envelope | [`neural-qr-auth-v1` / `NQA1`](../apps/qr-neural-key/AUTHENTICATION.md) |

The experiments accept their current formats only. Older encodings are not
supported through compatibility readers or migrations. See the
[source inventory](SOURCE_INVENTORY.md) for the current canonical components.
