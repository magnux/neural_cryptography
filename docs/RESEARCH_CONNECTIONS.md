# Connections to neural cryptography research

Literature note · 18 November 2026

The project began with a question about dynamics: if the weights of a recurrent
convolutional network are the key, what happens to two inputs that differ by
one bit? That led to a second question: can a device prove it holds a key
without giving the verifier the key? A third arose in the phone demo: can a
camera reliably recover the bytes of that proof from another screen?

These questions lead to different bodies of prior work. **Pointcheval is an
early precedent for the identification goal; synchronization research is the
closest comparison for the dynamical intuition; ZKBoo and Picnic explain the
proof construction.** The trained receiver belongs to communications research.
The comparisons below describe conceptual connections, not claims that the
papers inspired the original experiment or establish its security.

| Question | Where the secret or learning lives | What this project does |
| --- | --- | --- |
| How does a fixed key mix different inputs? | In the original experiment, the secret parameters stay fixed while activations evolve. | The [Python dynamics experiment](../experiments/neural-dynamics/README.md) compares recurrent CNN and Feistel variants using diffusion, bias and numerical agreement. |
| How can a device prove it holds the key? | A private 256-bit vector supplies secret biases to a public, untrained Boolean circuit. | [Neural-ZKBoo](../experiments/python-zkboo/README.md) proves that one witness explains both public reference outputs and a response to this challenge. This is a separate circuit, not a quantized copy of the original CNN. |
| How can the verifier read the proof? | A trained neural corrector has public weights. Its inputs are rectified camera frames. | [Neural-QR](NEURAL_QR.md) carries the bytes. The corrector estimates symbols; the proof verifier decides whether to accept. |

## Neural identification before synchronization

David Pointcheval's [*A New Identification Scheme Based on the Perceptrons
Problem* (EUROCRYPT 1995)](https://www.di.ens.fr/david.pointcheval/Documents/Papers/1995_eurocrypt.pdf)
is a close precedent. The verifier knows a matrix and an unordered collection of
values. The prover knows a vector of signs that, multiplied by that matrix,
produces those values in some order. Pointcheval proposed proving knowledge of
such a vector interactively, with smart cards in mind. No one has to train a
network or establish a shared key during identification.

The overlap is the public statement: *I know a secret that explains these
public observations*. Pointcheval uses a permuted perceptron relation and a
specialized proof; Neural-ZKBoo uses outputs of a keyed circuit and a
ZKBoo-style proof. The distinction between the **proof** and the **difficulty
of finding a witness** matters. [Clark and Jacob (EUROCRYPT 2002)](https://eprints.whiterose.ac.uk/id/eprint/5547/)
reported that all the recommended sizes for Pointcheval's scheme were unsafe.
An adversary who finds a satisfying witness can run an honest prover. More
proof repetitions do not repair an easy underlying problem.

## Synchronization and fixed-weight dynamics

In [Kanter, Kinzel and Kanter (2002)](https://arxiv.org/abs/cond-mat/0202112),
two parties start with different private weights and use public inputs and
exchanged output bits to update them. Their weights synchronize (antiparallel
in the original convention), yielding material from which both can derive a
shared key. [Kinzel and Kanter's account](https://arxiv.org/abs/cond-mat/0208453)
places this in the broader neural cryptography setting. [Ruttor, Kinzel and
Kanter (2007)](https://arxiv.org/abs/cond-mat/0612537) model how the overlap
between weight vectors evolves under attractive and repulsive updates.

The apparent opposition between synchronization and diffusion disappears once
we specify **what changes**. In synchronization, two *weight states* should
converge. In our original experiment, one weight state stays fixed and nearby
*input states* should produce widely different outputs. The rising output bias
seen in some recurrent ReLU runs mattered because many challenges were being
mapped to similar responses. Good diffusion addresses that observation; it
does not establish that the transformation is hard to predict or invert
without the key. Likewise, convergence to a common key does not by itself
identify a peer.

[Klimov, Mityagin and Shamir (ASIACRYPT 2002)](https://weizmann.elsevierpure.com/en/publications/analysis-of-neural-cryptography-2/)
showed why a simple imitator could lag behind the participants yet the original
protocol could still be broken. [Seoane and Ruttor (2012)](https://arxiv.org/abs/1111.5792)
obtained a related result for a different, permutation-parity-machine protocol.
These findings concern specific schemes. Their wider lesson is that an observer
need not use the same learning rule as the participants.

## Learning encryption and learning to analyze ciphers

[Abadi and Andersen (2016)](https://arxiv.org/abs/1610.06918) asked whether
Alice and Bob could *learn to use* a supplied secret key to communicate while
a trained eavesdropper, Eve, failed to recover the plaintext. This is a
confidentiality experiment: the key is an input and training selects the
transformation. In our first POC the weights themselves played the role of the
key, and in the current POC the verifier wants evidence of possession, not a
decrypted message. The useful methodological lesson is to evaluate a frozen
design against independent adversaries; failing to train one Eve is limited
evidence.

[Gohr (CRYPTO 2019)](https://eprint.iacr.org/2019/037) put a network in a
different role: a distinguisher that found structure in ciphertext pairs from
**reduced-round** Speck32/64. This is a precedent for evaluating a cipher,
not a learned encryption method. Similarly, the trained surrogate in our
dynamics experiment is a diagnostic: chance-level bit prediction under one
test does not establish pseudorandomness or unforgeability.

## Proofs for Boolean circuits

[Giacomelli, Madsen and Orlandi's *ZKBoo* (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/giacomelli)
provides a way to prove a statement about a Boolean circuit by simulating
several participants' views of its computation and selectively opening them
(MPC-in-the-head). [Chase et al. (CCS 2017)](https://eprint.iacr.org/2017/279)
used this approach with a symmetric primitive to construct signatures:
the public key contains an image of a secret, and the proof binds knowledge of
a preimage to the message being signed. This explains the Picnic analogy.
Neural-ZKBoo is a custom authentication POC, not an implementation of Picnic.

In our [browser proof engine](../shared/browser-crypto/crypto/proof.js), public
reference inputs (“anchors”) have public outputs. The prover uses **one and
the same secret** for those reference pairs and the response to a fresh
challenge. The verifier checks that claim without needing the secret. In the
[mobile demo](NEURAL_AUTH_DEMO.md) the public key is exchanged for this attempt;
automatic pairing does not independently establish who owns that key.

Two circuit comparisons help explain design choices without implying equivalent
security:

- **SIMON:** its [round function](https://eprint.iacr.org/2013/404) also combines
  AND, XOR and rotations. Our Feistel round computes a nonlinear term
  `u XOR (v AND w)` from three secret-biased sparse XOR convolutions. The
  linear maps, secret-bit selection and spatial permutations differ.
- **LowMC:** [Albrecht et al. (EUROCRYPT 2015)](https://eprint.iacr.org/2016/687)
  design for low multiplicative cost. In our proof engine an AND gate adds
  transcript work that a public permutation or XOR does not. Circuit rounds
  change the transformation; proof repetitions change amplification and proof
  size. Optical grid size changes neither. See the [proof format](FORMATS.md)
  for concrete sizes and parameters.

## Learning the optical channel

[O’Shea and Hoydis (2017)](https://arxiv.org/abs/1702.00832) illustrate how
deep learning can address the *physical layer* of communication, including
jointly learning a transmitter and receiver. The analogy here is narrower:
Neural-QR has a specified encoder and palette; only the [neural
corrector](NEURAL_CORRECTOR.md) learns how to recover the symbols from camera
bursts affected by moiré, blur, exposure and reflections. Reed–Solomon and CRC
handle transmission errors. Once the bytes are recovered, the verifier checks
the same proof it would receive over another transport. A better camera model
can improve recovery without changing the public relation or its security.

The [device results](../README.md#status-and-scope) document what the working
stack demonstrates. The [open research guide](../CONTRIBUTING.md#open-research-questions) turns
these comparisons into runnable first experiments; the [roadmap](ROADMAP.md)
tracks integration and device validation. A successful exchange establishes feasibility on the tested
devices; it does not establish a security level, classical or post-quantum.
