# Connections to neural cryptography research

Updated: 2026-10-19.

The immediate goal is a working device-to-device authentication POC using our
Neural ZKBoo proof and **color-matrix optical transport**. The code and device
observations so far form the exploratory phase of this research. This document
records relevant literature and possible later experiments; a new cryptanalytic
bench or learned cryptographic primitive is not a prerequisite for completing
that POC. See the [development plan](ROADMAP.md) and the
[learned optical decoder design](OPTICAL_DECODER.md) for the active work.

The [Wikipedia article on neural cryptography](https://en.wikipedia.org/wiki/Neural_cryptography)
was the starting point for this review. The technical connections below use
primary papers, and distinguish results in those papers from proposed work here.

## Where the project fits

| Research direction | Role of the network | Connection to this repository |
| --- | --- | --- |
| Tree parity machine synchronization | Participants adapt private weights through public interaction until their states agree. | Related dynamical-systems intuition; a separate possible key-establishment experiment. |
| Adversarially learned encryption | Sender and receiver learn against a trained eavesdropper. | Possible future architecture search and attacker evaluation. |
| Fixed neural keyed transformation | Secret parameters define a recurrent mapping from inputs to outputs. | The original floating-point CNN and Feistel [Python experiment](../experiments/neural-dynamics/README.md). |
| Proof of knowledge of a circuit secret | An exact circuit defines the public relation; a proof establishes knowledge of a satisfying secret. | Our [Python](../experiments/python-zkboo/README.md) and [browser](../shared/browser-crypto/README.md) ZKBoo-style implementations. |
| Learned optical reception | A trained model estimates transmitted symbols from camera observations. | The next transport component; its weights are public and have no key-establishment role. |

The original CNN experiment does **not** learn its secret weights during use.
Its surrogate attacker is trained, but the keyed forward transformation is
fixed. The proof circuit is a separate, untrained binary construction, not a
quantized export of the floating-point network. Its secret is a 256-bit bias/key
vector; the sparse XOR convolution structure, permutations, and round constants
are public.

## Synchronization and the space in which convergence occurs

Kinzel and Kanter describe networks initialized with different private weights,
fed common public inputs, and updated using exchanged output bits. Their
interacting learning dynamics can synchronize the weights, which are then used
as a shared secret [1].

This is connected to our interpretation of networks as dynamical systems, but
the desired convergence is different. Synchronization seeks agreement between
**two participants' weight states**. Our original diffusion experiment instead
seeks to preserve distinctions between **different inputs under one fixed key**.
The increasing output bias and clustering observed in some recurrent ReLU
configurations work against that second objective. Recentring activations and
clipping adaptive weights also act on different parts of these systems; one is
not a substitute for the other.

An optical implementation of synchronization would have to account for all
interactive steps, public inputs, and confirmation traffic. Small output bits
alone do not imply a fast screen-to-camera exchange. Agreement with a peer also
does not independently identify that peer as an enrolled device. For the current
POC we retain the fixed secret, public enrollment data, and challenge/proof flow.

## Lessons from attacks on synchronization schemes

Klimov, Mityagin, and Shamir explain why a straightforward imitating network can
lag behind the legitimate parties, then give three attacks against the original
protocol they analyze [2]. Seoane and Ruttor later attack permutation parity
machines by probabilistic inference over possible weights, rather than merely
imitating the participants' learning process [3]. These are results about those
specific constructions, not a general impossibility result for neural methods.

The practical lesson is that an attacker may use a different representation and
algorithm from the legitimate network. Our existing bias, avalanche, covariance,
and surrogate metrics are useful diagnostics. A weak surrogate reaching chance
accuracy does not bound a stronger attacker's success.

When cryptanalytic work resumes, include direct secret recovery and finding
**any satisfying witness** for the public relation. Recovering a different secret
that satisfies the enrolled anchors may already defeat a proof-of-knowledge
application; exact recovery of the originally sampled secret is not always the
right success criterion. Record the attack model and resulting forgery or fresh
response success, rather than relying solely on average bit prediction.

## Adversarial learning and differential distinguishers

Abadi and Andersen train communicating networks against an Eve network, using
shared keys in their principal symmetric experiment [4]. They also reset and
retrain Eve after freezing the communicating networks. Their guarantees are
relative to the explored adversaries, and their asymmetric experiments are
explicitly fragile: apparent secrecy often disappears after retraining Eve.

For this project, the reusable method is attacker diversity and fresh evaluation
after architecture selection. A future search could vary mixing patterns,
nonlinear units, and rotation schedules while measuring diffusion, attack
success, exact implementation cost, and proof size. The selected design would
still need independent analysis. Learned asymmetric encryption is not part of
the current authentication implementation.

Gohr demonstrates neural distinguishers and key-recovery improvements against
**reduced-round** Speck32/64, using structured input/output pairs [5]. This is a
stronger testing pattern than asking a small network to predict unrelated output
bits. An adaptation for our fixed circuit would compare output pairs from chosen
input differences against an appropriate random-permutation reference, then
sweep depth and secret keys. A distinguishing advantage would motivate an actual
key-recovery or forgery experiment; absence of an advantage in that experiment
would remain limited evidence.

## Boolean structure: a connection to SIMON

The current binary circuit forms three secret-biased sparse XOR convolutions,
then computes the nonlinear unit

$$
u \oplus (v \land w).
$$

This feeds a Feistel update followed by a public permutation. SIMON's round
function has a related AND/XOR structure [6]:

$$
f(x)=\bigl(\operatorname{rotl}_1(x)\land
\operatorname{rotl}_8(x)\bigr)\oplus\operatorname{rotl}_2(x).
$$

SIMON's rotations are cyclic bit shifts. Our circuit uses different linear maps,
secret-bit selections, spatial tap rotations, and permutations. This is a
structural analogy, not an equivalence or a transfer of security results. It
suggests useful future comparison methods: represent the public XOR convolution
as a binary linear map, study how differences propagate through AND gates, and
compare reduced-round variants under the same attack budget.

## ZKBoo, Picnic, and proof-friendly computation

ZKBoo provides a route to proofs for Boolean circuits through MPC-in-the-head
[7]. Picnic's original work connects proofs of knowledge to symmetric-key
primitives and improves proof efficiency [8]. LowMC deliberately optimizes the
number and depth of nonlinear operations for settings including MPC and ZK [9].

Our construction follows that broad separation: a public relation built from
the keyed binary circuit, plus a custom three-party MPC-in-the-head proof. It
commits simulated views and opens selected views; the compact encoding
reconstructs omitted data. It is a custom POC and does not implement Picnic.

In our current engine, XOR and permutation operations do not add the same
per-gate transcript cost as AND gates. They still cost computation. The
[current format documentation](FORMATS.md) describes the resulting compact proof:
the default side-8, 16-circuit-round configuration uses 17,474 bytes at 32 proof
repetitions and 119,249 bytes at 219. This gives a concrete reason to study
proof-friendly circuit design later, without changing the circuit while solving
the present optical bottleneck.

Keep the knobs separate:

- **Circuit rounds** change the keyed transformation and its attack surface.
- **Proof repetitions** change protocol amplification and proof cost.
- **Optical size, palette, phases, and coding** change transmission behavior.

More repetitions cannot make an easily solvable public relation hard to solve.
Reducing nonlinear gates is also not a free optimization: it changes the
primitive that needs analysis. The current POC keeps configurable repetitions
with verifier-owned acceptance requirements; these settings are not certified
security levels.

## The learned receiver directly advances the POC

The proposed small U-Net is a denoiser in purpose and a symbol classifier in
output: it estimates palette probabilities per logical cell from rectified
camera images. It learns the optical channel, including moving moiré, blur,
noise, and color changes. It does not generate cryptographic keys or decide
whether an authentication proof is valid.

This separation lets us improve transport without changing the proof relation.
The model feeds bounded recovery/reassembly, recovered bytes go through the
existing integrity checks, and the exact proof verifier makes the authentication
decision. Diagnostic payload labels belong only in data generation and
evaluation, never in the live decoder. See [OPTICAL_DECODER.md](OPTICAL_DECODER.md)
for architecture, browser execution, collection, synthetic training, and real
device evaluation.

## Future research backlog

These items follow the working color-transport authentication POC in the
[roadmap](ROADMAP.md). They are not additional completion gates for that POC.

| Experiment | First bounded scope | Evidence to retain |
| --- | --- | --- |
| Cryptanalytic comparison bench | Existing circuit versus explicitly identified reduced-round comparison circuits, including SIMON/LowMC where useful. | Parameters, attack model, query and compute budget, success rate, AND count, proof bytes, browser cost. |
| Differential neural attacks | A few controlled input differences and depths, with disjoint train/test keys. | Distinguisher advantage, repeated runs, and whether it enables secret recovery or forgery. |
| Direct relation solving | Small circuit instances, gradually increasing size and rounds. | Time/memory/query limits and success finding any satisfying witness. |
| Adversarial architecture search | A small search space and multiple independently retrained attackers. | Held-out attacks and reproducible costs, without declaring security from training loss. |
| Synchronization side experiment | Two legitimate parity machines plus passive imitation/probabilistic attackers. | Synchronization and attacker success, transcript bytes, interaction count, and latency model. |

No project claim of post-quantum security follows from the word “neural,” from
avoiding number-theoretic problems, or from using a ZK wrapper. Such a claim would
need justified primitive assumptions, protocol/transform analysis, and suitable
parameters. Likewise, universal approximation does not supply an efficient
cryptographic inversion algorithm, and membership in NP alone is not a hardness
argument. These distinctions keep exploratory results interpretable while the
functional POC progresses.

## Primary references

1. Wolfgang Kinzel and Ido Kanter, **Neural Cryptography** (2002).
   [arXiv:cond-mat/0208453](https://arxiv.org/abs/cond-mat/0208453).
2. Alexander Klimov, Anton Mityagin, and Adi Shamir, **Analysis of Neural
   Cryptography**, ASIACRYPT 2002.
   [IACR paper](https://www.iacr.org/archive/asiacrypt2002/25010286/25010286.pdf).
3. Luís F. Seoane and Andreas Ruttor, **Successful attack on
   permutation-parity-machine-based neural cryptography**, Physical Review E 85,
   025101(R), 2012. [arXiv:1111.5792](https://arxiv.org/abs/1111.5792).
4. Martín Abadi and David G. Andersen, **Learning to Protect Communications with
   Adversarial Neural Cryptography** (2016).
   [arXiv:1610.06918](https://arxiv.org/abs/1610.06918).
5. Aron Gohr, **Improving Attacks on Round-Reduced Speck32/64 using Deep
   Learning**, CRYPTO 2019. [ePrint 2019/037](https://eprint.iacr.org/2019/037).
6. Ray Beaulieu et al., **The SIMON and SPECK Families of Lightweight Block
   Ciphers** (2013). [ePrint 2013/404](https://eprint.iacr.org/2013/404);
   [authors' implementation guide](https://nsacyber.github.io/simon-speck/implementations/ImplementationGuide1.1.pdf).
7. Irene Giacomelli, Jesper Madsen, and Claudio Orlandi, **ZKBoo: Faster
   Zero-Knowledge for Boolean Circuits**, USENIX Security 2016.
   [ePrint 2016/163](https://eprint.iacr.org/2016/163).
8. Melissa Chase et al., **Post-Quantum Zero-Knowledge and Signatures from
   Symmetric-Key Primitives**, ACM CCS 2017.
   [ePrint 2017/279](https://eprint.iacr.org/2017/279).
9. Martin R. Albrecht et al., **Ciphers for MPC and FHE**, EUROCRYPT 2015.
   [ePrint 2016/687](https://eprint.iacr.org/2016/687);
   [LowMC project](https://lowmc.github.io/).
