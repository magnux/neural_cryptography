"""Crypto-only reference from the current neural ZKBoo experiment.

This is an unaudited research POC, not the Picnic standard. It is retained for
cross-language parity. The original OS-randomness behavior is unchanged.
See README.md in this directory for extraction provenance.
"""
import hashlib
import hmac
import json
import math
import secrets
import struct
import numpy as np

# ZKBoo: https://eprint.iacr.org/2016/163
# Picnic: https://eprint.iacr.org/2017/279
# The circuit and proof serialization are experimental choices.
ZK_PROTOCOL = "NeuralPicnic-POC-MPC3-v2"
ZK_MAGIC = b"NPZK0002"
ZK_HEADER = struct.Struct(">8sH32s")
ZK_KEY_BYTES = 32
ZK_MAX_PROOF = 8 * 1024 * 1024

def zk_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def zk_framed(label, *parts):
    fields = [ZK_PROTOCOL.encode("ascii"), label.encode("ascii"), *parts]
    return b"".join(struct.pack(">I", len(x)) + x for x in fields)


def zk_hash(label, *parts):
    return hashlib.sha256(zk_framed(label, *parts)).digest()


def zk_bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def zk_pack(bits):
    return np.packbits(bits, axis=-1, bitorder="big")


def zk_hex(text, count, name):
    if not isinstance(text, str) or len(text) != count * 2:
        raise ValueError(f"{name} must encode exactly {count} bytes")
    try:
        value = bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"invalid hexadecimal {name}") from exc
    if len(value) != count or value.hex() != text:
        raise ValueError(f"{name} must be canonical lowercase hexadecimal")
    return value


def zk_repetition_count(value, name="proof repetitions"):
    if type(value) is not int or not 1 <= value <= 512:
        raise ValueError(f"{name} must be an integer between 1 and 512")
    return value


def zk_pack_trits(challenges):
    """Four 0/1/2 challenges per byte, first challenge in the high two bits."""
    values = np.asarray(challenges)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Fiat-Shamir challenges must be a one-dimensional integer array")
    count = zk_repetition_count(len(values))
    if np.any((values < 0) | (values > 2)):
        raise ValueError("invalid Fiat-Shamir challenge")
    packed = bytearray((count + 3) // 4)
    for i, value in enumerate(values):
        packed[i // 4] |= int(value) << (6 - 2 * (i % 4))
    return bytes(packed)


def zk_unpack_trits(data, repetitions):
    zk_repetition_count(repetitions)
    if not isinstance(data, bytes) or len(data) != (repetitions + 3) // 4:
        raise ValueError("invalid packed challenge length")
    padding = (-repetitions) % 4
    if padding and data[-1] & ((1 << (2 * padding)) - 1):
        raise ValueError("nonzero packed challenge padding")
    values = np.array([(data[i // 4] >> (6 - 2 * (i % 4))) & 3
                       for i in range(repetitions)], np.uint8)
    if np.any(values == 3):
        raise ValueError("invalid packed challenge value")
    return values


class ZKBinaryCircuit:
    """An untrained binary convolutional Feistel circuit, NOT the float teacher.

    Three public sparse XOR convolutions on R produce u, v, w. Each receives a
    different selection of secret key bits as a bias. The nonlinear unit is
    u XOR (v AND w); its output enters a Feistel round, then a public bit shuffle.
    Filter offsets rotate by exact quarter turns. No float arithmetic is used.
    The private 256-bit vector is the network's recurrent bias/key material.
    All offsets, permutations, key selectors, and round constants are public.

    In a proof, all lanes (public-key anchors AND challenge) use the SAME input
    key shares. There is one fresh random tape bit per AND gate, party and
    repetition. Gates are ordered round -> lane -> row-major position.
    """
    def __init__(self, side=8, rounds=16):
        if type(side) is not int or side not in (4, 8, 16):
            raise ValueError("circuit side must be 4, 8, or 16")
        if type(rounds) is not int or not 2 <= rounds <= 64:
            raise ValueError("circuit rounds must be between 2 and 64")
        self.side, self.rounds = side, rounds
        self.half = side * side
        self.nbits = 2 * self.half
        self.nbytes = self.nbits // 8
        self.anchor_count = (256 + self.nbits - 1) // self.nbits
        self.lanes = self.anchor_count + 1
        self.gates = self.rounds * self.lanes * self.half
        self.trace_bytes = self.gates // 8
        self.output_bytes = self.lanes * self.nbytes
        self.descriptor = {"name": "binary-rotconv-feistel-v1", "side": side,
                           "rounds": rounds, "key_bits": 256}
        self.digest = zk_hash("circuit", zk_json(self.descriptor))
        # Sorting hash-derived labels is a fully specified public permutation.
        def permutation(count, label):
            labels = [zk_hash(label, self.digest, struct.pack(">I", i)) for i in range(count)]
            return np.array(sorted(range(count), key=lambda i: (labels[i], i)), dtype=np.int64)
        self.permutation = permutation(self.nbits, "state-permutation")
        self.inverse_permutation = np.argsort(self.permutation)
        key_permutation = permutation(256, "key-permutation")
        self.key_indices = []
        self.constants = []
        self.conv_indices = []
        yy, xx = np.indices((side, side))
        taps = (((0, 0), (-1, 0), (0, 1)),
                ((0, 0), (1, 1), (-1, 1)),
                ((0, 0), (0, -1), (1, 0)))
        for t in range(rounds):
            self.key_indices.append([key_permutation[(np.arange(self.half) +
                                      (3 * t + branch) * self.half) % 256] for branch in range(3)])
            self.constants.append(zk_bits(hashlib.shake_256(zk_framed(
                "round-constant", self.digest, struct.pack(">I", t))).digest(self.half // 8)))
            branches = []
            for branch in taps:
                indices = []
                for dy, dx in branch:
                    for _ in range((t + 1) % 4):
                        dy, dx = -dx, dy
                    indices.append((((yy + dy) % side) * side + (xx + dx) % side).ravel())
                branches.append(indices)
            self.conv_indices.append(branches)
        self.anchors = np.stack([zk_bits(hashlib.shake_256(zk_framed(
            "anchor", self.digest, struct.pack(">I", i))).digest(self.nbytes))
            for i in range(self.anchor_count)])

    def branches(self, right, key_shares, t, public_mask):
        result = []
        for branch in range(3):
            y = np.zeros_like(right)
            for indices in self.conv_indices[t][branch]:
                y ^= right[..., indices]
            y ^= key_shares[..., self.key_indices[t][branch]][..., None, :]
            if branch == 0:
                y ^= public_mask[..., None, None] * self.constants[t]
            result.append(y)
        return result

    def execute(self, key_shares, blocks, party_ids, tapes=None, supplied_trace=None):
        """One clear evaluation, three-party simulation, or two-view replay.

        key_shares: [repetitions, parties, 256]; blocks: [lanes, nbits].
        supplied_trace gives AND outputs of the second opened party. The first
        opened party's AND outputs are ALWAYS computed, never trusted.
        """
        batch, parties, _ = key_shares.shape
        lanes = len(blocks)
        public_mask = (party_ids == 0).astype(np.uint8)
        state = public_mask[..., None, None] * blocks[None, None, :, :]
        trace = None if parties == 1 else np.empty((batch, parties, self.rounds * lanes * self.half), np.uint8)
        stride = lanes * self.half
        for t in range(self.rounds):
            left, right = np.split(state, 2, axis=-1)
            u, v, w = self.branches(right, key_shares, t, public_mask)
            if parties == 1:
                nonlinear = v & w
            else:
                sl = slice(t * stride, (t + 1) * stride)
                random = tapes[..., sl].reshape(batch, parties, lanes, self.half)
                if parties == 3:
                    nonlinear = ((v & w) ^ (v & np.roll(w, -1, axis=1)) ^
                                 (np.roll(v, -1, axis=1) & w) ^ random ^ np.roll(random, -1, axis=1))
                elif parties == 2:
                    first = ((v[:, 0] & w[:, 0]) ^ (v[:, 0] & w[:, 1]) ^
                             (v[:, 1] & w[:, 0]) ^ random[:, 0] ^ random[:, 1])
                    second = supplied_trace[:, sl].reshape(batch, lanes, self.half)
                    nonlinear = np.stack((first, second), axis=1)
                else:
                    raise ValueError("expected one, two, or three parties")
                trace[..., sl] = nonlinear.reshape(batch, parties, stride)
            state = np.concatenate((right, left ^ u ^ nonlinear), axis=-1)[..., self.permutation]
        return state, trace

    def apply(self, secret, blocks):
        key = zk_bits(secret).reshape(1, 1, 256)
        return self.execute(key, blocks, np.zeros((1, 1), np.uint8))[0][0, 0]

    def inverse(self, secret, blocks):
        key = zk_bits(secret).reshape(1, 1, 256)
        state = blocks[None, None].copy()
        for t in reversed(range(self.rounds)):
            state = state[..., self.inverse_permutation]
            right, modified_left = np.split(state, 2, axis=-1)
            u, v, w = self.branches(right, key, t, np.ones((1, 1), np.uint8))
            state = np.concatenate((modified_left ^ u ^ (v & w), right), axis=-1)
        return state[0, 0]


def zk_public_key(secret, circuit, repetitions):
    zk_repetition_count(repetitions, "minimum repetitions")
    return {"protocol": ZK_PROTOCOL, "circuit": circuit.descriptor,
            "minimum_repetitions": repetitions,
            "anchor_outputs": zk_pack(circuit.apply(secret, circuit.anchors)).tobytes().hex()}


def zk_validate_public(public):
    if not isinstance(public, dict) or set(public) != {"protocol", "circuit", "minimum_repetitions", "anchor_outputs"}:
        raise ValueError("invalid public-key schema")
    if public["protocol"] != ZK_PROTOCOL:
        raise ValueError("unsupported proof protocol")
    descriptor = public["circuit"]
    if not isinstance(descriptor, dict) or set(descriptor) != {"name", "side", "rounds", "key_bits"}:
        raise ValueError("invalid circuit descriptor")
    circuit = ZKBinaryCircuit(descriptor["side"], descriptor["rounds"])
    if descriptor != circuit.descriptor:
        raise ValueError("unsupported circuit")
    zk_repetition_count(public["minimum_repetitions"], "minimum repetitions")
    zk_hex(public["anchor_outputs"], circuit.anchor_count * circuit.nbytes, "anchor outputs")
    return circuit


def zk_validate_challenge(challenge):
    if not isinstance(challenge, dict) or set(challenge) != {"protocol", "nonce", "lock_id", "action", "required_repetitions"}:
        raise ValueError("invalid challenge schema")
    if challenge["protocol"] != ZK_PROTOCOL:
        raise ValueError("unsupported challenge protocol")
    zk_hex(challenge["nonce"], 32, "challenge nonce")
    zk_repetition_count(challenge["required_repetitions"], "required repetitions")
    for field in ("lock_id", "action"):
        if not isinstance(challenge[field], str) or not 1 <= len(challenge[field].encode("utf-8")) <= 256:
            raise ValueError(f"{field} must contain 1..256 UTF-8 bytes")
    return zk_json(challenge)


def zk_statement(public, challenge, circuit, response, repetitions):
    zk_repetition_count(repetitions)
    if not isinstance(response, bytes) or len(response) != circuit.nbytes:
        raise ValueError("invalid circuit response length")
    message = zk_validate_challenge(challenge)
    # SHAKE here is PUBLIC message encoding, outside the secret computation.
    challenge_block = zk_bits(hashlib.shake_256(zk_framed("challenge-block", message)).digest(circuit.nbytes))
    blocks = np.concatenate((circuit.anchors, challenge_block[None]), axis=0)
    output = zk_hex(public["anchor_outputs"], circuit.anchor_count * circuit.nbytes, "anchor outputs") + response
    context = zk_hash("statement", zk_json(public), message, response, struct.pack(">H", repetitions))
    return blocks, output, context


def zk_tape(seed, salt, repetition, party, context, gate_count):
    return zk_bits(hashlib.shake_256(zk_framed(
        "random-tape", context, salt, struct.pack(">IB", repetition, party), seed)).digest(gate_count // 8))


def zk_commit(context, salt, repetition, party, seed, share, trace, output):
    return zk_hash("view-commitment", context, salt, struct.pack(">IB", repetition, party),
                   seed, share, trace, output)


def zk_challenges(context, salt, commitments, outputs, repetitions):
    source = hashlib.shake_256(zk_framed("fiat-shamir", context, salt, commitments, outputs))
    length = max(64, 2 * repetitions)
    while True:
        # 252 is a multiple of 3: rejection removes modulo bias.
        values = [x % 3 for x in source.digest(length) if x < 252]
        if len(values) >= repetitions:
            return np.array(values[:repetitions], np.uint8)
        length *= 2


def zk_proof_size(circuit, repetitions):
    zk_repetition_count(repetitions)
    per_rep = 32 + 4 * 32 + circuit.trace_bytes
    return ZK_HEADER.size + circuit.nbytes + (repetitions + 3) // 4 + repetitions * per_rep


def zk_resolve_repetitions(public, challenge, repetitions=None):
    """Enforce the enrolled floor and the verifier's outstanding challenge."""
    floor = max(zk_repetition_count(public["minimum_repetitions"], "minimum repetitions"),
                zk_repetition_count(challenge["required_repetitions"], "required repetitions"))
    repetitions = floor if repetitions is None else zk_repetition_count(repetitions)
    if repetitions < floor:
        raise ValueError("proof repetitions violate enrolled or challenge policy")
    return repetitions


def zk_prove(secret, public, challenge, repetitions=None):
    """Fiat-Shamir POC: commit three views, reveal adjacent views e and e+1.

    Only the hidden commitment and the second opened AND trace are transmitted.
    Opened commitments and every output share are reconstructed by the verifier.
    Fresh OS randomness is required for EVERY proof and EVERY repetition.
    """
    circuit = zk_validate_public(public)
    message = zk_validate_challenge(challenge)
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise ValueError("secret key must be exactly 32 bytes")
    repetitions = zk_resolve_repetitions(public, challenge, repetitions)
    actual = zk_pack(circuit.apply(secret, circuit.anchors)).tobytes()
    if not hmac.compare_digest(actual.hex(), public["anchor_outputs"]):
        raise ValueError("secret does not match public key")
    x = zk_bits(hashlib.shake_256(zk_framed("challenge-block", message)).digest(circuit.nbytes))
    response = zk_pack(circuit.apply(secret, x[None])).tobytes()
    blocks, expected, context = zk_statement(public, challenge, circuit, response, repetitions)
    salt = secrets.token_bytes(32)
    seeds = np.frombuffer(secrets.token_bytes(repetitions * 3 * 32), np.uint8).reshape(repetitions, 3, 32)
    shares = np.empty((repetitions, 3, 32), np.uint8)
    shares[:, :2] = np.frombuffer(secrets.token_bytes(repetitions * 2 * 32), np.uint8).reshape(repetitions, 2, 32)
    shares[:, 2] = shares[:, 0] ^ shares[:, 1] ^ np.frombuffer(secret, np.uint8)
    tapes = np.empty((repetitions, 3, circuit.gates), np.uint8)
    for rep in range(repetitions):
        for party in range(3):
            tapes[rep, party] = zk_tape(seeds[rep, party].tobytes(), salt, rep, party, context, circuit.gates)
    key_bits = np.unpackbits(shares, axis=-1, bitorder="big")
    output_bits, traces = circuit.execute(key_bits, blocks, np.broadcast_to(np.arange(3), (repetitions, 3)), tapes)
    outputs = zk_pack(output_bits.reshape(repetitions, 3, -1))
    traces = zk_pack(traces)
    if not np.all(np.bitwise_xor.reduce(outputs, axis=1) == np.frombuffer(expected, np.uint8)):
        raise AssertionError("MPC outputs do not reconstruct the clear circuit")
    commitments = np.empty((repetitions, 3, 32), np.uint8)
    for rep in range(repetitions):
        for party in range(3):
            commitments[rep, party] = np.frombuffer(zk_commit(context, salt, rep, party,
                seeds[rep, party].tobytes(), shares[rep, party].tobytes(),
                traces[rep, party].tobytes(), outputs[rep, party].tobytes()), np.uint8)
    challenges = zk_challenges(context, salt, commitments.tobytes(), outputs.tobytes(), repetitions)
    indices = np.stack((challenges, (challenges + 1) % 3), axis=1)
    rows = np.arange(repetitions)[:, None]
    proof = (ZK_HEADER.pack(ZK_MAGIC, repetitions, salt) + response + zk_pack_trits(challenges) +
             commitments[np.arange(repetitions), (challenges + 2) % 3].tobytes() +
             seeds[rows, indices].tobytes() + shares[rows, indices].tobytes() +
             traces[np.arange(repetitions), (challenges + 1) % 3].tobytes())
    if len(proof) != zk_proof_size(circuit, repetitions):
        raise AssertionError("incorrect proof serialization length")
    return proof


def zk_verify(public, challenge, proof):
    """Public-only verifier. Its inputs contain NO private key or secret model.

    This checks a mathematical proof for an expected challenge. Nonce consumption
    belongs to the lock protocol; the verify CLI adds an atomic replay cache.
    Malformed encodings fail closed, before allocating attacker-sized arrays.
    """
    try:
        circuit = zk_validate_public(public)
        zk_validate_challenge(challenge)
        if not isinstance(proof, bytes) or not ZK_HEADER.size <= len(proof) <= ZK_MAX_PROOF:
            return False
        magic, repetitions, salt = ZK_HEADER.unpack_from(proof)
        if magic != ZK_MAGIC:
            return False
        zk_resolve_repetitions(public, challenge, repetitions)
        if len(proof) != zk_proof_size(circuit, repetitions):
            return False
        pos = ZK_HEADER.size
        response = proof[pos:pos + circuit.nbytes]
        pos += circuit.nbytes
        blocks, expected, context = zk_statement(public, challenge, circuit, response, repetitions)
        trit_bytes = (repetitions + 3) // 4
        challenges = zk_unpack_trits(proof[pos:pos + trit_bytes], repetitions)
        pos += trit_bytes
        def take(shape):
            nonlocal pos
            count = math.prod(shape)
            result = np.frombuffer(proof, np.uint8, count=count, offset=pos).reshape(shape)
            pos += count
            return result
        hidden_commitments = take((repetitions, 32))
        seeds = take((repetitions, 2, 32))
        shares = take((repetitions, 2, 32))
        trace_second = take((repetitions, circuit.trace_bytes))
        if pos != len(proof):
            return False
        indices = np.stack((challenges, (challenges + 1) % 3), axis=1)
        tapes = np.empty((repetitions, 2, circuit.gates), np.uint8)
        for rep in range(repetitions):
            for slot in range(2):
                tapes[rep, slot] = zk_tape(seeds[rep, slot].tobytes(), salt, rep,
                                         int(indices[rep, slot]), context, circuit.gates)
        replay_outputs, replay_traces = circuit.execute(
            np.unpackbits(shares, axis=-1, bitorder="big"), blocks, indices, tapes,
            np.unpackbits(trace_second, axis=-1, bitorder="big"))
        replay_outputs = zk_pack(replay_outputs.reshape(repetitions, 2, -1))
        replay_traces = zk_pack(replay_traces)
        commitments = np.empty((repetitions, 3, 32), np.uint8)
        outputs = np.empty((repetitions, 3, circuit.output_bytes), np.uint8)
        for rep in range(repetitions):
            hidden = int((challenges[rep] + 2) % 3)
            commitments[rep, hidden] = hidden_commitments[rep]
            outputs[rep, hidden] = (np.frombuffer(expected, np.uint8) ^ replay_outputs[rep, 0] ^ replay_outputs[rep, 1])
            for slot in range(2):
                party = int(indices[rep, slot])
                outputs[rep, party] = replay_outputs[rep, slot]
                commitment = zk_commit(context, salt, rep, party, seeds[rep, slot].tobytes(),
                    shares[rep, slot].tobytes(), replay_traces[rep, slot].tobytes(), replay_outputs[rep, slot].tobytes())
                commitments[rep, party] = np.frombuffer(commitment, np.uint8)
        recomputed = zk_challenges(context, salt, commitments.tobytes(), outputs.tobytes(), repetitions)
        return hmac.compare_digest(challenges.tobytes(), recomputed.tobytes())
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, struct.error):
        return False


def zk_new_challenge(lock_id, action, required_repetitions=219):
    zk_repetition_count(required_repetitions, "required repetitions")
    challenge = {"protocol": ZK_PROTOCOL, "nonce": secrets.token_bytes(32).hex(), "lock_id": lock_id,
                 "action": action, "required_repetitions": required_repetitions}
    zk_validate_challenge(challenge)
    return challenge
