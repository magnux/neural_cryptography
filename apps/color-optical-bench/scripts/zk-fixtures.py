#!/usr/bin/env python3
"""Generate public Python/JavaScript parity vectors or verify foreign proofs.

Never use this deterministic entropy source outside these public test fixtures.
Production reference code continues to use OS randomness.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_REFERENCE = ROOT.parents[1] / "experiments" / "python-zkboo"
sys.path.insert(0, str(PYTHON_REFERENCE))
import neural_zkboo_reference as zk

CASES = (
    ("side4-rounds2-reps3", 4, 2, 3, 3),
    ("side8-rounds4-reps5-unicode", 8, 4, 5, 5),
    ("side16-rounds3-reps4", 16, 3, 4, 4),
    ("default-side8-rounds16-reps219", 8, 16, 219, 219),
    ("side8-rounds4-floor3-required32", 8, 4, 3, 32),
)


class FixtureEntropy:
    """One continuous SHAKE stream, independent of token_bytes call boundaries."""
    def __init__(self, seed):
        self.seed = seed
        self.offset = 0

    def __call__(self, length):
        if type(length) is not int or length < 0:
            raise ValueError("fixture entropy length must be a nonnegative integer")
        end = self.offset + length
        result = hashlib.shake_256(self.seed).digest(end)[self.offset:end]
        self.offset = end
        return result


@contextmanager
def fixture_entropy(seed):
    saved = zk.secrets.token_bytes
    entropy = FixtureEntropy(seed)
    zk.secrets.token_bytes = entropy
    try:
        yield entropy
    finally:
        zk.secrets.token_bytes = saved


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def packed(bits):
    return zk.zk_pack(bits).tobytes()


def public_permutation(circuit, count, label):
    labels = [zk.zk_hash(label, circuit.digest, struct.pack(">I", i)) for i in range(count)]
    return sorted(range(count), key=lambda i: (labels[i], i))


def intermediates(circuit, secret, public, challenge, proof, entropy_seed):
    """Parse the real proof, independently reconstruct all three private views.

    Reconstruction uses public fixture entropy only. It checks the real prover's
    hidden commitment, selected challenge trits, seed/share opening, and trace
    arrays byte for byte; full commitments and outputs are regenerated locally.
    """
    magic, repetitions, salt = zk.ZK_HEADER.unpack_from(proof)
    assert magic == zk.ZK_MAGIC
    pos = zk.ZK_HEADER.size
    response = proof[pos:pos + circuit.nbytes]
    pos += circuit.nbytes
    blocks, expected_output, context = zk.zk_statement(public, challenge, circuit, response, repetitions)

    def take(shape):
        nonlocal pos
        count = int(np.prod(shape))
        value = np.frombuffer(proof, np.uint8, count=count, offset=pos).reshape(shape)
        pos += count
        return value

    packed_trits = take(((repetitions + 3) // 4,))
    proof_trits = np.array([(int(packed_trits[i // 4]) >> (6 - 2 * (i % 4))) & 3
                           for i in range(repetitions)], dtype=np.uint8)
    assert np.all(proof_trits < 3)
    if repetitions % 4:
        assert int(packed_trits[-1]) & ((1 << (2 * (4 - repetitions % 4))) - 1) == 0
    proof_commitments = take((repetitions, 32))
    proof_seeds = take((repetitions, 2, 32))
    proof_shares = take((repetitions, 2, 32))
    proof_trace = take((repetitions, circuit.trace_bytes))
    assert pos == len(proof)

    entropy = FixtureEntropy(entropy_seed)
    assert entropy(32) == salt
    seeds = np.frombuffer(entropy(repetitions * 3 * 32), np.uint8).reshape(repetitions, 3, 32)
    shares = np.empty((repetitions, 3, 32), np.uint8)
    shares[:, :2] = np.frombuffer(entropy(repetitions * 2 * 32), np.uint8).reshape(repetitions, 2, 32)
    shares[:, 2] = shares[:, 0] ^ shares[:, 1] ^ np.frombuffer(secret, np.uint8)
    tapes = np.empty((repetitions, 3, circuit.gates), np.uint8)
    for repetition in range(repetitions):
        for party in range(3):
            tapes[repetition, party] = zk.zk_tape(
                seeds[repetition, party].tobytes(), salt, repetition, party, context, circuit.gates)
    key_bits = np.unpackbits(shares, axis=-1, bitorder="big")
    output_bits, trace_bits = circuit.execute(
        key_bits, blocks, np.broadcast_to(np.arange(3), (repetitions, 3)), tapes)
    outputs = zk.zk_pack(output_bits.reshape(repetitions, 3, -1))
    traces = zk.zk_pack(trace_bits)
    assert np.all(np.bitwise_xor.reduce(outputs, axis=1) == np.frombuffer(expected_output, np.uint8))
    commitments = np.empty((repetitions, 3, 32), np.uint8)
    for repetition in range(repetitions):
        for party in range(3):
            commitments[repetition, party] = np.frombuffer(zk.zk_commit(
                context, salt, repetition, party, seeds[repetition, party].tobytes(),
                shares[repetition, party].tobytes(), traces[repetition, party].tobytes(),
                outputs[repetition, party].tobytes()), np.uint8)
    trits = zk.zk_challenges(context, salt, commitments.tobytes(), outputs.tobytes(), repetitions)
    assert np.array_equal(trits, proof_trits)
    assert np.array_equal(commitments[np.arange(repetitions), (trits + 2) % 3], proof_commitments)
    opened = np.stack((trits, (trits + 1) % 3), axis=1)
    rows = np.arange(repetitions)[:, None]
    assert np.array_equal(seeds[rows, opened], proof_seeds)
    assert np.array_equal(shares[rows, opened], proof_shares)
    assert np.array_equal(traces[np.arange(repetitions), (trits + 1) % 3], proof_trace)
    inverse_bits = circuit.inverse(secret, circuit.apply(secret, blocks))
    assert np.array_equal(inverse_bits, blocks)

    result = {
        "descriptorCanonicalJSONHex": zk.zk_json(circuit.descriptor).hex(),
        "publicCanonicalJSONHex": zk.zk_json(public).hex(),
        "challengeCanonicalJSONHex": zk.zk_json(challenge).hex(),
        "circuitDigestHex": circuit.digest.hex(),
        "permutation": circuit.permutation.tolist(),
        "inversePermutation": circuit.inverse_permutation.tolist(),
        "keyPermutation": public_permutation(circuit, 256, "key-permutation"),
        "keyIndices": [[indices.tolist() for indices in branches] for branches in circuit.key_indices],
        "convIndices": [[[indices.tolist() for indices in branch] for branch in branches]
                        for branches in circuit.conv_indices],
        "roundConstantsHex": [packed(bits).hex() for bits in circuit.constants],
        "anchorsPackedHex": packed(circuit.anchors).hex(),
        "anchorOutputsHex": public["anchor_outputs"],
        "challengeBlockHex": packed(blocks[-1]).hex(),
        "responseHex": response.hex(),
        "contextHex": context.hex(),
        "saltHex": salt.hex(),
        "firstSeedsHex": [row.tobytes().hex() for row in seeds[0]],
        "firstSharesHex": [row.tobytes().hex() for row in shares[0]],
        "firstTapePackedHex": [packed(bits).hex() for bits in tapes[0]],
        "firstCommitmentsHex": [row.tobytes().hex() for row in commitments[0]],
        "commitmentsSha256": sha256(commitments.tobytes()),
        "firstOutputSharesHex": [row.tobytes().hex() for row in outputs[0]],
        "outputsSha256": sha256(outputs.tobytes()),
        "firstAndTraceSha256": [sha256(row.tobytes()) for row in traces[0]],
        "challengeTrits": trits.tolist(),
        "packedChallengesHex": packed_trits.tobytes().hex(),
        "hiddenCommitmentsSha256": sha256(proof_commitments.tobytes()),
        "repetitions": repetitions,
        "proofBytes": len(proof),
        "entropyBytes": entropy.offset,
        "gates": circuit.gates,
        "traceBytes": circuit.trace_bytes,
        "outputBytes": circuit.output_bytes,
    }
    if repetitions < 219:
        result["firstAndTraceHex"] = [row.tobytes().hex() for row in traces[0]]
    return result


def generate(output, browser_fixture=None):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 2, "warning": "Public deterministic test material; never use these keys or entropy in production.", "cases": []}
    for case_index, (case_id, side, rounds, minimum_repetitions, required_repetitions) in enumerate(CASES):
        circuit = zk.ZKBinaryCircuit(side, rounds)
        secret = bytes((i + case_index) % 256 for i in range(32))
        entropy_seed = ("NeuralPicnic parity fixture v2/" + case_id).encode("utf-8")
        public = zk.zk_public_key(secret, circuit, minimum_repetitions)
        challenge = {
            "protocol": zk.ZK_PROTOCOL,
            "nonce": bytes((255 - i - case_index) % 256 for i in range(32)).hex(),
            "lock_id": "puerta-ñ-🚪" if case_index == 1 else "fixture-lock/" + case_id,
            "action": "abrir\n€" if case_index == 1 else "unlock",
            "required_repetitions": required_repetitions,
        }
        with fixture_entropy(entropy_seed):
            proof = zk.zk_prove(secret, public, challenge)
        assert zk.zk_verify(public, challenge, proof)
        entry = {
            "id": case_id,
            "secretHex": secret.hex(),
            "entropySeedHex": entropy_seed.hex(),
            "public": public,
            "challenge": challenge,
            "proofFile": case_id + ".bin",
            "proofSha256": sha256(proof),
            "expected": intermediates(circuit, secret, public, challenge, proof, entropy_seed),
        }
        (output / entry["proofFile"]).write_bytes(proof)
        manifest["cases"].append(entry)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if browser_fixture is not None:
        entry = next(item for item in manifest["cases"] if "unicode" in item["id"])
        document = {"publicKey": entry["public"], "challenge": entry["challenge"],
                    "proofSha256": entry["proofSha256"]}
        proof = (output / entry["proofFile"]).read_bytes()
        proof_hex = proof.hex()
        lines = [json.dumps(proof_hex[start:start + 128]) for start in range(0, len(proof_hex), 128)]
        source = ("// Public-only Python compatibility fixture. This is not an enrollment key.\n"
                  "// Generated by scripts/zk-fixtures.py generate --browser-fixture <path>.\n"
                  "// Contains neither a secret key nor deterministic entropy.\n"
                  "import { fromHex } from '../crypto/bytes.js';\n\n"
                  "const DOCUMENT = " + json.dumps(document, indent=2, ensure_ascii=True) + ";\n"
                  "const PROOF_HEX =\n  " + " +\n  ".join(lines) + ";\n\n"
                  "export function referenceFixture() {\n"
                  "  return { ...structuredClone(DOCUMENT), proof: fromHex(PROOF_HEX, " + str(len(proof)) + ", 'reference proof') };\n"
                  "}\n")
        browser_fixture.parent.mkdir(parents=True, exist_ok=True)
        browser_fixture.write_text(source, encoding="utf-8")
    print(json.dumps({"ok": True, "cases": len(manifest["cases"]), "manifest": str(manifest_path),
                      "proofBytes": [entry["expected"]["proofBytes"] for entry in manifest["cases"]]}))


def verify(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("expected a nonempty manifest cases list")
    results = []
    for index, entry in enumerate(manifest["cases"]):
        proof = (path.parent / entry["proofFile"]).read_bytes()
        expected = entry.get("expectValid", True)
        if type(expected) is not bool:
            raise ValueError("expectValid must be a boolean")
        accepted = zk.zk_verify(entry["public"], entry["challenge"], proof)
        results.append({"id": entry.get("id", str(index)), "accepted": accepted,
                        "expected": expected, "match": accepted == expected})
    ok = all(result["match"] for result in results)
    summary = {"ok": ok, "cases": len(results),
               "verified": sum(result["accepted"] for result in results),
               "rejected": sum(not result["accepted"] for result in results)}
    if not ok:
        summary["mismatches"] = [result for result in results if not result["match"]]
    print(json.dumps(summary))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generation = sub.add_parser("generate", help="regenerate public deterministic fixtures")
    generation.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "zk")
    generation.add_argument("--browser-fixture", type=Path, help="also write the public-only browser fixture module")
    verification = sub.add_parser("verify", help="verify proof files listed in a JSON manifest")
    verification.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.out, args.browser_fixture)
        return 0
    return verify(args.manifest)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        raise SystemExit(1)
