#!/usr/bin/env python3
"""Neural cryptography ZKBoo POC: exact binary circuit + public proof verification.

    python neural_zkboo_experiment.py --camera
    python neural_zkboo_experiment.py --size 512 --colors 16 --frames 8
    python neural_zkboo_experiment.py --zk-action verify
    python neural_zkboo_experiment.py --self-test

--size/--matrix-size (8..512), --colors (2/4/8/16), and --frames (maximum 1..8)
control optical transport. The proof circuit uses a separate --state-size.
Requires NumPy only. This is an unaudited ZKBoo-style research POC, not an
implementation of the Picnic standard and not a post-quantum security claim.
Keys/proof randomness come from the OS; --seed affects the camera only.
The verifier must receive its public key and outstanding challenge from trusted
lock state. Keep private_key.json secret. The verify action consumes each nonce.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import time
import sys
import numpy as np
# Shared optics are simulation utilities; proof code has one canonical implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from neural_optical import (
    palette as zk_palette, frame_capacity as zk_frame_capacity,
    frame_encode as zk_frame_encode, frame_decode as zk_frame_decode,
    save_transport as zk_save_transport, camera_capture as zk_camera_capture,
    load_captures as zk_load_captures, add_optical_arguments, validate_optical_arguments,
)

from neural_zkboo_reference import (
    ZK_PROTOCOL, ZK_MAGIC, ZK_HEADER, ZK_MAX_PROOF, ZKBinaryCircuit,
    zk_json, zk_hash, zk_hex, zk_public_key, zk_validate_public,
    zk_validate_challenge, zk_new_challenge, zk_proof_size,
    zk_resolve_repetitions, zk_prove, zk_verify,
)

def zk_read_json(path, maximum=65536):
    if path.stat().st_size > maximum:
        raise ValueError(f"JSON file too large: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def zk_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def zk_load_secret(path):
    document = zk_read_json(path)
    if not isinstance(document, dict) or set(document) != {"protocol", "secret"} or document["protocol"] != ZK_PROTOCOL:
        raise ValueError("invalid private-key file")
    return zk_hex(document["secret"], 32, "private key")


def zk_create_key_files(secret_path, public_path, circuit, repetitions):
    if secret_path.exists() or public_path.exists():
        raise ValueError("key file already exists; choose fresh key paths or a new --out directory")
    secret = secrets.token_bytes(32)
    public = zk_public_key(secret, circuit, repetitions)
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately never overwrite an existing private key; do not print it.
    fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"protocol": ZK_PROTOCOL, "secret": secret.hex()}, handle)
        handle.write("\n")
    zk_write_json(public_path, public)
    return secret, public


def zk_accept_once(public, challenge, proof, cache):
    """Demo lock policy: verify expected challenge, atomically consume its nonce."""
    if not zk_verify(public, challenge, proof):
        return False, "proof rejected"
    cache.mkdir(parents=True, exist_ok=True)
    # Lock identity + nonce: changing action must not allow reuse of that nonce.
    tag = zk_hash("consumed-challenge", challenge["lock_id"].encode("utf-8"), bytes.fromhex(challenge["nonce"])).hex()
    try:
        fd = os.open(cache / tag, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False, "challenge already consumed"
    os.close(fd)
    return True, "proof accepted; challenge consumed"


def zk_tamper_checks(public, challenge, proof):
    circuit = zk_validate_public(public)
    repetitions = ZK_HEADER.unpack_from(proof)[1]
    trits = ZK_HEADER.size + circuit.nbytes
    hidden_commitments = trits + (repetitions + 3) // 4
    opened_seeds = hidden_commitments + repetitions * 32
    opened_shares = opened_seeds + repetitions * 2 * 32
    offsets = {"header": 0, "salt": 10, "response": ZK_HEADER.size,
               "packed_challenges": trits, "hidden_commitment": hidden_commitments,
               "opened_seed": opened_seeds, "opened_key_share": opened_shares,
               "and_transcript": len(proof) - 1}
    checks = {}
    for name, offset in offsets.items():
        altered = bytearray(proof)
        altered[offset] ^= 1
        checks[name] = not zk_verify(public, challenge, bytes(altered))
    checks["truncation"] = not zk_verify(public, challenge, proof[:-1])
    checks["trailing_data"] = not zk_verify(public, challenge, proof + b"\x00")
    for name in ("nonce", "lock_id", "action"):
        altered = dict(challenge)
        altered[name] = (("00" if altered[name][:2] != "00" else "01") + altered[name][2:]
                         if name == "nonce" else altered[name] + "!")
        checks["wrong_" + name] = not zk_verify(public, altered, proof)
    altered = dict(challenge, required_repetitions=challenge["required_repetitions"] % 512 + 1)
    checks["wrong_required_repetitions"] = not zk_verify(public, altered, proof)
    higher_requirement = dict(challenge, required_repetitions=min(512, repetitions + 1))
    if repetitions < 512:
        checks["higher_challenge_requirement"] = not zk_verify(public, higher_requirement, proof)
    wrong_public = dict(public)
    y = bytearray.fromhex(wrong_public["anchor_outputs"])
    y[0] ^= 1
    wrong_public["anchor_outputs"] = y.hex()
    checks["wrong_public_key"] = not zk_verify(wrong_public, challenge, proof)
    return checks


def zk_primitive_probe(secret, circuit):
    # A small diagnostic only: these scores DO NOT establish key-recovery hardness.
    data = np.frombuffer(secrets.token_bytes(128 * circuit.nbytes), np.uint8).reshape(128, circuit.nbytes)
    bits = np.unpackbits(data, axis=-1, bitorder="big")
    changed = bits.copy()
    changed[np.arange(128), np.arange(128) % circuit.nbits] ^= 1
    original = circuit.apply(secret, bits)
    flipped = circuit.apply(secret, changed)
    key_delta = bytearray(secret)
    key_delta[0] ^= 128
    keyed = circuit.apply(bytes(key_delta), bits)
    return {"samples": 128, "input_bit_flip_probability": float((original != flipped).mean()),
            "one_key_bit_flip_probability": float((original != keyed).mean()),
            "mean_absolute_bit_bias": float(np.abs(original.mean(0) - .5).mean())}


def zk_self_test():
    secret = bytes(range(32))  # Only the self-test uses a known test key.
    challenge = {"protocol": ZK_PROTOCOL, "nonce": "12" * 32, "lock_id": "test-lock", "action": "unlock", "required_repetitions": 32}
    for side in (4, 8, 16):
        circuit = ZKBinaryCircuit(side, 4)
        public = zk_public_key(secret, circuit, 32)
        proof = zk_prove(secret, public, challenge)
        assert zk_verify(public, challenge, proof)
        assert np.array_equal(circuit.inverse(secret, circuit.apply(secret, circuit.anchors)), circuit.anchors)
        checks = zk_tamper_checks(public, challenge, proof)
        assert all(checks.values()), checks
        higher_policy = dict(public, minimum_repetitions=33)
        assert not zk_verify(higher_policy, challenge, proof)
    # Gate identity exhaustively for all shares and random masks.
    for value in range(512):
        bits = np.array([(value >> i) & 1 for i in range(9)], np.uint8)
        x, y, r = bits[:3], bits[3:6], bits[6:]
        z = (x & y) ^ (x & np.roll(y, -1)) ^ (np.roll(x, -1) & y) ^ r ^ np.roll(r, -1)
        assert np.bitwise_xor.reduce(z) == (np.bitwise_xor.reduce(x) & np.bitwise_xor.reduce(y))
    for colors in (2, 4, 8, 16):
        for size in (32, 33, 512):
            capacity = zk_frame_capacity(size, colors)
            payload = hashlib.shake_256(b"frame-test").digest(2 * capacity + 1)
            grids = zk_frame_encode(payload, size, colors, 8)
            assert zk_frame_decode(grids[::-1], colors) == payload
            assert zk_frame_decode(np.concatenate((grids, grids[:1])), colors) == payload
            corrupted = grids.copy()
            corrupted[0, -1, -1] ^= 1
            for bad in (grids[:-1], corrupted):
                try:
                    zk_frame_decode(bad, colors)
                except ValueError:
                    pass
                else:
                    raise AssertionError("damaged transport accepted")
    for boundary_size in (32, 512):
        capacity = zk_frame_capacity(boundary_size, 16)
        payload = bytes(8 * capacity)
        boundary_frames = zk_frame_encode(payload, boundary_size, 16, 8)
        assert len(boundary_frames) == 8
        assert zk_frame_decode(boundary_frames[::-1], 16) == payload
        try:
            zk_frame_encode(payload + b"x", boundary_size, 16, 8)
        except ValueError:
            pass
        else:
            raise AssertionError("capacity overflow was silently accepted")
    capacity = zk_frame_capacity(32, 16)
    first = zk_frame_encode(b"A" * (capacity + 1), 32, 16, 8)
    second = zk_frame_encode(b"B" * (capacity + 1), 32, 16, 8)
    try:
        zk_frame_decode(np.stack((first[0], second[1])), 16)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed sequences accepted")
    captured = zk_camera_capture(first, 16, 7, 0., 8)
    assert np.array_equal(captured, first)
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        assert zk_accept_once(public, challenge, proof, cache)[0]
        assert not zk_accept_once(public, challenge, proof, cache)[0]
        captures = cache / "capture.npz"
        np.savez_compressed(captures, symbols=first, colors=16)
        frames, colors = zk_load_captures(captures)
        assert colors == 16 and np.array_equal(frames, first)
        np.savez_compressed(captures, symbols=np.zeros((1, 513, 513), np.uint8), colors=16)
        try:
            zk_load_captures(captures)
        except ValueError:
            pass
        else:
            raise AssertionError("oversized capture accepted")
    print("Neural Picnic self-test passed: gate identity, three circuit sizes, exact inversion, "
          "public-only proof verification, transcript/message/key tampering, repetition policy, "
          "palette framing through 512 cells and 8 frames, reordering/duplicates/loss/corruption, "
          "aligned capture and replay rejection.")


def zk_run(args):
    started = time.perf_counter()
    out = args.out
    secret_path = args.secret_key or out / "private_key.json"
    public_path = args.public_key or out / "public_key.json"
    challenge_path = args.challenge or out / "challenge.json"
    proof_path = args.proof or out / "proof.bin"
    action = args.zk_action
    enrollment_minimum = (args.minimum_repetitions if args.minimum_repetitions is not None
                          else args.repetitions if args.repetitions is not None else 219)
    challenge_requirement = (args.required_repetitions if args.required_repetitions is not None
                             else args.repetitions if args.repetitions is not None else enrollment_minimum)
    print("Neural ZKBoo POC | experimental ZKBoo-style MPC-in-the-head | exact NumPy/CPU circuit", flush=True)
    print("Research POC: unaudited neural primitive/protocol; no claimed security level or post-quantum guarantee.", flush=True)
    if action == "keygen":
        circuit = ZKBinaryCircuit(args.state_size, args.zk_rounds)
        zk_create_key_files(secret_path, public_path, circuit, enrollment_minimum)
        print(f"Created private key: {secret_path}\nPublic verifier key: {public_path}")
        return
    if action == "challenge":
        challenge = zk_new_challenge(args.lock_id, args.action, challenge_requirement)
        zk_write_json(challenge_path, challenge)
        grids = zk_save_transport(out, "challenge", zk_json(challenge), args)
        print(f"Fresh challenge: {challenge_path} | {len(grids)} optical frame(s)")
        return
    if action == "verify":
        # Deliberately no secret-key access anywhere in this branch.
        public = zk_read_json(public_path)
        challenge = zk_read_json(challenge_path)
        if args.captured:
            grids, colors = zk_load_captures(args.captured)
            proof = zk_frame_decode(grids, colors)
        else:
            if proof_path.stat().st_size > ZK_MAX_PROOF:
                raise ValueError("proof file too large")
            proof = proof_path.read_bytes()
        accepted, reason = zk_accept_once(public, challenge, proof, args.replay_cache or out / "consumed_challenges")
        print(f"{'ACCEPT' if accepted else 'REJECT'}: {reason}")
        if not accepted:
            raise SystemExit(1)
        return
    if action == "demo":
        circuit = ZKBinaryCircuit(args.state_size, args.zk_rounds)
        if secret_path.exists() and public_path.exists():
            secret, public = zk_load_secret(secret_path), zk_read_json(public_path)
            existing = zk_validate_public(public)
            if existing.descriptor != circuit.descriptor or public["minimum_repetitions"] != enrollment_minimum:
                raise ValueError("existing key uses different circuit/policy; choose a new --out directory")
            print("Reusing existing key; generating a fresh challenge and proof.", flush=True)
        else:
            secret, public = zk_create_key_files(secret_path, public_path, circuit, enrollment_minimum)
        challenge = zk_new_challenge(args.lock_id, args.action, challenge_requirement)
        zk_write_json(challenge_path, challenge)
    else:  # prove
        secret, public = zk_load_secret(secret_path), zk_read_json(public_path)
        challenge = zk_read_json(challenge_path)
        circuit = zk_validate_public(public)
    repetitions = zk_resolve_repetitions(public, challenge, args.repetitions)
    estimated_bytes = zk_proof_size(circuit, repetitions)
    capacity = zk_frame_capacity(args.size, args.colors)
    if capacity < 1 or estimated_bytes > capacity * args.frames:
        raise ValueError(f"Proof requires {estimated_bytes:,} bytes; configured transport holds "
                         f"{max(0, capacity) * args.frames:,}. Increase --size, --colors, or --frames.")
    print(f"Circuit: {circuit.side}x{circuit.side} binary halves, {circuit.rounds} rounds, "
          f"256 secret bits, {circuit.gates:,} AND gates per repetition.", flush=True)
    print(f"Proof: {repetitions} repetitions, {estimated_bytes:,} bytes. "
          f"Interactive (2/3)^t reference exponent: {repetitions * math.log2(1.5):.1f}; "
          "this is not a security claim for this implementation.", flush=True)
    t = time.perf_counter()
    proof = zk_prove(secret, public, challenge, repetitions)
    proving_seconds = time.perf_counter() - t
    t = time.perf_counter()
    verified = zk_verify(public, challenge, proof)
    verification_seconds = time.perf_counter() - t
    if not verified:
        raise AssertionError("newly generated proof rejected")
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_bytes(proof)
    challenge_grids = zk_save_transport(out, "challenge", zk_json(challenge), args)
    grids = zk_save_transport(out, "proof", proof, args)
    reconstructed = zk_frame_decode(grids[::-1], args.colors)
    if reconstructed != proof or not zk_verify(public, challenge, reconstructed):
        raise AssertionError("optical byte round trip failed")
    print(f"Public-only verification: ACCEPT | prove {proving_seconds:.3f}s, verify {verification_seconds:.3f}s", flush=True)
    print(f"Optical response: {len(grids)}/{args.frames} frames, {args.size}x{args.size}, {args.colors} colors; "
          f"{capacity:,} payload bytes/frame. Frame-order-independent round trip: PASS.", flush=True)
    report = {"report_format": "neural-zkboo-cli-v2", "protocol": ZK_PROTOCOL, "mode": action,
              "environment": {"numpy": np.__version__, "device": "cpu", "arithmetic": "exact GF(2)"},
              "circuit": circuit.descriptor, "key_bits": 256, "and_gates_per_repetition": circuit.gates,
              "repetitions": repetitions, "minimum_repetitions": public["minimum_repetitions"],
              "required_repetitions": challenge["required_repetitions"], "interactive_soundness_reference_exponent": repetitions * math.log2(1.5),
              "proof_bytes": len(proof), "public_key_json_bytes": len(zk_json(public)),
              "public_verification": verified, "proving_seconds": proving_seconds,
              "verification_seconds": verification_seconds,
              "transport": {"format": "NPM1-python-simulation", "size": args.size, "colors": args.colors, "bits_per_cell": args.colors.bit_length() - 1,
                            "frames_used": len(grids), "frame_limit": args.frames, "payload_bytes_per_frame": capacity,
                            "challenge_frames": len(challenge_grids), "crc": "CRC32 detection only; no ECC"},
              "interpretation": "Research POC, not Picnic-compatible. Neural key-recovery hardness is unestablished. "
                                "Fiat-Shamir instantiation is unaudited; no post-quantum security claim. "
                                "Display grids transport serialized proofs and are independent of circuit size. "
                                "Simulator does not validate real-camera reliability."}
    if action == "demo":
        checks = zk_tamper_checks(public, challenge, proof)
        if not all(checks.values()):
            raise AssertionError(f"tampering test failed: {checks}")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first_ok = zk_accept_once(public, challenge, proof, cache)[0]
            replay_rejected = not zk_accept_once(public, challenge, proof, cache)[0]
        checks["first_nonce_accepted"] = first_ok
        checks["replayed_nonce_rejected"] = replay_rejected
        report["negative_checks"] = checks
        report["primitive_probe"] = zk_primitive_probe(secret, circuit)
        print(f"Transcript/context/key tampering and replay checks: {sum(checks.values())}/{len(checks)} passed.", flush=True)
    if args.camera:
        decoded = zk_camera_capture(grids, args.colors, args.seed + 40, args.camera_noise, args.cell_pixels)
        np.savez_compressed(out / "captured_frames.npz", symbols=decoded, colors=args.colors, palette=zk_palette(args.colors))
        packet_ok = False
        camera_verified = False
        reason = None
        try:
            captured_bytes = zk_frame_decode(decoded, args.colors)
            packet_ok = captured_bytes == proof
            camera_verified = zk_verify(public, challenge, captured_bytes)
        except ValueError as exc:
            reason = str(exc)
        symbol_accuracy = float((decoded == grids).mean())
        report["camera"] = {"assumptions": "Aligned cell means, calibrated RGB gain/offset, Gaussian sensor noise; no ECC/perspective.",
                            "pixel_noise_std": args.camera_noise, "cell_pixels": args.cell_pixels,
                            "symbol_accuracy": symbol_accuracy, "packet_exact": packet_ok,
                            "public_verification": camera_verified, "rejection_reason": reason}
        print(f"Camera simulation: {100 * symbol_accuracy:.5f}% symbols, "
              f"packet {'exact' if packet_ok else 'rejected'}, proof {'ACCEPT' if camera_verified else 'REJECT'}.", flush=True)
        if reason:
            print(f"Capture diagnostic: {reason}", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    zk_write_json(out / "summary.json", report)
    print(f"Finished in {report['elapsed_seconds']:.2f}s. Open {out.resolve() / 'proof_display.html'} to animate the response.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_optical_arguments(p, size=256, colors=16, frames=8, frame_help="maximum optical frames, 1..8")
    p.add_argument('--zk-action', choices=('demo', 'keygen', 'challenge', 'prove', 'verify'), default='demo')
    p.add_argument('--state-size', type=int, choices=(4, 8, 16), default=8)
    p.add_argument('--zk-rounds', type=int, default=16)
    p.add_argument('--repetitions', type=int, help='proof repetitions; defaults to the enrolled/challenge floor (new keys default 219)')
    p.add_argument('--minimum-repetitions', type=int, help='enrolled floor for keygen/demo; defaults to --repetitions or 219')
    p.add_argument('--required-repetitions', type=int, help='challenge requirement; defaults to --repetitions or the new-key floor')
    p.add_argument('--lock-id', default='demo-lock')
    p.add_argument('--action', default='unlock')
    for option in ('secret-key', 'public-key', 'challenge', 'proof', 'captured', 'replay-cache'):
        p.add_argument('--' + option, type=Path)
    p.add_argument('--seed', type=int, default=7, help='camera seed; keys and proofs always use OS randomness')
    p.add_argument('--out', type=Path, default=Path('neural_zkboo_results'))
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    validate_optical_arguments(p, args)
    if not 2 <= args.zk_rounds <= 64:
        p.error('zk-rounds must be 2..64')
    for value in (args.repetitions, args.minimum_repetitions, args.required_repetitions):
        if value is not None and not 1 <= value <= 512:
            p.error('repetitions, minimum-repetitions and required-repetitions must be 1..512')
    if not 0 <= args.seed < 2**63 - 100:
        p.error('seed must be in [0, 2**63-100)')
    return args


def main():
    args = parse_args()
    try:
        if args.self_test:
            zk_self_test()
        else:
            zk_run(args)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'Neural ZKBoo POC: {exc}') from exc


if __name__ == '__main__':
    main()
