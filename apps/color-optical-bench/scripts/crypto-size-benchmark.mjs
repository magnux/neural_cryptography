#!/usr/bin/env node
/** Local measurements only: bytes, engine time, and challenge policy checks.
 * Does not measure cameras, transport, or the security of the neural primitive.
 */
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { PROTOCOL, hex } from '../dist/js/crypto/bytes.js';
import { randomBytes } from '../dist/js/crypto/hashes.js';
import { BinaryCircuit } from '../dist/js/crypto/circuit.js';
import { publicKey, proofSize, prove, verify } from '../dist/js/crypto/proof.js';

const counts = [32, 64, 110, 160, 219];
const circuit = await BinaryCircuit.create(8, 16);
const secret = randomBytes(32);
const document = await publicKey(secret, circuit, counts[0]);
const measurements = [];
try {
  for (const repetitions of counts) {
    const challenge = {
      protocol: PROTOCOL,
      nonce: hex(randomBytes(32)),
      lock_id: 'local-size-benchmark',
      action: 'measure-proof',
      required_repetitions: repetitions,
    };
    const start = performance.now();
    const proof = await prove(secret, document, challenge);
    const proved = performance.now();
    assert.equal(await verify(document, challenge, proof), true);
    const verified = performance.now();
    assert.equal(proof.length, proofSize(circuit, repetitions));
    assert.equal(proof[8] * 256 + proof[9], repetitions);
    // The expected verifier challenge is pinned, not read from the proof.
    assert.equal(await verify(document, {
      ...challenge, required_repetitions: repetitions + 1,
    }, proof), false);
    const previousBytes = 58 + 752 * repetitions;
    measurements.push({
      repetitions,
      proofBytes: proof.length,
      previousEncodingBytes: previousBytes,
      reductionPercent: Number((100 * (1 - proof.length / previousBytes)).toFixed(2)),
      repetitionTermExponent: Number((repetitions * Math.log2(1.5)).toFixed(2)),
      proveMs: Number((proved - start).toFixed(2)),
      verifyMs: Number((verified - proved).toFixed(2)),
      accepted: true,
      higherChallengeRejected: true,
    });
  }
} finally {
  secret.fill(0);
}
console.log(JSON.stringify({
  protocol: PROTOCOL,
  runtime: process.version,
  circuit: circuit.descriptor,
  keyMinimumRepetitions: document.minimum_repetitions,
  sameKeyAcrossMeasurements: true,
  note: 'Local engine timings only. Repetition exponents are not overall security ratings.',
  measurements,
}, null, 2));
