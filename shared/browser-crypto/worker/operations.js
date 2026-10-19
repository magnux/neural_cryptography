/** Worker-only orchestration. The reusable proof engine has no lab state. */
import { PROTOCOL, canonicalJSON, exactKeys, hex } from '../crypto/bytes.js';
import { randomBytes, sha256 } from '../crypto/hashes.js';
import { BinaryCircuit } from '../crypto/circuit.js';
import { publicKey, prove, verify, proofSize, validatePublic, validateChallenge, MAX_PROOF_BYTES } from '../crypto/proof.js';
import { referenceFixture } from './reference-fixture.js';

const now = () => performance.now();
const copy = value => structuredClone(value);
const median = values => {
  const sorted = values.slice().sort((a, b) => a - b), mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
function schema(value, keys, name) {
  if (!exactKeys(value, keys)) throw new TypeError(`invalid ${name} fields`);
}
function profile(value) {
  schema(value, ['side', 'rounds', 'repetitions'], 'profile');
  if (![4, 8, 16].includes(value.side) || !Number.isInteger(value.rounds) || value.rounds < 2 || value.rounds > 64 ||
      !Number.isInteger(value.repetitions) || value.repetitions < 1 || value.repetitions > 512) {
    throw new RangeError('profile requires side 4/8/16, rounds 2..64, and repetitions 1..512');
  }
  return { ...value };
}
function labels(value) {
  // Reuse the protocol's exact UTF-8 and lone-surrogate validation without RNG.
  validateChallenge({ protocol: PROTOCOL, nonce: '00'.repeat(32), lock_id: value.lockId, action: value.action, required_repetitions: value.requestedRepetitions });
}
function freshChallenge(lockId, action, requestedRepetitions) {
  labels({ lockId, action, requestedRepetitions });
  return { protocol: PROTOCOL, nonce: hex(randomBytes(32)), lock_id: lockId, action, required_repetitions: requestedRepetitions };
}
async function proofInput(value) {
  schema(value, ['publicKey', 'challenge', 'proof'], 'proof request');
  if (!(value.proof instanceof Uint8Array) || value.proof.length > MAX_PROOF_BYTES) throw new RangeError('proof must be bytes, at most 8 MiB');
  validateChallenge(value.challenge);
  await validatePublic(value.publicKey);
}
function changedHex(value) { return (value[0] === '0' ? '1' : '0') + value.slice(1); }
function changedText(value) { return value === 'different' ? 'other' : 'different'; }

export class CryptoOperations {
  #secret = null;
  #public = null;
  #keyId = null;

  clear() {
    // Best effort only: JavaScript does not guarantee secure memory erasure.
    this.#secret?.fill(0);
    this.#secret = this.#public = this.#keyId = null;
  }

  async #prove(challenge, progress) {
    if (!this.#secret) throw new Error('Create an ephemeral key first; the worker has no key');
    validateChallenge(challenge);
    const started = now();
    const proof = await prove(this.#secret, this.#public, challenge, { onProgress: progress });
    const proveMs = now() - started;
    return { proof, publicKey: copy(this.#public), keyId: this.#keyId, challenge: copy(challenge),
      proofBytes: proof.length, proofSha256: hex(await sha256(proof)), proveMs,
      keyMinimumRepetitions: this.#public.minimum_repetitions, requestedRepetitions: challenge.required_repetitions,
      usedRepetitions: proof[8] * 256 + proof[9] };
  }

  async run(type, payload, progress = () => {}) {
    switch (type) {
      case 'createKey': {
        schema(payload, ['profile'], 'key request');
        const chosen = profile(payload.profile), started = now();
        progress({ phase: 'keygen', completed: 0, total: 1 });
        const circuit = await BinaryCircuit.create(chosen.side, chosen.rounds);
        const proofBytes = proofSize(circuit, chosen.repetitions);
        if (proofBytes > MAX_PROOF_BYTES) throw new RangeError('profile exceeds the 8 MiB proof limit');
        this.clear();
        const secret = randomBytes(32);
        try {
          const document = await publicKey(secret, circuit, chosen.repetitions);
          const keyId = hex(await sha256(canonicalJSON(document)));
          this.#secret = secret; this.#public = document; this.#keyId = keyId;
          progress({ phase: 'keygen', completed: 1, total: 1 });
          return { publicKey: copy(document), keyId, profile: copy(chosen), proofBytes, keygenMs: now() - started };
        } catch (error) { secret.fill(0); throw error; }
      }
      case 'newChallenge': {
        schema(payload, ['lockId', 'action', 'requestedRepetitions'], 'challenge request');
        return { challenge: freshChallenge(payload.lockId, payload.action, payload.requestedRepetitions), requestedRepetitions: payload.requestedRepetitions };
      }
      case 'prove': {
        schema(payload, ['challenge'], 'proof creation request');
        return this.#prove(payload.challenge, progress);
      }
      case 'verify': {
        await proofInput(payload);
        progress({ phase: 'verify', completed: 0, total: 1 });
        // Match the benchmark's engine-only timing. The UI separately records
        // full request duration, including input validation and worker startup.
        const started = now();
        const accepted = await verify(payload.publicKey, payload.challenge, payload.proof);
        const verifyMs = now() - started;
        progress({ phase: 'verify', completed: 1, total: 1 });
        return { accepted, verifyMs, proofBytes: payload.proof.length, proofSha256: hex(await sha256(payload.proof)),
          keyMinimumRepetitions: payload.publicKey.minimum_repetitions, requestedRepetitions: payload.challenge.required_repetitions,
          ...(accepted ? { usedRepetitions: payload.proof[8] * 256 + payload.proof[9] } : {}) };
      }
      case 'tamper': {
        const started = now(), checks = [];
        await proofInput(payload);
        const check = async (name, expected, document, challenge, proof) => {
          const tick = now(), accepted = await verify(document, challenge, proof);
          checks.push({ name, expected, accepted, passed: accepted === expected, ms: now() - tick });
          progress({ phase: 'tamper', completed: checks.length, total: 10 });
        };
        const { publicKey: document, challenge, proof } = payload;
        progress({ phase: 'tamper', completed: 0, total: 10 });
        await check('Valid proof', true, document, challenge, proof);
        await check('Changed nonce', false, document, { ...challenge, nonce: changedHex(challenge.nonce) }, proof);
        await check('Changed action', false, document, { ...challenge, action: changedText(challenge.action) }, proof);
        await check('Changed lock ID', false, document, { ...challenge, lock_id: changedText(challenge.lock_id) }, proof);
        await check('Changed public key', false, { ...document, anchor_outputs: changedHex(document.anchor_outputs) }, challenge, proof);
        const repetitions = document.minimum_repetitions === 512 ? 511 : document.minimum_repetitions + 1;
        await check('Changed repetition policy', false, { ...document, minimum_repetitions: repetitions }, challenge, proof);
        const requested = challenge.required_repetitions === 512 ? 511 : challenge.required_repetitions + 1;
        await check('Changed challenge repetitions', false, document, { ...challenge, required_repetitions: requested }, proof);
        const trace = proof.slice(); if (trace.length) trace[trace.length - 1] ^= 1;
        await check('Changed trace', false, document, challenge, trace);
        const response = proof.slice(); if (response.length > 42) response[42] ^= 1;
        await check('Changed response', false, document, challenge, response);
        await check('Truncated proof', false, document, challenge, proof.slice(0, -1));
        return { passed: checks.every(check => check.passed), checks, totalMs: now() - started };
      }
      case 'benchmark': {
        schema(payload, ['lockId', 'action', 'requestedRepetitions', 'runs'], 'benchmark request'); labels(payload);
        if (!Number.isInteger(payload.runs) || payload.runs < 1 || payload.runs > 10) throw new RangeError('benchmark runs must be 1..10');
        if (!this.#secret) throw new Error('Create an ephemeral key first; the worker has no key');
        const started = now(), runs = []; let latest;
        progress({ phase: 'benchmark', completed: 0, total: payload.runs });
        for (let index = 0; index < payload.runs; index++) {
          latest = await this.#prove(freshChallenge(payload.lockId, payload.action, payload.requestedRepetitions), proofProgress =>
            progress({ ...proofProgress, phase: `benchmark-${index + 1}-${proofProgress.phase}` }));
          const tick = now(), accepted = await verify(latest.publicKey, latest.challenge, latest.proof), verifyMs = now() - tick;
          latest = { ...latest, accepted, verifyMs };
          runs.push({ proveMs: latest.proveMs, verifyMs, proofBytes: latest.proofBytes, accepted,
            keyMinimumRepetitions: latest.keyMinimumRepetitions, requestedRepetitions: latest.requestedRepetitions, usedRepetitions: latest.usedRepetitions });
          progress({ phase: 'benchmark', completed: index + 1, total: payload.runs });
          if (!accepted) throw new Error('A freshly generated benchmark proof was rejected');
        }
        return { runs, summary: { proveMedianMs: median(runs.map(run => run.proveMs)), verifyMedianMs: median(runs.map(run => run.verifyMs)) },
          latest, keyMinimumRepetitions: latest.keyMinimumRepetitions, requestedRepetitions: latest.requestedRepetitions, usedRepetitions: latest.usedRepetitions,
          totalMs: now() - started };
      }
      case 'selfTest': {
        schema(payload, [], 'reference check request');
        const started = now(), fixture = referenceFixture(), checks = [];
        progress({ phase: 'self-test', completed: 0, total: 2 });
        let tick = now();
        const matches = hex(await sha256(fixture.proof)) === fixture.proofSha256;
        checks.push({ name: 'Python fixture SHA-256', expected: true, accepted: matches, passed: matches, ms: now() - tick });
        progress({ phase: 'self-test', completed: 1, total: 2 }); tick = now();
        const accepted = await verify(fixture.publicKey, fixture.challenge, fixture.proof);
        checks.push({ name: 'Python Unicode proof', expected: true, accepted, passed: accepted, ms: now() - tick });
        progress({ phase: 'self-test', completed: 2, total: 2 });
        return { passed: checks.every(check => check.passed), checks, totalMs: now() - started };
      }
      default: throw new TypeError('Unsupported Crypto Lab operation');
    }
  }
}
