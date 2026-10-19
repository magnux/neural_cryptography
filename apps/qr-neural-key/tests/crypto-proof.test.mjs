import test from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import { PROTOCOL, concatBytes, u16 } from '../dist/js/crypto/bytes.js';
import { BinaryCircuit } from '../dist/js/crypto/circuit.js';
import {
  publicKey, validatePublic, validateChallenge, proofSize, prove, verify, MAX_PROOF_BYTES,
  packChallenges, unpackChallenges, statement,
} from '../dist/js/crypto/proof.js';

if (!globalThis.crypto) globalThis.crypto = webcrypto;
const secret = Uint8Array.from({ length: 32 }, (_, i) => i);
const challenge = (required_repetitions = 1) => ({ protocol: PROTOCOL, nonce: '12'.repeat(32), lock_id: 'test-lock', action: 'unlock', required_repetitions });
async function enrolled(side = 4, rounds = 2, repetitions = 6) {
  const circuit = await BinaryCircuit.create(side, rounds);
  return { circuit, document: await publicKey(secret, circuit, repetitions) };
}

test('browser proof engine round trips every supported state size', async () => {
  for (const side of [4, 8, 16]) {
    const { circuit, document } = await enrolled(side, 4, 6);
    const message = challenge();
    const proof = await prove(secret, document, message);
    assert.equal(proof.length, proofSize(circuit, 6));
    assert.equal(await verify(document, message, proof), true);
  }
});

test('default circuit proof size matches the Python wire formula', async () => {
  const circuit = await BinaryCircuit.create(8, 16);
  for (const repetitions of [1, 32, 219, 512]) {
    assert.equal(proofSize(circuit, repetitions), 58 + Math.ceil(repetitions / 4) + 544 * repetitions);
  }
  assert.throws(() => proofSize(circuit, 0));
  assert.throws(() => proofSize(circuit, true));
  assert.throws(() => proofSize(circuit, 513));
});

test('compact challenge encoding is MSB-first, bounded, and canonical', () => {
  assert.deepEqual(packChallenges(Uint8Array.of(0, 1, 2, 0, 2)), Uint8Array.of(0x18, 0x80));
  for (const count of [1, 2, 3, 4, 5, 31, 32, 33, 219, 512]) {
    const choices = Uint8Array.from({ length: count }, (_, i) => i % 3);
    assert.deepEqual(unpackChallenges(packChallenges(choices), count), choices);
  }
  assert.throws(() => packChallenges(Uint8Array.of(3)));
  assert.throws(() => packChallenges(new Uint8Array()));
  assert.throws(() => packChallenges(new Uint8Array(513)));
  assert.throws(() => unpackChallenges(Uint8Array.of(0xc0), 1));
  assert.throws(() => unpackChallenges(Uint8Array.of(0x01), 1));
  assert.throws(() => unpackChallenges(Uint8Array.of(0x01), 2));
  assert.throws(() => unpackChallenges(Uint8Array.of(0x01), 3));
  assert.throws(() => unpackChallenges(Uint8Array.of(0), 5));
  assert.throws(() => unpackChallenges(Uint8Array.of(0, 0), 4));
});

test('the same enrolled key accepts varying challenge requirements with fresh randomness', async () => {
  const { circuit, document } = await enrolled(4, 2, 4);
  for (const required of [1, 4, 7, 32]) {
    const message = challenge(required);
    const proof = await prove(secret, document, message);
    const actual = Math.max(4, required);
    assert.equal(proof[8] * 256 + proof[9], actual);
    assert.equal(proof.length, proofSize(circuit, actual));
    assert.equal(await verify(document, message, proof), true);
    assert.notDeepEqual(proof.subarray(10, 42), (await prove(secret, document, message)).subarray(10, 42));
  }
  const message = challenge(5);
  const proof = await prove(secret, document, message, { repetitions: 32 });
  assert.equal(proof.length, proofSize(circuit, 32));
  assert.equal(await verify(document, message, proof), true);
  for (const repetitions of [0, 3, 4, 513, 1.5, true, '32', null]) {
    await assert.rejects(prove(secret, document, message, { repetitions }), /repetitions/);
  }
});

test('proofs enforce both minimums and bind challenge requirements against downgrade', async () => {
  const { document } = await enrolled(4, 2, 4);
  const high = challenge(33);
  const low = challenge(32);
  const weakerProof = await prove(secret, document, low);
  assert.equal(await verify(document, high, weakerProof), false);
  const strongerProof = await prove(secret, document, high);
  assert.equal(await verify(document, low, strongerProof), false);
  await assert.rejects(prove(secret, document, high, { repetitions: 32 }), /below/);
  assert.equal(await verify({ ...document, minimum_repetitions: 34 }, high, strongerProof), false);
});

test('actual repetition count is mandatory in the statement and bound even above both floors', async () => {
  const { circuit, document } = await enrolled(4, 2, 1);
  const message = challenge();
  const response = new Uint8Array(circuit.nbytes);
  await assert.rejects(statement(document, message, circuit, response), /repetitions/);
  const first = await statement(document, message, circuit, response, 32);
  const second = await statement(document, message, circuit, response, 33);
  assert.notDeepEqual(first.context, second.context);

  const proof = await prove(secret, document, message, { repetitions: 33 });
  const choicesAt = 42 + circuit.nbytes;
  const commitmentsAt = choicesAt + 9;
  const seedsAt = commitmentsAt + 33 * 32;
  const sharesAt = seedsAt + 33 * 64;
  const tracesAt = sharesAt + 33 * 64;
  // Remove the 33rd repetition from every column, keeping a valid 32-repetition
  // wire shape above both floors. The statement hash must still reject it.
  const shortened = concatBytes(
    proof.subarray(0, 8), u16(32), proof.subarray(10, choicesAt),
    proof.subarray(choicesAt, choicesAt + 8),
    proof.subarray(commitmentsAt, commitmentsAt + 32 * 32),
    proof.subarray(seedsAt, seedsAt + 32 * 64),
    proof.subarray(sharesAt, sharesAt + 32 * 64),
    proof.subarray(tracesAt, tracesAt + 32 * circuit.traceBytes),
  );
  assert.equal(shortened.length, proofSize(circuit, 32));
  assert.equal(await verify(document, message, shortened), false);
});

test('wire parser rejects forbidden choices, nonzero padding, old magic, and reordered openings', async () => {
  const { circuit, document } = await enrolled(4, 2, 33);
  const message = challenge();
  const proof = await prove(secret, document, message);
  const choicesAt = 42 + circuit.nbytes;
  const invalid = proof.slice();
  invalid[choicesAt] |= 0xc0;
  assert.equal(await verify(document, message, invalid), false);
  const padding = proof.slice();
  padding[choicesAt + Math.ceil(33 / 4) - 1] |= 1;
  assert.equal(await verify(document, message, padding), false);
  const previousFormat = proof.slice();
  previousFormat[7] = 49;
  assert.equal(await verify(document, message, previousFormat), false);
  const seedsAt = choicesAt + Math.ceil(33 / 4) + 33 * 32;
  const swapped = proof.slice();
  swapped.set(proof.subarray(seedsAt + 32, seedsAt + 64), seedsAt);
  swapped.set(proof.subarray(seedsAt, seedsAt + 32), seedsAt + 32);
  assert.equal(await verify(document, message, swapped), false);
});

test('public and challenge schemas reject extra fields, noncanonical hex, and invalid types', async () => {
  const { document } = await enrolled();
  for (const changed of [
    { ...document, old_version: 1 },
    { ...document, protocol: `${PROTOCOL}-other` },
    { ...document, minimum_repetitions: true },
    { ...document, minimum_repetitions: 513 },
    { ...document, anchor_outputs: document.anchor_outputs.toUpperCase() },
    { ...document, circuit: { ...document.circuit, rounds: true } },
    { ...document, circuit: { ...document.circuit, key_bits: '256' } },
    { ...document, circuit: { ...document.circuit, old: true } },
  ]) await assert.rejects(validatePublic(changed));

  for (const changed of [
    { protocol: PROTOCOL, nonce: '12'.repeat(32), lock_id: 'test-lock', action: 'unlock' },
    { ...challenge(), extra: 1 },
    { ...challenge(), required_repetitions: 0 },
    { ...challenge(), required_repetitions: 513 },
    { ...challenge(), required_repetitions: true },
    { ...challenge(), required_repetitions: 1.5 },
    { ...challenge(), required_repetitions: '32' },
    { ...challenge(), nonce: 'AB'.repeat(32) },
    { ...challenge(), nonce: '12'.repeat(31) },
    { ...challenge(), action: '' },
    { ...challenge(), lock_id: '\ud800' },
    { ...challenge(), lock_id: '\udc00' },
    { ...challenge(), action: 'ñ'.repeat(129) },
  ]) assert.throws(() => validateChallenge(changed));
  assert.ok(validateChallenge({ ...challenge(), lock_id: '门🔑', action: 'é'.repeat(128) }) instanceof Uint8Array);
});

test('verification rejects corruption in every serialized proof component', async () => {
  const { circuit, document } = await enrolled(4, 2, 32);
  const message = challenge();
  const proof = await prove(secret, document, message);
  const reps = document.minimum_repetitions;
  const choicesAt = 42 + circuit.nbytes;
  const commitmentsAt = choicesAt + Math.ceil(reps / 4);
  const seedsAt = commitmentsAt + reps * 32;
  const sharesAt = seedsAt + reps * 2 * 32;
  for (const offset of [0, 8, 10, 42, choicesAt, commitmentsAt, seedsAt, sharesAt, proof.length - 1]) {
    const altered = proof.slice();
    altered[offset] ^= 1;
    assert.equal(await verify(document, message, altered), false, `accepted corruption at byte ${offset}`);
  }
  assert.equal(await verify(document, message, proof.subarray(0, proof.length - 1)), false);
  const extra = new Uint8Array(proof.length + 1);
  extra.set(proof);
  assert.equal(await verify(document, message, extra), false);
  assert.equal(await verify(document, message, new Uint8Array(MAX_PROOF_BYTES + 1)), false);
  assert.equal(await verify(document, message, []), false);
});

test('proofs bind enrolled public key, challenge identity, action, nonce, and repetition policy', async () => {
  const { document } = await enrolled(4, 2, 32);
  const message = challenge();
  const proof = await prove(secret, document, message);
  for (const [field, value] of [['nonce', '34'.repeat(32)], ['lock_id', 'other-lock'], ['action', 'other-action'], ['required_repetitions', 2]]) {
    assert.equal(await verify(document, { ...message, [field]: value }, proof), false);
  }
  // This tiny two-round test circuit does not necessarily read every key bit.
  const otherSecret = secret.map(value => value ^ 255);
  await assert.rejects(prove(otherSecret, document, message), /secret does not match/);
  assert.equal(await verify({ ...document, minimum_repetitions: 33 }, message, proof), false);
  // Even lowering a policy changes the canonical statement. It is not an
  // unauthenticated hint in the proof, and must remain pinned by enrollment.
  assert.equal(await verify({ ...document, minimum_repetitions: 31 }, message, proof), false);
  const otherPublic = structuredClone(document);
  otherPublic.anchor_outputs = (otherPublic.anchor_outputs.startsWith('00') ? '01' : '00') + otherPublic.anchor_outputs.slice(2);
  assert.equal(await verify(otherPublic, message, proof), false);
});

test('proof generation always uses fresh randomness and reports bounded progress', async () => {
  const { document } = await enrolled();
  const progress = [];
  const first = await prove(secret, document, challenge(), { onProgress: update => progress.push(update) });
  const second = await prove(secret, document, challenge());
  assert.notDeepEqual(first, second);
  assert.equal(progress[0].completed, 0);
  assert.deepEqual(progress.at(-1), { phase: 'complete', completed: 6, total: 6 });
  assert.ok(progress.every(update => update.completed >= 0 && update.completed <= update.total));
});

test('proof generation cancellation rejects before and between repetitions', async () => {
  const { document } = await enrolled();
  const before = new AbortController();
  before.abort();
  await assert.rejects(prove(secret, document, challenge(), { signal: before.signal }), { name: 'AbortError' });
  const during = new AbortController();
  await assert.rejects(prove(secret, document, challenge(), {
    signal: during.signal,
    onProgress(update) { if (update.completed === 2) during.abort(); },
  }), { name: 'AbortError' });
  assert.deepEqual(secret, Uint8Array.from({ length: 32 }, (_, i) => i));
});

test('asynchronous proving and verification snapshot caller-controlled documents and bytes', async () => {
  const { document } = await enrolled();
  const originalPublic = structuredClone(document);
  const localChallenge = challenge();
  const localSecret = secret.slice();
  const proving = prove(localSecret, document, localChallenge);
  localSecret.fill(0);
  document.minimum_repetitions = 512;
  document.anchor_outputs = '00'.repeat(originalPublic.anchor_outputs.length / 2);
  localChallenge.action = 'changed';
  localChallenge.required_repetitions = 512;
  const proof = await proving;
  assert.equal(await verify(originalPublic, challenge(), proof), true);

  const verifyPublic = structuredClone(originalPublic);
  const verifyChallenge = challenge();
  const verifying = verify(verifyPublic, verifyChallenge, proof);
  proof.fill(0);
  verifyPublic.minimum_repetitions = 512;
  verifyChallenge.action = 'changed';
  verifyChallenge.required_repetitions = 512;
  assert.equal(await verifying, true);
});

test('Node Buffer inputs are copied rather than wiped or retained as shared slices', async () => {
  const { document } = await enrolled();
  const keyBuffer = Buffer.from(secret);
  const proof = await prove(keyBuffer, document, challenge());
  assert.deepEqual(keyBuffer, Buffer.from(secret));
  const proofBuffer = Buffer.from(proof);
  const verifying = verify(document, challenge(), proofBuffer);
  proofBuffer.fill(0);
  assert.equal(await verifying, true);
});
