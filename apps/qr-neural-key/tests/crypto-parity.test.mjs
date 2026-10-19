import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { canonicalJSON, concatBytes, fromHex, hex, packBits, unpackBits } from '../dist/js/crypto/bytes.js';
import { sha256, shake256 } from '../dist/js/crypto/hashes.js';
import { publicKey, validatePublic, statement, tape, commit, challenges, packChallenges, unpackChallenges, prove, verify, proofSize } from '../dist/js/crypto/proof.js';

const directory = new URL('./fixtures/zk/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', directory), 'utf8'));
const bytes = value => fromHex(value, value.length / 2);

// Public fixture entropy is confined to this isolated test process. The served
// prover has no seed option and always uses crypto.getRandomValues().
async function withFixtureEntropy(fixture, action) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  const native = globalThis.crypto;
  const source = shake256(bytes(fixture.entropySeedHex), fixture.expected.entropyBytes);
  let position = 0;
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: {
    subtle: native.subtle,
    getRandomValues(array) {
      assert.ok(position + array.byteLength <= source.length, 'fixture entropy exhausted');
      new Uint8Array(array.buffer, array.byteOffset, array.byteLength)
        .set(source.subarray(position, position + array.byteLength));
      position += array.byteLength;
      return array;
    },
  } });
  try {
    const result = await action();
    assert.equal(position, source.length, 'Python and JavaScript consume the same entropy');
    return result;
  } finally {
    Object.defineProperty(globalThis, 'crypto', descriptor);
  }
}

// Reconstruct all three views independently of the compact proof parser. The
// wire only contains the hidden commitment and the two opened views; all full
// arrays below come from the public deterministic fixture stream.
async function reconstructViews(fixture, circuit, blocks, context, salt) {
  const repetitions = fixture.expected.repetitions;
  const source = shake256(bytes(fixture.entropySeedHex), fixture.expected.entropyBytes);
  assert.deepEqual(source.subarray(0, 32), salt);
  const seeds = source.subarray(32, 32 + repetitions * 96);
  const shared = source.subarray(32 + repetitions * 96);
  const secret = bytes(fixture.secretHex);
  const commitments = new Uint8Array(repetitions * 96);
  const outputs = new Uint8Array(repetitions * 3 * circuit.outputBytes);
  const openedSeeds = new Uint8Array(repetitions * 64);
  const openedShares = new Uint8Array(repetitions * 64);
  const secondTraces = new Uint8Array(repetitions * circuit.traceBytes);
  const hiddenCommitments = new Uint8Array(repetitions * 32);
  for (let repetition = 0; repetition < repetitions; repetition++) {
    const partySeeds = [0, 1, 2].map(p => seeds.subarray(repetition * 96 + p * 32, repetition * 96 + (p + 1) * 32));
    const shares = [shared.subarray(repetition * 64, repetition * 64 + 32),
      shared.subarray(repetition * 64 + 32, (repetition + 1) * 64)];
    shares.push(Uint8Array.from(secret, (value, i) => value ^ shares[0][i] ^ shares[1][i]));
    const tapes = partySeeds.map((seed, party) => tape(seed, salt, repetition, party, context, circuit.gates));
    const simulation = circuit.execute(concatBytes(...shares.map(unpackBits)), blocks, [0, 1, 2],
      { tapes: concatBytes(...tapes) });
    const traces = [0, 1, 2].map(p => packBits(simulation.trace.subarray(p * circuit.gates, (p + 1) * circuit.gates)));
    for (let party = 0; party < 3; party++) {
      const output = packBits(simulation.outputs.subarray(party * circuit.outputBytes * 8, (party + 1) * circuit.outputBytes * 8));
      outputs.set(output, (repetition * 3 + party) * circuit.outputBytes);
      const commitment = await commit(context, salt, repetition, party, partySeeds[party], shares[party], traces[party], output);
      commitments.set(commitment, repetition * 96 + party * 32);
      if (repetition === 0) {
        assert.equal(hex(commitment), fixture.expected.firstCommitmentsHex[party]);
        assert.equal(hex(output), fixture.expected.firstOutputSharesHex[party]);
      }
    }
    const first = fixture.expected.challengeTrits[repetition], second = (first + 1) % 3, hidden = (first + 2) % 3;
    openedSeeds.set(concatBytes(partySeeds[first], partySeeds[second]), repetition * 64);
    openedShares.set(concatBytes(shares[first], shares[second]), repetition * 64);
    secondTraces.set(traces[second], repetition * circuit.traceBytes);
    hiddenCommitments.set(commitments.subarray(repetition * 96 + hidden * 32, repetition * 96 + (hidden + 1) * 32), repetition * 32);
  }
  return { commitments, outputs, openedSeeds, openedShares, secondTraces, hiddenCommitments };
}

for (const fixture of manifest.cases) {
  test(`Python proof parity: ${fixture.id}`, async () => {
    const expected = fixture.expected;
    const pythonProof = new Uint8Array(await readFile(new URL(fixture.proofFile, directory)));
    const circuit = await validatePublic(fixture.public);
    const repetitions = expected.repetitions;
    assert.equal(hex(await sha256(pythonProof)), fixture.proofSha256, 'fixture SHA-256');
    assert.equal(pythonProof.length, expected.proofBytes);
    assert.equal(proofSize(circuit, repetitions), expected.proofBytes);
    assert.equal(hex(canonicalJSON(circuit.descriptor)), expected.descriptorCanonicalJSONHex);
    assert.equal(hex(canonicalJSON(fixture.public)), expected.publicCanonicalJSONHex);
    assert.equal(hex(canonicalJSON(fixture.challenge)), expected.challengeCanonicalJSONHex);
    assert.deepEqual(await publicKey(bytes(fixture.secretHex), circuit, fixture.public.minimum_repetitions), fixture.public);

    const { blocks, expected: reconstructed, context } = await statement(
      fixture.public, fixture.challenge, circuit, bytes(expected.responseHex), repetitions);
    assert.equal(hex(packBits(blocks.subarray(circuit.anchorCount * circuit.nbits))), expected.challengeBlockHex);
    assert.equal(hex(context), expected.contextHex);
    assert.equal(hex(reconstructed), expected.anchorOutputsHex + expected.responseHex);
    const salt = bytes(expected.saltHex);
    assert.deepEqual(pythonProof.subarray(10, 42), salt);
    const tapes = expected.firstSeedsHex.map((seed, party) => {
      const value = tape(bytes(seed), salt, 0, party, context, circuit.gates);
      assert.equal(hex(packBits(value)), expected.firstTapePackedHex[party], `party ${party} tape`);
      return value;
    });
    const simulation = circuit.execute(
      concatBytes(...expected.firstSharesHex.map(value => unpackBits(bytes(value)))),
      blocks, [0, 1, 2], { tapes: concatBytes(...tapes) });
    for (let party = 0; party < 3; party++) {
      const trace = packBits(simulation.trace.subarray(party * circuit.gates, (party + 1) * circuit.gates));
      const output = packBits(simulation.outputs.subarray(party * circuit.outputBytes * 8, (party + 1) * circuit.outputBytes * 8));
      assert.equal(hex(await commit(context, salt, 0, party, bytes(expected.firstSeedsHex[party]),
        bytes(expected.firstSharesHex[party]), trace, output)), expected.firstCommitmentsHex[party]);
    }
    const { commitments, outputs, openedSeeds, openedShares, secondTraces, hiddenCommitments } =
      await reconstructViews(fixture, circuit, blocks, context, salt);
    assert.equal(hex(await sha256(commitments)), expected.commitmentsSha256);
    assert.equal(hex(await sha256(outputs)), expected.outputsSha256);
    assert.deepEqual(Array.from(challenges(context, salt, commitments, outputs, repetitions)), expected.challengeTrits);
    const packed = packChallenges(Uint8Array.from(expected.challengeTrits));
    assert.equal(hex(packed), expected.packedChallengesHex);
    assert.deepEqual(Array.from(unpackChallenges(packed, repetitions)), expected.challengeTrits);
    assert.equal(hex(await sha256(hiddenCommitments)), expected.hiddenCommitmentsSha256);
    assert.deepEqual(pythonProof.subarray(42 + circuit.nbytes),
      concatBytes(packed, hiddenCommitments, openedSeeds, openedShares, secondTraces),
      'compact wire contains exactly packed trits, hidden commitments, opened seeds/shares, and second traces');

    assert.equal(await verify(fixture.public, fixture.challenge, pythonProof), true, 'Python proof verifies in JS');
    const jsProof = await withFixtureEntropy(fixture,
      () => prove(bytes(fixture.secretHex), fixture.public, fixture.challenge));
    assert.deepEqual(jsProof, pythonProof, 'entire serialized proof matches Python byte for byte');
    assert.equal(await verify(fixture.public, fixture.challenge, jsProof), true);
  });
}
