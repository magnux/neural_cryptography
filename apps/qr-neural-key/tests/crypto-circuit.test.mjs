import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { BinaryCircuit } from '../dist/js/crypto/circuit.js';
import { concatBytes, fromHex, hex, packBits, unpackBits } from '../dist/js/crypto/bytes.js';
import { sha256 } from '../dist/js/crypto/hashes.js';

function randomBits(length, initial = 0x6a09e667) {
  let word = initial >>> 0;
  return Uint8Array.from({ length }, () => {
    word ^= word << 13; word ^= word >>> 17; word ^= word << 5;
    return word & 1;
  });
}

function randomBytes(length, initial = 0xbb67ae85) {
  let word = initial >>> 0;
  return Uint8Array.from({ length }, () => {
    word ^= word << 13; word ^= word >>> 17; word ^= word << 5;
    return word & 255;
  });
}

const secret = randomBytes(32);
const circuits = new Map();
function circuit(side = 8, rounds = 4) {
  const name = `${side}/${rounds}`;
  if (!circuits.has(name)) circuits.set(name, BinaryCircuit.create(side, rounds));
  return circuits.get(name);
}

function simulation(c) {
  const blocks = randomBits(c.lanes * c.nbits);
  const shares = randomBits(3 * 256, 0x3c6ef372);
  const key = unpackBits(secret);
  for (let i = 0; i < 256; i++) shares[512 + i] = key[i] ^ shares[i] ^ shares[256 + i];
  const tapes = randomBits(3 * c.gates, 0xa54ff53a);
  return { blocks, shares, tapes, ...c.execute(shares, blocks, [0, 1, 2], { tapes }) };
}

test('binary circuit is exactly invertible for every supported side and lane count', async () => {
  for (const side of [4, 8, 16]) {
    const c = await circuit(side);
    for (const lanes of new Set([1, c.anchorCount, c.lanes])) {
      const blocks = randomBits(lanes * c.nbits, side * lanes + 1);
      const original = blocks.slice(), keyCopy = secret.slice();
      const encrypted = c.apply(secret, blocks);
      assert.deepEqual(c.inverse(secret, encrypted), blocks);
      assert.deepEqual(c.apply(secret, c.inverse(secret, blocks)), blocks);
      assert.deepEqual(blocks, original);
      assert.deepEqual(secret, keyCopy);
    }
    assert.equal(c.anchors.length, c.anchorCount * c.nbits);
    assert.equal(c.gates, c.rounds * c.lanes * c.half);
    assert.equal(c.traceBytes * 8, c.gates);
    assert.equal(c.outputBytes * 8, c.lanes * c.nbits);
  }
});

test('public tables use exact quarter-turn offsets and invertible state permutations', async () => {
  const c = await circuit();
  // Original tap (-1,0) rotates to (0,-1), (1,0), (0,1), (-1,0).
  const expectedOrigins = [7, 8, 1, 56];
  for (let t = 0; t < 4; t++) assert.equal(c.convIndices[t][0][1][0], expectedOrigins[t]);
  assert.equal(new Set(c.permutation).size, c.nbits);
  for (let i = 0; i < c.nbits; i++) assert.equal(c.permutation[c.inversePermutation[i]], i);
  for (const branches of c.keyIndices) for (const selectors of branches) {
    assert.equal(selectors.length, c.half);
    assert.ok(selectors.every(i => i >= 0 && i < 256));
  }
  const again = await BinaryCircuit.create(8, 4);
  assert.deepEqual(c.digest, again.digest);
  assert.deepEqual(c.permutation, again.permutation);
  assert.deepEqual(c.anchors, again.anchors);
});

test('all three MPC shares reconstruct the clear circuit for every supported side', async () => {
  for (const side of [4, 8, 16]) {
    const c = await circuit(side), sim = simulation(c);
    const size = c.lanes * c.nbits;
    const reconstructed = Uint8Array.from({ length: size }, (_, i) =>
      sim.outputs[i] ^ sim.outputs[size + i] ^ sim.outputs[2 * size + i]);
    assert.deepEqual(reconstructed, c.apply(secret, sim.blocks));
    assert.equal(sim.trace.length, 3 * c.gates);
    const sharesCopy = sim.shares.slice(), tapesCopy = sim.tapes.slice(), blocksCopy = sim.blocks.slice();
    c.execute(sim.shares, sim.blocks, [0, 1, 2], { tapes: sim.tapes });
    assert.deepEqual(sim.shares, sharesCopy);
    assert.deepEqual(sim.tapes, tapesCopy);
    assert.deepEqual(sim.blocks, blocksCopy);
  }
});

test('every adjacent pair replays its original outputs and AND traces', async () => {
  const c = await circuit(), sim = simulation(c), outputSize = c.lanes * c.nbits;
  for (let first = 0; first < 3; first++) {
    const second = (first + 1) % 3;
    const shares = new Uint8Array(512), tapes = new Uint8Array(2 * c.gates);
    shares.set(sim.shares.subarray(first * 256, (first + 1) * 256));
    shares.set(sim.shares.subarray(second * 256, (second + 1) * 256), 256);
    tapes.set(sim.tapes.subarray(first * c.gates, (first + 1) * c.gates));
    tapes.set(sim.tapes.subarray(second * c.gates, (second + 1) * c.gates), c.gates);
    const suppliedTrace = sim.trace.slice(second * c.gates, (second + 1) * c.gates);
    const replay = c.execute(shares, sim.blocks, [first, second], { tapes, suppliedTrace });
    for (const [opened, party] of [first, second].entries()) {
      assert.deepEqual(replay.outputs.subarray(opened * outputSize, (opened + 1) * outputSize),
        sim.outputs.subarray(party * outputSize, (party + 1) * outputSize));
      assert.deepEqual(replay.trace.subarray(opened * c.gates, (opened + 1) * c.gates),
        sim.trace.subarray(party * c.gates, (party + 1) * c.gates));
    }
    // A changed second-party view is used as supplied; it cannot be silently
    // replaced with a reconstructed trace that happens to verify.
    suppliedTrace[0] ^= 1;
    const altered = c.execute(shares, sim.blocks, [first, second], { tapes, suppliedTrace });
    assert.equal(altered.trace[c.gates], suppliedTrace[0]);
    assert.notDeepEqual(altered.outputs, replay.outputs);
  }
});

test('the three-party AND sharing identity holds exhaustively for input and tape bits', () => {
  for (let encoded = 0; encoded < 512; encoded++) {
    const v = [encoded & 1, (encoded >> 1) & 1, (encoded >> 2) & 1];
    const w = [(encoded >> 3) & 1, (encoded >> 4) & 1, (encoded >> 5) & 1];
    const r = [(encoded >> 6) & 1, (encoded >> 7) & 1, (encoded >> 8) & 1];
    const shares = v.map((value, p) => {
      const next = (p + 1) % 3;
      return (value & w[p]) ^ (value & w[next]) ^ (v[next] & w[p]) ^ r[p] ^ r[next];
    });
    assert.equal(shares[0] ^ shares[1] ^ shares[2], (v[0] ^ v[1] ^ v[2]) & (w[0] ^ w[1] ^ w[2]));
  }
});

test('circuit validates bounds and share shapes before evaluating', async () => {
  for (const side of [0, 7, 32, NaN, '8']) await assert.rejects(BinaryCircuit.create(side, 4));
  for (const rounds of [0, 1, 65, 2.5, '4']) await assert.rejects(BinaryCircuit.create(8, rounds));
  const c = await circuit();
  assert.throws(() => new BinaryCircuit().apply(secret, new Uint8Array(128)), /create/);
  assert.throws(() => c.apply(secret.subarray(1), c.anchors));
  assert.throws(() => c.apply(secret, new Uint8Array(c.nbits - 1)), /complete blocks/);
  assert.throws(() => c.apply(secret, new Uint8Array((c.lanes + 1) * c.nbits)), /complete blocks/);
  const badBlocks = c.anchors.slice(); badBlocks[0] = 2;
  assert.throws(() => c.apply(secret, badBlocks), /only bits/);
  assert.throws(() => c.execute(new Uint8Array(256), c.anchors, [1]), /party/);
  assert.throws(() => c.execute(new Uint8Array(512), c.anchors, [0, 2]), /party/);
  assert.throws(() => c.execute(new Uint8Array(768), c.anchors, [1, 2, 0]), /party/);
  assert.throws(() => c.execute(new Uint8Array(768), c.anchors, [0, 1, 2]), /tapes/i);
  assert.throws(() => c.execute(new Uint8Array(255), c.anchors, [0]));
  assert.throws(() => c.execute(new Uint8Array(256), c.anchors, [0], { tapes: new Uint8Array() }), /Clear evaluation/);
});

test('Python fixtures match public tables, clear results, output shares, and AND traces exactly', async () => {
  const manifest = JSON.parse(await readFile(new URL('./fixtures/zk/manifest.json', import.meta.url), 'utf8'));
  const fixtureBytes = value => fromHex(value, value.length / 2);
  for (const fixture of manifest.cases) {
    const { side, rounds } = fixture.public.circuit, expected = fixture.expected;
    const c = await circuit(side, rounds), secretKey = fixtureBytes(fixture.secretHex);
    assert.equal(hex(c.digest), expected.circuitDigestHex, `${fixture.id}: circuit digest`);
    assert.deepEqual(Array.from(c.permutation), expected.permutation);
    assert.deepEqual(Array.from(c.inversePermutation), expected.inversePermutation);
    assert.deepEqual(c.keyIndices.map(branches => branches.map(indices => Array.from(indices))), expected.keyIndices);
    assert.deepEqual(c.convIndices.map(branches => branches.map(taps => taps.map(indices => Array.from(indices)))), expected.convIndices);
    assert.deepEqual(c.constants.map(value => hex(packBits(value))), expected.roundConstantsHex);
    assert.equal(hex(packBits(c.anchors)), expected.anchorsPackedHex);
    const anchorOutputs = c.apply(secretKey, c.anchors);
    assert.equal(hex(packBits(anchorOutputs)), expected.anchorOutputsHex);
    assert.deepEqual(c.inverse(secretKey, anchorOutputs), c.anchors);
    const challengeBlock = unpackBits(fixtureBytes(expected.challengeBlockHex));
    assert.equal(hex(packBits(c.apply(secretKey, challengeBlock))), expected.responseHex);
    const blocks = concatBytes(c.anchors, challengeBlock);
    const shares = concatBytes(...expected.firstSharesHex.map(value => unpackBits(fixtureBytes(value))));
    const tapes = concatBytes(...expected.firstTapePackedHex.map(value => unpackBits(fixtureBytes(value))));
    const executed = c.execute(shares, blocks, [0, 1, 2], { tapes });
    for (let party = 0; party < 3; party++) {
      const output = executed.outputs.subarray(party * c.outputBytes * 8, (party + 1) * c.outputBytes * 8);
      const trace = packBits(executed.trace.subarray(party * c.gates, (party + 1) * c.gates));
      assert.equal(hex(packBits(output)), expected.firstOutputSharesHex[party], `${fixture.id}: party ${party} output`);
      assert.equal(hex(await sha256(trace)), expected.firstAndTraceSha256[party], `${fixture.id}: party ${party} AND trace`);
      if (expected.firstAndTraceHex) assert.equal(hex(trace), expected.firstAndTraceHex[party]);
    }
  }
});
