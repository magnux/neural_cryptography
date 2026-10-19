/** Exact browser port of the NeuralPicnic-POC-MPC3-v2 Python proof protocol.
 *
 * This module verifies mathematical statements. Enrollment, outstanding-nonce
 * policy, expiry, and one-time acceptance belong to the application layer.
 * No secret key is needed by verify(). This is an unaudited research POC.
 */
import {
  PROTOCOL, canonicalJSON, framed, concatBytes, packBits, unpackBits,
  hex, fromHex, u32, u16, sameBytes, exactKeys, assertBytes,
} from './bytes.js';
import { hash, shake256, randomBytes } from './hashes.js';
import { BinaryCircuit } from './circuit.js';

const MAGIC = Uint8Array.of(78, 80, 90, 75, 48, 48, 48, 50); // NPZK0002
const HEADER_BYTES = 42;
const KEY_BYTES = 32;
export const MAX_PROOF_BYTES = 8 * 1024 * 1024;

function checkRepetitions(repetitions) {
  if (!Number.isInteger(repetitions) || repetitions < 1 || repetitions > 512) {
    throw new RangeError('repetitions must be an integer from 1 to 512');
  }
}

function checkCircuit(circuit) {
  if (!(circuit instanceof BinaryCircuit)) throw new TypeError('expected a BinaryCircuit');
}

// Copy the small public documents before the first await. Callers cannot change
// a challenge, enrolled key, or repetition policy during an asynchronous proof.
function publicSnapshot(document) {
  if (!exactKeys(document, ['protocol', 'circuit', 'minimum_repetitions', 'anchor_outputs'])) {
    throw new TypeError('invalid public-key schema');
  }
  if (document.protocol !== PROTOCOL) throw new TypeError('unsupported proof protocol');
  const descriptor = document.circuit;
  if (!exactKeys(descriptor, ['name', 'side', 'rounds', 'key_bits']) ||
      descriptor.name !== 'binary-rotconv-feistel-v1' || descriptor.key_bits !== 256 ||
      ![4, 8, 16].includes(descriptor.side) ||
      !Number.isInteger(descriptor.rounds) || descriptor.rounds < 2 || descriptor.rounds > 64) {
    throw new TypeError('invalid circuit descriptor');
  }
  checkRepetitions(document.minimum_repetitions);
  const blockBits = 2 * descriptor.side * descriptor.side;
  fromHex(document.anchor_outputs, Math.ceil(256 / blockBits) * blockBits / 8, 'anchor outputs');
  return {
    protocol: PROTOCOL,
    circuit: { name: descriptor.name, side: descriptor.side, rounds: descriptor.rounds, key_bits: 256 },
    minimum_repetitions: document.minimum_repetitions,
    anchor_outputs: document.anchor_outputs,
  };
}

function validUtf8Text(text) {
  // UTF-8 never uses fewer bytes than UTF-16 code units for valid text.
  // Reject oversized input before allocating an encoded copy.
  if (typeof text !== 'string' || text.length < 1 || text.length > 256) return false;
  // TextEncoder replaces lone surrogates, whereas Python's UTF-8 encoding
  // rejects them. Reject explicitly to keep the accepted schemas identical.
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = text.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return false;
    } else if (code >= 0xdc00 && code <= 0xdfff) return false;
  }
  const size = new TextEncoder().encode(text).length;
  return size >= 1 && size <= 256;
}

export function validateChallenge(challenge) {
  if (!exactKeys(challenge, ['protocol', 'nonce', 'lock_id', 'action', 'required_repetitions'])) {
    throw new TypeError('invalid challenge schema');
  }
  if (challenge.protocol !== PROTOCOL) throw new TypeError('unsupported challenge protocol');
  checkRepetitions(challenge.required_repetitions);
  fromHex(challenge.nonce, 32, 'challenge nonce');
  for (const field of ['lock_id', 'action']) {
    if (!validUtf8Text(challenge[field])) throw new TypeError(`${field} must contain 1..256 UTF-8 bytes`);
  }
  return canonicalJSON(challenge);
}

function challengeSnapshot(challenge) {
  validateChallenge(challenge);
  return {
    protocol: challenge.protocol, nonce: challenge.nonce, lock_id: challenge.lock_id,
    action: challenge.action, required_repetitions: challenge.required_repetitions,
  };
}

export async function publicKey(secret, circuit, repetitions) {
  assertBytes(secret, KEY_BYTES, 'secret key');
  checkCircuit(circuit);
  checkRepetitions(repetitions);
  return {
    protocol: PROTOCOL,
    circuit: { ...circuit.descriptor },
    minimum_repetitions: repetitions,
    anchor_outputs: hex(packBits(circuit.apply(secret, circuit.anchors))),
  };
}

export async function validatePublic(document) {
  const snapshot = publicSnapshot(document);
  return BinaryCircuit.create(snapshot.circuit.side, snapshot.circuit.rounds);
}

export async function statement(document, challenge, circuit, response, repetitions) {
  checkCircuit(circuit);
  checkRepetitions(repetitions);
  assertBytes(response, circuit.nbytes, 'challenge response');
  const message = validateChallenge(challenge);
  const challengeBlock = unpackBits(shake256(framed('challenge-block', message), circuit.nbytes));
  const blocks = concatBytes(circuit.anchors, challengeBlock);
  const expected = concatBytes(fromHex(document.anchor_outputs, circuit.anchorCount * circuit.nbytes, 'anchor outputs'), response);
  const context = await hash('statement', canonicalJSON(document), message, response, u16(repetitions));
  return { blocks, expected, context };
}

function repetitionParty(repetition, party) {
  if (!Number.isInteger(repetition) || repetition < 0 || repetition >= 512 ||
      !Number.isInteger(party) || party < 0 || party > 2) {
    throw new RangeError('invalid repetition or party index');
  }
  return concatBytes(u32(repetition), Uint8Array.of(party));
}

export function tape(seed, salt, repetition, party, context, gateCount) {
  assertBytes(seed, 32, 'tape seed');
  assertBytes(salt, 32, 'proof salt');
  assertBytes(context, 32, 'statement context');
  if (!Number.isInteger(gateCount) || gateCount < 0 || gateCount > 32768 || gateCount % 8 !== 0) {
    throw new RangeError('invalid circuit gate count');
  }
  return unpackBits(shake256(framed('random-tape', context, salt, repetitionParty(repetition, party), seed), gateCount / 8));
}

export async function commit(context, salt, repetition, party, seed, share, trace, output) {
  assertBytes(context, 32, 'statement context');
  assertBytes(salt, 32, 'proof salt');
  assertBytes(seed, 32, 'tape seed');
  assertBytes(share, 32, 'key share');
  assertBytes(trace, undefined, 'AND trace');
  assertBytes(output, undefined, 'output share');
  return hash('view-commitment', context, salt, repetitionParty(repetition, party), seed, share, trace, output);
}

export function challenges(context, salt, commitments, outputs, repetitions) {
  assertBytes(context, 32, 'statement context');
  assertBytes(salt, 32, 'proof salt');
  checkRepetitions(repetitions);
  assertBytes(commitments, repetitions * 3 * 32, 'commitments');
  assertBytes(outputs, undefined, 'output shares');
  const input = framed('fiat-shamir', context, salt, commitments, outputs);
  const result = new Uint8Array(repetitions);
  let length = Math.max(64, 2 * repetitions);
  for (;;) {
    // SHAKE digest(n) is always the first n bytes, not a continued squeeze.
    // Re-scan that prefix when extending, exactly as the Python reference does.
    const source = shake256(input, length);
    let used = 0;
    for (const value of source) {
      if (value < 252) result[used++] = value % 3;
      if (used === repetitions) return result;
    }
    length *= 2;
  }
}

export function proofSize(circuit, repetitions) {
  checkCircuit(circuit);
  checkRepetitions(repetitions);
  const perRepetition = 32 + 4 * 32 + circuit.traceBytes;
  return HEADER_BYTES + circuit.nbytes + Math.ceil(repetitions / 4) + repetitions * perRepetition;
}

/** Four Fiat–Shamir choices per byte, most-significant pair first. */
export function packChallenges(selected) {
  assertBytes(selected, undefined, 'Fiat–Shamir choices');
  checkRepetitions(selected.length);
  const packed = new Uint8Array(Math.ceil(selected.length / 4));
  for (let i = 0; i < selected.length; i++) {
    if (selected[i] > 2) throw new RangeError('Fiat–Shamir choices must contain only 0, 1, or 2');
    packed[i >> 2] |= selected[i] << (6 - 2 * (i & 3));
  }
  return packed;
}

export function unpackChallenges(packed, repetitions) {
  checkRepetitions(repetitions);
  assertBytes(packed, Math.ceil(repetitions / 4), 'packed Fiat–Shamir choices');
  const remaining = repetitions & 3;
  if (remaining && (packed[packed.length - 1] & ((1 << (8 - 2 * remaining)) - 1))) {
    throw new RangeError('nonzero Fiat–Shamir padding');
  }
  const selected = new Uint8Array(repetitions);
  for (let i = 0; i < repetitions; i++) {
    selected[i] = (packed[i >> 2] >> (6 - 2 * (i & 3))) & 3;
    if (selected[i] === 3) throw new RangeError('invalid Fiat–Shamir choice');
  }
  return selected;
}

function checkAborted(signal) {
  if (signal?.aborted) {
    const error = new Error('proof generation was cancelled');
    error.name = 'AbortError';
    throw error;
  }
}

function outputReconstructs(outputs, expected, offset, outputBytes) {
  let difference = 0;
  for (let i = 0; i < outputBytes; i++) {
    difference |= outputs[offset + i] ^ outputs[offset + outputBytes + i] ^
      outputs[offset + 2 * outputBytes + i] ^ expected[i];
  }
  return difference === 0;
}

export async function prove(secret, document, challenge, { signal, onProgress, repetitions: requestedRepetitions } = {}) {
  assertBytes(secret, KEY_BYTES, 'secret key');
  const localSecret = new Uint8Array(secret);
  const publicDocument = publicSnapshot(document);
  const localChallenge = challengeSnapshot(challenge);
  if (onProgress !== undefined && typeof onProgress !== 'function') throw new TypeError('onProgress must be a function');
  const sensitive = [localSecret];
  try {
    checkAborted(signal);
    const circuit = await validatePublic(publicDocument);
    const minimum = Math.max(publicDocument.minimum_repetitions, localChallenge.required_repetitions);
    const repetitions = requestedRepetitions === undefined ? minimum : requestedRepetitions;
    checkRepetitions(repetitions);
    if (repetitions < minimum) throw new RangeError('repetitions are below the public-key or challenge requirement');
    const size = proofSize(circuit, repetitions);
    if (size > MAX_PROOF_BYTES) throw new RangeError('proof exceeds the size limit');
    const actual = packBits(circuit.apply(localSecret, circuit.anchors));
    if (!sameBytes(actual, fromHex(publicDocument.anchor_outputs, actual.length))) {
      throw new Error('secret does not match public key');
    }
    const message = validateChallenge(localChallenge);
    const x = unpackBits(shake256(framed('challenge-block', message), circuit.nbytes));
    const response = packBits(circuit.apply(localSecret, x));
    const { blocks, expected, context } = await statement(publicDocument, localChallenge, circuit, response, repetitions);
    checkAborted(signal);

    // Match Python's RNG request order exactly. Deterministic randomness belongs
    // exclusively to tests; this public API always obtains fresh OS randomness.
    const salt = randomBytes(32);
    const seeds = randomBytes(repetitions * 3 * 32);
    const randomShares = randomBytes(repetitions * 2 * 32);
    const shares = new Uint8Array(repetitions * 3 * 32);
    sensitive.push(seeds, randomShares, shares);
    for (let rep = 0; rep < repetitions; rep++) {
      for (let i = 0; i < 32; i++) {
        const first = randomShares[rep * 64 + i];
        const second = randomShares[rep * 64 + 32 + i];
        shares[rep * 96 + i] = first;
        shares[rep * 96 + 32 + i] = second;
        shares[rep * 96 + 64 + i] = first ^ second ^ localSecret[i];
      }
    }
    const outputs = new Uint8Array(repetitions * 3 * circuit.outputBytes);
    const traces = new Uint8Array(repetitions * 3 * circuit.traceBytes);
    const commitments = new Uint8Array(repetitions * 3 * 32);
    const tapes = new Uint8Array(3 * circuit.gates);
    sensitive.push(traces, tapes);
    onProgress?.({ phase: 'simulate', completed: 0, total: repetitions });
    for (let rep = 0; rep < repetitions; rep++) {
      checkAborted(signal);
      for (let party = 0; party < 3; party++) {
        const seedOffset = (rep * 3 + party) * 32;
        tapes.set(tape(seeds.subarray(seedOffset, seedOffset + 32), salt, rep, party, context, circuit.gates), party * circuit.gates);
      }
      const keyShares = unpackBits(shares.subarray(rep * 96, (rep + 1) * 96));
      const result = circuit.execute(keyShares, blocks, Uint8Array.of(0, 1, 2), { tapes });
      keyShares.fill(0);
      const packedOutputs = packBits(result.outputs);
      const packedTraces = packBits(result.trace);
      result.trace.fill(0);
      outputs.set(packedOutputs, rep * 3 * circuit.outputBytes);
      traces.set(packedTraces, rep * 3 * circuit.traceBytes);
      if (!outputReconstructs(packedOutputs, expected, 0, circuit.outputBytes)) {
        throw new Error('MPC outputs do not reconstruct the clear circuit');
      }
      for (let party = 0; party < 3; party++) {
        const index = rep * 3 + party;
        commitments.set(await commit(context, salt, rep, party,
          seeds.subarray(index * 32, (index + 1) * 32),
          shares.subarray(index * 32, (index + 1) * 32),
          traces.subarray(index * circuit.traceBytes, (index + 1) * circuit.traceBytes),
          outputs.subarray(index * circuit.outputBytes, (index + 1) * circuit.outputBytes)), index * 32);
      }
      packedTraces.fill(0);
      onProgress?.({ phase: 'simulate', completed: rep + 1, total: repetitions });
    }
    checkAborted(signal);
    const selected = challenges(context, salt, commitments, outputs, repetitions);
    const openedSeeds = new Uint8Array(repetitions * 2 * 32);
    const openedShares = new Uint8Array(repetitions * 2 * 32);
    const secondTraces = new Uint8Array(repetitions * circuit.traceBytes);
    const hiddenCommitments = new Uint8Array(repetitions * 32);
    sensitive.push(openedSeeds, openedShares, secondTraces);
    for (let rep = 0; rep < repetitions; rep++) {
      for (let slot = 0; slot < 2; slot++) {
        const party = (selected[rep] + slot) % 3;
        const inputIndex = (rep * 3 + party) * 32;
        const outputIndex = (rep * 2 + slot) * 32;
        openedSeeds.set(seeds.subarray(inputIndex, inputIndex + 32), outputIndex);
        openedShares.set(shares.subarray(inputIndex, inputIndex + 32), outputIndex);
      }
      const traceIndex = (rep * 3 + (selected[rep] + 1) % 3) * circuit.traceBytes;
      secondTraces.set(traces.subarray(traceIndex, traceIndex + circuit.traceBytes), rep * circuit.traceBytes);
      const hiddenIndex = (rep * 3 + (selected[rep] + 2) % 3) * 32;
      hiddenCommitments.set(commitments.subarray(hiddenIndex, hiddenIndex + 32), rep * 32);
    }
    // Opened commitments and all output shares are reconstructed by verification.
    const proof = concatBytes(MAGIC, u16(repetitions), salt, response, packChallenges(selected), hiddenCommitments, openedSeeds, openedShares, secondTraces);
    if (proof.length !== size) throw new Error('incorrect proof serialization length');
    onProgress?.({ phase: 'complete', completed: repetitions, total: repetitions });
    return proof;
  } finally {
    // Best-effort lifetime reduction; JavaScript does not promise secure erasure.
    for (const bytes of sensitive) bytes.fill(0);
  }
}

export async function verify(document, challenge, proof) {
  try {
    assertBytes(proof, undefined, 'proof');
    if (proof.length < HEADER_BYTES || proof.length > MAX_PROOF_BYTES) return false;
    const publicDocument = publicSnapshot(document);
    const localChallenge = challengeSnapshot(challenge);
    // Freeze the received bytes across asynchronous digest operations.
    const input = new Uint8Array(proof);
    if (!sameBytes(input.subarray(0, 8), MAGIC)) return false;
    const repetitions = input[8] * 256 + input[9];
    if (repetitions < Math.max(publicDocument.minimum_repetitions, localChallenge.required_repetitions) || repetitions > 512) return false;
    const circuit = await validatePublic(publicDocument);
    if (input.length !== proofSize(circuit, repetitions)) return false;
    let position = HEADER_BYTES;
    const take = length => {
      const part = input.subarray(position, position + length);
      position += length;
      return part;
    };
    const salt = input.subarray(10, HEADER_BYTES);
    const response = take(circuit.nbytes);
    // Validate the exact wire length and canonical choices before allocating the
    // expanded transcript or doing any replay work.
    const selected = unpackChallenges(take(Math.ceil(repetitions / 4)), repetitions);
    const hiddenCommitments = take(repetitions * 32);
    const seeds = take(repetitions * 2 * 32);
    const shares = take(repetitions * 2 * 32);
    const secondTraces = take(repetitions * circuit.traceBytes);
    if (position !== input.length) return false;
    const { blocks, expected, context } = await statement(publicDocument, localChallenge, circuit, response, repetitions);
    const commitments = new Uint8Array(repetitions * 3 * 32);
    const outputs = new Uint8Array(repetitions * 3 * circuit.outputBytes);
    const tapes = new Uint8Array(2 * circuit.gates);
    for (let rep = 0; rep < repetitions; rep++) {
      const parties = Uint8Array.of(selected[rep], (selected[rep] + 1) % 3);
      for (let slot = 0; slot < 2; slot++) {
        const index = (rep * 2 + slot) * 32;
        tapes.set(tape(seeds.subarray(index, index + 32), salt, rep, parties[slot], context, circuit.gates), slot * circuit.gates);
      }
      const replay = circuit.execute(
        unpackBits(shares.subarray(rep * 64, (rep + 1) * 64)), blocks, parties,
        { tapes, suppliedTrace: unpackBits(secondTraces.subarray(rep * circuit.traceBytes, (rep + 1) * circuit.traceBytes)) },
      );
      const replayOutputs = packBits(replay.outputs);
      const replayTraces = packBits(replay.trace);
      for (let slot = 0; slot < 2; slot++) {
        const party = parties[slot];
        const openedIndex = (rep * 2 + slot) * 32;
        const fullIndex = rep * 3 + party;
        const output = replayOutputs.subarray(slot * circuit.outputBytes, (slot + 1) * circuit.outputBytes);
        outputs.set(output, fullIndex * circuit.outputBytes);
        // Both commitments are recomputed. The first opened AND trace comes
        // from execution; only the adjacent party's trace is supplied on wire.
        const commitment = await commit(context, salt, rep, party,
          seeds.subarray(openedIndex, openedIndex + 32), shares.subarray(openedIndex, openedIndex + 32),
          replayTraces.subarray(slot * circuit.traceBytes, (slot + 1) * circuit.traceBytes), output);
        commitments.set(commitment, fullIndex * 32);
      }
      const hiddenIndex = rep * 3 + (selected[rep] + 2) % 3;
      commitments.set(hiddenCommitments.subarray(rep * 32, (rep + 1) * 32), hiddenIndex * 32);
      for (let i = 0; i < circuit.outputBytes; i++) {
        outputs[hiddenIndex * circuit.outputBytes + i] = expected[i] ^ replayOutputs[i] ^ replayOutputs[circuit.outputBytes + i];
      }
    }
    // The transmitted choices are instructions for reconstruction, not trusted
    // challenges. Accept only an exact match to the rebuilt Fiat–Shamir hash.
    return sameBytes(selected, challenges(context, salt, commitments, outputs, repetitions));
  } catch {
    return false;
  }
}
