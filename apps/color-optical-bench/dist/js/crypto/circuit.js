// Exact binary convolutional Feistel circuit from neural_zkboo_experiment.py.
// Public tables depend only on the profile. Secret state stays in local arrays.
import { assertBytes, canonicalJSON, framed, u32, unpackBits } from './bytes.js';
import { hash, shake256 } from './hashes.js';

function profile(side, rounds) {
  if (!Number.isInteger(side) || ![4, 8, 16].includes(side)) {
    throw new RangeError('Circuit side must be 4, 8, or 16');
  }
  if (!Number.isInteger(rounds) || rounds < 2 || rounds > 64) {
    throw new RangeError('Circuit rounds must be between 2 and 64');
  }
}

function bits(value, length, name) {
  assertBytes(value, length, name);
  for (let i = 0; i < value.length; i++) {
    if (value[i] > 1) throw new RangeError(`${name} must contain only bits`);
  }
  return value;
}

async function publicPermutation(count, label, digest) {
  const labels = new Array(count);
  // Keep Web Crypto's pending operation count bounded, even at side 16.
  for (let start = 0; start < count; start += 32) {
    const end = Math.min(start + 32, count);
    const values = await Promise.all(Array.from({ length: end - start },
      (_, offset) => hash(label, digest, u32(start + offset))));
    for (let i = start; i < end; i++) labels[i] = values[i - start];
  }
  const indices = Array.from({ length: count }, (_, i) => i);
  indices.sort((a, b) => {
    for (let i = 0; i < 32; i++) {
      const difference = labels[a][i] - labels[b][i];
      if (difference) return difference;
    }
    return a - b;
  });
  return Uint16Array.from(indices);
}

export class BinaryCircuit {
  #ready = false;

  constructor(side = 8, rounds = 16) {
    profile(side, rounds);
    this.side = side;
    this.rounds = rounds;
    this.half = side * side;
    this.nbits = 2 * this.half;
    this.nbytes = this.nbits / 8;
    this.anchorCount = Math.ceil(256 / this.nbits);
    this.lanes = this.anchorCount + 1;
    this.gates = rounds * this.lanes * this.half;
    this.traceBytes = this.gates / 8;
    this.outputBytes = this.lanes * this.nbytes;
    this.descriptor = Object.freeze({
      name: 'binary-rotconv-feistel-v1', side, rounds, key_bits: 256,
    });
  }

  static async create(side = 8, rounds = 16) {
    const circuit = new BinaryCircuit(side, rounds);
    circuit.digest = await hash('circuit', canonicalJSON(circuit.descriptor));
    circuit.permutation = await publicPermutation(circuit.nbits, 'state-permutation', circuit.digest);
    circuit.inversePermutation = new Uint16Array(circuit.nbits);
    for (let i = 0; i < circuit.nbits; i++) circuit.inversePermutation[circuit.permutation[i]] = i;
    const keyPermutation = await publicPermutation(256, 'key-permutation', circuit.digest);
    circuit.keyIndices = [];
    circuit.constants = [];
    circuit.convIndices = [];
    const taps = [[[0, 0], [-1, 0], [0, 1]], [[0, 0], [1, 1], [-1, 1]], [[0, 0], [0, -1], [1, 0]]];
    for (let t = 0; t < rounds; t++) {
      circuit.keyIndices.push(Array.from({ length: 3 }, (_, branch) =>
        Uint16Array.from({ length: circuit.half }, (_, i) =>
          keyPermutation[(i + (3 * t + branch) * circuit.half) % 256])));
      circuit.constants.push(unpackBits(shake256(framed('round-constant', circuit.digest, u32(t)), circuit.half / 8)));
      circuit.convIndices.push(taps.map(branch => branch.map(([initialY, initialX]) => {
        let dy = initialY, dx = initialX;
        for (let turn = 0; turn < (t + 1) % 4; turn++) [dy, dx] = [-dx, dy];
        return Uint16Array.from({ length: circuit.half }, (_, i) => {
          const y = Math.floor(i / side), x = i % side;
          return ((y + dy + side) % side) * side + (x + dx + side) % side;
        });
      })));
    }
    circuit.anchors = new Uint8Array(circuit.anchorCount * circuit.nbits);
    for (let i = 0; i < circuit.anchorCount; i++) {
      circuit.anchors.set(unpackBits(shake256(framed('anchor', circuit.digest, u32(i)), circuit.nbytes)), i * circuit.nbits);
    }
    circuit.#ready = true;
    return circuit;
  }

  #blockCount(blocks) {
    if (!this.#ready) throw new Error('Construct the circuit with await BinaryCircuit.create()');
    assertBytes(blocks, undefined, 'Circuit blocks');
    const lanes = blocks.length / this.nbits;
    if (!Number.isInteger(lanes) || lanes < 1 || lanes > this.lanes) {
      throw new RangeError('Circuit blocks must contain 1 to profile.lanes complete blocks');
    }
    bits(blocks, blocks.length, 'Circuit blocks');
    return lanes;
  }

  /** One repetition; arrays are flat party-major, then lane-major, MSB-first bits.
   * The same 256 key-share bits feed every lane. Traces/tapes order gates by
   * round, lane, row-major cell. Public blocks/constants belong only to party 0.
   */
  execute(keySharesBits, blocksBits, partyIds, { tapes = null, suppliedTrace = null } = {}) {
    const lanes = this.#blockCount(blocksBits);
    if (!(Array.isArray(partyIds) || partyIds instanceof Uint8Array)) throw new TypeError('Invalid party identifiers');
    const parties = partyIds.length;
    const validParties = parties === 1 ? partyIds[0] === 0
      : parties === 2 ? Number.isInteger(partyIds[0]) && partyIds[0] >= 0 && partyIds[0] < 3 && partyIds[1] === (partyIds[0] + 1) % 3
      : parties === 3 && partyIds[0] === 0 && partyIds[1] === 1 && partyIds[2] === 2;
    if (!validParties) throw new RangeError('Expected party 0, two adjacent parties, or parties 0,1,2');
    bits(keySharesBits, parties * 256, 'Key shares');
    const stride = lanes * this.half;
    const traceLength = this.rounds * stride;
    if (parties > 1) bits(tapes, parties * traceLength, 'Random tapes');
    else if (tapes !== null) throw new TypeError('Clear evaluation does not use random tapes');
    if (parties === 2) bits(suppliedTrace, traceLength, 'Supplied trace');
    else if (suppliedTrace !== null) throw new TypeError('Only two-view replay accepts a supplied trace');

    const stateLength = parties * lanes * this.nbits;
    let state = new Uint8Array(stateLength), next = new Uint8Array(stateLength);
    for (let p = 0; p < parties; p++) {
      if (partyIds[p] === 0) state.set(blocksBits, p * blocksBits.length);
    }
    const u = new Uint8Array(parties * stride), v = new Uint8Array(parties * stride);
    const w = new Uint8Array(parties * stride), nonlinear = new Uint8Array(parties * stride);
    const trace = parties === 1 ? null : new Uint8Array(parties * traceLength);
    for (let t = 0; t < this.rounds; t++) {
      const branchArrays = [u, v, w];
      for (let p = 0; p < parties; p++) {
        for (let lane = 0; lane < lanes; lane++) {
          const stateBase = (p * lanes + lane) * this.nbits + this.half;
          const branchBase = p * stride + lane * this.half;
          for (let branch = 0; branch < 3; branch++) {
            const indices = this.convIndices[t][branch], selectors = this.keyIndices[t][branch];
            const output = branchArrays[branch];
            for (let j = 0; j < this.half; j++) {
              output[branchBase + j] = state[stateBase + indices[0][j]] ^ state[stateBase + indices[1][j]]
                ^ state[stateBase + indices[2][j]] ^ keySharesBits[p * 256 + selectors[j]]
                ^ (branch === 0 && partyIds[p] === 0 ? this.constants[t][j] : 0);
            }
          }
        }
      }
      for (let p = 0; p < parties; p++) {
        for (let j = 0; j < stride; j++) {
          const at = p * stride + j;
          if (parties === 1) nonlinear[at] = v[at] & w[at];
          else if (parties === 2 && p === 1) nonlinear[at] = suppliedTrace[t * stride + j];
          else {
            const nextParty = (p + 1) % parties, other = nextParty * stride + j;
            nonlinear[at] = (v[at] & w[at]) ^ (v[at] & w[other]) ^ (v[other] & w[at])
              ^ tapes[p * traceLength + t * stride + j] ^ tapes[nextParty * traceLength + t * stride + j];
          }
          if (trace) trace[p * traceLength + t * stride + j] = nonlinear[at];
        }
        for (let lane = 0; lane < lanes; lane++) {
          const stateBase = (p * lanes + lane) * this.nbits, branchBase = p * stride + lane * this.half;
          for (let j = 0; j < this.nbits; j++) {
            const source = this.permutation[j];
            next[stateBase + j] = source < this.half ? state[stateBase + this.half + source]
              : state[stateBase + source - this.half] ^ u[branchBase + source - this.half] ^ nonlinear[branchBase + source - this.half];
          }
        }
      }
      [state, next] = [next, state];
    }
    return { outputs: state, trace };
  }

  apply(secretBytes, blocksBits) {
    assertBytes(secretBytes, 32, 'Secret key');
    return this.execute(unpackBits(secretBytes), blocksBits, [0]).outputs;
  }

  inverse(secretBytes, blocksBits) {
    assertBytes(secretBytes, 32, 'Secret key');
    const lanes = this.#blockCount(blocksBits), keyBits = unpackBits(secretBytes);
    let state = blocksBits.slice(), next = new Uint8Array(state.length);
    const unshuffled = new Uint8Array(state.length);
    for (let t = this.rounds - 1; t >= 0; t--) {
      for (let lane = 0; lane < lanes; lane++) {
        const base = lane * this.nbits;
        for (let j = 0; j < this.nbits; j++) unshuffled[base + j] = state[base + this.inversePermutation[j]];
        for (let j = 0; j < this.half; j++) {
          const branches = [0, 0, 0];
          for (let branch = 0; branch < 3; branch++) {
            const indices = this.convIndices[t][branch];
            branches[branch] = unshuffled[base + indices[0][j]] ^ unshuffled[base + indices[1][j]]
              ^ unshuffled[base + indices[2][j]] ^ keyBits[this.keyIndices[t][branch][j]];
          }
          next[base + j] = unshuffled[base + this.half + j] ^ branches[0] ^ this.constants[t][j] ^ (branches[1] & branches[2]);
          next[base + this.half + j] = unshuffled[base + j];
        }
      }
      [state, next] = [next, state];
    }
    return state;
  }
}
