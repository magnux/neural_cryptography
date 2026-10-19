/** Current optical message framing. CRC and SHA-256 detect damage, not forgery. */
import { crc32 } from '../format.js';

export const DATA_OVERHEAD = 44;
export const CONTROL_MAX_BYTES = 88;
export const LIMITS = Object.freeze({ maxMessageBytes: 8 * 1024 * 1024, maxChunks: 131072, maxChunkBytes: 131072, maxWindowSize: 8 });
export const KIND = Object.freeze({ START: 1, DATA: 2, ACK: 3, POLL: 4, CANCEL: 5 });
export const KINDS = Object.freeze({ start: 1, data: 2, ack: 3, poll: 4, cancel: 5 });
export const MESSAGE_TYPES = Object.freeze({ raw: 0, hello: 1, proof: 2, finish: 3, 'fast-challenge': 4, 'fast-response': 5, result: 6 });
const STATUS = Object.freeze({ ready: 0, progress: 1, complete: 2 });
const MAGIC = Uint8Array.of(0x4f, 0x4c, 0x4d, 0x31); // OLM1: one current format.
const HEX = /^[0-9a-f]+$/;
const nameOf = (table, n) => Object.keys(table).find(k => table[k] === n);
export function bytesToHex(bytes) { return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join(''); }
export function hexToBytes(hex, size) {
  if (typeof hex !== 'string' || hex.length !== size * 2 || !HEX.test(hex)) throw new Error(`Expected ${size}-byte lowercase hex value`);
  return Uint8Array.from(hex.match(/../g), s => parseInt(s, 16));
}
export function randomId(size = 16) {
  if (size !== 8 && size !== 16) throw new Error('Identifier must be 8 or 16 bytes');
  return bytesToHex(crypto.getRandomValues(new Uint8Array(size)));
}
export async function sha256(bytes) {
  if (!(bytes instanceof Uint8Array)) throw new Error('Expected Uint8Array');
  return bytesToHex(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)));
}
function uint(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`Invalid ${name}`);
  return value;
}
function validateStart(p) {
  uint(p.totalBytes, 0, LIMITS.maxMessageBytes, 'totalBytes');
  uint(p.chunkBytes, 1, LIMITS.maxChunkBytes, 'chunkBytes');
  uint(p.chunkCount, 1, LIMITS.maxChunks, 'chunkCount');
  if (p.chunkCount !== Math.max(1, Math.ceil(p.totalBytes / p.chunkBytes))) throw new Error('Inconsistent chunk count');
  uint(p.windowSize, 1, LIMITS.maxWindowSize, 'windowSize');
  hexToBytes(p.digest, 32);
}
function bodySize(p) {
  switch (p.kind) {
    case 'start': validateStart(p); return 48;
    case 'data':
      uint(p.index, 0, LIMITS.maxChunks - 1, 'chunk index');
      if (!(p.data instanceof Uint8Array) || p.data.length > LIMITS.maxChunkBytes) throw new Error('Invalid data bytes');
      return 4 + p.data.length;
    case 'ack':
      uint(p.base, 0, LIMITS.maxChunks - 1, 'ACK base');
      uint(p.bitmap, 0, 0xffffffff, 'ACK bitmap');
      if (!Object.hasOwn(STATUS, p.status)) throw new Error('Invalid ACK status');
      hexToBytes(p.digest, 32); return 44;
    case 'poll': uint(p.base, 0, LIMITS.maxChunks - 1, 'POLL base'); return 4;
    case 'cancel': uint(p.reason ?? 0, 0, 3, 'cancel reason'); return 4;
    default: throw new Error('Unknown packet kind');
  }
}
export function encodePacket(p) {
  if (!Object.hasOwn(MESSAGE_TYPES, p.messageType)) throw new Error('Unknown message type');
  const n = bodySize(p), out = new Uint8Array(40 + n), v = new DataView(out.buffer);
  out.set(MAGIC); out[4] = KINDS[p.kind]; out[5] = MESSAGE_TYPES[p.messageType];
  out.set(hexToBytes(p.sessionId, 16), 8); out.set(hexToBytes(p.messageId, 8), 24);
  v.setUint32(32, n); const b = 36;
  if (p.kind === 'start') {
    v.setUint32(b, p.totalBytes); v.setUint32(b + 4, p.chunkBytes); v.setUint32(b + 8, p.chunkCount);
    out[b + 12] = p.windowSize; out.set(hexToBytes(p.digest, 32), b + 16);
  } else if (p.kind === 'data') { v.setUint32(b, p.index); out.set(p.data, b + 4); }
  else if (p.kind === 'ack') {
    v.setUint32(b, p.base); v.setUint32(b + 4, p.bitmap); out[b + 8] = STATUS[p.status];
    out.set(hexToBytes(p.digest, 32), b + 12);
  } else if (p.kind === 'poll') v.setUint32(b, p.base);
  else out[b] = p.reason ?? 0;
  v.setUint32(out.length - 4, crc32(out.subarray(0, out.length - 4)));
  return out;
}
export function decodePacket(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.length < 40 || bytes.length > LIMITS.maxChunkBytes + DATA_OVERHEAD) throw new Error('Invalid packet length');
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), n = v.getUint32(32), b = 36;
  if (MAGIC.some((x, i) => bytes[i] !== x) || bytes[6] || bytes[7]) throw new Error('Invalid packet header');
  if (n !== bytes.length - 40 || v.getUint32(bytes.length - 4) !== crc32(bytes.subarray(0, bytes.length - 4))) throw new Error('Packet CRC or length mismatch');
  const p = { kind: nameOf(KINDS, bytes[4]), messageType: nameOf(MESSAGE_TYPES, bytes[5]), sessionId: bytesToHex(bytes.subarray(8, 24)), messageId: bytesToHex(bytes.subarray(24, 32)) };
  if (!p.kind || !p.messageType) throw new Error('Unknown packet kind or message type');
  if (p.kind === 'start') {
    if (n !== 48 || bytes[b + 13] || bytes[b + 14] || bytes[b + 15]) throw new Error('Invalid START body');
    Object.assign(p, { totalBytes: v.getUint32(b), chunkBytes: v.getUint32(b + 4), chunkCount: v.getUint32(b + 8), windowSize: bytes[b + 12], digest: bytesToHex(bytes.subarray(b + 16, b + 48)) });
  } else if (p.kind === 'data') {
    if (n < 4) throw new Error('Invalid DATA body');
    p.index = v.getUint32(b); p.data = bytes.slice(b + 4, bytes.length - 4);
  } else if (p.kind === 'ack') {
    if (n !== 44 || bytes[b + 9] || bytes[b + 10] || bytes[b + 11]) throw new Error('Invalid ACK body');
    Object.assign(p, { base: v.getUint32(b), bitmap: v.getUint32(b + 4), status: nameOf(STATUS, bytes[b + 8]), digest: bytesToHex(bytes.subarray(b + 12, b + 44)) });
  } else if (p.kind === 'poll') { if (n !== 4) throw new Error('Invalid POLL body'); p.base = v.getUint32(b); }
  else {
    if (n !== 4 || bytes[b + 1] || bytes[b + 2] || bytes[b + 3]) throw new Error('Invalid CANCEL body'); p.reason = bytes[b];
  }
  bodySize(p); return p;
}
