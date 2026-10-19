/** QRT, the sole current binary wire format. All multi-byte integers are big-endian. */
import { bytesToHex, hexToBytes, crc32 } from './bytes.js';
export { bytesToHex, hexToBytes, sha256, randomId, crc32 } from './bytes.js';
export const DATA_OVERHEAD = 18;
export const ACK_BYTES = 28;
export const CONTROL_MAX_BYTES = 68;
// The largest DATA frame fills a version 40-L QR: 2,953 bytes minus its framing.
export const LIMITS = Object.freeze({ maxMessageBytes: 8 * 1024 * 1024, maxChunks: 65535, maxChunkBytes: 2953 - DATA_OVERHEAD, maxWindowSize: 8 });
export const KIND = Object.freeze({ START: 1, DATA: 2, ACK: 3, POLL: 4, CANCEL: 5, RESTART: 6 });
export const KINDS = Object.freeze({ start: 1, data: 2, ack: 3, poll: 4, cancel: 5, restart: 6 });
const STATUS = Object.freeze({ ready: 0, progress: 1, complete: 2, stored: 3 });
const MAGIC = Uint8Array.of(0x51, 0x52, 0x54); // QRT. No compatibility formats.
const nameOf = (table, value) => Object.keys(table).find(k => table[k] === value);
function uint(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw Error(`Invalid ${name}`);
}
function bodySize(p) {
  switch (p.kind) {
    case 'start': case 'restart':
      uint(p.totalBytes, 0, LIMITS.maxMessageBytes, 'totalBytes');
      uint(p.chunkBytes, 1, LIMITS.maxChunkBytes, 'chunkBytes');
      uint(p.chunkCount, 1, LIMITS.maxChunks, 'chunkCount');
      if (p.chunkCount !== Math.max(1, Math.ceil(p.totalBytes / p.chunkBytes))) throw Error('Inconsistent chunk count');
      uint(p.windowSize, 1, LIMITS.maxWindowSize, 'windowSize');
      hexToBytes(p.digest, 32);
      if (p.kind === 'restart') {
        hexToBytes(p.previousTransferId, 8);
        if (p.previousTransferId === p.transferId) throw Error('RESTART requires a new transfer ID');
      }
      return p.kind === 'restart' ? 52 : 44;
    case 'data':
      uint(p.index, 0, LIMITS.maxChunks - 1, 'chunk index');
      if (!(p.data instanceof Uint8Array) || p.data.length > LIMITS.maxChunkBytes) throw Error('Invalid data bytes');
      return 2 + p.data.length;
    case 'ack':
      uint(p.base, 0, LIMITS.maxChunks - 1, 'ACK base'); uint(p.bitmap, 0, 255, 'ACK bitmap');
      if (!Object.hasOwn(STATUS, p.status)) throw Error('Invalid ACK status');
      if (p.status === 'stored' && (p.base !== 0 || p.bitmap !== 0)) throw Error('Invalid stored ACK');
      hexToBytes(p.digestPrefix, 8); return 12;
    case 'poll': uint(p.base, 0, LIMITS.maxChunks - 1, 'POLL base'); return 2;
    case 'cancel': uint(p.reason ?? 0, 0, 3, 'cancel reason'); return 1;
    default: throw Error('Unknown packet kind');
  }
}
export function encodePacket(p) {
  const n = bodySize(p), out = new Uint8Array(16 + n), view = new DataView(out.buffer), b = 12;
  out.set(MAGIC); out[3] = KINDS[p.kind]; out.set(hexToBytes(p.transferId, 8), 4);
  if (p.kind === 'start' || p.kind === 'restart') {
    view.setUint32(b, p.totalBytes); view.setUint16(b + 4, p.chunkBytes); view.setUint16(b + 6, p.chunkCount);
    out[b + 8] = p.windowSize; out.set(hexToBytes(p.digest, 32), b + 12);
    if (p.kind === 'restart') out.set(hexToBytes(p.previousTransferId, 8), b + 44);
  } else if (p.kind === 'data') { view.setUint16(b, p.index); out.set(p.data, b + 2); }
  else if (p.kind === 'ack') {
    view.setUint16(b, p.base); out[b + 2] = p.bitmap; out[b + 3] = STATUS[p.status]; out.set(hexToBytes(p.digestPrefix, 8), b + 4);
  } else if (p.kind === 'poll') view.setUint16(b, p.base);
  else out[b] = p.reason ?? 0;
  view.setUint32(out.length - 4, crc32(out.subarray(0, -4))); return out;
}
export function decodePacket(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.length < 17 || bytes.length > LIMITS.maxChunkBytes + DATA_OVERHEAD) throw Error('Invalid packet length');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), b = 12, n = bytes.length - 16;
  if (MAGIC.some((value, i) => bytes[i] !== value)) throw Error('Invalid packet magic');
  if (view.getUint32(bytes.length - 4) !== crc32(bytes.subarray(0, -4))) throw Error('Packet CRC mismatch');
  const p = { kind: nameOf(KINDS, bytes[3]), transferId: bytesToHex(bytes.subarray(4, 12)) };
  if (p.kind === 'start' || p.kind === 'restart') {
    if (n !== (p.kind === 'restart' ? 52 : 44) || bytes[b + 9] || bytes[b + 10] || bytes[b + 11]) throw Error('Invalid START/RESTART body');
    Object.assign(p, { totalBytes: view.getUint32(b), chunkBytes: view.getUint16(b + 4), chunkCount: view.getUint16(b + 6), windowSize: bytes[b + 8], digest: bytesToHex(bytes.subarray(b + 12, b + 44)) });
    if (p.kind === 'restart') p.previousTransferId = bytesToHex(bytes.subarray(b + 44, b + 52));
  } else if (p.kind === 'data') {
    if (n < 2) throw Error('Invalid DATA body'); p.index = view.getUint16(b); p.data = bytes.slice(b + 2, -4);
  } else if (p.kind === 'ack') {
    if (n !== 12) throw Error('Invalid ACK body');
    Object.assign(p, { base: view.getUint16(b), bitmap: bytes[b + 2], status: nameOf(STATUS, bytes[b + 3]), digestPrefix: bytesToHex(bytes.subarray(b + 4, b + 12)) });
  } else if (p.kind === 'poll') { if (n !== 2) throw Error('Invalid POLL body'); p.base = view.getUint16(b); }
  else if (p.kind === 'cancel') { if (n !== 1) throw Error('Invalid CANCEL body'); p.reason = bytes[b]; }
  else throw Error('Unknown packet kind');
  bodySize(p); return p;
}
