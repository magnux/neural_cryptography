/** Bounded selective-repeat message transport. Delivery means integrity, not identity. */
import { encodePacket, decodePacket, sha256, randomId, LIMITS, MESSAGE_TYPES, hexToBytes } from './protocol.js';

const TERMINAL = new Set(['delivered', 'failed', 'timed-out', 'cancelled']);
const result = (accepted, reason, outbound = [], delivery) => ({ accepted, ...(reason ? { reason } : {}), outbound, ...(delivery ? { delivery } : {}) });
const same = (a, b) => a.sessionId === b.sessionId && a.messageId === b.messageId && a.messageType === b.messageType;
const baseFields = p => ({ sessionId: p.sessionId, messageId: p.messageId, messageType: p.messageType });
const mask = (base, count, window) => (1 << Math.min(window, count - base)) - 1;
const equalBytes = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
function numberOption(n, min, max, label) {
  if (!Number.isSafeInteger(n) || n < min || n > max) throw new Error(`Invalid ${label}`);
  return n;
}
function time(n) { if (!Number.isFinite(n) || n < 0) throw new Error('Invalid clock value'); return n; }
function metadataEqual(a, b) {
  return same(a, b) && ['totalBytes', 'chunkBytes', 'chunkCount', 'windowSize', 'digest'].every(k => a[k] === b[k]);
}
function ack(p, base, bitmap, status) { return encodePacket({ ...baseFields(p), kind: 'ack', base, bitmap, status, digest: p.digest }); }
function cancellation(p, reason) { return encodePacket({ ...baseFields(p), kind: 'cancel', reason }); }
function expectedLength(p, index) { return Math.min(p.chunkBytes, p.totalBytes - index * p.chunkBytes); }

export async function createSender(payload, options = {}) {
  if (!(payload instanceof Uint8Array) || payload.length > LIMITS.maxMessageBytes) throw new Error('Payload must be Uint8Array of at most 8 MiB');
  const bytes = payload.slice();
  const metadata = {
    sessionId: options.sessionId ?? randomId(16), messageId: options.messageId ?? randomId(8), messageType: options.messageType ?? 'raw',
    totalBytes: bytes.length, chunkBytes: options.chunkBytes ?? 106, windowSize: options.windowSize ?? 4,
  };
  hexToBytes(metadata.sessionId, 16); hexToBytes(metadata.messageId, 8);
  if (!Object.hasOwn(MESSAGE_TYPES, metadata.messageType)) throw new Error('Unknown message type');
  numberOption(metadata.chunkBytes, 1, LIMITS.maxChunkBytes, 'chunkBytes');
  numberOption(metadata.windowSize, 1, LIMITS.maxWindowSize, 'windowSize');
  metadata.chunkCount = Math.max(1, Math.ceil(bytes.length / metadata.chunkBytes));
  numberOption(metadata.chunkCount, 1, LIMITS.maxChunks, 'chunkCount');
  metadata.digest = await sha256(bytes);
  return new Sender(bytes, metadata, options);
}

class Sender {
  constructor(bytes, metadata, options) {
    this.bytes = bytes; this.meta = metadata;
    this.retryMs = numberOption(options.retryMs ?? 800, 1, 86400000, 'retryMs');
    this.maxRetries = numberOption(options.maxRetries ?? 12, 0, 10000, 'maxRetries');
    this.timeoutMs = numberOption(options.timeoutMs ?? 120000, 1, 86400000, 'timeoutMs');
    this.startedAt = this.clock = time(options.now ?? 0); this.deadline = this.startedAt + this.timeoutMs;
    this.state = 'awaiting-start'; this.base = 0; this.acked = new Uint8Array(metadata.chunkCount);
    this.everSent = new Uint8Array(metadata.chunkCount); this.queue = []; this.attempts = 0;
    this.lastSendAt = null; this.sentPackets = 0; this.retries = 0; this.error = null;
    this.ackedChunks = 0; this.ackedBytes = 0;
  }
  _clock(now) { this.clock = Math.max(this.clock, time(now)); return this.clock; }
  _fail(state, error) { this.state = state; this.error = error; this.queue = []; this.bytes = null; }
  _emit(packet, now) { this.sentPackets++; this.lastSendAt = now; return encodePacket({ ...this.meta, ...packet }); }
  _retry(now, packet) {
    if (this.lastSendAt !== null) {
      if (now - this.lastSendAt < this.retryMs) return null;
      if (this.attempts >= this.maxRetries) { this._fail('failed', 'Retry limit reached'); return cancellation(this.meta, 1); }
      this.attempts++; this.retries++;
    }
    return this._emit(packet, now);
  }
  next(now) {
    now = this._clock(now);
    if (TERMINAL.has(this.state)) return null;
    if (now >= this.deadline) { this._fail('timed-out', 'Absolute transfer deadline reached'); return cancellation(this.meta, 1); }
    if (this.state === 'awaiting-start') return this._retry(now, { kind: 'start' });
    if (this.state === 'awaiting-ack') return this._retry(now, { kind: 'poll', base: this.base });
    if (this.queue.length) {
      const index = this.queue.shift(), data = this.bytes.subarray(index * this.meta.chunkBytes, (index + 1) * this.meta.chunkBytes);
      this.everSent[index] = 1; return this._emit({ kind: 'data', index, data }, now);
    }
    this.state = 'awaiting-ack'; this.lastSendAt = null;
    return this._emit({ kind: 'poll', base: this.base }, now);
  }
  receive(bytes, now) {
    now = this._clock(now);
    if (TERMINAL.has(this.state)) return { accepted: false, reason: 'terminal' };
    if (now >= this.deadline) { this._fail('timed-out', 'Absolute transfer deadline reached'); return { accepted: false, reason: 'deadline' }; }
    let p; try { p = decodePacket(bytes); } catch { return { accepted: false, reason: 'malformed' }; }
    if (!same(p, this.meta)) return { accepted: false, reason: 'foreign' };
    if (p.kind === 'cancel') { this._fail('cancelled', `Peer cancelled transfer (${p.reason})`); return { accepted: true }; }
    if (p.kind !== 'ack' || p.digest !== this.meta.digest) return { accepted: false, reason: 'unexpected' };
    const m = this.meta;
    if (p.base >= m.chunkCount || p.base % m.windowSize || (p.bitmap & ~mask(p.base, m.chunkCount, m.windowSize))) return { accepted: false, reason: 'invalid-bitmap' };
    for (let j = 0; j < m.windowSize && p.base + j < m.chunkCount; j++) {
      if ((p.bitmap & (1 << j)) && !this.everSent[p.base + j]) return { accepted: false, reason: 'unsent-ack' };
    }
    if (p.status === 'complete') {
      const lastBase = Math.floor((m.chunkCount - 1) / m.windowSize) * m.windowSize;
      if (p.base !== lastBase || p.bitmap !== mask(lastBase, m.chunkCount, m.windowSize) || this.everSent.some(x => !x)) return { accepted: false, reason: 'premature-complete' };
      this.acked.fill(1); this.ackedChunks = m.chunkCount; this.ackedBytes = m.totalBytes;
      this.state = 'delivered'; this.bytes = null; this.queue = []; return { accepted: true };
    }
    if (p.base !== this.base || (this.state === 'awaiting-start' ? p.status !== 'ready' : this.state !== 'awaiting-ack' || p.status !== 'progress')) return { accepted: false, reason: 'stale-ack' };
    const starting = this.state === 'awaiting-start';
    for (let j = 0; j < m.windowSize && this.base + j < m.chunkCount; j++) {
      const i = this.base + j;
      if ((p.bitmap & (1 << j)) && !this.acked[i]) { this.acked[i] = 1; this.ackedChunks++; this.ackedBytes += expectedLength(m, i); }
    }
    const full = Array.from({ length: Math.min(m.windowSize, m.chunkCount - this.base) }, (_, j) => this.acked[this.base + j]).every(Boolean);
    if (full) {
      if (this.base + m.windowSize >= m.chunkCount) return { accepted: false, reason: 'awaiting-integrity-confirmation' };
      this.base += m.windowSize; this.attempts = 0;
    } else if (!starting) {
      if (this.attempts >= this.maxRetries) { this._fail('failed', 'Retry limit reached'); return { accepted: false, reason: 'retry-limit' }; }
      this.attempts++; this.retries++;
    } else this.attempts = 0;
    this.queue = [];
    for (let i = this.base; i < Math.min(this.base + m.windowSize, m.chunkCount); i++) if (!this.acked[i]) this.queue.push(i);
    this.state = 'sending'; this.lastSendAt = null; return { accepted: true };
  }
  cancel(now) {
    this._clock(now); if (TERMINAL.has(this.state)) return null;
    this._fail('cancelled', 'Cancelled locally'); return cancellation(this.meta, 0);
  }
  snapshot() {
    const missingChunks = [];
    for (let i = this.base; i < Math.min(this.base + this.meta.windowSize, this.meta.chunkCount); i++) if (!this.acked[i]) missingChunks.push(i);
    return { ...this.meta, state: this.state, ackedChunks: this.ackedChunks, ackedBytes: this.ackedBytes,
      progress: this.state === 'delivered' ? 1 : this.meta.totalBytes ? this.ackedBytes / this.meta.totalBytes : 0,
      windowBase: this.base, missingChunks, sentPackets: this.sentPackets, retries: this.retries, error: this.error };
  }
}

export class Receiver {
  constructor(options = {}) {
    this.timeoutMs = numberOption(options.timeoutMs ?? 120000, 1, 86400000, 'timeoutMs');
    this.maxMessageBytes = numberOption(options.maxMessageBytes ?? LIMITS.maxMessageBytes, 0, LIMITS.maxMessageBytes, 'maxMessageBytes');
    this.maxCompleted = numberOption(options.maxCompleted ?? 8, 1, 128, 'maxCompleted');
    this.completedTtlMs = numberOption(options.completedTtlMs ?? 120000, 1, 86400000, 'completedTtlMs');
    this.clock = 0; this.active = null; this.receipts = []; this.last = { state: 'idle' };
    this.completedMessages = 0; this.duplicatePackets = 0; this.malformedPackets = 0; this.foreignPackets = 0;
  }
  _clock(now) { this.clock = Math.max(this.clock, time(now)); this.receipts = this.receipts.filter(r => r.expiresAt > this.clock); return this.clock; }
  _bitmap(a, base) {
    let bits = 0;
    for (let j = 0; j < a.meta.windowSize && base + j < a.meta.chunkCount; j++) if (a.seen[base + j]) bits |= 1 << j;
    return bits;
  }
  _finish(a, state, now, error, reason) {
    if (this.active !== a) return;
    this.last = { ...a.meta, state, receivedBytes: a.receivedBytes, receivedChunks: a.receivedChunks, error: error ?? null };
    this.receipts.push({ meta: a.meta, state, reason: reason ?? 0, expiresAt: now + this.completedTtlMs });
    if (this.receipts.length > this.maxCompleted) this.receipts.shift();
    this.active = null;
  }
  tick(now) {
    now = this._clock(now);
    if (this.active && now >= this.active.deadline) {
      const a = this.active; this._finish(a, 'timed-out', now, 'Absolute transfer deadline reached', 1);
      return result(false, 'deadline', [cancellation(a.meta, 1)]);
    }
    return result(false, 'idle');
  }
  async receive(bytes, now) {
    now = this._clock(now);
    const expired = this.tick(now);
    let p;
    try { p = decodePacket(bytes); } catch { this.malformedPackets++; return result(false, 'malformed', expired.outbound); }
    const receipt = this.receipts.find(r => same(r.meta, p));
    if (receipt) {
      if (p.kind === 'start' && !metadataEqual(receipt.meta, p)) return result(false, 'conflicting-start', expired.outbound);
      if (p.kind === 'ack' || p.kind === 'cancel') return result(false, 'terminal', expired.outbound);
      this.duplicatePackets++;
      if (receipt.state === 'delivered') {
        const base = Math.floor((receipt.meta.chunkCount - 1) / receipt.meta.windowSize) * receipt.meta.windowSize;
        return result(true, 'already-delivered', [ack(receipt.meta, base, mask(base, receipt.meta.chunkCount, receipt.meta.windowSize), 'complete')]);
      }
      return result(false, 'terminal', [cancellation(receipt.meta, receipt.reason)]);
    }
    if (this.active && !same(this.active.meta, p)) { this.foreignPackets++; return result(false, 'foreign', expired.outbound); }
    if (p.kind === 'start') {
      if (p.totalBytes > this.maxMessageBytes) return result(false, 'message-limit', expired.outbound);
      if (this.active) {
        if (!metadataEqual(this.active.meta, p)) return result(false, 'conflicting-start', expired.outbound);
        this.duplicatePackets++;
      } else {
        this.active = { meta: p, bytes: new Uint8Array(p.totalBytes), seen: new Uint8Array(p.chunkCount), receivedChunks: 0, receivedBytes: 0, deadline: now + this.timeoutMs, verifying: false };
      }
      return result(true, undefined, [ack(this.active.meta, 0, this._bitmap(this.active, 0), 'ready')]);
    }
    if (!this.active) return result(false, 'no-active-message', expired.outbound);
    const a = this.active;
    if (p.kind === 'cancel') { this._finish(a, 'cancelled', now, `Peer cancelled transfer (${p.reason})`, p.reason); return result(true); }
    if (p.kind === 'ack') return result(false, 'unexpected');
    if (a.verifying) return result(false, 'verifying');
    if (p.kind === 'poll') {
      if (p.base >= a.meta.chunkCount || p.base % a.meta.windowSize) return result(false, 'invalid-window');
      return result(true, undefined, [ack(a.meta, p.base, this._bitmap(a, p.base), 'progress')]);
    }
    if (p.index >= a.meta.chunkCount || p.data.length !== expectedLength(a.meta, p.index)) return result(false, 'invalid-chunk');
    const offset = p.index * a.meta.chunkBytes;
    if (a.seen[p.index]) {
      this.duplicatePackets++;
      return result(false, equalBytes(a.bytes.subarray(offset, offset + p.data.length), p.data) ? 'duplicate' : 'conflicting-chunk');
    }
    a.bytes.set(p.data, offset); a.seen[p.index] = 1; a.receivedChunks++; a.receivedBytes += p.data.length;
    if (a.receivedChunks !== a.meta.chunkCount) return result(true);
    a.verifying = true;
    let digest;
    try { digest = await sha256(a.bytes); } catch {
      if (this.active !== a) return result(false, 'superseded');
      this._finish(a, 'failed', this.clock, 'SHA-256 unavailable', 2); return result(false, 'digest-error', [cancellation(a.meta, 2)]);
    }
    if (this.active !== a) return result(false, 'superseded');
    if (digest !== a.meta.digest) {
      this._finish(a, 'failed', this.clock, 'Full-message SHA-256 mismatch', 2); return result(false, 'digest-mismatch', [cancellation(a.meta, 2)]);
    }
    const delivery = { ...baseFields(a.meta), digest, bytes: a.bytes };
    this.completedMessages++; this._finish(a, 'delivered', this.clock);
    const base = Math.floor((a.meta.chunkCount - 1) / a.meta.windowSize) * a.meta.windowSize;
    return result(true, undefined, [ack(a.meta, base, mask(base, a.meta.chunkCount, a.meta.windowSize), 'complete')], delivery);
  }
  cancel(now) {
    now = this._clock(now); if (!this.active) return null;
    const a = this.active; this._finish(a, 'cancelled', now, 'Cancelled locally', 0); return cancellation(a.meta, 0);
  }
  snapshot() {
    const a = this.active, s = a ? { ...a.meta, state: 'receiving', receivedChunks: a.receivedChunks, receivedBytes: a.receivedBytes, error: null } : this.last;
    return { ...s, receivedBytes: s.receivedBytes ?? 0, receivedChunks: s.receivedChunks ?? 0, totalBytes: s.totalBytes ?? 0, chunkCount: s.chunkCount ?? 0,
      progress: s.state === 'delivered' ? 1 : s.totalBytes ? s.receivedBytes / s.totalBytes : 0,
      bufferedBytes: a ? a.bytes.byteLength + a.seen.byteLength : 0, terminalCount: this.receipts.length,
      completedMessages: this.completedMessages, duplicatePackets: this.duplicatePackets, malformedPackets: this.malformedPackets, foreignPackets: this.foreignPackets };
  }
}
