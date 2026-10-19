/** Bounded QR multipart transport. Completion checks integrity, not peer identity. */
import { encodePacket, decodePacket, sha256, randomId, LIMITS, hexToBytes } from './protocol.js';
import { equalBytes } from './bytes.js';
const TERMINAL = new Set(['delivered', 'failed', 'timed-out', 'cancelled']);
const result = (accepted, reason, outbound = [], delivery) => ({ accepted, ...(reason ? { reason } : {}), outbound, ...(delivery ? { delivery } : {}) });
const mask = (base, count, window) => (1 << Math.min(window, count - base)) - 1;
function numberOption(n, min, max, label) {
  if (!Number.isSafeInteger(n) || n < min || n > max) throw Error(`Invalid ${label}`); return n;
}
function time(n) { if (!Number.isFinite(n) || n < 0) throw Error('Invalid clock value'); return n; }
function metadataEqual(a, b) {
  return ['transferId', 'previousTransferId', 'totalBytes', 'chunkBytes', 'chunkCount', 'windowSize', 'digest'].every(k => a[k] === b[k]);
}
function smallerReplacement(previous, next) {
  return previous.totalBytes === next.totalBytes && previous.digest === next.digest && next.chunkBytes < previous.chunkBytes && next.windowSize <= previous.windowSize;
}
function ack(p, base, bitmap, status) {
  return encodePacket({ transferId: p.transferId, kind: 'ack', base, bitmap, status, digestPrefix: p.digest.slice(0, 16) });
}
function cancellation(p, reason) { return encodePacket({ transferId: p.transferId, kind: 'cancel', reason }); }
function expectedLength(p, index) { return Math.min(p.chunkBytes, p.totalBytes - index * p.chunkBytes); }

export async function createSender(payload, options = {}) {
  if (!(payload instanceof Uint8Array) || payload.length > LIMITS.maxMessageBytes) throw Error('Payload must be Uint8Array of at most 8 MiB');
  const bytes = payload.slice();
  const metadata = { transferId: options.transferId ?? randomId(), totalBytes: bytes.length, chunkBytes: options.chunkBytes ?? 128, windowSize: options.windowSize ?? 1 };
  hexToBytes(metadata.transferId, 8);
  if (options.previousTransferId !== undefined) {
    hexToBytes(options.previousTransferId, 8);
    if (options.previousTransferId === metadata.transferId) throw Error('RESTART requires a new transfer ID');
    metadata.previousTransferId = options.previousTransferId;
  }
  numberOption(metadata.chunkBytes, 1, LIMITS.maxChunkBytes, 'chunkBytes');
  numberOption(metadata.windowSize, 1, LIMITS.maxWindowSize, 'windowSize');
  metadata.chunkCount = Math.max(1, Math.ceil(bytes.length / metadata.chunkBytes));
  numberOption(metadata.chunkCount, 1, LIMITS.maxChunks, 'chunkCount (increase chunk size for this payload)');
  metadata.digest = await sha256(bytes);
  return new Sender(bytes, metadata, options);
}

class Sender {
  constructor(bytes, metadata, options) {
    this.bytes = bytes; this.meta = metadata;
    this.retryMs = numberOption(options.retryMs ?? 2500, 1, 86400000, 'retryMs');
    this.maxRetries = numberOption(options.maxRetries ?? 20, 0, 10000, 'maxRetries');
    this.timeoutMs = numberOption(options.timeoutMs ?? 120000, 1, 86400000, 'timeoutMs');
    this.startedAt = this.clock = time(options.now ?? 0); this.deadline = this.startedAt + this.timeoutMs;
    this.state = 'awaiting-start'; this.base = 0;
    this.acked = new Uint8Array(metadata.chunkCount); this.everSent = new Uint8Array(metadata.chunkCount);
    this.queue = []; this.attempts = 0; this.lastSendAt = null; this.pollPending = false;
    this.sentPackets = 0; this.retries = 0; this.error = null; this.ackedChunks = 0; this.ackedBytes = 0;
  }
  _clock(now) { this.clock = Math.max(this.clock, time(now)); return this.clock; }
  _fail(state, error) { this.state = state; this.error = error; this.queue = []; this.bytes = null; }
  _emit(packet, now) {
    this.sentPackets++; this.lastSendAt = now;
    if (packet.kind === 'poll') this.pollPending = true;
    return encodePacket({ ...this.meta, ...packet });
  }
  _retry(now, packet) {
    if (this.lastSendAt !== null) {
      if (now - this.lastSendAt < this.retryMs) return null;
      if (this.attempts >= this.maxRetries) { this._fail('failed', 'Retry limit reached'); return cancellation(this.meta, 1); }
      this.attempts++; this.retries++;
    }
    return this._emit(packet, now);
  }
  _data(index) { return { kind: 'data', index, data: this.bytes.subarray(index * this.meta.chunkBytes, (index + 1) * this.meta.chunkBytes) }; }
  _queueMissing() {
    this.queue = [];
    for (let i = this.base; i < Math.min(this.base + this.meta.windowSize, this.meta.chunkCount); i++) if (!this.acked[i]) this.queue.push(i);
    this.state = this.queue.length ? 'sending' : 'awaiting-ack'; this.lastSendAt = null; this.pollPending = false;
  }
  next(now) {
    now = this._clock(now);
    if (TERMINAL.has(this.state)) return null;
    if (now >= this.deadline) { this._fail('timed-out', 'Absolute transfer deadline reached'); return cancellation(this.meta, 1); }
    if (this.state === 'awaiting-start') return this._retry(now, { kind: this.meta.previousTransferId ? 'restart' : 'start' });
    if (this.state === 'awaiting-ack') {
      // Stop-and-wait keeps the same DATA QR visible while awaiting its ACK.
      // Larger windows poll for the receiver's cumulative bitmap.
      return this._retry(now, this.meta.windowSize === 1 ? this._data(this.base) : { kind: 'poll', base: this.base });
    }
    const index = this.queue.shift();
    this.everSent[index] = 1;
    if (!this.queue.length) this.state = 'awaiting-ack';
    return this._emit(this._data(index), now);
  }
  receive(bytes, now) {
    now = this._clock(now);
    if (TERMINAL.has(this.state)) return { accepted: false, reason: 'terminal' };
    if (now >= this.deadline) { this._fail('timed-out', 'Absolute transfer deadline reached'); return { accepted: false, reason: 'deadline' }; }
    let p; try { p = decodePacket(bytes); } catch { return { accepted: false, reason: 'malformed' }; }
    if (p.transferId !== this.meta.transferId) return { accepted: false, reason: 'foreign' };
    if (p.kind === 'cancel') { this._fail('cancelled', `Peer cancelled transfer (${p.reason})`); return { accepted: true }; }
    if (p.kind !== 'ack' || p.digestPrefix !== this.meta.digest.slice(0, 16)) return { accepted: false, reason: 'unexpected' };
    const m = this.meta;
    // A restart can refer to bytes already delivered under the previous ID.
    // This separate status never bypasses ordinary COMPLETE's sent-bit checks.
    if (p.status === 'stored') {
      if (!m.previousTransferId || this.state !== 'awaiting-start' || p.base || p.bitmap) return { accepted: false, reason: 'unexpected-stored' };
      this.acked.fill(1); this.ackedChunks = m.chunkCount; this.ackedBytes = m.totalBytes;
      this.state = 'delivered'; this.bytes = null; this.queue = []; return { accepted: true };
    }
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
    let newlyAcked = 0;
    for (let j = 0; j < m.windowSize && this.base + j < m.chunkCount; j++) {
      const i = this.base + j;
      if ((p.bitmap & (1 << j)) && !this.acked[i]) { this.acked[i] = 1; this.ackedChunks++; newlyAcked++; this.ackedBytes += expectedLength(m, i); }
    }
    const full = Array.from({ length: Math.min(m.windowSize, m.chunkCount - this.base) }, (_, j) => this.acked[this.base + j]).every(Boolean);
    if (full) {
      if (this.base + m.windowSize >= m.chunkCount) return { accepted: false, reason: 'awaiting-integrity-confirmation' };
      this.base += m.windowSize; this.attempts = 0;
    } else if (starting) this.attempts = 0;
    else {
      // Repeated camera observations of the same ACK must not consume retries
      // or keep postponing the clock. New information can trigger one resend.
      if (!newlyAcked && !this.pollPending && this.lastSendAt !== null && now - this.lastSendAt < this.retryMs) return { accepted: false, reason: 'duplicate-ack' };
      // An explicit POLL must accept even an unchanged bitmap: the previous
      // retransmission may also have been lost. That POLL already consumed
      // one retry; repeated camera reads after requeuing cannot consume more.
      if (!this.pollPending) {
        if (this.attempts >= this.maxRetries) { this._fail('failed', 'Retry limit reached'); return { accepted: false, reason: 'retry-limit' }; }
        this.attempts++; this.retries++;
      }
    }
    this._queueMissing(); return { accepted: true };
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
    this.completedTtlMs = numberOption(options.completedTtlMs ?? this.timeoutMs, 1, 86400000, 'completedTtlMs');
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
    this._remember({ meta: a.meta, state, reason: reason ?? 0, deadline: a.deadline, expiresAt: now + this.completedTtlMs });
    this.active = null;
  }
  _remember(receipt) {
    this.receipts.push(receipt);
    if (this.receipts.length > this.maxCompleted) this.receipts.shift();
  }
  _begin(meta, deadline) {
    this.active = { meta, bytes: new Uint8Array(meta.totalBytes), seen: new Uint8Array(meta.chunkCount), receivedChunks: 0, receivedBytes: 0, deadline, verifying: false };
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
    now = this._clock(now); const expired = this.tick(now);
    let p; try { p = decodePacket(bytes); } catch { this.malformedPackets++; return result(false, 'malformed', expired.outbound); }
    const receipt = this.receipts.find(r => r.meta.transferId === p.transferId);
    if (receipt) {
      if ((p.kind === 'start' || p.kind === 'restart') && !metadataEqual(receipt.meta, p)) return result(false, 'conflicting-start', expired.outbound);
      if (p.kind === 'ack' || p.kind === 'cancel') return result(false, 'terminal', expired.outbound);
      if (p.kind === 'data' && (p.index >= receipt.meta.chunkCount || p.data.length !== expectedLength(receipt.meta, p.index))) return result(false, 'invalid-chunk');
      if (p.kind === 'poll' && (p.base >= receipt.meta.chunkCount || p.base % receipt.meta.windowSize)) return result(false, 'invalid-window');
      this.duplicatePackets++;
      if (receipt.state === 'delivered') {
        if (receipt.stored) return result(true, 'already-delivered', [ack(receipt.meta, 0, 0, 'stored')]);
        const base = Math.floor((receipt.meta.chunkCount - 1) / receipt.meta.windowSize) * receipt.meta.windowSize;
        return result(true, 'already-delivered', [ack(receipt.meta, base, mask(base, receipt.meta.chunkCount, receipt.meta.windowSize), 'complete')]);
      }
      return result(false, 'terminal', [cancellation(receipt.meta, receipt.reason)]);
    }
    if (p.kind === 'restart' && this.active?.meta.transferId !== p.transferId) {
      if (p.totalBytes > this.maxMessageBytes) return result(false, 'message-limit', expired.outbound);
      const previous = this.receipts.find(r => r.meta.transferId === p.previousTransferId);
      let deadline = now + this.timeoutMs;
      if (this.active) {
        const a = this.active;
        if (a.meta.transferId !== p.previousTransferId) { this.foreignPackets++; return result(false, 'foreign', expired.outbound); }
        if (!smallerReplacement(a.meta, p)) return result(false, 'invalid-restart', expired.outbound);
        deadline = a.deadline;
        // A fresh generation discards all partial bytes. Delayed old packets
        // receive cancellation for their own ID and cannot roll back this one.
        this._finish(a, 'superseded', now, 'Replaced by smaller chunks', 3);
      } else if (previous) {
        if (previous.state !== 'delivered') return result(false, 'stale-restart', expired.outbound);
        if (!smallerReplacement(previous.meta, p)) return result(false, 'invalid-restart', expired.outbound);
        if (now >= previous.deadline) return result(false, 'deadline', expired.outbound);
        // Final ACK may have been lost even though the application already
        // received the payload. Do not deliver again or request its data again.
        this._remember({ meta: p, state: 'delivered', reason: 0, stored: true, deadline: previous.deadline, expiresAt: previous.expiresAt });
        return result(true, 'already-delivered', [ack(p, 0, 0, 'stored')]);
      }
      // Unknown predecessor is permitted: its original START may never have
      // been captured. The caller also bounds the whole authentication attempt.
      this._begin(p, deadline);
      return result(true, undefined, [ack(p, 0, 0, 'ready')]);
    }
    if (this.active && this.active.meta.transferId !== p.transferId) { this.foreignPackets++; return result(false, 'foreign', expired.outbound); }
    if (p.kind === 'start' || p.kind === 'restart') {
      if (p.totalBytes > this.maxMessageBytes) return result(false, 'message-limit', expired.outbound);
      if (this.active) {
        if (!metadataEqual(this.active.meta, p)) return result(false, 'conflicting-start', expired.outbound);
        this.duplicatePackets++;
      } else this._begin(p, now + this.timeoutMs);
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
    const offset = p.index * a.meta.chunkBytes, base = Math.floor(p.index / a.meta.windowSize) * a.meta.windowSize;
    if (a.seen[p.index]) {
      this.duplicatePackets++;
      if (!equalBytes(a.bytes.subarray(offset, offset + p.data.length), p.data)) return result(false, 'conflicting-chunk');
      return result(true, 'duplicate', [ack(a.meta, base, this._bitmap(a, base), 'progress')]);
    }
    a.bytes.set(p.data, offset); a.seen[p.index] = 1; a.receivedChunks++; a.receivedBytes += p.data.length;
    if (a.receivedChunks !== a.meta.chunkCount) return result(true, undefined, [ack(a.meta, base, this._bitmap(a, base), 'progress')]);
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
    const delivery = { transferId: a.meta.transferId, digest, bytes: a.bytes };
    this.completedMessages++; this._finish(a, 'delivered', this.clock);
    const finalBase = Math.floor((a.meta.chunkCount - 1) / a.meta.windowSize) * a.meta.windowSize;
    return result(true, undefined, [ack(a.meta, finalBase, mask(finalBase, a.meta.chunkCount, a.meta.windowSize), 'complete')], delivery);
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
