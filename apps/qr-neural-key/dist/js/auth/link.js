/** Camera-facing, bounded duplex adapter for the existing QRT transport. */
import { createSender, Receiver } from '../session.js';
import { decodePacket, LIMITS, bytesToHex } from '../protocol.js';
import { SPEED_PRESETS, validateProfile } from '../profiles.js';
import { smallerChunk, largerChunk, validateAckHold, MAX_AUTO_RESTARTS, CLEAN_WINDOWS_TO_GROW } from './adaptation.js';

const TERMINAL = new Set(['delivered', 'failed', 'timed-out', 'cancelled']);
const MAX_SENDS = 4;
const MAX_READS = 8;
const MAX_CONTROLS = 8;
const failure = reason => reason instanceof Error ? reason : new Error(String(reason));

/**
 * next() must be polled while active, including while proof work runs in a worker.
 * Returned bytes may renew a visible QR. Keep the previous QR when next() is null.
 * Receipts survive outgoing message handoffs, so a lost COMPLETE can be recovered.
 */
export class OpticalLink {
  constructor({ profile, adaptive = false, ackHoldMs = 500, now = () => performance.now(), onDelivery = () => {}, onChange = () => {}, onError = () => {}, timeoutMs = 600000, maxMessageBytes = 524288 } = {}) {
    this.configuredProfile = Object.freeze(validateProfile(profile ?? SPEED_PRESETS.find(p => p.id === 'fast')));
    if (typeof adaptive !== 'boolean') throw new TypeError('Adaptive must be a boolean');
    this.profile = this.configuredProfile; this.nextProfile = this.profile;
    this.adaptive = adaptive; this.ackHoldMs = validateAckHold(ackHoldMs);
    this.adaptation = { restarts: 0, downshifts: 0, upshifts: 0, reason: 'Configured local outgoing settings', events: [] };
    if (typeof now !== 'function' || [onDelivery, onChange, onError].some(fn => typeof fn !== 'function')) throw new TypeError('Link callbacks must be functions');
    this.now = now; this.onDelivery = onDelivery; this.onChange = onChange; this.onError = onError;
    this.receiver = new Receiver({ timeoutMs, maxMessageBytes, maxCompleted: 16, completedTtlMs: timeoutMs });
    this.timeoutMs = timeoutMs; this.maxMessageBytes = maxMessageBytes;
    this.closed = false; this.clock = 0; this.epoch = 0;
    this.queue = []; this.current = null; this.lastSender = null;
    this.controls = []; this.visible = null; this.lastSenderWire = null; this.preferSender = false;
    this.readTail = Promise.resolve(); this.pendingReads = 0;
  }

  _time() {
    const value = this.now();
    if (!Number.isFinite(value) || value < 0) throw new Error('Invalid link clock');
    this.clock = Math.max(this.clock, value); return this.clock;
  }
  _error(error) { try { this.onError(failure(error)); } catch { /* A UI callback cannot break the transport. */ } }
  _change() { try { this.onChange(this.snapshot()); } catch (error) { this._error(error); } }

  _record(kind, reason, fromChunkBytes = this.profile.chunkBytes, toChunkBytes = this.profile.chunkBytes) {
    this.adaptation.reason = reason;
    this.adaptation.events.push({ at: this._time(), kind, reason, fromChunkBytes, toChunkBytes });
    if (this.adaptation.events.length > 32) this.adaptation.events.shift();
  }

  /** Timing takes effect now; wire metadata changes at the next message unless
   * smaller chunks explicitly restart this same bounded message. */
  setProfile(profile, { adaptive = this.adaptive, ackHoldMs = this.ackHoldMs } = {}) {
    if (this.closed) throw new Error('Optical link is closed');
    const target = Object.freeze(validateProfile(profile));
    if (typeof adaptive !== 'boolean') throw new TypeError('Adaptive must be a boolean');
    const hold = validateAckHold(ackHoldMs); this._time();
    this.configuredProfile = target; this.nextProfile = target; this.adaptive = adaptive; this.ackHoldMs = hold;
    const current = this.current;
    if (!current) this.profile = target;
    else {
      this.profile = Object.freeze({ ...this.profile, dwellMs: target.dwellMs, retryMs: target.retryMs, controlEcc: target.controlEcc });
      if (current.sender) current.sender.retryMs = target.retryMs;
      current.cleanWindows = 0; current.cleanRetries = current.sender?.retries ?? 0; current.upshiftScheduled = false;
      current.deferredShrink = false;
      if (!current.sender) current.pendingBuildProfile = target;
      else if (target.chunkBytes < current.sender.meta.chunkBytes) {
        // Once RESTART is visible its READY may be lost: the receiver could be
        // on either generation. Confirm that handoff before naming it again.
        if (current.sender.meta.previousTransferId && current.sender.sentPackets && current.sender.state === 'awaiting-start') current.deferredShrink = true;
        else this._restart(current, target, 'Manual smaller QR setting; retransmitting the same message');
      }
    }
    // An explicit shorter/longer hold applies to the currently displayed code.
    if (this.visible) this.visible.until = this.visible.shownAt + (this.visible.control ? Math.max(this.ackHoldMs, this.profile.dwellMs) : this.profile.dwellMs);
    this._record('manual', current ? 'Local settings applied; larger chunks, window and correction changes wait for the next message' : 'Local outgoing settings applied');
    this._change(); return this.snapshot();
  }

  send(bytes) {
    if (this.closed) return Promise.reject(new Error('Optical link is closed'));
    if (!(bytes instanceof Uint8Array) || bytes.length > this.maxMessageBytes) return Promise.reject(new Error(`Message must be Uint8Array of at most ${this.maxMessageBytes} bytes`));
    if (this.queue.length + Number(!!this.current) >= MAX_SENDS) return Promise.reject(new Error('Optical send queue is full'));
    const startedAt = this._time(), payload = bytes.slice();
    const promise = new Promise((resolve, reject) => this.queue.push({ payload, startedAt, resolve, reject }));
    this._pump(); this._change(); return promise;
  }

  async _pump() {
    if (this.closed || this.current || !this.queue.length) return;
    const request = this.queue.shift(), current = { request, sender: null, generation: 0, autoRestarts: 0, dataStartedAt: null, ackWaitStartedAt: null, lastProgressAt: null, lastActionAt: null, cleanWindows: 0 };
    this.current = current; this.lastSenderWire = null;
    this.profile = Object.freeze({ ...this.nextProfile });
    await this._build(current, this.profile);
  }

  async _build(current, profile, previousTransferId) {
    const { request } = current, epoch = this.epoch, generation = ++current.generation;
    current.sender = null;
    try {
      const sender = await createSender(request.payload, {
        ...profile, ...(previousTransferId ? { previousTransferId } : {}), now: request.startedAt, timeoutMs: this.timeoutMs,
        // The absolute deadline bounds retries even at the fastest retry setting.
        maxRetries: Math.min(10000, Math.ceil(this.timeoutMs / 500) + 1),
      });
      if (this.closed || epoch !== this.epoch || this.current !== current || generation !== current.generation) return;
      if (current.pendingBuildProfile) {
        const pending = current.pendingBuildProfile; current.pendingBuildProfile = null;
        // The just-created sender has never emitted a code. Only the previous
        // visible transfer needs a RESTART link; never create a phantom chain.
        const next = previousTransferId ? { ...pending,
          ...(pending.chunkBytes >= profile.chunkBytes ? { chunkBytes: profile.chunkBytes, ecc: profile.ecc } : {}),
          windowSize: Math.min(pending.windowSize, current.restartWindowCeiling),
        } : pending;
        this.profile = Object.freeze(next);
        return this._build(current, next, previousTransferId);
      }
      sender.retryMs = this.profile.retryMs;
      current.sender = sender;
      this._change();
    } catch (error) {
      if (this.closed || epoch !== this.epoch || this.current !== current || generation !== current.generation) return;
      request.payload = null;
      this.current = null; request.reject(error); this._error(error); this._change(); this._pump();
    }
  }

  _restart(current, profile, reason) {
    if (!current.sender || this.current !== current) return;
    const previousTransferId = current.sender.sentPackets ? current.sender.meta.transferId : current.sender.meta.previousTransferId, from = current.sender.meta.chunkBytes;
    // A prepared but never displayed sender cannot be named as a predecessor:
    // the camera has never seen it. Keep the last actually emitted ancestor.
    current.restartWindowCeiling = current.sender.sentPackets || !previousTransferId ? current.sender.meta.windowSize : current.restartWindowCeiling;
    this.profile = Object.freeze({ ...profile, windowSize: Math.min(profile.windowSize, current.restartWindowCeiling) }); this.nextProfile = Object.freeze({ ...profile });
    current.deferredShrink = false;
    current.dataStartedAt = null; current.ackWaitStartedAt = null; current.lastProgressAt = null; current.lastActionAt = this._time(); current.cleanWindows = 0; current.cleanRetries = 0; current.upshiftScheduled = false;
    this.lastSenderWire = null;
    this.adaptation.restarts++; this._record('restart', reason, from, profile.chunkBytes);
    this._build(current, this.profile, previousTransferId);
  }

  _progress(current, before, after, accepted, now) {
    if (!accepted || !current || after.ackedBytes <= before.ackedBytes) return;
    current.lastProgressAt = now; current.ackWaitStartedAt = null;
    if (after.retries > before.retries || after.retries > (current.cleanRetries ?? 0)) current.cleanWindows = 0;
    else if (after.windowBase > before.windowBase || after.state === 'delivered') current.cleanWindows++;
    current.cleanRetries = after.retries;
    if (!this.adaptive || current.cleanWindows < CLEAN_WINDOWS_TO_GROW || current.upshiftScheduled) return;
    const next = largerChunk(this.profile.chunkBytes, this.configuredProfile.chunkBytes);
    if (next === this.profile.chunkBytes) return;
    current.upshiftScheduled = true;
    this.nextProfile = Object.freeze({ ...this.profile, chunkBytes: next, ecc: this.configuredProfile.ecc });
    this.adaptation.upshifts++; this._record('upshift', 'Eight clean ACK windows; larger chunks are queued for the next message', this.profile.chunkBytes, next);
  }

  _adapt(now) {
    const current = this.current, sender = current?.sender;
    if (!this.adaptive || !sender || TERMINAL.has(sender.state)) return;
    // A large window at a slow display rate must finish before judging its ACK.
    // Re-reading empty/duplicate ACKs never resets this stall clock.
    if (sender.state === 'sending' || (current.dataStartedAt !== null && current.ackWaitStartedAt === null)) return;
    const since = Math.max(current.ackWaitStartedAt ?? current.request.startedAt, current.lastActionAt ?? 0);
    if (now - since < 3 * sender.retryMs) return;
    current.lastActionAt = now; current.cleanWindows = 0;
    const smaller = smallerChunk(sender.meta.chunkBytes);
    if (current.dataStartedAt !== null && smaller < sender.meta.chunkBytes && current.autoRestarts < MAX_AUTO_RESTARTS) {
      const target = Object.freeze({ ...this.profile, chunkBytes: smaller, windowSize: 1, dwellMs: Math.max(350, this.profile.dwellMs) });
      this.ackHoldMs = Math.max(750, this.ackHoldMs); current.autoRestarts++; this.adaptation.downshifts++;
      this._restart(current, target, 'No new acknowledged bytes for three retry intervals; retrying the same message with smaller QRs');
    } else {
      const dwellMs = Math.min(2000, Math.max(350, Math.ceil(this.profile.dwellMs * 1.5)));
      if (dwellMs === this.profile.dwellMs && this.ackHoldMs >= 750) return;
      this.profile = Object.freeze({ ...this.profile, dwellMs, retryMs: Math.max(dwellMs, this.profile.retryMs) });
      this.nextProfile = Object.freeze({ ...this.nextProfile, dwellMs, retryMs: this.profile.retryMs });
      sender.retryMs = this.profile.retryMs; this.ackHoldMs = Math.max(750, this.ackHoldMs);
      this._record('timing', current.dataStartedAt === null ? 'Waiting for START acknowledgment; allowing more camera reading time' : 'Minimum chunk size or restart limit reached; allowing more camera reading time');
    }
    this._change();
  }

  _settle() {
    const current = this.current;
    if (!current?.sender) return;
    const snapshot = current.sender.snapshot();
    if (!TERMINAL.has(snapshot.state)) return;
    this.lastSender = snapshot; this.current = null; this.lastSenderWire = null;
    current.request.payload = null;
    if (snapshot.state === 'delivered') current.request.resolve(snapshot);
    else { const error = new Error(snapshot.error || `Optical transfer ${snapshot.state}`); current.request.reject(error); this._error(error); }
    this._change(); this._pump();
  }

  _control(bytes) {
    const packet = decodePacket(bytes), key = bytesToHex(bytes);
    // Re-reading the same QR does not extend its lease or build an ACK backlog.
    if (this.visible?.key === key) return;
    const existing = this.controls.findIndex(item => item.transferId === packet.transferId);
    const item = { bytes, key, transferId: packet.transferId };
    if (existing >= 0) this.controls[existing] = item;
    else { if (this.controls.length >= MAX_CONTROLS) this.controls.shift(); this.controls.push(item); }
  }

  receive(wire) {
    if (this.closed) return Promise.resolve({ accepted: false, reason: 'closed' });
    if (!(wire instanceof Uint8Array) || wire.length > LIMITS.maxChunkBytes + 18) return Promise.resolve({ accepted: false, reason: 'malformed' });
    if (this.pendingReads >= MAX_READS) return Promise.resolve({ accepted: false, reason: 'read-queue-full' });
    const bytes = wire.slice(), epoch = this.epoch;
    this.pendingReads++;
    const operation = this.readTail.then(async () => {
      if (this.closed || epoch !== this.epoch) return { accepted: false, reason: 'closed' };
      let packet;
      try { packet = decodePacket(bytes); } catch { return { accepted: false, reason: 'malformed' }; }
      const now = this._time();
      if (packet.kind === 'ack' || (packet.kind === 'cancel' && packet.transferId === this.current?.sender?.meta.transferId)) {
        const current = this.current, sender = current?.sender, before = sender?.snapshot();
        const result = sender?.receive(bytes, now) ?? { accepted: false, reason: 'no-sender' };
        if (sender) this._progress(current, before, sender.snapshot(), result.accepted, now);
        if (sender && result.accepted && current.deferredShrink && sender.state !== 'awaiting-start' && !TERMINAL.has(sender.state)) this._restart(current, this.nextProfile, 'Previous restart confirmed; applying the queued smaller QR setting');
        this._settle(); this._change(); return result;
      }
      const result = await this.receiver.receive(bytes, now);
      if (this.closed || epoch !== this.epoch) return { accepted: false, reason: 'closed' };
      for (const outbound of result.outbound) this._control(outbound);
      this._change();
      if (result.delivery) {
        // The application may queue a reply or start proof generation immediately.
        // It must not hold up reads or the final transport ACK for this delivery.
        try {
          Promise.resolve(this.onDelivery(result.delivery.bytes)).catch(error => { if (!this.closed && epoch === this.epoch) this._error(error); });
        } catch (error) { this._error(error); }
      }
      return result;
    });
    this.readTail = operation.catch(error => { if (!this.closed && epoch === this.epoch) this._error(error); }).finally(() => { this.pendingReads--; });
    return operation;
  }

  _show(bytes, now, control) {
    this.visible = { bytes, key: bytesToHex(bytes), shownAt: now, control, until: now + (control ? Math.max(this.ackHoldMs, this.profile.dwellMs) : this.profile.dwellMs) };
    this.preferSender = control;
    return bytes.slice();
  }

  _outgoing(now) {
    const current = this.current, sender = current?.sender;
    if (!sender) return null;
    const bytes = sender.next(now);
    this._settle();
    if (bytes) {
      const kind = decodePacket(bytes).kind;
      if (kind === 'data' && current.dataStartedAt === null) current.dataStartedAt = now;
      if (kind === 'data' && sender.state === 'awaiting-ack' && current.ackWaitStartedAt === null) current.ackWaitStartedAt = now;
      if (kind !== 'cancel') this.lastSenderWire = bytes;
      this._change(); return this._show(bytes, now, false);
    }
    // After yielding the display to an ACK, restore the still-current outgoing
    // packet. This does not reset Sender retry clocks or consume another retry.
    if (this.lastSenderWire && this.visible?.key !== bytesToHex(this.lastSenderWire)) return this._show(this.lastSenderWire, now, false);
    return null;
  }

  next() {
    if (this.closed) return null;
    const now = this._time();
    for (const bytes of this.receiver.tick(now).outbound) this._control(bytes);
    for (let i = this.queue.length - 1; i >= 0; i--) {
      if (now >= this.queue[i].startedAt + this.timeoutMs) {
        const [request] = this.queue.splice(i, 1); request.payload = null;
        const error = new Error('Queued optical transfer deadline reached'); request.reject(error); this._error(error); this._change();
      }
    }
    if (this.current?.sender && now >= this.current.sender.deadline) {
      const cancellation = this.current.sender.next(now);
      if (cancellation) this._control(cancellation);
      this._settle();
    }
    if (this.current && !this.current.sender && now >= this.current.request.startedAt + this.timeoutMs) {
      const current = this.current; this.current = null; current.generation++; current.request.payload = null;
      const error = new Error('Absolute transfer deadline reached'); current.request.reject(error); this._error(error);
      this.lastSender = { state: 'timed-out', error: error.message }; this._change(); this._pump();
    }
    this._adapt(now);
    if (this.visible && now < this.visible.until) return null;
    if (this.preferSender) {
      const outgoing = this._outgoing(now);
      if (outgoing) return outgoing;
    }
    if (this.controls.length) return this._show(this.controls.shift().bytes, now, true);
    return this._outgoing(now);
  }

  snapshot() {
    const sender = this.current?.sender?.snapshot() ?? (this.current ? { state: 'preparing', totalBytes: this.current.request.payload?.length ?? 0, ackedBytes: 0, progress: 0 } : this.lastSender);
    const differs = ['chunkBytes', 'ecc', 'windowSize'].some(key => this.nextProfile[key] !== this.profile[key]);
    return { sender, receiver: this.receiver.snapshot(), queued: this.queue.length, closed: this.closed,
      configuredProfile: { ...this.configuredProfile }, effectiveProfile: { ...this.profile }, ackHoldMs: this.ackHoldMs, adaptive: this.adaptive,
      adaptation: { ...this.adaptation, events: this.adaptation.events.map(event => ({ ...event })), pendingProfile: differs ? { ...this.nextProfile } : null } };
  }

  close(reason = 'Optical link closed') {
    if (this.closed) return;
    this.closed = true; this.epoch++;
    const error = failure(reason), now = this._time();
    if (this.current) {
      this.current.sender?.cancel(now);
      this.lastSender = this.current.sender?.snapshot() ?? { state: 'cancelled', error: error.message };
      this.current.request.payload = null; this.current.request.reject(error); this.current = null;
    }
    for (const request of this.queue) { request.payload = null; request.reject(error); }
    this.queue = []; this.receiver.cancel(now); this.controls = []; this.visible = null; this.lastSenderWire = null;
    this._change();
  }
}
