/** Ephemeral, one-direction authentication. Optical delivery is not authentication. */
import { canonicalJSON, hex } from '../crypto/bytes.js';
import { randomBytes, sha256 } from '../crypto/hashes.js';
import { AUTH_PROTOCOL, decodeEnvelope, encodeEnvelope, validDeviceId, validateAuthPublic } from './envelope.js';

const TERMINAL = new Set(['verified', 'rejected', 'peer-accepted', 'peer-rejected', 'failed', 'timed-out', 'closed']);
const copy = value => structuredClone(value);
const hashDocument = async value => hex(await sha256(canonicalJSON(value)));
const STATUS = {
  idle: 'Ready to create a local key', 'generating-key': 'Generating an ephemeral key locally',
  'awaiting-hello': 'Waiting for the verifier identity', 'awaiting-identity': 'Sending verifier identity; waiting for the key holder',
  pairing: 'Compare the pairing code on both devices, then confirm on each',
  'creating-challenge': 'Creating a fresh challenge', 'awaiting-challenge': 'Pairing confirmed; waiting for the challenge',
  'awaiting-proof': 'Challenge sent; waiting for the proof', proving: 'Generating the proof locally',
  'sending-proof': 'Sending the proof; waiting for the verifier result', verifying: 'Checking the proof locally',
  verified: 'Proof accepted locally', rejected: 'Proof rejected locally',
  'peer-accepted': 'Peer reports acceptance (peer report is not authenticated)',
  'peer-rejected': 'Peer reports rejection (peer report is not authenticated)',
  failed: 'Authentication attempt failed', 'timed-out': 'Authentication attempt expired', closed: 'Authentication attempt closed',
};

export class AuthSession {
  #role; #deviceId; #crypto; #send; #now; #onChange; #timeout; #repetitions;
  #epoch = 0; #state = 'idle'; #deadline = null; #error = null; #progress = null;
  #startedAt = null; #finishedAt = null; #timings = { keygenMs: null, proveMs: null, verifyMs: null };
  #local = null; #peer = null; #localKeyId = null; #peerKeyId = null; #peerDeviceId = null;
  #sessionId = null; #binding = null; #confirmed = false; #challenge = null; #challengeConsumed = false;
  #proofBytes = 0; #proofDigest = null; #result = null; #resultDelivered = false; #deferredHello = null;
  #receiving = false;

  constructor({ role, deviceId, cryptoClient, send, now = () => performance.now(), onChange = () => {}, timeoutMs = 600000, repetitions = 219 }) {
    if (!['verifier', 'prover'].includes(role) || !validDeviceId(deviceId) || typeof cryptoClient?.request !== 'function' ||
        typeof send !== 'function' || typeof now !== 'function' || typeof onChange !== 'function' ||
        !Number.isInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 3600000 ||
        !Number.isInteger(repetitions) || repetitions < 1 || repetitions > 512) throw new TypeError('Invalid authentication session options');
    this.#role = role; this.#deviceId = deviceId; this.#crypto = cryptoClient; this.#send = send;
    this.#now = now; this.#onChange = onChange; this.#timeout = timeoutMs; this.#repetitions = repetitions;
  }

  snapshot() {
    return {
      role: this.#role, state: this.#state, status: STATUS[this.#state], pairingCode: this.#binding?.slice(0, 24) ?? null,
      localKeyId: this.#localKeyId, peerKeyId: this.#peerKeyId, confirmed: this.#confirmed,
      proofBytes: this.#proofBytes, result: this.#result, deadline: this.#deadline, error: this.#error,
      progress: this.#progress ? copy(this.#progress) : null, canConfirm: this.#state === 'pairing' && !!this.#binding && !this.#confirmed,
      challengeReceived: !!this.#challenge, repetitions: this.#challenge?.required_repetitions ?? this.#repetitions,
      sessionId: this.#sessionId, binding: this.#binding, proofDigest: this.#proofDigest,
      resultDelivered: this.#resultDelivered, peerReported: this.#role === 'prover' && this.#result !== null,
      timings: { ...this.#timings, elapsedMs: this.#startedAt === null ? 0 : Math.max(0, (this.#finishedAt ?? this.#now()) - this.#startedAt) },
    };
  }
  #notify() { try { this.#onChange(this.snapshot()); } catch { /* Observers cannot alter the protocol. */ } }
  #set(state) {
    this.#state = state; this.#progress = null;
    if (TERMINAL.has(state) && this.#finishedAt === null) this.#finishedAt = this.#now();
    this.#notify();
  }
  #active(epoch = this.#epoch) {
    if (epoch !== this.#epoch || TERMINAL.has(this.#state)) return false;
    if (this.#deadline !== null && this.#now() >= this.#deadline) { this.#stop('timed-out', 'The attempt deadline expired'); return false; }
    return true;
  }
  #stop(state, reason) {
    ++this.#epoch; this.#error = String(reason); this.#deferredHello = null; this.#challengeConsumed = true;
    this.#set(state);
    try { this.#crypto.cancel?.(); } catch { /* Epoch protection remains effective if cleanup fails. */ }
  }
  #fail(error, epoch) { if (this.#active(epoch)) this.#stop('failed', error?.message ?? String(error)); }
  tick() { this.#active(); return this.snapshot(); }
  close(reason = 'Closed by the user') { if (this.#state !== 'closed') this.#stop('closed', reason); }

  async #request(type, payload, epoch) {
    const result = await this.#crypto.request(type, payload, { onProgress: progress => {
      if (this.#active(epoch)) { this.#progress = copy(progress); this.#notify(); }
    } });
    if (!this.#active(epoch)) return null;
    for (const field of ['keygenMs', 'proveMs', 'verifyMs']) {
      if (Number.isFinite(result?.[field]) && result[field] >= 0 && result[field] <= this.#timeout) this.#timings[field] = result[field];
    }
    return result;
  }
  #transmit(type, fields, body, onDelivered) {
    const epoch = this.#epoch;
    const bytes = encodeEnvelope({ protocol: AUTH_PROTOCOL, type, sessionId: this.#sessionId, ...fields }, body);
    // Delivery acknowledgments run independently of application messages and crypto work.
    Promise.resolve().then(() => {
      if (epoch !== this.#epoch) return;
      return this.#send(bytes);
    }).then(() => {
      if (epoch === this.#epoch) onDelivered?.();
    }).catch(error => {
      if (epoch !== this.#epoch) return;
      if (this.#state === 'verified' || this.#state === 'rejected') {
        this.#error = `Local result is final, but result delivery failed: ${error?.message ?? error}`; this.#notify();
      } else this.#fail(error, epoch);
    });
  }

  async start() {
    if (this.#state !== 'idle') throw new Error('This authentication attempt has already started');
    this.#startedAt = this.#now(); this.#deadline = this.#startedAt + this.#timeout; this.#set('generating-key');
    const epoch = this.#epoch;
    try {
      const result = await this.#request('createKey', { profile: { side: 8, rounds: 16, repetitions: this.#repetitions } }, epoch);
      if (!result) return;
      validateAuthPublic(result.publicKey); this.#local = copy(result.publicKey);
      if (this.#local.minimum_repetitions !== this.#repetitions) throw new Error('Local key policy mismatch');
      this.#localKeyId = await hashDocument(this.#local);
      if (!this.#active(epoch)) return;
      if (this.#role === 'verifier') {
        this.#sessionId = hex(randomBytes(16)); this.#set('awaiting-identity');
        this.#transmit('HELLO', { deviceId: this.#deviceId, publicKey: this.#local });
      } else {
        this.#set('awaiting-hello');
        if (this.#deferredHello) { const pending = this.#deferredHello; this.#deferredHello = null; await this.receive(pending); }
      }
    } catch (error) { this.#fail(error, epoch); }
  }

  async #pair(header, epoch) {
    const peer = copy(header.publicKey);
    const peerKeyId = await hashDocument(peer);
    if (!this.#active(epoch)) return;
    const verifier = this.#role === 'verifier' ? { deviceId: this.#deviceId, keyId: this.#localKeyId } : { deviceId: header.deviceId, keyId: peerKeyId };
    const prover = this.#role === 'prover' ? { deviceId: this.#deviceId, keyId: this.#localKeyId } : { deviceId: header.deviceId, keyId: peerKeyId };
    const binding = await hashDocument({ protocol: AUTH_PROTOCOL, sessionId: header.sessionId, verifier, prover });
    if (!this.#active(epoch)) return;
    this.#sessionId = header.sessionId; this.#peer = peer; this.#peerKeyId = peerKeyId;
    this.#peerDeviceId = header.deviceId; this.#binding = binding; this.#set('pairing');
  }

  async confirmPairing() {
    if (!this.#active() || !this.snapshot().canConfirm) return false;
    this.#confirmed = true;
    const epoch = this.#epoch;
    try {
      if (this.#role === 'verifier') {
        this.#set('creating-challenge');
        const required = Math.max(this.#repetitions, this.#peer.minimum_repetitions);
        const result = await this.#request('newChallenge', { lockId: `qr:${this.#binding}`, action: 'authenticate', requestedRepetitions: required }, epoch);
        if (!result) return false;
        this.#challenge = copy(result.challenge); this.#set('awaiting-proof');
        this.#transmit('CHALLENGE', { binding: this.#binding, challenge: this.#challenge });
      } else if (this.#challenge) await this.#prove(epoch);
      else this.#set('awaiting-challenge');
      return true;
    } catch (error) { this.#fail(error, epoch); return false; }
  }

  async #prove(epoch) {
    this.#set('proving');
    const result = await this.#request('prove', { challenge: copy(this.#challenge) }, epoch);
    if (!result) return;
    const digest = hex(await sha256(result.proof));
    if (!this.#active(epoch)) return;
    this.#proofBytes = result.proof.length; this.#proofDigest = digest; this.#set('sending-proof');
    this.#transmit('PROOF', { binding: this.#binding, nonce: this.#challenge.nonce }, result.proof);
  }

  async receive(bytes) {
    if (!this.#active() || this.#state === 'idle') return { accepted: false, reason: 'Attempt is not active' };
    let envelope;
    try { envelope = decodeEnvelope(bytes); } catch (error) { return { accepted: false, reason: error.message }; }
    const { header, body } = envelope;
    if (this.#state === 'generating-key' && this.#role === 'prover' && header.type === 'HELLO') {
      if (!this.#deferredHello) { this.#deferredHello = bytes.slice(); return { accepted: true, deferred: true }; }
      return { accepted: false, reason: 'Already waiting on an identity' };
    }
    if (this.#receiving) return { accepted: false, reason: 'Application message is already being processed' };
    const epoch = this.#epoch;
    this.#receiving = true;
    try {
      if (header.type === 'HELLO' && this.#role === 'prover' && this.#state === 'awaiting-hello') {
        await this.#pair(header, epoch);
        if (!this.#active(epoch)) return { accepted: false, reason: 'Attempt ended' };
        this.#transmit('IDENTITY', { deviceId: this.#deviceId, publicKey: this.#local });
        return { accepted: true };
      }
      if (header.sessionId !== this.#sessionId) return { accepted: false, reason: 'Session ID mismatch' };
      if (header.type === 'IDENTITY' && this.#role === 'verifier' && this.#state === 'awaiting-identity') {
        await this.#pair(header, epoch);
        return { accepted: this.#active(epoch) };
      }
      if (!this.#binding || header.binding !== this.#binding) return { accepted: false, reason: 'Pairing binding mismatch' };
      if (header.type === 'CHALLENGE' && this.#role === 'prover' && ['pairing', 'awaiting-challenge'].includes(this.#state) && !this.#challenge) {
        const expected = Math.max(this.#repetitions, this.#peer.minimum_repetitions);
        if (header.challenge.lock_id !== `qr:${this.#binding}` || header.challenge.action !== 'authenticate' || header.challenge.required_repetitions !== expected) {
          return { accepted: false, reason: 'Challenge does not match the confirmed pairing and repetition policy' };
        }
        this.#challenge = copy(header.challenge); this.#notify();
        if (this.#confirmed) await this.#prove(epoch);
        return { accepted: this.#active(epoch) };
      }
      if (header.type === 'PROOF' && this.#role === 'verifier' && this.#state === 'awaiting-proof' && this.#confirmed && !this.#challengeConsumed && header.nonce === this.#challenge.nonce) {
        // Consume before asynchronous verification. No second proof can race this one.
        this.#challengeConsumed = true; this.#proofBytes = body.length; this.#set('verifying');
        const digest = hex(await sha256(body));
        if (!this.#active(epoch)) return { accepted: false, reason: 'Attempt ended' };
        const result = await this.#request('verify', { publicKey: copy(this.#peer), challenge: copy(this.#challenge), proof: body }, epoch);
        if (!result) return { accepted: false, reason: 'Attempt ended' };
        this.#proofDigest = digest; this.#result = result.accepted === true;
        this.#set(this.#result ? 'verified' : 'rejected');
        this.#transmit('RESULT', { binding: this.#binding, nonce: this.#challenge.nonce, proofDigest: digest, accepted: this.#result }, undefined,
          () => { this.#resultDelivered = true; this.#notify(); });
        return { accepted: true, verified: this.#result };
      }
      if (header.type === 'RESULT' && this.#role === 'prover' && this.#state === 'sending-proof' && header.nonce === this.#challenge.nonce && header.proofDigest === this.#proofDigest) {
        this.#result = header.accepted; this.#set(header.accepted ? 'peer-accepted' : 'peer-rejected');
        return { accepted: true, peerReported: true };
      }
      return { accepted: false, reason: 'Unexpected or duplicate authentication message' };
    } catch (error) { this.#fail(error, epoch); return { accepted: false, reason: error.message }; }
    finally { this.#receiving = false; }
  }
}
