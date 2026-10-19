import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate as yieldTurn } from 'node:timers/promises';
import { AuthSession } from '../dist/js/auth/session.js';
import { OpticalLink } from '../dist/js/auth/link.js';
import { decodeEnvelope, encodeEnvelope } from '../dist/js/auth/envelope.js';
import { CryptoOperations } from '../dist/js/crypto-worker/operations.js';
import { decodePacket } from '../dist/js/protocol.js';
import { SPEED_PRESETS } from '../dist/js/profiles.js';
import { encodeQr, renderQr, decodeQr } from '../dist/js/qr.js';

const DEVICE_IDS = {
  verifier: '11111111-1111-4111-8111-111111111111',
  prover: '22222222-2222-4222-8222-222222222222',
};
const opposite = role => role === 'verifier' ? 'prover' : 'verifier';
const roles = ['verifier', 'prover'];
const fast = SPEED_PRESETS.find(profile => profile.id === 'fast');
const phone = { ...SPEED_PRESETS.find(profile => profile.id === 'slow'), ackHoldMs: 750 };

/** The actual worker operations, with observable calls and no substitute proof algorithm. */
class LocalCryptoClient {
  constructor(role, hooks) {
    this.role = role; this.hooks = hooks; this.operations = new CryptoOperations();
    this.calls = []; this.pending = 0;
  }
  get busy() { return this.pending > 0; }
  async request(type, payload, { onProgress } = {}) {
    const entry = { type, payload: structuredClone(payload) };
    this.calls.push(entry); this.pending++;
    try {
      const actual = this.hooks.request?.(this.role, type, structuredClone(payload)) ?? payload;
      const result = await this.operations.run(type, actual, onProgress);
      entry.result = structuredClone(result);
      await this.hooks.after?.(this.role, type, result);
      return result;
    } finally { this.pending--; }
  }
  cancel() { this.operations.clear(); }
  count(type) { return this.calls.filter(call => call.type === type).length; }
}

/** Two independent devices. The only exchange is their persistent displayed QR. */
class DevicePair {
  constructor({ repetitions = 32, qrImages = false, transform, drop, cryptoHooks = {}, timeoutMs = 600000,
    profiles = {}, adaptive = false } = {}) {
    this.clock = 0; this.qrImages = qrImages; this.transform = transform; this.drop = drop;
    this.links = {}; this.sessions = {}; this.clients = {}; this.visible = {};
    this.sent = []; this.received = []; this.errors = []; this.qrCache = new Map(); this.scanCount = 0;
    this.displayed = [];
    for (const role of roles) {
      const client = this.clients[role] = new LocalCryptoClient(role, cryptoHooks);
      this.sessions[role] = new AuthSession({
        role, deviceId: DEVICE_IDS[role], cryptoClient: client, repetitions, timeoutMs,
        now: () => this.clock,
        send: bytes => {
          const original = decodeEnvelope(bytes);
          const transmitted = this.transform?.(role, original, bytes.slice()) ?? bytes;
          const envelope = decodeEnvelope(transmitted);
          this.sent.push({ role, ...envelope, bytes: transmitted.slice() });
          return this.links[role].send(transmitted);
        },
      });
      this.links[role] = new OpticalLink({
        profile: profiles[role] ?? fast, adaptive, ackHoldMs: profiles[role]?.ackHoldMs ?? 500,
        now: () => this.clock, timeoutMs,
        onDelivery: bytes => {
          this.received.push({ role, ...decodeEnvelope(bytes) });
          return this.sessions[role].receive(bytes);
        },
        onError: error => this.errors.push({ role, message: error.message }),
      });
    }
  }
  async start() {
    await this.sessions.prover.start();
    await this.sessions.verifier.start();
  }
  snapshot() { return Object.fromEntries(roles.map(role => [role, this.sessions[role].snapshot()])); }
  async scan(from) {
    const wire = this.visible[from];
    if (!wire) return;
    const packet = decodePacket(wire), to = opposite(from);
    if (this.drop?.({ from, to, packet, wire, clock: this.clock })) return;
    let decoded = wire;
    if (this.qrImages) {
      const key = Buffer.from(wire).toString('hex');
      decoded = this.qrCache.get(key);
      if (!decoded) {
        const ecc = packet.kind === 'data' ? (this.links[from].snapshot().effectiveProfile ?? this.links[from].profile).ecc : 'Q';
        const image = renderQr(encodeQr(wire, { ecc }), { scale: 3 });
        const result = decodeQr(image);
        assert.ok(result, `A clean ${packet.kind} QR must decode`);
        assert.deepEqual(result.bytes, wire);
        decoded = result.bytes; this.qrCache.set(key, decoded);
      }
    }
    this.scanCount++;
    await this.links[to].receive(decoded);
  }
  async step({ advanceClock = true } = {}) {
    for (const role of roles) {
      const next = this.links[role].next();
      if (next) {
        this.visible[role] = next;
        this.displayed.push({ role, packet: decodePacket(next), length: next.length, clock: this.clock });
      }
    }
    await Promise.all(roles.map(role => this.scan(role)));
    // Browser workers continue independently of the camera. Do not burn fake
    // minutes during one real asynchronous hash or an intentionally held job.
    await yieldTurn();
    if (advanceClock && !roles.some(role => this.clients[role].busy)) this.clock += 50;
    for (const role of roles) this.sessions[role].tick();
  }
  async until(predicate, { confirm = false, maxSteps = 20000 } = {}) {
    for (let step = 0; step < maxSteps; step++) {
      const states = this.snapshot();
      if (predicate(states)) return states;
      if (confirm && states.verifier.canConfirm && states.prover.canConfirm) {
        assert.equal(states.verifier.pairingCode, states.prover.pairingCode);
        await Promise.all(roles.map(role => this.sessions[role].confirmPairing()));
      }
      await this.step();
    }
    assert.fail(`Authentication did not reach the expected state: ${JSON.stringify({ clock: this.clock, states: this.snapshot(), links: Object.fromEntries(roles.map(role => [role, this.links[role].snapshot()])), errors: this.errors })}`);
  }
  async finish() {
    return this.until(states => states.verifier.resultDelivered && ['peer-accepted', 'peer-rejected'].includes(states.prover.state), { confirm: true });
  }
  close() {
    for (const role of roles) this.sessions[role].close('Test cleanup');
    for (const role of roles) this.links[role].close('Test cleanup');
  }
}

test('32-repetition authentication completes using rendered QR images and the real proof engine', async t => {
  const pair = new DevicePair({ qrImages: true }); t.after(() => pair.close());
  await pair.start();
  const states = await pair.finish();
  assert.equal(states.verifier.state, 'verified');
  assert.equal(states.prover.state, 'peer-accepted');
  assert.equal(states.verifier.result, true);
  assert.equal(states.prover.peerReported, true);
  assert.equal(states.verifier.proofBytes, 17474);
  assert.equal(states.verifier.proofDigest, states.prover.proofDigest);
  assert.equal(states.verifier.binding, states.prover.binding);
  assert.equal(states.verifier.pairingCode.length, 24);
  assert.deepEqual(pair.sent.map(message => message.header.type), ['HELLO', 'IDENTITY', 'CHALLENGE', 'PROOF', 'RESULT']);
  assert.equal(pair.clients.prover.count('prove'), 1);
  assert.equal(pair.clients.verifier.count('verify'), 1);
  assert.ok(pair.qrCache.size > 20, 'Every distinct displayed QR is rendered and decoded at least once');
  assert.deepEqual(pair.errors, []);
});

test('real QR authentication uses independent 128-byte phone and 1024-byte desktop directions', async t => {
  const pair = new DevicePair({
    qrImages: true, profiles: { verifier: fast, prover: phone },
    // Model a PC webcam that cannot decode the phone's denser DATA symbols.
    drop: ({ from, packet, wire }) => from === 'prover' && packet.kind === 'data' && wire.length > 146,
  }); t.after(() => pair.close());
  await pair.start();
  const states = await pair.finish();
  assert.equal(states.verifier.state, 'verified');
  assert.equal(states.prover.state, 'peer-accepted');
  assert.equal(states.verifier.proofBytes, 17474);
  assert.equal(states.verifier.proofDigest, states.prover.proofDigest);
  assert.equal(pair.clients.prover.count('prove'), 1);
  assert.equal(pair.clients.verifier.count('verify'), 1);
  assert.equal(pair.received.length, 5);
  const phoneData = pair.displayed.filter(frame => frame.role === 'prover' && frame.packet.kind === 'data');
  assert.ok(phoneData.length > 130, 'The complete proof is split into small phone QRs');
  assert.ok(phoneData.every(frame => frame.length <= 146), 'Every phone DATA QR obeys its local chunk cap');
  assert.ok(pair.displayed.some(frame => frame.role === 'verifier' && frame.packet.kind === 'data' && frame.length > 146),
    'The desktop retains its independently larger outbound chunk size');
  assert.ok(pair.qrCache.size > 150, 'Both directions pass through the actual QR renderer and decoder');
  assert.deepEqual(pair.errors, []);
});

test('small outbound phone QRs remain independent of the authentication role', async t => {
  const pair = new DevicePair({
    repetitions: 8, profiles: { verifier: phone, prover: fast },
    drop: ({ from, packet, wire }) => from === 'verifier' && packet.kind === 'data' && wire.length > 146,
  }); t.after(() => pair.close());
  await pair.start();
  const states = await pair.finish();
  assert.equal(states.verifier.state, 'verified');
  assert.equal(states.prover.state, 'peer-accepted');
  const phoneData = pair.displayed.filter(frame => frame.role === 'verifier' && frame.packet.kind === 'data');
  assert.ok(phoneData.length > 0);
  assert.ok(phoneData.every(frame => frame.length <= 146));
  assert.ok(pair.displayed.some(frame => frame.role === 'prover' && frame.packet.kind === 'data' && frame.length > 146));
  assert.equal(pair.clients.prover.count('prove'), 1);
  assert.equal(pair.clients.verifier.count('verify'), 1);
  assert.deepEqual(pair.errors, []);
});

test('unreadable dense proof QRs downshift and recover without regenerating the proof', async t => {
  let pair;
  let blockedProofPackets = 0;
  pair = new DevicePair({
    adaptive: true, profiles: { verifier: fast, prover: fast },
    drop: ({ from, packet, wire }) => {
      const proofQueued = pair.sent.some(message => message.header.type === 'PROOF');
      if (proofQueued && from === 'prover' && packet.kind === 'data' && wire.length > 146) {
        blockedProofPackets++; return true;
      }
      return false;
    },
  }); t.after(() => pair.close());
  await pair.start();
  const originalDeadlines = Object.fromEntries(roles.map(role => [role, pair.snapshot()[role].deadline]));
  const states = await pair.finish();
  assert.equal(states.verifier.state, 'verified');
  assert.equal(states.prover.state, 'peer-accepted');
  assert.ok(blockedProofPackets > 0, 'The initially dense phone proof cannot reach the desktop');
  assert.equal(pair.clients.prover.count('prove'), 1, 'Fallback reuses the original proof bytes');
  assert.equal(pair.clients.verifier.count('verify'), 1);
  assert.equal(pair.received.filter(message => message.header.type === 'PROOF').length, 1);
  assert.equal(pair.sent.filter(message => message.header.type === 'PROOF').length, 1);
  const proof = pair.sent.find(message => message.header.type === 'PROOF');
  assert.deepEqual(pair.clients.verifier.calls.find(call => call.type === 'verify').payload.proof, proof.body);
  const adaptation = pair.links.prover.snapshot().adaptation;
  assert.ok(adaptation.downshifts >= 3);
  assert.ok(adaptation.events.some(event => event.fromChunkBytes === 1024 && event.toChunkBytes === 512));
  assert.ok(adaptation.events.some(event => event.fromChunkBytes === 512 && event.toChunkBytes === 256));
  assert.ok(adaptation.events.some(event => event.fromChunkBytes === 256 && event.toChunkBytes === 128));
  assert.equal(pair.links.prover.snapshot().configuredProfile.chunkBytes, 1024);
  for (const role of roles) assert.equal(states[role].deadline, originalDeadlines[role], 'Fallback preserves the attempt deadline');
  assert.deepEqual(pair.errors, []);
});

test('adaptive restart after a lost proof completion ACK does not redeliver or reverify', async t => {
  let pair;
  let completionBlockedUntil = null;
  let originalCompletionTransferId = null;
  pair = new DevicePair({
    repetitions: 8, adaptive: true,
    drop: ({ from, to, packet, wire, clock }) => {
      const proof = pair.sent.find(message => message.header.type === 'PROOF');
      if (!proof) return false;
      if (from === 'prover' && packet.kind === 'data' && wire.length > 146) return true;
      if (to !== 'prover' || packet.kind !== 'ack' || packet.status !== 'complete') return false;
      const isProof = pair.displayed.some(frame => frame.role === 'prover' &&
        ['start', 'restart'].includes(frame.packet.kind) && frame.packet.transferId === packet.transferId && frame.packet.totalBytes === proof.bytes.length);
      if (!isProof) return false;
      if (completionBlockedUntil === null) {
        originalCompletionTransferId = packet.transferId;
        completionBlockedUntil = clock + 10000;
      }
      // A smaller restart may recover a stored receipt under a new transfer ID.
      return packet.transferId === originalCompletionTransferId && clock < completionBlockedUntil;
    },
  }); t.after(() => pair.close());
  await pair.start();
  const states = await pair.until(value => value.verifier.resultDelivered && value.prover.state === 'peer-accepted' &&
    pair.links.prover.snapshot().sender?.state === 'delivered', { confirm: true });
  assert.equal(states.verifier.state, 'verified');
  assert.notEqual(originalCompletionTransferId, null, 'The initial proof completion ACK is lost');
  assert.ok(pair.links.prover.snapshot().adaptation.downshifts >= 4, 'The already delivered 128-byte transfer restarts at 64 bytes');
  const receiptRestart = pair.displayed.find(frame => frame.role === 'prover' && frame.packet.kind === 'restart' &&
    frame.packet.previousTransferId === originalCompletionTransferId);
  assert.ok(receiptRestart, 'The restart explicitly identifies the already delivered transfer');
  assert.equal(receiptRestart.packet.chunkBytes, 64);
  assert.equal(pair.clients.prover.count('prove'), 1);
  assert.equal(pair.clients.verifier.count('verify'), 1);
  assert.equal(pair.received.filter(message => message.header.type === 'PROOF').length, 1);
  assert.equal(pair.received.length, 5);
  assert.deepEqual(pair.errors, []);
});

test('219-repetition compact proof crosses multipart transport and verifies at 119,249 bytes', async t => {
  const pair = new DevicePair({ repetitions: 219 }); t.after(() => pair.close());
  await pair.start();
  const states = await pair.finish();
  assert.equal(states.verifier.state, 'verified');
  assert.equal(states.verifier.proofBytes, 119249);
  const verification = pair.clients.verifier.calls.find(call => call.type === 'verify');
  assert.equal(verification.result.usedRepetitions, 219);
  assert.equal(verification.payload.challenge.required_repetitions, 219);
  assert.deepEqual(pair.errors, []);
});

test('confirmation is required on each device before the prover creates a proof', async t => {
  const pair = new DevicePair({ repetitions: 4 }); t.after(() => pair.close());
  await pair.start();
  await pair.until(states => states.verifier.canConfirm && states.prover.canConfirm);
  for (let scan = 0; scan < 20; scan++) await pair.step();
  assert.equal(pair.clients.prover.count('prove'), 0);
  assert.equal(pair.clients.verifier.count('verify'), 0);
  await pair.sessions.verifier.confirmPairing();
  await pair.until(states => states.prover.challengeReceived);
  assert.equal(pair.clients.prover.count('prove'), 0);
  assert.equal(pair.snapshot().prover.confirmed, false);
  await pair.sessions.prover.confirmPairing();
  assert.equal((await pair.finish()).verifier.state, 'verified');
});

test('a changed proof with valid transport hashes and CRCs is rejected cryptographically', async t => {
  const pair = new DevicePair({
    repetitions: 32,
    transform: (role, envelope) => {
      if (role !== 'prover' || envelope.header.type !== 'PROOF') return;
      const changed = envelope.body.slice(); changed[changed.length - 1] ^= 1;
      return encodeEnvelope(envelope.header, changed);
    },
  }); t.after(() => pair.close());
  await pair.start();
  // The proof sender remembers its original digest, so it correctly refuses a
  // result naming the altered proof. The verifier's local rejection is final.
  const states = await pair.until(value => value.verifier.resultDelivered, { confirm: true });
  assert.equal(states.verifier.state, 'rejected');
  assert.equal(states.verifier.result, false);
  assert.equal(states.prover.state, 'sending-proof');
  assert.equal(pair.clients.verifier.count('verify'), 1);
  assert.equal(pair.received.filter(message => message.header.type === 'PROOF').length, 1);
  assert.deepEqual(pair.errors, []);
});

test('proof for a different nonce fails against the exact outstanding challenge', async t => {
  const pair = new DevicePair({
    repetitions: 32,
    cryptoHooks: { request: (role, type, payload) => {
      if (role === 'prover' && type === 'prove') {
        const nonce = payload.challenge.nonce;
        payload.challenge.nonce = (nonce[0] === '0' ? '1' : '0') + nonce.slice(1);
      }
      return payload;
    } },
  }); t.after(() => pair.close());
  await pair.start();
  const states = await pair.finish();
  assert.equal(states.verifier.state, 'rejected');
  assert.equal(states.prover.state, 'peer-rejected');
  assert.equal(pair.clients.verifier.count('verify'), 1);
});

test('lost completion ACKs recover across all five application message handoffs', async t => {
  const blockedUntil = new Map();
  const pair = new DevicePair({
    repetitions: 8,
    drop: ({ packet, clock }) => {
      if (packet.kind !== 'ack' || packet.status !== 'complete') return false;
      if (!blockedUntil.has(packet.transferId)) blockedUntil.set(packet.transferId, clock + 3000);
      return clock < blockedUntil.get(packet.transferId);
    },
  }); t.after(() => pair.close());
  await pair.start();
  const states = await pair.finish();
  assert.equal(states.verifier.state, 'verified');
  assert.equal(blockedUntil.size, 5);
  assert.equal(pair.clients.prover.count('prove'), 1);
  assert.equal(pair.clients.verifier.count('verify'), 1);
  assert.equal(pair.received.length, 5, 'Retries do not redeliver application messages');
  assert.deepEqual(pair.errors, []);
});

test('a completed proof cannot be replayed to trigger another verification', async t => {
  const pair = new DevicePair({ repetitions: 4 }); t.after(() => pair.close());
  await pair.start(); await pair.finish();
  const proof = pair.sent.find(message => message.header.type === 'PROOF');
  const result = await pair.sessions.verifier.receive(proof.bytes);
  assert.equal(result.accepted, false);
  assert.equal(pair.clients.verifier.count('verify'), 1);
  assert.equal(pair.snapshot().verifier.state, 'verified');
});

for (const termination of ['expiry', 'close']) {
  test(`a valid proof finishing after ${termination} cannot authenticate`, async t => {
    let release;
    const held = new Promise(resolve => { release = resolve; });
    const pair = new DevicePair({
      repetitions: 4,
      cryptoHooks: { after: async (role, type) => {
        if (role === 'verifier' && type === 'verify') await held;
      } },
    });
    t.after(() => { release(); pair.close(); });
    await pair.start();
    await pair.until(() => pair.clients.verifier.calls.some(call => call.type === 'verify' && call.result), { confirm: true });
    assert.equal(pair.snapshot().verifier.state, 'verifying');
    assert.equal(pair.clients.verifier.calls.find(call => call.type === 'verify').result.accepted, true);
    if (termination === 'expiry') {
      pair.clock = pair.snapshot().verifier.deadline;
      pair.sessions.verifier.tick();
    } else pair.sessions.verifier.close('Cancelled during verification');
    release();
    for (let scan = 0; scan < 20; scan++) await pair.step({ advanceClock: false });
    const state = pair.snapshot().verifier;
    assert.equal(state.state, termination === 'expiry' ? 'timed-out' : 'closed');
    assert.equal(state.result, null);
    assert.equal(pair.sent.filter(message => message.header.type === 'RESULT').length, 0);
  });
}
