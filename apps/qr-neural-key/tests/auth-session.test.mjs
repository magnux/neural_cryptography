import test from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import { AuthSession } from '../dist/js/auth/session.js';
import { AUTH_PROTOCOL, MAX_AUTH_BYTES, MAX_DISPLAY_NAME_CHARACTERS, MAX_DISPLAY_NAME_BYTES, decodeEnvelope, encodeEnvelope, normalizeDisplayName } from '../dist/js/auth/envelope.js';
import { CryptoOperations } from '../dist/js/crypto-worker/operations.js';
import { canonicalJSON } from '../dist/js/crypto/bytes.js';

globalThis.crypto ??= webcrypto;
const DEVICES = ['00000000-0000-4000-8000-000000000001', '00000000-0000-4000-8000-000000000002'];
const settle = () => new Promise(resolve => setImmediate(resolve));
function client(overrides = {}) {
  const ops = new CryptoOperations(), calls = [];
  return { calls, cancel: () => ops.clear(), request: async (type, payload, { onProgress } = {}) => {
    calls.push({ type, payload: structuredClone(payload) });
    return overrides[type] ? overrides[type](payload, onProgress, ops) : ops.run(type, payload, onProgress);
  } };
}
async function pair({ verifierRepetitions = 2, proverRepetitions = 2, verifierClient, proverClient, timeoutMs = 600000,
  displayNames = [], editHello = value => value, editIdentity = value => value, matchingNames = true } = {}) {
  const queues = [[], []], clients = [verifierClient ?? client(), proverClient ?? client()];
  const state = { time: 0 }, sessions = ['verifier', 'prover'].map((role, index) => new AuthSession({
    role, deviceId: DEVICES[index], cryptoClient: clients[index], now: () => state.time, timeoutMs,
    displayName: displayNames[index],
    repetitions: index ? proverRepetitions : verifierRepetitions,
    send: bytes => { queues[1 - index].push(bytes.slice()); return Promise.resolve(); },
  }));
  await Promise.all(sessions.map(session => session.start())); await settle();
  async function deliver(index, edit = value => value) {
    await settle(); assert.ok(queues[index].length, `message exists for side ${index}`);
    const bytes = edit(queues[index].shift()); const result = await sessions[index].receive(bytes); await settle();
    return { bytes, result };
  }
  await deliver(1, editHello); await deliver(0, editIdentity);
  if (matchingNames) assert.equal(sessions[0].snapshot().pairingCode, sessions[1].snapshot().pairingCode);
  return { sessions, clients, queues, state, deliver };
}
async function pendingProof(options) {
  const harness = await pair(options);
  await harness.sessions[0].confirmPairing(); await harness.sessions[1].confirmPairing();
  await harness.deliver(1);
  assert.equal(harness.sessions[0].snapshot().state, 'awaiting-proof');
  assert.equal(harness.sessions[1].snapshot().state, 'sending-proof');
  return harness;
}
function editHeader(bytes, transform) { const { header, body } = decodeEnvelope(bytes); return encodeEnvelope(transform(header), body); }

test('one local proof authenticates the pinned peer; result on prover remains a peer report', async () => {
  const { sessions, deliver, clients } = await pendingProof({ displayNames: ['  Mari\u0301a  desktop ', '電話 📱'] });
  assert.equal(sessions[0].snapshot().localDisplayName, 'María desktop');
  assert.equal(sessions[0].snapshot().peerDisplayName, '電話 📱');
  assert.equal(sessions[1].snapshot().localDisplayName, '電話 📱');
  assert.equal(sessions[1].snapshot().peerDisplayName, 'María desktop');
  const { bytes, result } = await deliver(0);
  assert.deepEqual(result, { accepted: true, verified: true });
  assert.equal(sessions[0].snapshot().state, 'verified');
  assert.equal(sessions[0].snapshot().peerReported, false);
  assert.equal(clients[0].calls.filter(call => call.type === 'verify').length, 1);
  assert.equal((await sessions[0].receive(bytes)).accepted, false);
  await deliver(1);
  assert.equal(sessions[1].snapshot().state, 'peer-accepted');
  assert.equal(sessions[1].snapshot().peerReported, true);
  assert.equal(sessions[0].snapshot().peerDisplayName, '電話 📱');
  assert.equal(sessions[1].snapshot().peerDisplayName, 'María desktop');
  assert.equal(sessions[0].snapshot().resultDelivered, true);
  assert.ok(Number.isFinite(sessions[0].snapshot().timings.keygenMs));
  assert.ok(Number.isFinite(sessions[0].snapshot().timings.verifyMs));
  assert.ok(Number.isFinite(sessions[1].snapshot().timings.proveMs));
});

test('display names normalize Unicode and spaces with bounded, printable values', () => {
  assert.equal(normalizeDisplayName('  Mari\u0301a\u00a0  👩🏽‍💻  '), 'María 👩🏽‍💻');
  assert.equal(normalizeDisplayName('📱'.repeat(MAX_DISPLAY_NAME_CHARACTERS)), '📱'.repeat(64));
  assert.equal(new TextEncoder().encode(normalizeDisplayName('📱'.repeat(64))).length, MAX_DISPLAY_NAME_BYTES);
  // Markup is a label, never HTML; UI consumers must insert it with textContent.
  assert.equal(normalizeDisplayName('<b>My phone</b>'), '<b>My phone</b>');
  for (const value of [null, 1, '', '   ', 'A'.repeat(65), '📱'.repeat(65), 'x'.repeat(257),
    'line\nbreak', 'tab\tname', 'null\0name', '\u202ephone', '\u2066phone\u2069', 'phone\u200b', '\ufeffphone', '\ud800']) {
    assert.throws(() => normalizeDisplayName(value), /display name/);
  }
});

test('existing callers get stable device names and names are pinned for an attempt', async () => {
  const { sessions } = await pair();
  assert.equal(sessions[0].snapshot().localDisplayName, 'Device 00000001');
  assert.equal(sessions[0].snapshot().peerDisplayName, 'Device 00000002');
  assert.equal(sessions[1].snapshot().peerDisplayName, 'Device 00000001');
  const before = sessions[0].snapshot();
  const changed = sessions[0].snapshot();
  changed.localDisplayName = 'Another desktop'; changed.peerDisplayName = 'Another phone';
  assert.equal(sessions[0].snapshot().localDisplayName, before.localDisplayName);
  assert.equal(sessions[0].snapshot().peerDisplayName, before.peerDisplayName);
  assert.throws(() => new AuthSession({ role: 'verifier', deviceId: DEVICES[0], displayName: '',
    cryptoClient: client(), send: () => {} }), /display name/);
});

test('relabeling either exchanged name changes pairing and cannot trigger a proof', async () => {
  for (const message of ['editHello', 'editIdentity']) {
    const { sessions, clients, deliver } = await pair({ displayNames: ['Desktop', 'Phone'], matchingNames: false,
      [message]: bytes => editHeader(bytes, header => ({ ...header, displayName: 'Relabeled device' })) });
    assert.notEqual(sessions[0].snapshot().pairingCode, sessions[1].snapshot().pairingCode);
    assert.notEqual(sessions[0].snapshot().binding, sessions[1].snapshot().binding);
    await sessions[0].confirmPairing(); await sessions[1].confirmPairing();
    const { result } = await deliver(1);
    assert.equal(result.accepted, false);
    assert.match(result.reason, /Pairing binding mismatch/);
    assert.equal(clients[1].calls.some(call => call.type === 'prove'), false);
    assert.equal(sessions[0].snapshot().result, null);
  }
});

test('rewriting transport bindings cannot authenticate a proof under a relabeled identity', async () => {
  const { sessions, deliver } = await pair({ verifierRepetitions: 32, proverRepetitions: 32,
    displayNames: ['Desktop', 'Phone'], matchingNames: false,
    editIdentity: bytes => editHeader(bytes, header => ({ ...header, displayName: 'Relabeled phone' })) });
  await sessions[0].confirmPairing(); await sessions[1].confirmPairing();
  const verifierBinding = sessions[0].snapshot().binding, proverBinding = sessions[1].snapshot().binding;
  // An active relay can rewrite unauthenticated envelope fields and challenges.
  // The resulting proof must still fail against the verifier's saved challenge.
  const challenge = await deliver(1, bytes => editHeader(bytes, header => ({ ...header, binding: proverBinding,
    challenge: { ...header.challenge, lock_id: `neural-zkboo:${proverBinding}` } })));
  assert.equal(challenge.result.accepted, true);
  const proof = await deliver(0, bytes => editHeader(bytes, header => ({ ...header, binding: verifierBinding })));
  assert.deepEqual(proof.result, { accepted: true, verified: false });
  assert.equal(sessions[0].snapshot().state, 'rejected');
  assert.equal(sessions[0].snapshot().peerDisplayName, 'Relabeled phone');
});

test('identity envelopes require canonical names and reject the previous schema', async () => {
  const outgoing = [], session = new AuthSession({ role: 'verifier', deviceId: DEVICES[0], displayName: 'María',
    repetitions: 2, cryptoClient: client(), send: bytes => { outgoing.push(bytes); } });
  await session.start(); await settle();
  const { header } = decodeEnvelope(outgoing[0]);
  assert.equal(header.displayName, 'María');
  for (const type of ['HELLO', 'IDENTITY']) {
    for (const displayName of [undefined, '', '  Phone ', 'Mari\u0301a', 'A'.repeat(65), '\u202ePhone', '\ud800']) {
      const malformed = { ...header, type, displayName };
      if (displayName === undefined) delete malformed.displayName;
      assert.throws(() => encodeEnvelope(malformed));
      const encoded = canonicalJSON(malformed), bytes = new Uint8Array(8 + encoded.length);
      bytes.set(outgoing[0].subarray(0, 4)); new DataView(bytes.buffer).setUint32(4, encoded.length);
      bytes.set(encoded, 8);
      assert.throws(() => decodeEnvelope(bytes));
    }
  }
  assert.throws(() => encodeEnvelope({ ...header, protocol: 'neural-zkboo-auth-v1' }));
});

test('both pairing confirmations are required and challenge can wait for prover confirmation', async () => {
  const { sessions, deliver, clients, queues } = await pair();
  assert.equal(queues[1].length, 0);
  await sessions[0].confirmPairing();
  await deliver(1);
  assert.equal(sessions[1].snapshot().state, 'pairing');
  assert.equal(sessions[1].snapshot().challengeReceived, true);
  assert.equal(clients[1].calls.some(call => call.type === 'prove'), false);
  await sessions[1].confirmPairing(); await settle();
  assert.equal(clients[1].calls.filter(call => call.type === 'prove').length, 1);
  assert.equal(await sessions[1].confirmPairing(), false);
  await deliver(0);
  assert.equal(sessions[0].snapshot().result, true);
});

test('requested repetitions use the larger local policy and paired key floor', async () => {
  const { sessions, deliver, clients } = await pendingProof({ verifierRepetitions: 2, proverRepetitions: 5 });
  assert.equal(sessions[0].snapshot().repetitions, 5);
  assert.equal(sessions[1].snapshot().repetitions, 5);
  await deliver(0);
  const { payload } = clients[0].calls.find(call => call.type === 'verify');
  assert.equal(payload.challenge.required_repetitions, 5);
  assert.equal(payload.publicKey.minimum_repetitions, 5);
  assert.equal(sessions[0].snapshot().result, true);
});

test('wrong session, pairing binding, or nonce cannot consume the outstanding challenge', async () => {
  const { sessions, queues, clients, deliver } = await pendingProof();
  const proof = queues[0][0];
  for (const field of ['sessionId', 'binding', 'nonce']) {
    const mutated = editHeader(proof, header => ({ ...header, [field]: '0'.repeat(header[field].length) }));
    assert.equal((await sessions[0].receive(mutated)).accepted, false);
    assert.equal(sessions[0].snapshot().state, 'awaiting-proof');
  }
  assert.equal(clients[0].calls.some(call => call.type === 'verify'), false);
  await deliver(0);
  assert.equal(sessions[0].snapshot().result, true);
});

test('challenge policy/action/binding changes do not trigger proof generation', async () => {
  const { sessions, queues, clients, deliver } = await pair();
  await sessions[0].confirmPairing(); await sessions[1].confirmPairing(); await settle();
  const challenge = queues[1][0];
  for (const changed of [{ required_repetitions: 1 }, { required_repetitions: 3 }, { action: 'unlock' }, { lock_id: 'neural-zkboo:wrong' }]) {
    const mutation = editHeader(challenge, header => ({ ...header, challenge: { ...header.challenge, ...changed } }));
    assert.equal((await sessions[1].receive(mutation)).accepted, false);
  }
  assert.equal(clients[1].calls.some(call => call.type === 'prove'), false);
  await deliver(1);
  assert.equal(clients[1].calls.filter(call => call.type === 'prove').length, 1);
  assert.equal((await sessions[1].receive(challenge)).accepted, false);
});

test('changed proof is rejected locally, and no second proof can reuse that challenge', async () => {
  const { sessions, queues, deliver, clients } = await pendingProof();
  const original = queues[0][0].slice();
  await deliver(0, bytes => { const { header, body } = decodeEnvelope(bytes); body[0] ^= 1; return encodeEnvelope(header, body); });
  assert.equal(sessions[0].snapshot().state, 'rejected');
  assert.equal((await sessions[0].receive(original)).accepted, false);
  assert.equal(clients[0].calls.filter(call => call.type === 'verify').length, 1);
  // The altered proof digest does not match what the prover sent.
  assert.equal((await deliver(1)).result.accepted, false);
  assert.equal(sessions[1].snapshot().result, null);
});

test('proof for another public key is rejected against the already paired key', async () => {
  const { sessions, queues, deliver } = await pendingProof({ verifierRepetitions: 32, proverRepetitions: 32 });
  const outsider = client();
  await outsider.request('createKey', { profile: { side: 8, rounds: 16, repetitions: 32 } });
  const { header } = decodeEnvelope(queues[0][0]);
  const challenge = { protocol: 'NeuralPicnic-POC-MPC3-v2', nonce: header.nonce, lock_id: `neural-zkboo:${header.binding}`, action: 'authenticate', required_repetitions: 32 };
  const result = await outsider.request('prove', { challenge });
  await deliver(0, () => encodeEnvelope(header, result.proof));
  assert.equal(sessions[0].snapshot().result, false);
});

test('deadline is checked before accepting incoming data and after asynchronous verification', async () => {
  const first = await pendingProof({ timeoutMs: 1000 });
  first.state.time = 1000;
  await first.deliver(0);
  assert.equal(first.sessions[0].snapshot().state, 'timed-out');
  assert.equal(first.clients[0].calls.some(call => call.type === 'verify'), false);

  let finish;
  const delayed = client({ verify: () => new Promise(resolve => { finish = resolve; }) });
  const second = await pendingProof({ timeoutMs: 1000, verifierClient: delayed });
  const pending = second.sessions[0].receive(second.queues[0].shift());
  while (!finish) await settle();
  second.state.time = 1000; finish({ accepted: true });
  await pending;
  assert.equal(second.sessions[0].snapshot().state, 'timed-out');
  assert.equal(second.sessions[0].snapshot().result, null);
  assert.equal(second.queues[1].length, 0);
});

test('close invalidates a pending verification and concurrent proof cannot run twice', async () => {
  let finish;
  const delayed = client({ verify: () => new Promise(resolve => { finish = resolve; }) });
  const harness = await pendingProof({ verifierClient: delayed });
  const bytes = harness.queues[0].shift(), pending = harness.sessions[0].receive(bytes);
  while (!finish) await settle();
  assert.equal((await harness.sessions[0].receive(bytes)).accepted, false);
  harness.sessions[0].close('cancelled in test'); finish({ accepted: true });
  await pending;
  assert.equal(harness.sessions[0].snapshot().state, 'closed');
  assert.equal(harness.sessions[0].snapshot().result, null);
  assert.equal(delayed.calls.filter(call => call.type === 'verify').length, 1);
});

test('a fresh attempt changes session, pairing code and challenge nonce', async () => {
  const first = await pendingProof(), second = await pendingProof();
  for (const field of ['sessionId', 'pairingCode']) assert.notEqual(first.sessions[0].snapshot()[field], second.sessions[0].snapshot()[field]);
  assert.notEqual(decodeEnvelope(first.queues[0][0]).header.nonce, decodeEnvelope(second.queues[0][0]).header.nonce);
  assert.equal((await second.sessions[0].receive(first.queues[0][0])).accepted, false);
});

test('a HELLO received during key generation is retained once and processed after key readiness', async () => {
  let release;
  const gate = new Promise(resolve => { release = resolve; }), waiting = client({ createKey: async (payload, progress, ops) => {
    await gate; return ops.run('createKey', payload, progress);
  } });
  const outgoing = [], verifier = new AuthSession({ role: 'verifier', deviceId: DEVICES[0], repetitions: 2,
    cryptoClient: client(), send: bytes => { outgoing.push(bytes); } });
  const replies = [], prover = new AuthSession({ role: 'prover', deviceId: DEVICES[1], repetitions: 2,
    cryptoClient: waiting, send: bytes => { replies.push(bytes); } });
  const started = prover.start(); await verifier.start(); await settle();
  assert.equal((await prover.receive(outgoing[0])).deferred, true);
  assert.equal((await prover.receive(outgoing[0])).accepted, false);
  release(); await started; await settle();
  assert.equal(prover.snapshot().state, 'pairing');
  assert.equal(replies.length, 1);
  await verifier.receive(replies[0]);
  assert.equal(verifier.snapshot().pairingCode, prover.snapshot().pairingCode);
});

test('an identity cannot be replaced after pairing, and snapshot mutation does not alter pinned state', async () => {
  const { sessions } = await pair();
  const outsider = client(), created = await outsider.request('createKey', { profile: { side: 8, rounds: 16, repetitions: 2 } });
  const before = sessions[0].snapshot();
  const forged = encodeEnvelope({ protocol: AUTH_PROTOCOL, type: 'IDENTITY', sessionId: before.sessionId,
    deviceId: DEVICES[1], displayName: 'Forged replacement', publicKey: created.publicKey });
  assert.equal((await sessions[0].receive(forged)).accepted, false);
  const snapshot = sessions[0].snapshot(); snapshot.binding = '00'.repeat(32); snapshot.localKeyId = '00'.repeat(32);
  assert.equal(sessions[0].snapshot().peerKeyId, before.peerKeyId);
  assert.equal(sessions[0].snapshot().binding, before.binding);
  assert.equal(sessions[0].snapshot().localKeyId, before.localKeyId);
  assert.equal(sessions[0].snapshot().peerDisplayName, before.peerDisplayName);
});

test('outbound transport failure fails the active attempt without accepting a late operation', async () => {
  const session = new AuthSession({ role: 'verifier', deviceId: DEVICES[0], repetitions: 2, cryptoClient: client(),
    send: () => Promise.reject(new Error('camera transfer expired')) });
  await session.start(); await settle();
  assert.equal(session.snapshot().state, 'failed');
  assert.match(session.snapshot().error, /camera transfer expired/);
  assert.equal(session.snapshot().result, null);
});

test('envelope carries raw proof bytes and rejects malformed, noncanonical and oversized inputs', async () => {
  const harness = await pendingProof(), bytes = harness.queues[0][0];
  const { header, body } = decodeEnvelope(bytes);
  assert.deepEqual(decodeEnvelope(encodeEnvelope(header, body)), { header, body });
  assert.equal(bytes.length, 8 + canonicalJSON(header).length + body.length);
  assert.throws(() => encodeEnvelope({ ...header, extra: true }, body));
  assert.throws(() => encodeEnvelope(header));
  assert.throws(() => encodeEnvelope(header, new Uint8Array(MAX_AUTH_BYTES)));
  assert.throws(() => decodeEnvelope(new Uint8Array(MAX_AUTH_BYTES + 1)));
  assert.throws(() => decodeEnvelope(bytes.slice(0, 7)));
  const badLength = bytes.slice(); new DataView(badLength.buffer).setUint32(4, 4097); assert.throws(() => decodeEnvelope(badLength));
  const oldMagic = bytes.slice(); oldMagic[3] = 49; assert.throws(() => decodeEnvelope(oldMagic));
  const text = new TextDecoder().decode(canonicalJSON(header)), noncanonical = new TextEncoder().encode(` ${text}`);
  const spaced = new Uint8Array(8 + noncanonical.length + body.length); spaced.set(bytes.subarray(0, 4));
  new DataView(spaced.buffer).setUint32(4, noncanonical.length); spaced.set(noncanonical, 8); spaced.set(body, 8 + noncanonical.length);
  assert.throws(() => decodeEnvelope(spaced), /Noncanonical/);
  const duplicate = new TextEncoder().encode(text.replace('{', '{"protocol":"ignored",'));
  const duplicated = new Uint8Array(8 + duplicate.length + body.length); duplicated.set(bytes.subarray(0, 4));
  new DataView(duplicated.buffer).setUint32(4, duplicate.length); duplicated.set(duplicate, 8); duplicated.set(body, 8 + duplicate.length);
  assert.throws(() => decodeEnvelope(duplicated), /Noncanonical/);
  assert.equal(header.protocol, AUTH_PROTOCOL);
  assert.equal(AUTH_PROTOCOL, 'neural-zkboo-auth-v2');
  assert.equal(new TextDecoder().decode(bytes.subarray(0, 4)), 'NZA2');
});
