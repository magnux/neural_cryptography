import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker as NodeWorker } from 'node:worker_threads';
import { readFileSync } from 'node:fs';
import { CryptoClient } from '../dist/js/crypto-lab/client.js';
import { referenceFixture } from '../dist/js/crypto-lab/reference-fixture.js';

// Translate only the browser Worker event API; the actual crypto module runs in
// a separate Node worker with native Web Crypto, not a mocked proof engine.
class BrowserWorkerBridge {
  static instances = [];
  onmessage = null; onerror = null; onmessageerror = null;
  constructor(url, options) {
    assert.equal(url.pathname.endsWith('/crypto-lab/worker.js'), true);
    assert.equal(options.type, 'module');
    this.worker = new NodeWorker(new URL('./crypto-worker-harness.mjs', import.meta.url));
    this.worker.on('message', data => this.onmessage?.({ data }));
    this.worker.on('error', error => this.onerror?.({ message: error.message, preventDefault() {} }));
    this.worker.on('messageerror', error => this.onmessageerror?.(error));
    BrowserWorkerBridge.instances.push(this);
  }
  postMessage(data, transfer) {
    this.lastId = data.id;
    this.lastProofBackingBytes = data.payload.proof?.buffer.byteLength;
    this.worker.postMessage(data, transfer);
  }
  terminate() { this.worker.terminate(); }
}
globalThis.Worker = BrowserWorkerBridge;
const small = { side: 4, rounds: 2, repetitions: 3 };
const labels = { lockId: 'lab-puerta-ñ-🚪', action: 'abrir\n€', requestedRepetitions: 3 };
function closeAfter(t) { const client = new CryptoClient(); t.after(() => client.destroy()); return client; }
function envelope(proof) { return { publicKey: proof.publicKey, challenge: proof.challenge, proof: proof.proof }; }
const manifest = JSON.parse(readFileSync(new URL('./fixtures/zk/manifest.json', import.meta.url), 'utf8'));

test('browser reference fixture contains only public material and matches the Python source', () => {
  const known = manifest.cases.find(item => item.id.includes('unicode')), actual = referenceFixture();
  assert.deepEqual(Object.keys(actual).sort(), ['challenge', 'proof', 'proofSha256', 'publicKey']);
  assert.deepEqual(actual.publicKey, known.public);
  assert.deepEqual(actual.challenge, known.challenge);
  assert.equal(actual.proofSha256, known.proofSha256);
  assert.deepEqual(actual.proof, new Uint8Array(readFileSync(new URL(`./fixtures/zk/${known.proofFile}`, import.meta.url))));
  actual.publicKey.anchor_outputs = 'changed'; actual.proof[0] ^= 1;
  assert.deepEqual(referenceFixture().publicKey, known.public);
  const source = readFileSync(new URL('../dist/js/crypto-lab/reference-fixture.js', import.meta.url), 'utf8');
  assert.equal(source.includes(known.secretHex), false);
  assert.equal(source.includes(known.entropySeedHex), false);
});

test('worker creates fresh ephemeral keys and challenges, proves, verifies and reports progress', async t => {
  const client = closeAfter(t), events = [];
  assert.equal(client.alive, false); assert.equal(client.busy, false);
  const key = await client.request('createKey', { profile: small }, { onProgress: value => events.push(value) });
  assert.equal(client.alive, true); assert.equal(client.busy, false);
  assert.deepEqual(Object.keys(key).sort(), ['keyId', 'keygenMs', 'profile', 'proofBytes', 'publicKey']);
  assert.match(key.keyId, /^[0-9a-f]{64}$/); assert.ok(key.keygenMs >= 0);
  const { challenge } = await client.request('newChallenge', labels);
  const second = await client.request('newChallenge', labels);
  assert.notEqual(challenge.nonce, second.challenge.nonce);
  assert.equal(challenge.lock_id, labels.lockId);
  const result = await client.request('prove', { challenge }, { onProgress: value => events.push(value) });
  assert.equal(result.proof.length, key.proofBytes); assert.equal(result.keyId, key.keyId);
  assert.deepEqual(Object.keys(result).sort(), ['challenge', 'keyId', 'keyMinimumRepetitions', 'proof', 'proofBytes', 'proofSha256', 'proveMs', 'publicKey', 'requestedRepetitions', 'usedRepetitions']);
  const original = result.proof.slice();
  const checked = await client.request('verify', envelope(result));
  assert.equal(checked.accepted, true); assert.equal(checked.proofSha256, result.proofSha256);
  assert.deepEqual(result.proof, original, 'verification must preserve the main-thread proof');
  assert.ok(events.some(value => value.phase === 'simulate' && value.completed === small.repetitions));
  for (const value of events) assert.ok(value.completed >= 0 && value.completed <= value.total);
  const replacement = await client.request('createKey', { profile: small });
  assert.notEqual(replacement.keyId, key.keyId);
  assert.notEqual(replacement.publicKey.anchor_outputs, key.publicKey.anchor_outputs);
});

test('fresh worker verifies public-only Python fixture, rejects tampering, and needs no key', async t => {
  const client = closeAfter(t), fixture = referenceFixture();
  const checks = await client.request('selfTest');
  assert.equal(checks.passed, true); assert.equal(checks.checks.length, 2);
  const input = envelope(fixture), originalProof = fixture.proof.slice(), originalChallenge = structuredClone(fixture.challenge);
  const checked = await client.request('verify', input);
  assert.equal(checked.accepted, true);
  const tampered = await client.request('tamper', input);
  assert.equal(tampered.passed, true); assert.equal(tampered.checks.length, 10);
  assert.equal(tampered.checks[0].accepted, true);
  for (const check of tampered.checks.slice(1)) assert.equal(check.accepted, false, check.name);
  assert.deepEqual(fixture.proof, originalProof); assert.deepEqual(fixture.challenge, originalChallenge);
  await assert.rejects(client.request('prove', { challenge: fixture.challenge }), /no key/);
});

test('benchmark uses fresh challenges and produces bounded timings with a verifiable latest proof', async t => {
  const client = closeAfter(t);
  await client.request('createKey', { profile: small });
  const report = await client.request('benchmark', { ...labels, runs: 2 });
  assert.equal(report.runs.length, 2); assert.ok(report.runs.every(run => run.accepted));
  assert.ok(report.summary.proveMedianMs >= 0); assert.ok(report.summary.verifyMedianMs >= 0);
  assert.equal((await client.request('verify', envelope(report.latest))).accepted, true);
  const other = await client.request('benchmark', { ...labels, runs: 1 });
  assert.notEqual(other.latest.challenge.nonce, report.latest.challenge.nonce);
  assert.notEqual(other.latest.proofSha256, report.latest.proofSha256);
  await assert.rejects(client.request('benchmark', { ...labels, runs: 11 }), /1..10/);
});

test('default 219-repetition roundtrip stays off the parent event loop', async t => {
  const client = closeAfter(t);
  const key = await client.request('createKey', { profile: { side: 8, rounds: 16, repetitions: 219 } });
  assert.equal(key.proofBytes, 119249);
  const { challenge } = await client.request('newChallenge', labels);
  let heartbeats = 0;
  const heartbeat = setInterval(() => heartbeats++, 2);
  try {
    const proof = await client.request('prove', { challenge });
    assert.equal(proof.proofBytes, 119249);
    assert.equal((await client.request('verify', envelope(proof))).accepted, true);
    assert.ok(heartbeats > 10, `parent event loop continued (${heartbeats} heartbeats)`);
  } finally { clearInterval(heartbeat); }
});

test('cancellation destroys the key; stale worker and stale job events cannot complete a later request', async t => {
  const client = closeAfter(t), { challenge } = await client.request('newChallenge', labels);
  await client.request('createKey', { profile: { side: 8, rounds: 16, repetitions: 219 } });
  let started;
  const working = new Promise(resolve => { started = resolve; });
  const pending = client.request('prove', { challenge }, { onProgress: progress => {
    if (progress.phase === 'simulate' && progress.completed >= 1) started();
  } });
  const oldWorker = BrowserWorkerBridge.instances.at(-1), staleHandler = oldWorker.onmessage, staleId = oldWorker.lastId;
  const aborted = assert.rejects(pending, { name: 'AbortError' });
  await assert.rejects(client.request('selfTest'), { name: 'InvalidStateError' });
  await working;
  client.cancel(); await aborted;
  assert.equal(client.alive, false); assert.equal(client.busy, false);
  const fresh = client.request('selfTest');
  staleHandler({ data: { id: staleId, kind: 'result', result: { passed: 'stale' } } });
  const replacement = BrowserWorkerBridge.instances.at(-1);
  replacement.onmessage({ data: { id: staleId, kind: 'result', result: { passed: 'wrong job' } } });
  assert.equal((await fresh).passed, true);
  await assert.rejects(client.request('prove', { challenge }), /no key/);
  await client.request('createKey', { profile: small });
  assert.equal((await client.request('prove', { challenge })).proofBytes > 0, true);
  client.destroy(); assert.equal(client.alive, false);
});

test('worker constructor failures reject immediately and later requests can recover', async t => {
  const client = closeAfter(t);
  globalThis.Worker = class { constructor() { throw new Error('Worker startup blocked'); } };
  try {
    await assert.rejects(client.request('selfTest'), /startup blocked/);
    assert.equal(client.alive, false); assert.equal(client.busy, false);
  } finally { globalThis.Worker = BrowserWorkerBridge; }
  assert.equal((await client.request('selfTest')).passed, true);
});

test('worker errors, message errors and timeouts reject work and discard keys', async t => {
  const client = closeAfter(t), fixture = referenceFixture();
  for (const failure of ['error', 'messageerror']) {
    await client.request('createKey', { profile: small });
    const pending = client.request('selfTest'), worker = BrowserWorkerBridge.instances.at(-1);
    const rejected = assert.rejects(pending, { name: 'WorkerError' });
    if (failure === 'error') worker.onerror({ preventDefault() {} }); else worker.onmessageerror({});
    await rejected; assert.equal(client.alive, false); assert.equal(client.busy, false);
    await assert.rejects(client.request('prove', { challenge: fixture.challenge }), /no key/);
    client.destroy();
  }
  await assert.rejects(client.request('selfTest', {}, { timeoutMs: 1 }), { name: 'TimeoutError' });
  assert.equal(client.alive, false);
  assert.equal((await client.request('selfTest')).passed, true);
});

test('strict request bounds reject malformed payloads without wedging the client', async t => {
  const client = closeAfter(t);
  await assert.rejects(client.request('unknown'), /Unsupported/);
  await assert.rejects(client.request('selfTest', {}, { timeoutMs: 120001 }), /timeoutMs/);
  await assert.rejects(client.request('benchmark', {}, { timeoutMs: 300001 }), /timeoutMs/);
  await assert.rejects(client.request('selfTest', { proof: new Uint8Array(8 * 1024 * 1024 + 1) }), /8 MiB/);
  assert.equal(client.alive, false);
  await assert.rejects(client.request('createKey', { profile: { ...small, rounds: 1 } }), /profile/);
  await assert.rejects(client.request('createKey', { profile: { ...small, secret: 'forbidden' } }), /fields/);
  await assert.rejects(client.request('newChallenge', { ...labels, action: '\ud800' }), /UTF-8/);
  await assert.rejects(client.request('newChallenge', { ...labels, action: 'a'.repeat(257) }), /UTF-8/);
  await assert.rejects(client.request('selfTest', { unexpected: 1 }), /fields/);
  assert.equal((await client.request('selfTest')).passed, true);
});

test('client copies a proof view without cloning its oversized backing buffer and caps aggregate bytes', async t => {
  const client = closeAfter(t), fixture = referenceFixture();
  const backing = new Uint8Array(9 * 1024 * 1024);
  const proof = backing.subarray(113, 113 + fixture.proof.length);
  proof.set(fixture.proof);
  const result = await client.request('verify', { publicKey: fixture.publicKey, challenge: fixture.challenge, proof });
  assert.equal(result.accepted, true);
  assert.equal(BrowserWorkerBridge.instances.at(-1).lastProofBackingBytes, fixture.proof.length);
  assert.deepEqual(proof, fixture.proof, 'caller view remains intact');
  client.destroy();
  await assert.rejects(client.request('selfTest', {
    first: new Uint8Array(5 * 1024 * 1024), second: new Uint8Array(4 * 1024 * 1024),
  }), /8 MiB/);
  assert.equal(client.alive, false, 'reject aggregate size before creating a worker');
});

test('idle worker failure notifies the UI that the key is lost before the next request', async t => {
  let keyAvailable = false, notifications = 0;
  const client = new CryptoClient({ onInvalidate(error) {
    keyAvailable = false; notifications++;
    assert.ok(['WorkerError', 'AbortError'].includes(error.name));
  } });
  t.after(() => client.destroy());
  await client.request('createKey', { profile: small }); keyAvailable = true;
  BrowserWorkerBridge.instances.at(-1).onerror({ preventDefault() {} });
  assert.equal(client.alive, false); assert.equal(keyAvailable, false); assert.equal(notifications, 1);
  if (!keyAvailable) { await client.request('createKey', { profile: small }); keyAvailable = true; }
  const { challenge } = await client.request('newChallenge', labels);
  const proof = await client.request('prove', { challenge });
  assert.equal((await client.request('verify', envelope(proof))).accepted, true);
});


test('challenge repetitions vary above the key floor without generating another key', async t => {
  const client = closeAfter(t);
  const key = await client.request('createKey', { profile: small });
  for (const requestedRepetitions of [7, 4, 1]) {
    const { challenge } = await client.request('newChallenge', { ...labels, requestedRepetitions });
    assert.equal(challenge.required_repetitions, requestedRepetitions);
    const proof = await client.request('prove', { challenge });
    assert.equal(proof.keyId, key.keyId);
    assert.equal(proof.keyMinimumRepetitions, 3);
    assert.equal(proof.requestedRepetitions, requestedRepetitions);
    assert.equal(proof.usedRepetitions, Math.max(3, requestedRepetitions));
    const checked = await client.request('verify', envelope(proof));
    assert.equal(checked.accepted, true);
    assert.equal(checked.usedRepetitions, proof.usedRepetitions);
    assert.equal(checked.requestedRepetitions, requestedRepetitions);
    const changed = { ...proof.challenge, required_repetitions: requestedRepetitions + 1 };
    const rejected = await client.request('verify', { ...envelope(proof), challenge: changed });
    assert.equal(rejected.accepted, false);
    assert.equal(Object.hasOwn(rejected, 'usedRepetitions'), false);
  }
  const benchmark = await client.request('benchmark', { ...labels, requestedRepetitions: 5, runs: 2 });
  assert.equal(benchmark.requestedRepetitions, 5);
  assert.equal(benchmark.usedRepetitions, 5);
  assert.ok(benchmark.runs.every(run => run.requestedRepetitions === 5 && run.usedRepetitions === 5 && run.keyMinimumRepetitions === 3));
});

test('worker requests require a bounded integer challenge repetition count', async t => {
  const client = closeAfter(t);
  for (const requestedRepetitions of [0, 513, 3.5]) {
    await assert.rejects(client.request('newChallenge', { ...labels, requestedRepetitions }));
    await assert.rejects(client.request('benchmark', { ...labels, requestedRepetitions, runs: 1 }));
  }
  const { requestedRepetitions, ...missing } = labels;
  await assert.rejects(client.request('newChallenge', missing), /fields/);
  await assert.rejects(client.request('benchmark', { ...missing, runs: 1 }), /fields/);
});
