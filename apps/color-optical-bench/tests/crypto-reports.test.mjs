import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  CONTEXT_MAX_BYTES, encodeContext, decodeContext, LabReport,
  describeEnvironment, frameGapStatistics, startResponsivenessMonitor,
} from '../dist/js/crypto-lab/reports.js';

const fixtures = JSON.parse(readFileSync(new URL('./fixtures/zk/manifest.json', import.meta.url), 'utf8')).cases;
const context = (index = 0) => ({ publicKey: structuredClone(fixtures[index].public), challenge: structuredClone(fixtures[index].challenge) });

test('public context exchange preserves every fixture, including Unicode, without private material', () => {
  for (let index = 0; index < fixtures.length; index++) {
    const input = context(index);
    const encoded = encodeContext(input);
    const decoded = decodeContext(encoded);
    assert.ok(new TextEncoder().encode(encoded).length <= CONTEXT_MAX_BYTES);
    assert.deepEqual(decoded, input);
    assert.notEqual(decoded.publicKey, input.publicKey);
    assert.notEqual(decoded.publicKey.circuit, input.publicKey.circuit);
    assert.deepEqual(Object.keys(JSON.parse(encoded)).sort(), ['challenge', 'format', 'publicKey']);
    decoded.publicKey.circuit.side = 128;
    assert.notEqual(input.publicKey.circuit.side, 128);
  }
});

test('context exchange rejects old formats, extra fields, malformed keys and challenges', () => {
  const input = context();
  const serialized = JSON.parse(encodeContext(input));
  for (const mutation of [
    value => { value.format = 'old-format'; },
    value => { value.secret = 'not-allowed'; },
    value => { value.publicKey.anchor_outputs = 'ab'; },
    value => { value.publicKey.anchor_outputs = value.publicKey.anchor_outputs.toUpperCase(); },
    value => { value.publicKey.minimum_repetitions = true; },
    value => { value.publicKey.minimum_repetitions = 513; },
    value => { value.publicKey.circuit.side = 128; },
    value => { value.publicKey.circuit.rounds = 1; },
    value => { value.publicKey.circuit.key_bits = 128; },
    value => { value.publicKey.circuit.extra = true; },
    value => { value.challenge.protocol = 'different'; },
    value => { value.challenge.nonce = 'AB'.repeat(32); },
    value => { value.challenge.action = ''; },
    value => { value.challenge.action = 'ñ'.repeat(129); },
    value => { value.challenge.lock_id = '\ud800'; },
    value => { value.challenge.lock_id = '\udc00'; },
    value => { value.challenge.lock_id = '\ud800A'; },
    value => { value.challenge.extra = true; },
    value => { delete value.challenge.required_repetitions; },
    value => { value.challenge.required_repetitions = 0; },
    value => { value.challenge.required_repetitions = 513; },
    value => { value.challenge.required_repetitions = 1.5; },
    value => { value.challenge.required_repetitions = true; },
  ]) {
    const changed = structuredClone(serialized);
    mutation(changed);
    assert.throws(() => decodeContext(JSON.stringify(changed)));
  }
  assert.throws(() => encodeContext({ ...input, proof: new Uint8Array(1) }));
  assert.throws(() => decodeContext('{broken'));
  assert.throws(() => decodeContext('null'));
  assert.throws(() => decodeContext('[]'));
});

test('context parser limits encoded bytes before parsing and rejects accessor documents', () => {
  assert.throws(() => decodeContext(' '.repeat(CONTEXT_MAX_BYTES + 1)), /8 KiB/);
  assert.throws(() => decodeContext('€'.repeat(3000)), /8 KiB/);
  assert.throws(() => decodeContext(null));
  let called = false;
  assert.throws(() => encodeContext({
    get publicKey() { called = true; return context().publicKey; },
    challenge: context().challenge,
  }));
  assert.equal(called, false);
  const unicode = context();
  unicode.challenge.lock_id = '门🔑';
  unicode.challenge.action = 'é'.repeat(128);
  assert.deepEqual(decodeContext(encodeContext(unicode)), unicode);
});

test('report history keeps bounded measurement snapshots and excludes cryptographic material', () => {
  const privateData = 'DO-NOT-EXPORT-RAW-CRYPTOGRAPHIC-MATERIAL';
  const report = new LabReport({
    userAgent: 'test browser', platform: 'test', hardwareConcurrency: 4,
    screen: { width: 1280, height: 720, secret: privateData },
    privateData,
  });
  const raw = {
    keyId: 'ab'.repeat(32), proofSha256: 'cd'.repeat(32),
    profile: { side: 8, rounds: 16, repetitions: 219, secret: privateData },
    proofBytes: 119249, proveMs: 100, verifyMs: 25, passed: true,
    keyMinimumRepetitions: 32, requestedRepetitions: 160, usedRepetitions: 219,
    proof: privateData, secret: privateData, challenge: { nonce: privateData }, publicKey: privateData,
    checks: [{ name: 'changed nonce', expected: false, accepted: false, passed: true, ms: 2, proof: privateData }],
    runs: [{ proveMs: 100, verifyMs: 25, proofBytes: 119249, accepted: true, keyMinimumRepetitions: 32, requestedRepetitions: 160, usedRepetitions: 219, secret: privateData }],
    summary: { proveMedianMs: 100, verifyMedianMs: 25, nonce: privateData },
    latest: { proof: privateData, challenge: privateData },
  };
  const added = report.add('benchmark', raw, { durationMs: 130, visibleFrames: 8, hiddenDuringRun: false, proof: privateData });
  raw.profile.side = 16;
  raw.checks[0].passed = false;
  added.result.profile.side = 4;
  const exported = report.export('phone test');
  assert.equal(exported.format, 'NeuralPicnicLab-report-v2');
  assert.equal(exported.notes, 'phone test');
  assert.equal(exported.entries[0].result.profile.side, 8);
  assert.equal(exported.entries[0].result.checks[0].passed, true);
  assert.equal(exported.entries[0].result.proofBytes, 119249);
  assert.equal(exported.entries[0].result.summary.proveMedianMs, 100);
  for (const result of [exported.entries[0].result, exported.entries[0].result.runs[0]]) {
    assert.equal(result.keyMinimumRepetitions, 32);
    assert.equal(result.requestedRepetitions, 160);
    assert.equal(result.usedRepetitions, 219);
  }
  assert.ok(!JSON.stringify(exported).includes(privateData));
  assert.deepEqual(Object.keys(exported.environment).sort(), ['hardwareConcurrency', 'platform', 'screen', 'userAgent']);
  exported.entries[0].result.proofBytes = 0;
  assert.equal(report.export().entries[0].result.proofBytes, 119249);
  for (let i = 0; i < 60; i++) report.add(`run-${i}`, { proveMs: i });
  const bounded = report.export('x'.repeat(3000));
  assert.equal(bounded.entries.length, 50);
  assert.equal(bounded.entries[0].operation, 'run-10');
  assert.equal(bounded.entries.at(-1).operation, 'run-59');
  assert.equal(bounded.notes.length, 2048);
  report.clear();
  assert.deepEqual(report.export().entries, []);
});

test('report sanitation bounds nested results and ignores nonfinite numbers or getters', () => {
  const report = new LabReport();
  let called = false;
  const raw = {
    get proofBytes() { called = true; return 20; },
    proveMs: NaN, verifyMs: Infinity, totalMs: -1,
    checks: Array.from({ length: 100 }, () => ({ name: 'x'.repeat(200), accepted: false })),
    runs: Array.from({ length: 100 }, () => ({ proveMs: 3 })),
  };
  report.add('test', raw);
  assert.equal(called, false);
  const result = report.export().entries[0].result;
  assert.equal(result.checks.length, 32);
  assert.equal(result.checks[0].name.length, 96);
  assert.equal(result.runs.length, 20);
  assert.ok(!Object.hasOwn(result, 'proveMs'));
  assert.ok(!Object.hasOwn(result, 'proofBytes'));
  assert.throws(() => report.add('', {}));
  assert.throws(() => report.export({}));
});

test('reports retain failure and cancellation status without raw error details', () => {
  const report = new LabReport();
  const cases = [
    ['completed', undefined], ['failed', 'TypeError'],
    ['canceled', 'AbortError'], ['timeout', 'TimeoutError'],
  ];
  for (const [status, errorName] of cases) {
    report.add('prove', { status, errorName, message: 'DO-NOT-EXPORT', error: { stack: 'DO-NOT-EXPORT' } });
  }
  const entries = report.export().entries;
  for (let index = 0; index < cases.length; index++) {
    assert.equal(entries[index].result.status, cases[index][0]);
    assert.equal(entries[index].result.errorName, cases[index][1]);
  }
  assert.ok(!JSON.stringify(entries).includes('DO-NOT-EXPORT'));
  const rejected = report.add('prove', { status: 'unrecognized-status', errorName: 'secret-error-payload' });
  assert.deepEqual(rejected.result, {});
});

test('frame statistics use nearest-rank p95 without mutating the input', () => {
  assert.deepEqual(frameGapStatistics([]), { maxFrameGapMs: null, p95FrameGapMs: null });
  const samples = [100, ...Array(19).fill(16)];
  assert.deepEqual(frameGapStatistics(samples), { maxFrameGapMs: 100, p95FrameGapMs: 16 });
  assert.equal(samples[0], 100);
  assert.deepEqual(frameGapStatistics([20]), { maxFrameGapMs: 20, p95FrameGapMs: 20 });
  for (const invalid of [[NaN], [Infinity], [-1], Array(4097).fill(16), null]) {
    assert.throws(() => frameGapStatistics(invalid));
  }
});

test('responsiveness monitoring separates background pauses from measured visible frame gaps', () => {
  const saved = new Map();
  for (const name of ['document', 'requestAnimationFrame', 'cancelAnimationFrame', 'PerformanceObserver']) {
    saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
  }
  let callback;
  let listener;
  let cancelled = false;
  const document = {
    hidden: false,
    addEventListener(name, value) { assert.equal(name, 'visibilitychange'); listener = value; },
    removeEventListener(name, value) { assert.equal(value, listener); listener = null; },
  };
  try {
    Object.defineProperty(globalThis, 'document', { configurable: true, value: document });
    Object.defineProperty(globalThis, 'requestAnimationFrame', { configurable: true, value(fn) { callback = fn; return 1; } });
    Object.defineProperty(globalThis, 'cancelAnimationFrame', { configurable: true, value() { cancelled = true; } });
    Object.defineProperty(globalThis, 'PerformanceObserver', { configurable: true, value: undefined });
    const monitor = startResponsivenessMonitor();
    callback(100); callback(116);
    document.hidden = true; listener();
    callback(1000);
    document.hidden = false; listener();
    callback(2000); callback(2020);
    const result = monitor.stop();
    assert.equal(result.visibleFrames, 4);
    assert.equal(result.hiddenDuringRun, true);
    assert.equal(result.maxFrameGapMs, 20);
    assert.equal(result.p95FrameGapMs, 20);
    assert.equal(result.longTasksSupported, false);
    assert.equal(result.measurementAvailable, true);
    assert.equal(listener, null);
    assert.equal(cancelled, true);
    assert.deepEqual(monitor.stop(), result);
    result.visibleFrames = 0;
    assert.equal(monitor.stop().visibleFrames, 4);
    const immediate = startResponsivenessMonitor().stop();
    assert.equal(immediate.measurementAvailable, false);
    assert.equal(immediate.maxFrameGapMs, null);
  } finally {
    for (const [name, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  }
});

test('environment helper remains usable outside a browser without inventing device measurements', () => {
  const environment = describeEnvironment();
  assert.equal(typeof environment, 'object');
  assert.ok(!Object.hasOwn(environment, 'proof'));
  assert.deepEqual(environment.viewport, {});
  const sample = startResponsivenessMonitor().stop();
  assert.equal(sample.measurementAvailable, false);
  assert.equal(sample.visibleFrames, 0);
  assert.equal(sample.p95FrameGapMs, null);
});
