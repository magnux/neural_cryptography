import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { runSimulation, SIMULATION_LIMITS } from '../dist/js/simulation.js';
import { SPEED_PRESETS, maxChunkBytes } from '../dist/js/profiles.js';

function delivered(r, bytes) {
  assert.equal(r.status, 'delivered');
  assert.equal(r.exact, true);
  assert.equal(r.confirmationReceived, true);
  assert.equal(r.payloadBytes, bytes);
  assert.equal(r.expectedDigest, r.receivedDigest);
  assert.equal(r.stats.deliveries, 1);
  assert.equal(r.receiver.bufferedBytes, 0);
  assert.equal(r.receiver.completedMessages, 1);
  assert.ok(r.stats.qrDecoded > 0);
}

test('default 1024-byte exchange passes through standard QR images and confirms', async () => {
  const r = await runSimulation();
  delivered(r, 1024);
  assert.equal(r.expectedDigest, '114d8ebe680728cadf08c6b27b1cf2524f7191006a14b1c5adff01453c4bc343');
  assert.equal(r.qr.data.version, 10);
  assert.equal(r.qr.data.modules, 57);
  assert.equal(r.qr.control.packetBytes, 60);
});

test('4096-byte exchange survives actual drops, duplicate observations and CRC damage', async () => {
  const r = await runSimulation({ bytes: 4096, seed: 12 });
  delivered(r, 4096);
  for (const key of ['dropped', 'duplicated', 'corrupted', 'protocolRejected']) assert.ok(r.stats[key] > 0, key);
  assert.ok(r.receiver.malformedPackets > 0);
  assert.equal(r.stats.qrRejected, 0, 'CRC faults are injected before QR encoding, not into QR pixels');
  assert.ok(r.events.length <= 100);
});

for (const windowSize of [4, 8]) test(`faulted transfer recovers missing data with window ${windowSize}`, async () => {
  const r = await runSimulation({ bytes: 4096, seed: 12, windowSize });
  delivered(r, 4096);
  assert.ok(r.stats.dropped > 0);
  assert.ok(r.sender.retries > 0);
});

test('blackholed link terminates at the retry bound without false success', async () => {
  const r = await runSimulation({ dropRate: 1, maxRetries: 2, dwellMs: 30, retryMs: 100 });
  assert.equal(r.status, 'failed');
  assert.equal(r.sender.retries, 2);
  assert.equal(r.exact, false);
  assert.equal(r.confirmationReceived, false);
  assert.equal(r.stats.deliveries, 0);
  assert.equal(r.stats.qrDecoded, 0);
  assert.ok(r.simulatedMs < 1000);
});

test('universally corrupted wire packets decode as QR but cannot establish a transfer', async () => {
  const r = await runSimulation({ dropRate: 0, duplicateRate: 0, corruptRate: 1, maxRetries: 2, retryMs: 100, dwellMs: 30 });
  assert.equal(r.status, 'failed');
  assert.equal(r.exact, false);
  assert.equal(r.receiver.state, 'idle');
  assert.ok(r.receiver.malformedPackets >= 3);
  assert.equal(r.stats.qrRejected, 0);
  assert.ok(r.stats.qrDecoded >= 3);
});

test('absolute deadline ends a stalled session even with retries remaining', async () => {
  const r = await runSimulation({ dropRate: 1, dwellMs: 100, retryMs: 1000, timeoutMs: 500 });
  assert.equal(r.status, 'timed-out');
  assert.equal(r.exact, false);
  assert.ok(r.simulatedMs <= 600);
});

test('arbitrary supplied binary payload round-trips and is omitted from reports', async () => {
  const payload = Uint8Array.from({ length: 1024 }, (_, i) => i & 255);
  const original = payload.slice();
  const r = await runSimulation({ payload, dropRate: 0, duplicateRate: 0, corruptRate: 0 });
  delivered(r, payload.length);
  assert.deepEqual(payload, original);
  assert.equal(Object.hasOwn(r.options, 'payload'), false);
  assert.equal(Object.hasOwn(r, 'payload'), false);
});

test('progress reports are bounded snapshots with separate wall and virtual clocks', async () => {
  const progress = [];
  const r = await runSimulation({ bytes: 1024 }, { onProgress: p => progress.push(p) });
  delivered(r, 1024);
  assert.ok(progress.length > 1);
  assert.equal(progress[0].stats.transmissions, 0);
  for (let i = 1; i < progress.length; i++) {
    assert.ok(progress[i].stats.transmissions >= progress[i - 1].stats.transmissions);
    assert.ok(progress[i].simulatedMs >= progress[i - 1].simulatedMs);
    assert.ok(progress[i].events.length <= 100);
  }
  assert.ok(r.elapsedMs >= 0);
  assert.ok(r.simulatedMs > 0);
});

test('invalid inputs are rejected before unbounded simulation allocation', async () => {
  for (const options of [{ bytes: SIMULATION_LIMITS.maxBytes + 1 }, { dropRate: -1 }, { duplicateRate: 2 },
    { corruptRate: NaN }, { timeoutMs: 0 }, { dwellMs: 0 }, { seed: -1 }, { payload: [1, 2] }, { ecc: 'X' }, { controlEcc: 'X' }, { presetId: 'unbounded-metadata' }]) {
    await assert.rejects(runSimulation(options));
  }
});

test('ECC capacity is validated before a supplied payload can be copied', async () => {
  class GuardedBytes extends Uint8Array { slice() { throw Error('Payload copied before validation'); } }
  for (const ecc of ['L', 'M', 'Q', 'H']) {
    await assert.rejects(runSimulation({ payload: new GuardedBytes(3), ecc, chunkBytes: maxChunkBytes(ecc) + 1 }), /chunkBytes/);
  }
});

for (const preset of SPEED_PRESETS) test(`${preset.label} preset delivers across a window boundary through real QR images`, async () => {
  const bytes = preset.chunkBytes * (preset.windowSize + 1) + 17;
  const r = await runSimulation({ ...preset, presetId: preset.id, bytes, dropRate: 0, duplicateRate: 0, corruptRate: 0 });
  delivered(r, bytes);
  assert.equal(r.sender.chunkCount, preset.windowSize + 2);
  assert.equal(r.options.ecc, preset.ecc);
  assert.equal(r.options.presetId, preset.id);
  assert.equal(r.qr.data.ecc, preset.ecc);
  assert.equal(r.options.controlEcc, 'Q');
  assert.equal(r.qr.control.ecc, 'Q');
  assert.equal(r.qr.data.packetBytes, preset.chunkBytes + 18);
});

test('independent control correction is applied while DATA uses its selected level', async () => {
  const r = await runSimulation({ bytes: 257, presetId: 'custom', ecc: 'L', controlEcc: 'H', dropRate: 0, duplicateRate: 0, corruptRate: 0 });
  delivered(r, 257);
  assert.equal(r.qr.data.ecc, 'L');
  assert.equal(r.qr.control.ecc, 'H');
  assert.equal(r.qr.controlEcc, 'H');
  assert.equal(r.options.presetId, 'custom');
});

test('actual simulation worker module emits progress and a correct final report', { timeout: 15000 }, async t => {
  const worker = new Worker(new URL('./simulation-worker-harness.mjs', import.meta.url));
  t.after(() => worker.terminate());
  const messages = [];
  const finished = new Promise((resolve, reject) => {
    worker.on('error', reject);
    worker.on('message', m => {
      messages.push(m);
      if (m.type === 'error') reject(Error(m.error));
      if (m.type === 'result') resolve(m);
    });
  });
  worker.postMessage({ type: 'run', id: 'worker-check', options: { bytes: 1024 } });
  const result = await finished;
  assert.equal(result.id, 'worker-check');
  delivered(result.report, 1024);
  assert.ok(messages.some(m => m.type === 'progress'));
});

test('worker validates malformed requests and can run again after an error', { timeout: 15000 }, async t => {
  const worker = new Worker(new URL('./simulation-worker-harness.mjs', import.meta.url));
  t.after(() => worker.terminate());
  const result = await new Promise((resolve, reject) => {
    worker.on('error', reject);
    worker.on('message', m => {
      if (m.type === 'error') {
        assert.equal(m.id, 'bad-input');
        worker.postMessage({ type: 'run', id: 'recovered', options: { bytes: 128 } });
      } else if (m.type === 'result') resolve(m);
    });
    worker.postMessage({ type: 'run', id: 'bad-input', options: { bytes: -1 } });
  });
  assert.equal(result.id, 'recovered');
  delivered(result.report, 128);
});
