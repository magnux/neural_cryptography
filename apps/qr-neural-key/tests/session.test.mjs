import test from 'node:test';
import assert from 'node:assert/strict';
import { createSender, Receiver } from '../dist/js/session.js';
import { decodePacket, encodePacket, sha256 } from '../dist/js/protocol.js';
import { diagnosticPayload } from '../dist/js/payload.js';
const transferId = 'ab'.repeat(8);
const terminal = state => ['delivered', 'failed', 'cancelled', 'timed-out'].includes(state);

async function transfer(payload, options = {}, fault = () => false) {
  const sender = await createSender(payload, { transferId, chunkBytes: 128, retryMs: 10, maxRetries: 30, timeoutMs: 100000, ...options });
  const receiver = new Receiver({ timeoutMs: 100000 }), deliveries = [], trace = [];
  for (let now = 0; now < 100000; now++) {
    const bytes = sender.next(now);
    if (bytes) {
      const packet = decodePacket(bytes); trace.push(packet);
      if (!fault(packet, bytes, trace)) {
        const result = await receiver.receive(bytes, now);
        if (result.delivery) deliveries.push(result.delivery);
        for (const outbound of result.outbound) sender.receive(outbound, now);
      }
    }
    if (terminal(sender.snapshot().state)) break;
  }
  return { sender, receiver, deliveries, trace };
}

for (const [length, windowSize] of [[0, 1], [35, 8], [1024, 1], [165000, 1], [165000, 4], [165000, 8]]) {
  test(`exact ${length}-byte transfer, window ${windowSize}`, async () => {
    const original = diagnosticPayload(length), r = await transfer(original, { windowSize });
    assert.equal(r.sender.snapshot().state, 'delivered'); assert.equal(r.deliveries.length, 1);
    assert.deepEqual(r.deliveries[0].bytes, original); assert.equal(r.deliveries[0].digest, await sha256(original));
    assert.equal(r.receiver.snapshot().bufferedBytes, 0); assert.equal(r.sender.snapshot().ackedChunks, r.sender.snapshot().chunkCount);
  });
}

for (const windowSize of [1, 4, 8]) {
  test(`window ${windowSize} recovers dropped and CRC-damaged DATA`, async () => {
    const seen = new Map();
    const r = await transfer(diagnosticPayload(165000), { windowSize }, (p, bytes) => {
      if (p.kind !== 'data') return false;
      const n = (seen.get(p.index) ?? 0) + 1; seen.set(p.index, n);
      if (n === 1 && p.index === 2) return true;
      if (n === 1 && p.index === 5) bytes[bytes.length - 5] ^= 1;
      return false;
    });
    assert.equal(r.sender.snapshot().state, 'delivered'); assert.equal(r.deliveries.length, 1);
    assert.equal(seen.get(2), 2); assert.equal(seen.get(5), 2);
    for (const [index, count] of seen) if (index !== 2 && index !== 5) assert.equal(count, 1);
    assert.equal(r.receiver.snapshot().malformedPackets, 1);
  });
}

test('stop-and-wait retains the same DATA while ACK is lost, then advances immediately', async () => {
  const sender = await createSender(diagnosticPayload(256), { transferId, retryMs: 10 }); const receiver = new Receiver();
  const ready = await receiver.receive(sender.next(0), 0); sender.receive(ready.outbound[0], 0);
  const data = sender.next(1), first = await receiver.receive(data, 1);
  assert.equal(sender.snapshot().state, 'awaiting-ack'); assert.equal(decodePacket(first.outbound[0]).status, 'progress');
  assert.equal(sender.next(2), null); assert.deepEqual(sender.next(11), data);
  const repeated = await receiver.receive(data, 11); assert.equal(repeated.reason, 'duplicate');
  assert.equal(repeated.outbound[0].length, 28); sender.receive(repeated.outbound[0], 11);
  assert.equal(decodePacket(sender.next(11)).index, 1);
});

for (const windowSize of [1, 2, 3, 4, 5, 6, 7, 8]) {
  test(`window ${windowSize} never starts the next batch before its full ACK`, async () => {
    const sender = await createSender(diagnosticPayload((windowSize * 2 + 1) * 16), {
      transferId, chunkBytes: 16, windowSize, retryMs: 10,
    });
    const receiver = new Receiver();
    sender.receive((await receiver.receive(sender.next(0), 0)).outbound[0], 0);
    let lastData, latestAck;
    for (let index = 0; index < windowSize; index++) {
      const packet = sender.next(index + 1);
      assert.equal(decodePacket(packet).index, index);
      lastData = packet;
      // For a multipart batch, withhold its last chunk to obtain a partial ACK.
      if (windowSize === 1 || index < windowSize - 1) {
        latestAck = (await receiver.receive(packet, index + 1)).outbound[0];
      }
    }
    assert.equal(sender.snapshot().state, 'awaiting-ack');
    assert.equal(sender.next(windowSize + 1), null);
    const retry = decodePacket(sender.next(windowSize + 10));
    assert.equal(retry.kind, windowSize === 1 ? 'data' : 'poll');
    assert.equal(retry.kind === 'data' ? retry.index : retry.base, 0);
    assert.equal(sender.snapshot().windowBase, 0);

    if (windowSize > 1) {
      assert.equal(sender.receive(latestAck, windowSize + 10).accepted, true);
      const missing = sender.next(windowSize + 11);
      assert.equal(decodePacket(missing).index, windowSize - 1);
      assert.equal(sender.snapshot().windowBase, 0);
      assert.equal(sender.next(windowSize + 12), null);
      latestAck = (await receiver.receive(missing, windowSize + 12)).outbound[0];
    } else {
      assert.deepEqual(encodePacket(retry), lastData);
    }

    assert.equal(sender.receive(latestAck, windowSize + 12).accepted, true);
    assert.equal(sender.snapshot().windowBase, windowSize);
    assert.equal(decodePacket(sender.next(windowSize + 13)).index, windowSize);
  });
}

test('non-power-of-two windows recover exact bytes through a partial final batch', async () => {
  for (const windowSize of [2, 3, 5, 6, 7]) {
    const original = diagnosticPayload(windowSize * 2 * 32 + 9);
    const r = await transfer(original, { chunkBytes: 32, windowSize });
    assert.equal(r.sender.snapshot().state, 'delivered', `window ${windowSize}`);
    assert.equal(r.deliveries.length, 1, `window ${windowSize}`);
    assert.deepEqual(r.deliveries[0].bytes, original, `window ${windowSize}`);
    assert.equal(r.receiver.snapshot().bufferedBytes, 0, `window ${windowSize}`);
  }
});

test('lost COMPLETE recovers through repeated DATA without duplicate delivery', async () => {
  const sender = await createSender(diagnosticPayload(10), { transferId, retryMs: 10 }); const receiver = new Receiver();
  sender.receive((await receiver.receive(sender.next(0), 0)).outbound[0], 0);
  const last = sender.next(1), first = await receiver.receive(last, 1); assert.ok(first.delivery);
  const retry = sender.next(11); assert.deepEqual(retry, last);
  const repeated = await receiver.receive(retry, 11); assert.equal(repeated.delivery, undefined);
  sender.receive(repeated.outbound[0], 11); assert.equal(sender.snapshot().state, 'delivered');
  assert.equal(receiver.snapshot().completedMessages, 1); assert.equal(decodePacket(repeated.outbound[0]).status, 'complete');
});

test('unchanged partial ACK after explicit POLL retries a chunk lost twice', async () => {
  const sender = await createSender(diagnosticPayload(1024), { transferId, windowSize: 4, retryMs: 10 });
  const receiver = new Receiver();
  sender.receive((await receiver.receive(sender.next(0), 0)).outbound[0], 0);
  for (let now = 1; now <= 4; now++) {
    const data = sender.next(now);
    if (decodePacket(data).index !== 1) {
      const received = await receiver.receive(data, now);
      for (const outbound of received.outbound) sender.receive(outbound, now);
    }
  }
  assert.equal(decodePacket(sender.next(5)).index, 1); // Lose the retransmission too.
  const poll = sender.next(15); assert.equal(decodePacket(poll).kind, 'poll');
  const unchanged = await receiver.receive(poll, 15);
  assert.equal(decodePacket(unchanged.outbound[0]).bitmap, 13);
  assert.equal(sender.receive(unchanged.outbound[0], 15).accepted, true);
  assert.equal(sender.snapshot().retries, 2);
  const recovered = sender.next(16); assert.equal(decodePacket(recovered).index, 1);
  sender.receive((await receiver.receive(recovered, 16)).outbound[0], 16);
  assert.equal(sender.snapshot().windowBase, 4);
});

test('full SHA-256 rejects a corrupted payload with freshly computed valid packet CRC', async () => {
  const sender = await createSender(diagnosticPayload(10), { transferId }); const receiver = new Receiver();
  sender.receive((await receiver.receive(sender.next(0), 0)).outbound[0], 0);
  const data = decodePacket(sender.next(1)); data.data[0] ^= 1;
  const result = await receiver.receive(encodePacket(data), 1);
  assert.equal(result.reason, 'digest-mismatch'); assert.equal(result.delivery, undefined);
  assert.equal(receiver.snapshot().state, 'failed'); assert.equal(receiver.snapshot().bufferedBytes, 0);
  sender.receive(result.outbound[0], 1); assert.equal(sender.snapshot().state, 'cancelled');
});

test('cancellation is idempotent and not echoed', async () => {
  const sender = await createSender(diagnosticPayload(20), { transferId }); const receiver = new Receiver();
  await receiver.receive(sender.next(0), 0);
  const cancellation = sender.cancel(1); assert.equal((await receiver.receive(cancellation, 1)).outbound.length, 0);
  assert.equal((await receiver.receive(cancellation, 2)).outbound.length, 0);
  assert.equal(sender.cancel(3), null); assert.equal(sender.next(1000), null); assert.equal(receiver.snapshot().state, 'cancelled');
});

test('deadline and retry exhaustion release sender payloads and stop transmissions', async () => {
  const deadline = await createSender(new Uint8Array(10), { transferId, timeoutMs: 10 }); deadline.next(0);
  assert.equal(decodePacket(deadline.next(10)).kind, 'cancel'); assert.equal(deadline.snapshot().state, 'timed-out'); assert.equal(deadline.next(11), null);
  const retry = await createSender(new Uint8Array(10), { transferId, retryMs: 2, maxRetries: 1 }); retry.next(0); retry.next(2);
  assert.equal(decodePacket(retry.next(4)).kind, 'cancel'); assert.equal(retry.snapshot().state, 'failed');
});

test('chunk count exceeding 16-bit representation is rejected before START', async () => {
  await assert.rejects(createSender(new Uint8Array(65536), { chunkBytes: 1 }), /chunkCount/);
});
