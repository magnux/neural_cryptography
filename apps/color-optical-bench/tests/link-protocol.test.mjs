import test from 'node:test';
import assert from 'node:assert/strict';
import { encodePacket, decodePacket, sha256, LIMITS, DATA_OVERHEAD, CONTROL_MAX_BYTES } from '../dist/js/link/protocol.js';
import { createSender, Receiver } from '../dist/js/link/session.js';
import { crc32 } from '../dist/js/format.js';

const ids = { sessionId: 'ab'.repeat(16), messageId: 'cd'.repeat(8), messageType: 'proof' };
const bytes = n => Uint8Array.from({ length: n }, (_, i) => (i * 17 + (i >> 8)) & 255);
const resign = p => { new DataView(p.buffer, p.byteOffset, p.byteLength).setUint32(p.length - 4, crc32(p.subarray(0, -4))); return p; };
const digest = 'ef'.repeat(32);

test('current binary packet kinds round trip with bounded overhead', () => {
  for (const body of [
    { kind: 'start', totalBytes: 101, chunkBytes: 100, chunkCount: 2, windowSize: 8, digest },
    { kind: 'data', index: 1, data: bytes(101) },
    { kind: 'ack', base: 0, bitmap: 3, status: 'progress', digest },
    { kind: 'poll', base: 8 }, { kind: 'cancel', reason: 2 },
  ]) {
    const p = { ...ids, ...body }, encoded = encodePacket(p);
    assert.deepEqual(decodePacket(encoded), p);
    assert.ok(encoded.length <= (p.kind === 'data' ? p.data.length + DATA_OVERHEAD : CONTROL_MAX_BYTES));
  }
});

test('parser rejects corrupted, unknown, noncanonical and oversized packets', () => {
  const packet = encodePacket({ ...ids, kind: 'poll', base: 0 });
  const damaged = packet.slice(); damaged[25] ^= 1; assert.throws(() => decodePacket(damaged), /CRC/);
  for (const index of [4, 5, 6, 7]) {
    const p = packet.slice(); p[index] = 255; assert.throws(() => decodePacket(resign(p)));
  }
  const trailing = new Uint8Array(packet.length + 1); trailing.set(packet); assert.throws(() => decodePacket(trailing));
  assert.throws(() => decodePacket(new Uint8Array(LIMITS.maxChunkBytes + DATA_OVERHEAD + 1)));
  assert.throws(() => encodePacket({ ...ids, messageType: 'constructor', kind: 'poll', base: 0 }));
  assert.throws(() => encodePacket({ ...ids, kind: 'ack', base: 0, bitmap: 0, status: 'toString', digest }));
});

test('START requires consistent bounded dimensions before allocating', () => {
  const start = { ...ids, kind: 'start', totalBytes: 20, chunkBytes: 10, chunkCount: 2, windowSize: 4, digest };
  for (const patch of [{ totalBytes: LIMITS.maxMessageBytes + 1 }, { chunkBytes: 0 }, { chunkCount: 3 }, { windowSize: 9 }, { digest: 'EF'.repeat(32) }]) {
    assert.throws(() => encodePacket({ ...start, ...patch }));
  }
  const bad = encodePacket(start); new DataView(bad.buffer).setUint32(44, LIMITS.maxChunks + 1);
  assert.throws(() => decodePacket(resign(bad)), /chunkCount|chunk count/);
});

async function transfer(payload, options = {}, fault = () => false) {
  const sender = await createSender(payload, { ...ids, chunkBytes: 5428, retryMs: 10, maxRetries: 30, timeoutMs: 50000, ...options });
  const receiver = new Receiver({ timeoutMs: 50000 });
  const deliveries = [], trace = [];
  for (let now = 0; now < 50000; now += 5) {
    const packet = sender.next(now);
    if (packet) {
      const p = decodePacket(packet); trace.push(p);
      if (!fault(p, packet, trace)) {
        const r = await receiver.receive(packet, now);
        if (r.delivery) deliveries.push(r.delivery);
        for (const outbound of r.outbound) sender.receive(outbound, now);
      }
    }
    if (['delivered', 'failed', 'cancelled', 'timed-out'].includes(sender.snapshot().state)) break;
  }
  return { sender, receiver, deliveries, trace };
}

for (const [length, windowSize] of [[0, 1], [35, 8], [164746, 1], [164746, 8]]) {
  test(`byte exact transfer of ${length} bytes with window ${windowSize}`, async () => {
    const original = bytes(length), r = await transfer(original, { windowSize });
    assert.equal(r.sender.snapshot().state, 'delivered');
    assert.equal(r.deliveries.length, 1); assert.deepEqual(r.deliveries[0].bytes, original);
    assert.equal(r.deliveries[0].digest, await sha256(original));
    assert.equal(r.receiver.snapshot().bufferedBytes, 0);
  });
}

test('missing and CRC damaged chunks alone are retransmitted', async () => {
  const seen = new Map();
  const r = await transfer(bytes(164746), { windowSize: 8 }, (p, packet) => {
    if (p.kind !== 'data') return false;
    const n = (seen.get(p.index) ?? 0) + 1; seen.set(p.index, n);
    if (n === 1 && p.index === 2) return true;
    if (n === 1 && p.index === 5) packet[packet.length - 5] ^= 1;
    return false;
  });
  assert.equal(r.sender.snapshot().state, 'delivered'); assert.equal(r.deliveries.length, 1);
  assert.equal(seen.get(2), 2); assert.equal(seen.get(5), 2);
  for (const [i, n] of seen) if (i !== 2 && i !== 5) assert.equal(n, 1);
  assert.equal(r.receiver.snapshot().malformedPackets, 1);
});

test('lost final receipt recovers without delivering the message twice', async () => {
  const sender = await createSender(bytes(10), { ...ids, chunkBytes: 20, retryMs: 10 });
  const receiver = new Receiver(); let delivered = 0;
  const first = await receiver.receive(sender.next(0), 0); sender.receive(first.outbound[0], 0);
  const last = sender.next(1), received = await receiver.receive(last, 1); delivered += !!received.delivery;
  assert.equal(received.outbound.length, 1); // Drop this COMPLETE receipt.
  const poll = sender.next(2), recovered = await receiver.receive(poll, 2); delivered += !!recovered.delivery;
  sender.receive(recovered.outbound[0], 2);
  const repeated = await receiver.receive(last, 3); delivered += !!repeated.delivery;
  assert.equal(delivered, 1); assert.equal(sender.snapshot().state, 'delivered');
  assert.equal(decodePacket(repeated.outbound[0]).status, 'complete');
});

test('receiver rejects SHA mismatch even when every packet CRC is valid', async () => {
  const sender = await createSender(bytes(10), { ...ids, chunkBytes: 20 }); const receiver = new Receiver();
  const ready = await receiver.receive(sender.next(0), 0); sender.receive(ready.outbound[0], 0);
  const data = decodePacket(sender.next(1)); data.data[0] ^= 1;
  const r = await receiver.receive(encodePacket(data), 1);
  assert.equal(r.reason, 'digest-mismatch'); assert.equal(r.delivery, undefined);
  assert.equal(receiver.snapshot().state, 'failed'); assert.equal(receiver.snapshot().bufferedBytes, 0);
});

test('sender rejects ACK bits for unsent chunks and early COMPLETE', async () => {
  const sender = await createSender(bytes(30), { ...ids, chunkBytes: 10, windowSize: 2 });
  const start = decodePacket(sender.next(0));
  const invalid = { ...ids, kind: 'ack', base: 0, bitmap: 1, status: 'ready', digest: start.digest };
  assert.equal(sender.receive(encodePacket(invalid), 1).reason, 'unsent-ack');
  assert.equal(sender.snapshot().state, 'awaiting-start');
  const early = { ...invalid, base: 2, bitmap: 1, status: 'complete' };
  assert.equal(sender.receive(encodePacket(early), 2).accepted, false);
  assert.equal(sender.snapshot().ackedChunks, 0);
});

test('repeated START does not extend receiver deadline or resurrect timed out state', async () => {
  const sender = await createSender(bytes(30), { ...ids }); const receiver = new Receiver({ timeoutMs: 20 });
  const start = sender.next(0); await receiver.receive(start, 0); await receiver.receive(start, 19);
  assert.equal(receiver.tick(20).reason, 'deadline');
  assert.equal((await receiver.receive(start, 21)).reason, 'terminal');
  assert.equal(receiver.snapshot().state, 'timed-out'); assert.equal(receiver.snapshot().bufferedBytes, 0);
});

test('cancel is terminal, idempotent and never echoed in a cancellation loop', async () => {
  const sender = await createSender(bytes(30), { ...ids }); const receiver = new Receiver();
  await receiver.receive(sender.next(0), 0); const cancel = sender.cancel(1);
  assert.equal((await receiver.receive(cancel, 1)).outbound.length, 0);
  assert.equal((await receiver.receive(cancel, 2)).outbound.length, 0);
  assert.equal(sender.cancel(3), null); assert.equal(sender.next(1000), null);
  assert.equal(receiver.snapshot().state, 'cancelled');
});

test('sender snapshots do not expose mutable metadata and payload ownership is copied', async () => {
  const input = bytes(10), sender = await createSender(input, { ...ids, chunkBytes: 20 }); input.fill(0);
  const snapshot = sender.snapshot(); snapshot.sessionId = '00'.repeat(16); snapshot.missingChunks.push(99);
  const receiver = new Receiver(); const ready = await receiver.receive(sender.next(0), 0); sender.receive(ready.outbound[0], 0);
  const r = await receiver.receive(sender.next(1), 1);
  assert.deepEqual(r.delivery.bytes, bytes(10)); assert.equal(sender.snapshot().sessionId, ids.sessionId);
});
