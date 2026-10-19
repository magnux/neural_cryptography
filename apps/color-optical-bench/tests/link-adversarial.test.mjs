import test from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import { crc32 } from '../dist/js/format.js';
import { decodePacket, encodePacket, LIMITS, sha256 } from '../dist/js/link/protocol.js';
import { Receiver, createSender } from '../dist/js/link/session.js';

if (!globalThis.crypto) globalThis.crypto = webcrypto;
const sessionId = '00112233445566778899aabbccddeeff';
const messageId = '0123456789abcdef';
const digest = '42'.repeat(32);
const identity = { sessionId, messageId, messageType: 'raw' };
const createReceiver = options => new Receiver(options);
const start = overrides => ({ ...identity, kind: 'start', totalBytes: 256,
  chunkBytes: 64, chunkCount: 4, windowSize: 4, digest, ...overrides });

function altered(packet, change) {
  const bytes = encodePacket(packet);
  change(bytes, new DataView(bytes.buffer));
  new DataView(bytes.buffer).setUint32(bytes.length - 4, crc32(bytes.subarray(0, bytes.length - 4)));
  return bytes;
}

test('link parser rejects CRC-valid inconsistent lengths, huge declarations, and reserved bytes', () => {
  for (const change of [
    (b, v) => v.setUint32(36, LIMITS.maxMessageBytes + 1),
    (b, v) => v.setUint32(40, 0),
    (b, v) => v.setUint32(40, LIMITS.maxChunkBytes + 1),
    (b, v) => v.setUint32(44, 0),
    (b, v) => v.setUint32(44, LIMITS.maxChunks + 1),
    (b, v) => v.setUint32(44, 5),
    b => { b[48] = 0; },
    b => { b[48] = 9; },
    b => { b[4] = 255; },
    b => { b[5] = 255; },
    b => { b[6] = 1; },
    b => { b[7] = 1; },
    b => { b[49] = 1; },
    b => { b[50] = 1; },
    b => { b[51] = 1; },
    (b, v) => v.setUint32(32, 47),
  ]) assert.throws(() => decodePacket(altered(start(), change)));

  const good = encodePacket(start());
  assert.throws(() => decodePacket(good.subarray(0, good.length - 1)));
  const padded = new Uint8Array(good.length + 1); padded.set(good);
  assert.throws(() => decodePacket(padded));
});

test('link control packets reject noncanonical fields even when their CRC is valid', () => {
  const ack = { ...identity, kind: 'ack', base: 0, bitmap: 0, status: 'ready', digest };
  for (const offset of [45, 46, 47]) {
    assert.throws(() => decodePacket(altered(ack, b => { b[offset] = 1; })));
  }
  assert.throws(() => decodePacket(altered(ack, b => { b[44] = 255; })));
  assert.throws(() => decodePacket(altered(ack, (b, v) => v.setUint32(36, LIMITS.maxChunks))));
  for (const offset of [37, 38, 39]) {
    assert.throws(() => decodePacket(altered({ ...identity, kind: 'cancel', reason: 0 }, b => { b[offset] = 1; })));
  }
  assert.throws(() => decodePacket(altered({ ...identity, kind: 'cancel', reason: 0 }, b => { b[36] = 4; })));
  assert.throws(() => decodePacket(altered({ ...identity, kind: 'poll', base: 0 }, (b, v) => v.setUint32(36, LIMITS.maxChunks))));
});

test('link packet encoder rejects inherited enum names instead of converting them into valid packets', () => {
  for (const value of ['constructor', '__proto__', 'toString', 'hasOwnProperty']) {
    assert.throws(() => encodePacket(start({ messageType: value })));
    assert.throws(() => encodePacket({ ...identity, kind: 'ack', base: 0, bitmap: 0, status: value, digest }));
  }
});

test('link decoding owns DATA bytes and enforces the independent chunk bounds', () => {
  const wire = encodePacket({ ...identity, kind: 'data', index: 0, data: Uint8Array.of(1, 2, 3) });
  const decoded = decodePacket(wire);
  wire[40] = 200;
  assert.deepEqual(decoded.data, Uint8Array.of(1, 2, 3));
  assert.throws(() => decodePacket(altered({ ...identity, kind: 'data', index: 0, data: new Uint8Array() },
    (b, v) => v.setUint32(36, LIMITS.maxChunks))));
  assert.throws(() => encodePacket({ ...identity, kind: 'data', index: 0, data: new Uint8Array(LIMITS.maxChunkBytes + 1) }));
});

async function transferPackets(options = {}) {
  const bytes = options.bytes ?? Uint8Array.from({ length: 150 }, (_, i) => (i * 37 + 11) & 255);
  const chunkBytes = options.chunkBytes ?? 64;
  const header = start({ ...options, totalBytes: bytes.length, chunkBytes,
    chunkCount: Math.max(1, Math.ceil(bytes.length / chunkBytes)), digest: await sha256(bytes) });
  return {
    bytes, header, start: encodePacket(header),
    data: Array.from({ length: header.chunkCount }, (_, index) => encodePacket({
      ...header, kind: 'data', index, data: bytes.slice(index * chunkBytes, (index + 1) * chunkBytes),
    })),
    poll: encodePacket({ ...header, kind: 'poll', base: 0 }),
  };
}

test('receiver rejects foreign identifiers and conflicting metadata without delivering bytes', async () => {
  for (const change of [{ sessionId: 'ff'.repeat(16) }, { messageId: 'ff'.repeat(8) }, { messageType: 'proof' }]) {
    const receiver = createReceiver();
    const packets = await transferPackets();
    assert.equal((await receiver.receive(packets.start, 0)).accepted, true);
    const foreign = encodePacket({ ...decodePacket(packets.data[0]), ...change });
    const result = await receiver.receive(foreign, 1);
    assert.equal(result.accepted, false);
    assert.equal(result.delivery, undefined);
  }
  for (const change of [{ digest: '77'.repeat(32) }, { windowSize: 3 }, { chunkBytes: 50, chunkCount: 3 }]) {
    const receiver = createReceiver();
    const packets = await transferPackets();
    await receiver.receive(packets.start, 0);
    const result = await receiver.receive(encodePacket({ ...packets.header, ...change }), 1);
    assert.equal(result.accepted, false);
    assert.equal(result.delivery, undefined);
  }
});

test('receiver rejects conflicting duplicate DATA and incorrectly sized last chunks', async () => {
  const packets = await transferPackets();
  const receiver = createReceiver();
  await receiver.receive(packets.start, 0);
  await receiver.receive(packets.data[0], 1);
  const conflict = decodePacket(packets.data[0]); conflict.data[0] ^= 1;
  const result = await receiver.receive(encodePacket(conflict), 2);
  assert.equal(result.accepted, false);
  assert.equal(result.delivery, undefined);

  for (const length of [0, 21, 23, 64]) {
    const r = createReceiver(); await r.receive(packets.start, 0);
    const wrongLast = encodePacket({ ...packets.header, kind: 'data', index: 2, data: new Uint8Array(length) });
    assert.equal((await r.receive(wrongLast, 1)).accepted, false);
  }
});

test('reordered and duplicated DATA delivers once and lost final ACK is recoverable', async () => {
  const receiver = createReceiver();
  const packets = await transferPackets();
  await receiver.receive(packets.start, 0);
  const results = [];
  for (const [at, index] of [2, 0, 2, 1, 0].entries()) results.push(await receiver.receive(packets.data[index], at + 1));
  const deliveries = results.filter(r => r.delivery);
  assert.equal(deliveries.length, 1);
  assert.deepEqual(deliveries[0].delivery.bytes, packets.bytes);
  // The sender missed the completion reply. A repeated START/POLL must recover it.
  for (const wire of [packets.start, packets.poll, packets.data[2]]) {
    const result = await receiver.receive(wire, 10);
    assert.equal(result.delivery, undefined);
    assert.ok(result.outbound.some(b => decodePacket(b).kind === 'ack' && decodePacket(b).status === 'complete'));
  }
});

test('repeated traffic cannot keep an incomplete receiver alive past its absolute deadline', async () => {
  const receiver = createReceiver({ timeoutMs: 10000 });
  const packets = await transferPackets();
  await receiver.receive(packets.start, 0);
  await receiver.receive(packets.data[0], 100);
  for (const at of [1000, 5000, 9999]) {
    await receiver.receive(packets.start, at);
    await receiver.receive(packets.poll, at);
  }
  receiver.tick(10001);
  for (const wire of [packets.start, packets.data[1], packets.data[2], packets.poll]) {
    const result = await receiver.receive(wire, 10002);
    assert.equal(result.accepted, false);
    assert.equal(result.delivery, undefined);
  }
});

test('cancelled messages cannot be recreated by stale START or completed by late DATA', async () => {
  const receiver = createReceiver();
  const packets = await transferPackets();
  await receiver.receive(packets.start, 0);
  await receiver.receive(packets.data[0], 1);
  const cancellation = receiver.cancel(2);
  assert.equal(decodePacket(cancellation).kind, 'cancel');
  for (const wire of [packets.start, ...packets.data]) {
    const result = await receiver.receive(wire, 3);
    assert.equal(result.accepted, false);
    assert.equal(result.delivery, undefined);
  }
  const fresh = await transferPackets({ messageId: '9988776655443322' });
  assert.equal((await receiver.receive(fresh.start, 4)).accepted, true);
});

test('concurrent completion and cancellation do not emit a stale or duplicate delivery', async () => {
  const packets = await transferPackets({ bytes: Uint8Array.of(5, 6, 7) });
  const receiver = createReceiver();
  await receiver.receive(packets.start, 0);
  const simultaneous = await Promise.all([
    receiver.receive(packets.data[0], 1), receiver.receive(packets.data[0], 1), receiver.receive(packets.poll, 1),
  ]);
  assert.equal(simultaneous.filter(r => r.delivery).length, 1);

  const cancelled = createReceiver();
  await cancelled.receive(packets.start, 0);
  const pending = cancelled.receive(packets.data[0], 1);
  cancelled.cancel(2);
  const result = await pending;
  assert.equal(result.delivery, undefined);
  assert.ok(result.outbound.every(b => decodePacket(b).status !== 'complete'));
});

test('sender rejects ACK claims for unsent chunks, wrong digest, and foreign identity', async () => {
  const bytes = Uint8Array.from({ length: 256 }, (_, i) => i);
  const sender = await createSender(bytes, { ...identity, chunkBytes: 64, windowSize: 4, now: 0 });
  const announced = decodePacket(sender.next(0));
  const ack = { ...announced, kind: 'ack', base: 0, bitmap: 0, status: 'ready' };
  for (const change of [{ digest: '77'.repeat(32) }, { sessionId: 'ff'.repeat(16) }, { messageId: 'ff'.repeat(8) },
    { bitmap: 15, status: 'complete' }, { bitmap: 1, status: 'progress' }, { base: 4 }]) {
    assert.equal(sender.receive(encodePacket({ ...ack, ...change }), 1).accepted, false);
  }
  assert.equal(sender.receive(encodePacket(ack), 2).accepted, true);
  const first = decodePacket(sender.next(3));
  assert.equal(first.kind, 'data');
  assert.equal(first.index, 0);
  // Bits outside the negotiated four-part window are never valid.
  assert.equal(sender.receive(encodePacket({ ...ack, status: 'progress', bitmap: 16 }), 4).accepted, false);
});

test('full-message digest rejects a complete stream of individually CRC-valid altered chunks', async () => {
  const receiver = createReceiver();
  const packets = await transferPackets();
  await receiver.receive(packets.start, 0);
  const bad = decodePacket(packets.data[0]); bad.data[12] ^= 128;
  await receiver.receive(encodePacket(bad), 1);
  await receiver.receive(packets.data[1], 2);
  const result = await receiver.receive(packets.data[2], 3);
  assert.equal(result.accepted, false);
  assert.equal(result.delivery, undefined);
  assert.equal(result.reason, 'digest-mismatch');
  assert.ok(result.outbound.some(b => decodePacket(b).kind === 'cancel'));
  // Repair requires a new message ID, rather than mixing a new attempt into a failed one.
  assert.equal((await receiver.receive(packets.start, 4)).accepted, false);
});

test('sender deadline is absolute despite timely progress and rejects late completion', async () => {
  const bytes = Uint8Array.from({ length: 256 }, (_, i) => i);
  const sender = await createSender(bytes, { ...identity, chunkBytes: 64, windowSize: 2, now: 0,
    retryMs: 1000, timeoutMs: 10000, maxRetries: 100 });
  const receiver = createReceiver({ timeoutMs: 20000 });
  for (const at of [0, 1, 2, 3]) {
    const packet = sender.next(at);
    assert.ok(packet);
    const response = await receiver.receive(packet, at);
    for (const reply of response.outbound) sender.receive(reply, at);
  }
  for (const at of [9997, 9998]) {
    const packet = sender.next(at);
    assert.ok(packet);
    const response = await receiver.receive(packet, at);
    for (const reply of response.outbound) {
      // Even the real receiver's valid final confirmation cannot extend the deadline.
      assert.equal(sender.receive(reply, 10001).accepted, false);
    }
  }
  sender.next(10002);
  assert.equal(sender.snapshot().state, 'timed-out');
});

test('receiver enforces its message budget and releases buffers on every terminal path', async () => {
  const packets = await transferPackets();
  const tooSmall = createReceiver({ maxMessageBytes: 149 });
  assert.equal((await tooSmall.receive(packets.start, 0)).accepted, false);
  assert.equal(tooSmall.snapshot().bufferedBytes, 0);

  for (const terminal of ['cancel', 'timeout', 'success', 'digest-failure']) {
    const receiver = createReceiver({ maxMessageBytes: 150, timeoutMs: 10000 });
    await receiver.receive(packets.start, 0);
    assert.ok(receiver.snapshot().bufferedBytes <= 150 + packets.data.length);
    assert.ok(receiver.snapshot().bufferedBytes > 0);
    if (terminal === 'cancel') receiver.cancel(1);
    else if (terminal === 'timeout') receiver.tick(10001);
    else {
      for (const [index, wire] of packets.data.entries()) {
        const data = decodePacket(wire);
        if (terminal === 'digest-failure' && index === 0) data.data[0] ^= 1;
        await receiver.receive(encodePacket(data), index + 1);
      }
    }
    assert.equal(receiver.snapshot().bufferedBytes, 0, terminal);
    assert.equal(receiver.snapshot().terminalCount, 1, terminal);
  }
});

test('terminal receipts are bounded, contain no payload buffers, and duplicate traffic does not extend retention', async () => {
  const receiver = createReceiver({ maxCompleted: 2, completedTtlMs: 1000 });
  let last;
  for (let i = 0; i < 6; i++) {
    const packets = await transferPackets({ bytes: Uint8Array.of(i), messageId: i.toString(16).padStart(16, '0') });
    await receiver.receive(packets.start, i * 10);
    const result = await receiver.receive(packets.data[0], i * 10 + 1);
    assert.ok(result.delivery);
    assert.ok(receiver.snapshot().terminalCount <= 2);
    assert.equal(receiver.snapshot().bufferedBytes, 0);
    last = packets;
  }
  assert.equal(receiver.snapshot().terminalCount, 2);
  for (const at of [100, 500, 1000]) await receiver.receive(last.poll, at);
  receiver.tick(1052);
  assert.equal(receiver.snapshot().terminalCount, 0);
  assert.equal(receiver.snapshot().bufferedBytes, 0);
});
