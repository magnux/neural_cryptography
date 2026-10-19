import test from 'node:test';
import assert from 'node:assert/strict';
import { createSender, Receiver } from '../dist/js/session.js';
import { encodePacket, decodePacket, CONTROL_MAX_BYTES, crc32 } from '../dist/js/protocol.js';
import { diagnosticPayload } from '../dist/js/payload.js';

const oldId = '11'.repeat(8), newId = '22'.repeat(8), thirdId = '33'.repeat(8), foreignId = '44'.repeat(8);
const options = { transferId: oldId, chunkBytes: 128, windowSize: 1, retryMs: 2, timeoutMs: 1000 };
const smaller = { ...options, transferId: newId, previousTransferId: oldId, chunkBytes: 64 };
const wire = (sender, now = 0) => sender.next(now);
async function begin(sender, receiver, now = 0) {
  const response = await receiver.receive(wire(sender, now), now);
  assert.equal(response.accepted, true);
  assert.equal(sender.receive(response.outbound[0], now).accepted, true);
  return response;
}
async function drain(sender, receiver, now = 1) {
  const deliveries = [];
  for (; now < sender.deadline; now++) {
    const packet = sender.next(now);
    if (packet) {
      const response = await receiver.receive(packet, now);
      if (response.delivery) deliveries.push(response.delivery);
      for (const ack of response.outbound) sender.receive(ack, now);
    }
    if (sender.snapshot().state === 'delivered') return deliveries;
  }
  assert.fail('Transfer did not finish');
}

test('RESTART is a strict 68-byte control packet; stored ACK has zero base and bitmap', async () => {
  const sender = await createSender(diagnosticPayload(257), smaller);
  const bytes = wire(sender), packet = decodePacket(bytes);
  assert.equal(bytes.length, 68); assert.equal(CONTROL_MAX_BYTES, 68);
  assert.equal(packet.kind, 'restart'); assert.equal(packet.previousTransferId, oldId);
  assert.deepEqual(decodePacket(encodePacket(packet)), packet);
  assert.throws(() => encodePacket({ ...packet, previousTransferId: newId }), /new transfer/);
  const stored = { kind: 'ack', transferId: newId, digestPrefix: packet.digest.slice(0, 16), status: 'stored', base: 0, bitmap: 0 };
  assert.deepEqual(decodePacket(encodePacket(stored)), stored);
  for (const patch of [{ base: 1 }, { bitmap: 1 }]) assert.throws(() => encodePacket({ ...stored, ...patch }), /stored ACK/);
  for (const index of [21, 22, 23]) {
    const bad = bytes.slice(); bad[index] = 1;
    new DataView(bad.buffer).setUint32(bad.length - 4, crc32(bad.subarray(0, -4)));
    assert.throws(() => decodePacket(bad), /START\/RESTART/);
  }
  await assert.rejects(createSender(new Uint8Array(1), { ...smaller, previousTransferId: newId }), /new transfer/);
});

test('smaller restart discards partial data and delivers full retransmission exactly once', async () => {
  const payload = diagnosticPayload(333), receiver = new Receiver({ timeoutMs: 1000 });
  const old = await createSender(payload, options); const start = old.next(0);
  old.receive((await receiver.receive(start, 0)).outbound[0], 0);
  const data = old.next(1); const progress = await receiver.receive(data, 1);
  assert.equal(receiver.snapshot().receivedBytes, 128);
  const replacement = await createSender(payload, smaller); const restart = replacement.next(2);
  replacement.receive((await receiver.receive(restart, 2)).outbound[0], 2);
  assert.equal(receiver.snapshot().receivedBytes, 0);
  assert.equal(receiver.active.deadline, 1000);
  for (const stale of [start, data]) {
    const response = await receiver.receive(stale, 3);
    assert.equal(response.accepted, false); assert.equal(decodePacket(response.outbound[0]).reason, 3);
    assert.equal(receiver.snapshot().transferId, newId);
  }
  assert.equal(replacement.receive(progress.outbound[0], 3).reason, 'foreign');
  const deliveries = await drain(replacement, receiver, 4);
  assert.equal(deliveries.length, 1); assert.deepEqual(deliveries[0].bytes, payload);
  assert.equal(receiver.snapshot().completedMessages, 1); assert.equal(receiver.snapshot().bufferedBytes, 0);
});

test('lost final ACK restart is completed from receipt without retransmission or duplicate delivery', async () => {
  const payload = diagnosticPayload(100), old = await createSender(payload, options), receiver = new Receiver({ timeoutMs: 1000 });
  await begin(old, receiver);
  const complete = await receiver.receive(old.next(1), 1); assert.ok(complete.delivery);
  const replacement = await createSender(payload, smaller), restart = replacement.next(2);
  const response = await receiver.receive(restart, 2);
  assert.equal(response.delivery, undefined); assert.equal(decodePacket(response.outbound[0]).status, 'stored');
  assert.equal(receiver.snapshot().bufferedBytes, 0);
  // The old generation's COMPLETE can never finish its replacement.
  assert.equal(replacement.receive(complete.outbound[0], 2).reason, 'foreign');
  const duplicate = await receiver.receive(restart, 3);
  assert.deepEqual(duplicate.outbound, response.outbound); assert.equal(duplicate.delivery, undefined);
  replacement.receive(duplicate.outbound[0], 3);
  assert.equal(replacement.snapshot().state, 'delivered'); assert.equal(replacement.next(4), null);
  assert.equal(receiver.snapshot().completedMessages, 1);
});

test('lost stored ACK chains to another smaller generation with original receipt lifetime', async () => {
  const payload = diagnosticPayload(100), receiver = new Receiver({ timeoutMs: 1000, completedTtlMs: 2000 });
  const old = await createSender(payload, options); await begin(old, receiver); await receiver.receive(old.next(1), 1);
  const replacement = await createSender(payload, smaller); await receiver.receive(replacement.next(2), 2);
  const third = await createSender(payload, { ...smaller, previousTransferId: newId, transferId: thirdId, chunkBytes: 32 });
  const response = await receiver.receive(third.next(3), 3);
  assert.equal(decodePacket(response.outbound[0]).status, 'stored'); third.receive(response.outbound[0], 3);
  assert.equal(third.snapshot().state, 'delivered'); assert.equal(receiver.completedMessages, 1);
  assert.ok(receiver.receipts.every(r => r.deadline === 1000 && r.expiresAt === 2001));
});

test('repeated restart and lost initial START preserve one active generation and its deadline', async () => {
  const receiver = new Receiver({ timeoutMs: 100 }), replacement = await createSender(diagnosticPayload(100), smaller);
  const restart = replacement.next(5);
  assert.equal((await receiver.receive(restart, 5)).accepted, true);
  assert.equal(receiver.active.deadline, 105);
  assert.equal((await receiver.receive(restart, 30)).accepted, true);
  assert.equal(receiver.active.deadline, 105);
  const conflicting = { ...decodePacket(restart), previousTransferId: foreignId };
  assert.equal((await receiver.receive(encodePacket(conflicting), 31)).reason, 'conflicting-start');
  assert.equal(receiver.active.deadline, 105);
  assert.equal(receiver.tick(105).reason, 'deadline');
  assert.equal((await receiver.receive(restart, 106)).reason, 'terminal');
});

test('restart rejects changed payload, non-decreasing chunks, increased window, or foreign active transfer', async () => {
  const payload = diagnosticPayload(333), receiver = new Receiver();
  const old = await createSender(payload, options); await begin(old, receiver);
  const replacement = await createSender(payload, smaller), packet = decodePacket(replacement.next(1));
  for (const patch of [
    { digest: 'aa'.repeat(32) },
    { totalBytes: 332, chunkCount: 6 },
    { chunkBytes: 128, chunkCount: 3 },
    { chunkBytes: 256, chunkCount: 2 },
    { windowSize: 2 },
  ]) {
    const response = await receiver.receive(encodePacket({ ...packet, ...patch }), 2);
    assert.equal(response.reason, 'invalid-restart'); assert.equal(receiver.snapshot().transferId, oldId);
  }
  assert.equal((await receiver.receive(encodePacket({ ...packet, previousTransferId: foreignId }), 3)).reason, 'foreign');
  assert.equal(receiver.snapshot().transferId, oldId);
});

test('superseded predecessor cannot replace a newer generation after cancellation', async () => {
  const payload = diagnosticPayload(333), receiver = new Receiver();
  await begin(await createSender(payload, options), receiver);
  await begin(await createSender(payload, smaller), receiver, 1);
  receiver.cancel(2);
  const stale = await createSender(payload, { ...smaller, transferId: thirdId, chunkBytes: 32 });
  assert.equal((await receiver.receive(stale.next(3), 3)).reason, 'stale-restart');
  assert.equal(receiver.active, null);
});

test('restart never extends the original active deadline, including repeated generations', async () => {
  const payload = diagnosticPayload(333), receiver = new Receiver({ timeoutMs: 100 });
  await begin(await createSender(payload, options), receiver, 10);
  await begin(await createSender(payload, smaller), receiver, 80);
  const third = await createSender(payload, { ...smaller, previousTransferId: newId, transferId: thirdId, chunkBytes: 32 });
  await begin(third, receiver, 109);
  assert.equal(receiver.active.deadline, 110);
  assert.equal(receiver.tick(110).reason, 'deadline');
  assert.equal(receiver.snapshot().bufferedBytes, 0);
});

test('expired delivered receipt cannot establish another restart generation', async () => {
  const payload = diagnosticPayload(100), receiver = new Receiver({ timeoutMs: 10, completedTtlMs: 100 });
  const old = await createSender(payload, options); await begin(old, receiver); await receiver.receive(old.next(1), 1);
  const replacement = await createSender(payload, smaller);
  assert.equal((await receiver.receive(replacement.next(10), 10)).reason, 'deadline');
  assert.equal(receiver.receipts.length, 1); assert.equal(receiver.active, null);
});

test('stored ACK cannot bypass normal COMPLETE validation or finish a started restart', async () => {
  const payload = diagnosticPayload(333), ordinary = await createSender(payload, options);
  const stored = sender => encodePacket({ kind: 'ack', transferId: sender.meta.transferId, base: 0, bitmap: 0, status: 'stored', digestPrefix: sender.meta.digest.slice(0, 16) });
  assert.equal(ordinary.receive(stored(ordinary), 0).reason, 'unexpected-stored');
  const replacement = await createSender(payload, smaller), receiver = new Receiver();
  await begin(replacement, receiver);
  assert.equal(replacement.receive(stored(replacement), 1).reason, 'unexpected-stored');
  const premature = encodePacket({ kind: 'ack', transferId: newId, base: 5, bitmap: 1, status: 'complete', digestPrefix: replacement.meta.digest.slice(0, 16) });
  assert.equal(replacement.receive(premature, 2).reason, 'unsent-ack');
  assert.notEqual(replacement.snapshot().state, 'delivered');
});

test('restart receipt aliases stay bounded and their duplicates do not prolong expiry', async () => {
  const payload = diagnosticPayload(100), receiver = new Receiver({ timeoutMs: 1000, completedTtlMs: 100, maxCompleted: 2 });
  const old = await createSender(payload, { ...options, chunkBytes: 256 });
  await begin(old, receiver); await receiver.receive(old.next(1), 1);
  let previousTransferId = oldId;
  for (const [i, chunkBytes] of [128, 64, 32, 16].entries()) {
    const transferId = (i + 5).toString(16).padStart(2, '0').repeat(8);
    const sender = await createSender(payload, { ...options, previousTransferId, transferId, chunkBytes });
    const bytes = sender.next(i + 2), response = await receiver.receive(bytes, i + 2);
    assert.equal(decodePacket(response.outbound[0]).status, 'stored');
    assert.ok(receiver.receipts.length <= 2);
    await receiver.receive(bytes, i + 2.5);
    assert.ok(receiver.receipts.every(r => r.expiresAt === 101 && r.deadline === 1000));
    previousTransferId = transferId;
  }
  assert.equal(receiver.completedMessages, 1); receiver.tick(101); assert.equal(receiver.receipts.length, 0);
});

test('restart during asynchronous digest checking supersedes the old delivery callback', async () => {
  const payload = diagnosticPayload(100), receiver = new Receiver({ timeoutMs: 1000 });
  const old = await createSender(payload, options), replacement = await createSender(payload, smaller);
  await begin(old, receiver);
  const originalDigest = crypto.subtle.digest;
  let resolveDigest;
  const deferred = new Promise(resolve => { resolveDigest = resolve; });
  const digest = await originalDigest.call(crypto.subtle, 'SHA-256', payload);
  crypto.subtle.digest = () => deferred;
  try {
    const completing = receiver.receive(old.next(1), 1);
    assert.equal(receiver.active.verifying, true);
    const response = await receiver.receive(replacement.next(2), 2);
    assert.equal(response.accepted, true); assert.equal(receiver.active.meta.transferId, newId);
    resolveDigest(digest);
    const staleCompletion = await completing;
    assert.equal(staleCompletion.reason, 'superseded'); assert.equal(staleCompletion.delivery, undefined);
    assert.equal(receiver.completedMessages, 0); assert.equal(receiver.active.deadline, 1000);
    replacement.receive(response.outbound[0], 2);
  } finally { crypto.subtle.digest = originalDigest; }
  const deliveries = await drain(replacement, receiver, 3);
  assert.equal(deliveries.length, 1); assert.deepEqual(deliveries[0].bytes, payload);
});
