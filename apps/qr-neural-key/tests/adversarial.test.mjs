import test from 'node:test';
import assert from 'node:assert/strict';
import { createSender, Receiver } from '../dist/js/session.js';
import { encodePacket, decodePacket, hexToBytes, crc32 } from '../dist/js/protocol.js';

const ID = '0011223344556677';
const make = (length = 16, options = {}) => createSender(Uint8Array.from({ length }, (_, i) => i), {
  transferId: ID, chunkBytes: 4, windowSize: 4, retryMs: 10, maxRetries: 3, timeoutMs: 100, ...options,
});
const ack = (sender, fields = {}) => encodePacket({ kind: 'ack', transferId: ID,
  base: 0, bitmap: 0, status: 'progress', digestPrefix: sender.meta.digest.slice(0, 16), ...fields });
const begin = sender => { sender.next(0); assert.equal(sender.receive(ack(sender, { status: 'ready' }), 1).accepted, true); };
const sendWindow = (sender, now = 2) => Array.from({ length: Math.min(sender.meta.windowSize, sender.meta.chunkCount - sender.base) }, () => sender.next(now));

test('sender rejects ACKs for unsent data and an early final confirmation', async () => {
  const sender = await make();
  sender.next(0);
  assert.equal(sender.receive(ack(sender, { status: 'ready', bitmap: 1 }), 1).reason, 'unsent-ack');
  assert.equal(sender.receive(ack(sender, { status: 'complete', bitmap: 15 }), 1).reason, 'unsent-ack');
  assert.equal(sender.snapshot().ackedChunks, 0);
  assert.equal(sender.snapshot().state, 'awaiting-start');
});

test('stale or foreign ACKs cannot advance the current window', async () => {
  const sender = await make(32);
  begin(sender); sendWindow(sender);
  assert.equal(sender.receive(ack(sender, { bitmap: 15 }), 3).accepted, true);
  assert.equal(sender.snapshot().windowBase, 4);
  assert.equal(sender.receive(ack(sender, { bitmap: 15 }), 4).reason, 'stale-ack');
  assert.equal(sender.receive(ack(sender, { transferId: 'ffeeddccbbaa9988', base: 4 }), 4).reason, 'foreign');
  assert.equal(sender.receive(ack(sender, { digestPrefix: '0000000000000000', base: 4 }), 4).reason, 'unexpected');
  assert.equal(sender.snapshot().ackedChunks, 4);
});

test('repeated camera observations of a partial ACK cannot reset retry exhaustion', async () => {
  const sender = await make();
  begin(sender); sendWindow(sender);
  const partial = ack(sender, { bitmap: 1 });
  assert.equal(sender.receive(partial, 3).accepted, true);
  for (let i = 0; i < 3; i++) sender.next(4);
  for (let t = 5; t < 14; t++) assert.equal(sender.receive(partial, t).reason, 'duplicate-ack');
  assert.equal(sender.receive(partial, 14).accepted, true);
  for (let i = 0; i < 3; i++) sender.next(15);
  assert.equal(sender.receive(partial, 25).accepted, true);
  for (let i = 0; i < 3; i++) sender.next(26);
  assert.equal(sender.receive(partial, 36).reason, 'retry-limit');
  assert.equal(sender.snapshot().state, 'failed');
  assert.equal(sender.snapshot().retries, 3);
});

test('a regressing partial bitmap never erases already acknowledged chunks', async () => {
  const sender = await make();
  begin(sender); sendWindow(sender);
  sender.receive(ack(sender, { bitmap: 3 }), 3);
  sender.next(4); sender.next(4);
  assert.equal(sender.receive(ack(sender, { bitmap: 1 }), 5).reason, 'duplicate-ack');
  assert.equal(sender.snapshot().ackedChunks, 2);
  assert.deepEqual(sender.snapshot().missingChunks, [2, 3]);
});

test('sender copies source bytes and an expired deadline wins over a valid final ACK', async () => {
  const source = Uint8Array.of(1, 2, 3, 4);
  const sender = await createSender(source, { transferId: ID, chunkBytes: 4, timeoutMs: 10 });
  source.fill(99); begin(sender);
  assert.deepEqual(decodePacket(sender.next(2)).data, Uint8Array.of(1, 2, 3, 4));
  assert.equal(sender.receive(ack(sender, { status: 'complete', bitmap: 1 }), 10).reason, 'deadline');
  assert.equal(sender.snapshot().state, 'timed-out');
});

test('a conflicting START cannot replace active transfer metadata or allocate again', async () => {
  const sender = await make(); const receiver = new Receiver({ timeoutMs: 100 });
  const start = sender.next(0); await receiver.receive(start, 0);
  const original = receiver.snapshot(); const packet = decodePacket(start);
  const conflict = encodePacket({ ...packet, digest: '0'.repeat(64) });
  assert.equal((await receiver.receive(conflict, 1)).reason, 'conflicting-start');
  assert.equal(receiver.snapshot().digest, original.digest);
  assert.equal(receiver.snapshot().bufferedBytes, original.bufferedBytes);
  assert.equal((await receiver.receive(encodePacket({ ...packet, transferId: '1111111111111111' }), 2)).reason, 'foreign');
  assert.equal(receiver.snapshot().bufferedBytes, original.bufferedBytes);
});

test('CRC-valid alteration fails the end-to-end digest and releases the buffer', async () => {
  const sender = await make(4, { windowSize: 1 }); const receiver = new Receiver();
  await receiver.receive(sender.next(0), 0);
  sender.receive(ack(sender, { status: 'ready' }), 1);
  const packet = decodePacket(sender.next(2)); packet.data[0] ^= 1;
  const outcome = await receiver.receive(encodePacket(packet), 2);
  assert.equal(outcome.reason, 'digest-mismatch');
  assert.equal(outcome.delivery, undefined);
  assert.equal(receiver.snapshot().state, 'failed');
  assert.equal(receiver.snapshot().bufferedBytes, 0);
  assert.equal((await receiver.receive(encodePacket(packet), 3)).reason, 'terminal');
});

test('pending SHA-256 completion cannot resurrect a cancelled or expired receiver', async () => {
  for (const finish of ['cancel', 'timeout']) {
    const sender = await make(4, { windowSize: 1 });
    const receiver = new Receiver({ timeoutMs: 10 });
    await receiver.receive(sender.next(0), 0); sender.receive(ack(sender, { status: 'ready' }), 1);
    const data = sender.next(2), deferred = Promise.withResolvers();
    const originalDigest = crypto.subtle.digest;
    crypto.subtle.digest = () => deferred.promise;
    try {
      const pending = receiver.receive(data, 2);
      if (finish === 'cancel') receiver.cancel(3); else receiver.tick(10);
      deferred.resolve(hexToBytes(sender.meta.digest, 32).buffer);
      const outcome = await pending;
      assert.equal(outcome.reason, 'superseded');
      assert.equal(outcome.delivery, undefined);
      assert.equal(receiver.snapshot().state, finish === 'cancel' ? 'cancelled' : 'timed-out');
      assert.equal(receiver.snapshot().bufferedBytes, 0);
    } finally { crypto.subtle.digest = originalDigest; }
  }
});

test('terminal receipts are count-bounded, expire, and do not redeliver repeated packets', async () => {
  const receiver = new Receiver({ maxCompleted: 2, completedTtlMs: 10 });
  let latestData;
  for (let i = 0; i < 3; i++) {
    const id = (i + 1).toString(16).padStart(16, '0');
    const sender = await make(4, { transferId: id, windowSize: 1 });
    const ready = await receiver.receive(sender.next(i), i);
    sender.receive(ready.outbound[0], i);
    latestData = sender.next(i);
    assert.ok((await receiver.receive(latestData, i)).delivery);
  }
  assert.equal(receiver.snapshot().terminalCount, 2);
  const duplicate = await receiver.receive(latestData, 3);
  assert.equal(duplicate.reason, 'already-delivered');
  assert.equal(duplicate.delivery, undefined);
  assert.equal(receiver.snapshot().completedMessages, 3);
  receiver.tick(12);
  assert.equal(receiver.snapshot().terminalCount, 0);
  assert.equal((await receiver.receive(latestData, 12)).reason, 'no-active-message');
});

test('decoder rejects reserved fields even with a valid CRC and bounded oversized input', async () => {
  const sender = await make();
  const start = sender.next(0); start[21] = 1;
  new DataView(start.buffer).setUint32(start.length - 4, crc32(start.subarray(0, -4)));
  assert.throws(() => decodePacket(start), /START body/);
  assert.throws(() => decodePacket(new Uint8Array(2954)), /length/);
});
