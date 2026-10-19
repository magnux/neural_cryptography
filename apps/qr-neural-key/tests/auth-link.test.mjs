import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate as yieldTask, setTimeout as delay } from 'node:timers/promises';
import { OpticalLink } from '../dist/js/auth/link.js';
import { createSender } from '../dist/js/session.js';
import { decodePacket } from '../dist/js/protocol.js';
import { diagnosticPayload } from '../dist/js/payload.js';

const profile = { chunkBytes: 128, windowSize: 4, ecc: 'Q', dwellMs: 100, retryMs: 500 };
async function prepared(link) {
  const deadline = performance.now() + 3000;
  while (link.snapshot().sender?.state === 'preparing' && performance.now() < deadline) await delay(1);
  assert.notEqual(link.snapshot().sender?.state, 'preparing', 'WebCrypto sender preparation completes within three seconds');
}

/** The displayed QR persists, and each camera reads it repeatedly. */
async function cameras(a, b, clock, { until, fault = () => false, duration = 30000 } = {}) {
  let visibleA = null, visibleB = null;
  const trace = [];
  for (; clock.now < duration; clock.now += 25) {
    // Real hashing shares the thread pool with the rest of the test suite.
    // Keep its scheduling delay out of this deterministic optical clock.
    await prepared(a); await prepared(b);
    for (const [name, link] of [['a', a], ['b', b]]) {
      const next = link.next();
      if (next) {
        trace.push({ from: name, at: clock.now, packet: decodePacket(next) });
        if (name === 'a') visibleA = next; else visibleB = next;
      }
    }
    if (clock.now % 50 === 0) {
      for (const [from, receiver, bytes] of [['a', b, visibleA], ['b', a, visibleB]]) {
        if (bytes && !fault(from, decodePacket(bytes), clock.now)) await receiver.receive(bytes);
      }
    }
    await yieldTask();
    if (until?.()) return trace;
  }
  throw new Error(`Camera transfer did not complete: ${JSON.stringify({ a: a.snapshot(), b: b.snapshot() })}`);
}

test('duplex staged transfers retain receipt ACKs across immediate reply handoffs', async () => {
  const clock = { now: 0 }, receivedA = [], receivedB = [], sends = [], errors = [];
  const payloads = [diagnosticPayload(1800), diagnosticPayload(2900), diagnosticPayload(700), diagnosticPayload(5000)];
  // Mark messages to distinguish stages independently of diagnostic pattern.
  payloads.forEach((bytes, i) => { bytes[0] = i; });
  let a, b;
  const track = promise => { sends.push(promise); promise.catch(error => errors.push(error)); return promise; };
  a = new OpticalLink({ profile, now: () => clock.now, onError: error => errors.push(error), onDelivery: async bytes => {
    receivedA.push(bytes);
    if (bytes[0] === 1) await track(a.send(payloads[2]));
  } });
  b = new OpticalLink({ profile, now: () => clock.now, onError: error => errors.push(error), onDelivery: async bytes => {
    receivedB.push(bytes);
    if (bytes[0] === 0) await track(b.send(payloads[1]));
    if (bytes[0] === 2) await track(b.send(payloads[3]));
  } });
  track(a.send(payloads[0])); await prepared(a);
  let droppedId, firstCompleteAt, droppedReads = 0;
  const trace = await cameras(a, b, clock, {
    fault: (from, packet, now) => {
      if (from === 'b' && packet.kind === 'ack' && packet.status === 'complete' && !droppedId) { droppedId = packet.transferId; firstCompleteAt = now; }
      if (from === 'b' && packet.kind === 'ack' && packet.transferId === droppedId && packet.status === 'complete' && now - firstCompleteAt < 1500) { droppedReads++; return true; }
      return false;
    },
    until: () => receivedA.length === 2 && receivedB.length === 2 && a.snapshot().sender?.state === 'delivered' && b.snapshot().sender?.state === 'delivered',
  });
  await Promise.all(sends);
  assert.ok(droppedReads >= 5);
  assert.deepEqual(receivedA, [payloads[1], payloads[3]]); assert.deepEqual(receivedB, [payloads[0], payloads[2]]);
  assert.deepEqual(errors, []);
  assert.equal(a.snapshot().receiver.completedMessages, 2); assert.equal(b.snapshot().receiver.completedMessages, 2);
  assert.ok(b.snapshot().receiver.duplicatePackets > 0);
  assert.ok(trace.some(item => item.from === 'b' && item.packet.kind === 'start' && item.at < firstCompleteAt + 1500), 'Reply starts while the previous COMPLETE is still lost');
  a.close(); b.close();
});

test('repeated camera observations do not extend an ACK lease or starve outgoing QRs', async () => {
  let now = 0;
  const incoming = await createSender(new Uint8Array(10));
  const start = incoming.next(0), link = new OpticalLink({ profile, ackHoldMs: 250, now: () => now });
  const send = link.send(new Uint8Array(20)); send.catch(() => {}); await prepared(link);
  await link.receive(start);
  assert.equal(decodePacket(link.next()).kind, 'ack');
  let ownStartAt = null;
  for (now = 25; now <= 600; now += 25) {
    await link.receive(start);
    const wire = link.next();
    if (wire && decodePacket(wire).kind === 'start') { ownStartAt = now; break; }
  }
  assert.equal(ownStartAt, 250);
  now = 275; await link.receive(start); assert.equal(link.next(), null, 'Own packet keeps its minimum dwell');
  now = 350; assert.equal(decodePacket(link.next()).kind, 'ack');
  link.close(); await assert.rejects(send, /closed/);
});

test('simultaneous transfers with repeated frames complete fairly in both directions', async () => {
  const clock = { now: 0 }, receivedA = [], receivedB = [];
  const a = new OpticalLink({ profile, now: () => clock.now, onDelivery: bytes => receivedA.push(bytes) });
  const b = new OpticalLink({ profile, now: () => clock.now, onDelivery: bytes => receivedB.push(bytes) });
  const first = diagnosticPayload(3100), second = diagnosticPayload(4200);
  const pa = a.send(first), pb = b.send(second); await prepared(a); await prepared(b);
  const trace = await cameras(a, b, clock, { until: () => a.snapshot().sender?.state === 'delivered' && b.snapshot().sender?.state === 'delivered' });
  await Promise.all([pa, pb]); assert.deepEqual(receivedA, [second]); assert.deepEqual(receivedB, [first]);
  for (const from of ['a', 'b']) {
    const emissions = trace.filter(p => p.from === from);
    for (let i = 1; i < emissions.length; i++) {
      const minimum = emissions[i - 1].packet.kind === 'ack' ? 500 : profile.dwellMs;
      assert.ok(emissions[i].at - emissions[i - 1].at >= minimum);
    }
  }
  a.close(); b.close();
});

test('outgoing buffers and camera-read backlog are bounded, close releases pending sends', async () => {
  const link = new OpticalLink({ maxMessageBytes: 100 });
  await assert.rejects(link.send(new Uint8Array(101)), /at most 100/);
  const sends = Array.from({ length: 4 }, () => link.send(new Uint8Array(10)));
  const outcomes = sends.map(promise => promise.catch(error => error.message));
  await assert.rejects(link.send(new Uint8Array(10)), /queue is full/);
  assert.equal(link.snapshot().queued, 3);
  const incoming = await createSender(new Uint8Array(1));
  const wire = incoming.next(0);
  const reads = Array.from({ length: 9 }, () => link.receive(wire));
  assert.equal((await reads[8]).reason, 'read-queue-full');
  link.close('Test cancelled'); await Promise.all(reads);
  assert.deepEqual(await Promise.all(outcomes), Array(4).fill('Test cancelled'));
  assert.equal(link.next(), null); assert.equal(link.snapshot().receiver.bufferedBytes, 0);
  await assert.rejects(link.send(new Uint8Array()), /closed/);
});

test('absolute send and queue deadlines expire without ACKs and cannot be renewed by reads', async () => {
  let now = 0;
  const errors = [], link = new OpticalLink({ profile, now: () => now, timeoutMs: 600, onError: error => errors.push(error.message) });
  const first = link.send(new Uint8Array(10)), second = link.send(new Uint8Array(10));
  const results = [first, second].map(p => p.catch(error => error.message));
  await prepared(link); assert.equal(decodePacket(link.next()).kind, 'start');
  now = 599; link.next(); assert.notEqual(link.snapshot().sender.state, 'timed-out');
  now = 600; link.next();
  assert.deepEqual(await Promise.all(results), ['Absolute transfer deadline reached', 'Queued optical transfer deadline reached']);
  assert.equal(link.snapshot().sender.state, 'timed-out'); assert.equal(link.snapshot().queued, 0);
  assert.equal(errors.length, 2); link.close();
});

test('delivery callback does not block receipt ACK display or subsequent camera reads', async () => {
  let now = 0, release;
  const callbackBlocked = new Promise(resolve => { release = resolve; });
  const link = new OpticalLink({ profile, now: () => now, onDelivery: () => callbackBlocked });
  const sender = await createSender(new Uint8Array(20));
  sender.receive((await link.receive(sender.next(0))).outbound[0], 0);
  const data = sender.next(1), delivered = await link.receive(data);
  assert.ok(delivered.delivery); assert.equal(decodePacket(link.next()).status, 'complete');
  now = 500;
  assert.equal((await link.receive(data)).reason, 'already-delivered');
  assert.equal(link.snapshot().receiver.completedMessages, 1);
  link.close(); release();
});

test('send snapshots caller payload before asynchronous hashing', async () => {
  const clock = { now: 0 }, received = [];
  const a = new OpticalLink({ profile, now: () => clock.now });
  const b = new OpticalLink({ profile, now: () => clock.now, onDelivery: bytes => received.push(bytes) });
  const original = diagnosticPayload(512), expected = original.slice(), sent = a.send(original); original.fill(0);
  await prepared(a);
  await cameras(a, b, clock, { until: () => a.snapshot().sender?.state === 'delivered' });
  await sent; assert.deepEqual(received, [expected]); a.close(); b.close();
});
