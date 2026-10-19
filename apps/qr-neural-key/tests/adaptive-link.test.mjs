import test from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { OpticalLink } from '../dist/js/auth/link.js';
import { PHONE_PROFILE, DESKTOP_PROFILE, smallerChunk, largerChunk } from '../dist/js/auth/adaptation.js';
import { decodePacket, encodePacket } from '../dist/js/protocol.js';
import { Receiver } from '../dist/js/session.js';

const fast = { ...DESKTOP_PROFILE, chunkBytes: 512, windowSize: 1, dwellMs: 100, retryMs: 500 };
async function prepared(link) {
  for (let i = 0; i < 3000 && link.snapshot().sender?.state === 'preparing'; i++) await delay(1);
  assert.notEqual(link.snapshot().sender?.state, 'preparing');
}
function start(profile = fast, options = {}) {
  const clock = { now: 0 }, link = new OpticalLink({ profile, now: () => clock.now, ...options });
  const sent = link.send(new Uint8Array(8192)); sent.catch(() => {});
  return { clock, link, sent };
}
async function ready(link, receiver, clock) {
  await prepared(link);
  const wire = link.next(); assert.ok(wire);
  const result = await receiver.receive(wire, clock.now);
  assert.ok(result.outbound.length); await link.receive(result.outbound[0]);
  return decodePacket(wire);
}

test('device targets and ladder keep a configured phone cap', () => {
  assert.equal(PHONE_PROFILE.chunkBytes, 128); assert.equal(DESKTOP_PROFILE.chunkBytes, 1024);
  assert.equal(smallerChunk(1000), 512); assert.equal(smallerChunk(32), 32);
  assert.equal(largerChunk(64, 128), 128); assert.equal(largerChunk(128, 128), 128);
});

test('manual smaller QR restarts same payload and deadline; larger window waits for next message', async () => {
  const { clock, link, sent } = start(), receiver = new Receiver();
  const initial = await ready(link, receiver, clock), digest = initial.digest;
  clock.now = 100; await receiver.receive(link.next(), clock.now);
  const deadline = link.current.sender.deadline;
  link.setProfile({ ...fast, chunkBytes: 128, windowSize: 4, dwellMs: 200 }); await prepared(link);
  assert.equal(link.current.sender.deadline, deadline);
  assert.equal(link.snapshot().effectiveProfile.windowSize, 1);
  assert.equal(link.snapshot().adaptation.pendingProfile.windowSize, 4);
  clock.now = 300;
  const restart = decodePacket(link.next());
  assert.equal(restart.kind, 'restart'); assert.equal(restart.previousTransferId, initial.transferId);
  assert.equal(restart.digest, digest); assert.equal(restart.chunkBytes, 128);
  const accepted = await receiver.receive(encodePacket(restart), clock.now); assert.ok(accepted.accepted);
  link.close(); await assert.rejects(sent);
});

test('consecutive unapplied smaller settings never name a phantom transfer', async () => {
  const { clock, link, sent } = start(), receiver = new Receiver();
  const first = await ready(link, receiver, clock);
  link.setProfile({ ...fast, chunkBytes: 256 }); await prepared(link);
  link.setProfile({ ...fast, chunkBytes: 128 }); await prepared(link);
  clock.now = 100;
  const restart = decodePacket(link.next());
  assert.equal(restart.previousTransferId, first.transferId); assert.equal(restart.chunkBytes, 128);
  assert.ok((await receiver.receive(encodePacket(restart), clock.now)).accepted);
  link.close(); await assert.rejects(sent);
});

test('a second shrink waits for acknowledgement of an already emitted restart', async () => {
  const { clock, link, sent } = start(), receiver = new Receiver();
  const first = await ready(link, receiver, clock);
  link.setProfile({ ...fast, chunkBytes: 256 }); await prepared(link); clock.now = 100;
  const intermediate = link.next(); // Camera misses this restart initially.
  const intermediatePacket = decodePacket(intermediate); assert.equal(intermediatePacket.previousTransferId, first.transferId);
  link.setProfile({ ...fast, chunkBytes: 128 }); await prepared(link);
  assert.equal(link.snapshot().effectiveProfile.chunkBytes, 256);
  assert.equal(link.snapshot().adaptation.pendingProfile.chunkBytes, 128);
  clock.now = 600; const retry = link.next(); assert.equal(decodePacket(retry).transferId, intermediatePacket.transferId);
  const response = await receiver.receive(retry, clock.now); await link.receive(response.outbound[0]); await prepared(link);
  clock.now = 700; const final = decodePacket(link.next());
  assert.equal(final.previousTransferId, intermediatePacket.transferId); assert.equal(final.chunkBytes, 128);
  assert.ok((await receiver.receive(encodePacket(final), clock.now)).accepted);
  link.close(); await assert.rejects(sent);
});

test('duplicate and foreign ACKs cannot postpone downshifting; deadline is never extended', async () => {
  const { clock, link, sent } = start(fast, { adaptive: true, timeoutMs: 10000 }), receiver = new Receiver();
  const initial = await ready(link, receiver, clock); clock.now = 100;
  const data = link.next(); assert.equal(decodePacket(data).kind, 'data');
  const empty = encodePacket({ kind: 'ack', transferId: initial.transferId, digestPrefix: initial.digest.slice(0, 16), base: 0, bitmap: 0, status: 'progress' });
  const foreign = encodePacket({ kind: 'ack', transferId: '0000000000000000', digestPrefix: initial.digest.slice(0, 16), base: 0, bitmap: 1, status: 'complete' });
  for (clock.now = 200; clock.now <= 1800; clock.now += 100) { await link.receive(empty); await link.receive(foreign); link.next(); await prepared(link); }
  assert.equal(link.snapshot().adaptation.downshifts, 1); assert.equal(link.snapshot().effectiveProfile.chunkBytes, 256);
  assert.equal(link.current.sender.deadline, 10000); assert.equal(link.snapshot().ackHoldMs, 750);
  clock.now = 10000; link.next(); await assert.rejects(sent, /deadline/);
  link.close();
});

test('missing START ACK only slows timing, and a slow eight-code window is allowed to finish', async () => {
  const first = start(fast, { adaptive: true }); await prepared(first.link); first.link.next();
  first.clock.now = 1600; first.link.next();
  assert.equal(first.link.snapshot().effectiveProfile.chunkBytes, 512); assert.equal(first.link.snapshot().adaptation.downshifts, 0);
  assert.equal(first.link.snapshot().effectiveProfile.dwellMs, 350); first.link.close(); await assert.rejects(first.sent);
  const { clock, link, sent } = start({ ...fast, windowSize: 8, dwellMs: 2000, retryMs: 2000 }, { adaptive: true });
  const receiver = new Receiver(); await ready(link, receiver, clock);
  for (clock.now = 2000; clock.now <= 16000; clock.now += 2000) {
    const wire = link.next(); assert.equal(decodePacket(wire).kind, 'data');
    const reply = await receiver.receive(wire, clock.now); for (const ack of reply.outbound) await link.receive(ack);
    assert.equal(link.snapshot().adaptation.downshifts, 0);
  }
  assert.equal(link.snapshot().sender.ackedBytes, 4096);
  link.close(); await assert.rejects(sent);
});

test('eight clean ACK windows queue one bounded next-message upshift', async () => {
  const { clock, link, sent } = start({ ...fast, chunkBytes: 128 }, { adaptive: true }), receiver = new Receiver();
  await ready(link, receiver, clock);
  clock.now = 100; link.next(); // Dense DATA is unreadable; force a real fallback.
  clock.now = 1700; link.next(); await prepared(link);
  clock.now = 1800; const restart = link.next();
  const response = await receiver.receive(restart, clock.now); await link.receive(response.outbound[0]);
  for (let i = 0; i < 12; i++) {
    clock.now += 350; const wire = link.next();
    if (!wire) continue;
    const result = await receiver.receive(wire, clock.now); for (const ack of result.outbound) await link.receive(ack);
  }
  assert.equal(link.snapshot().effectiveProfile.chunkBytes, 64);
  assert.equal(link.snapshot().adaptation.pendingProfile.chunkBytes, 128); assert.equal(link.snapshot().adaptation.upshifts, 1);
  assert.ok(link.snapshot().adaptation.events.length <= 32);
  link.close(); await assert.rejects(sent);
});
