import test from 'node:test';
import assert from 'node:assert/strict';
import { encodePacket, decodePacket, crc32, LIMITS, DATA_OVERHEAD, ACK_BYTES, CONTROL_MAX_BYTES } from '../dist/js/protocol.js';
import { diagnosticPayload, seededRandom } from '../dist/js/payload.js';
const transferId = 'ab'.repeat(8), digest = 'cd'.repeat(32), digestPrefix = digest.slice(0, 16);
const resign = bytes => { new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).setUint32(bytes.length - 4, crc32(bytes.subarray(0, -4))); return bytes; };

test('QRT packets round trip with compact fixed ACK and control sizes', () => {
  for (const body of [
    { kind: 'start', totalBytes: 257, chunkBytes: 128, chunkCount: 3, windowSize: 1, digest },
    { kind: 'data', index: 2, data: Uint8Array.of(255) },
    { kind: 'ack', base: 0, bitmap: 3, status: 'progress', digestPrefix },
    { kind: 'poll', base: 8 }, { kind: 'cancel', reason: 2 },
  ]) {
    const packet = { transferId, ...body }, encoded = encodePacket(packet);
    assert.deepEqual(decodePacket(encoded), packet);
    if (packet.kind === 'ack') assert.equal(encoded.length, ACK_BYTES);
    else if (packet.kind === 'data') assert.equal(encoded.length, packet.data.length + DATA_OVERHEAD);
    else assert.ok(encoded.length <= CONTROL_MAX_BYTES);
  }
});

test('CRC32 uses the standard known check value', () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
});

test('parser rejects corruption, wrong magic, unknown kind and malformed lengths', () => {
  const packet = encodePacket({ transferId, kind: 'poll', base: 0 });
  const corrupt = packet.slice(); corrupt[5] ^= 128; assert.throws(() => decodePacket(corrupt), /CRC/);
  for (const index of [0, 1, 2, 3]) { const bad = packet.slice(); bad[index] = 255; assert.throws(() => decodePacket(resign(bad))); }
  const trailing = new Uint8Array(packet.length + 1); trailing.set(packet); assert.throws(() => decodePacket(resign(trailing)));
  assert.throws(() => decodePacket(new Uint8Array(16)), /length/);
  assert.throws(() => decodePacket(new Uint8Array(LIMITS.maxChunkBytes + DATA_OVERHEAD + 1)), /length/);
  assert.throws(() => encodePacket({ transferId, kind: 'constructor' }));
  assert.throws(() => encodePacket({ transferId, kind: 'ack', base: 0, bitmap: 1, status: 'toString', digestPrefix }));
});

test('START validates consistent dimensions and reserved bytes before allocation', () => {
  const start = { transferId, kind: 'start', totalBytes: 10, chunkBytes: 8, chunkCount: 2, windowSize: 4, digest };
  for (const patch of [{ totalBytes: LIMITS.maxMessageBytes + 1 }, { chunkBytes: 0 }, { chunkBytes: LIMITS.maxChunkBytes + 1 }, { chunkCount: 65536 }, { chunkCount: 3 }, { windowSize: 9 }, { digest: digest.toUpperCase() }, { transferId: 'AA'.repeat(8) }]) assert.throws(() => encodePacket({ ...start, ...patch }));
  for (const index of [21, 22, 23]) { const bad = encodePacket(start); bad[index] = 1; assert.throws(() => decodePacket(resign(bad)), /START/); }
  const badCount = encodePacket(start); new DataView(badCount.buffer).setUint16(18, 0); assert.throws(() => decodePacket(resign(badCount)), /chunkCount/);
});

test('maximum DATA frame fills QR-L capacity and larger frame rejects', () => {
  const data = diagnosticPayload(LIMITS.maxChunkBytes, 19);
  const packet = { transferId, kind: 'data', index: 65534, data };
  const wire = encodePacket(packet);
  assert.equal(wire.length, 2953);
  assert.deepEqual(decodePacket(wire), packet);
  assert.throws(() => encodePacket({ ...packet, data: new Uint8Array(LIMITS.maxChunkBytes + 1) }), /data bytes/);
  const start = { transferId, kind: 'start', totalBytes: LIMITS.maxChunkBytes * 2 + 1,
    chunkBytes: LIMITS.maxChunkBytes, chunkCount: 3, windowSize: 8, digest };
  assert.deepEqual(decodePacket(encodePacket(start)), start);
});

test('decoder handles sliced buffers and owns DATA bytes', () => {
  const encoded = encodePacket({ transferId, kind: 'data', index: 4, data: Uint8Array.of(1, 2, 3) });
  const backing = new Uint8Array(encoded.length + 8); backing.set(encoded, 3);
  const decoded = decodePacket(backing.subarray(3, 3 + encoded.length)); backing.fill(0);
  assert.deepEqual(decoded.data, Uint8Array.of(1, 2, 3));
});

test('diagnostic payloads are deterministic and bounded', () => {
  assert.deepEqual(diagnosticPayload(1024, 1234), diagnosticPayload(1024, 1234));
  assert.notDeepEqual(diagnosticPayload(1024, 1234), diagnosticPayload(1024, 1235));
  assert.equal(diagnosticPayload(0).length, 0);
  for (const length of [-1, 1.1, 8388609, NaN]) assert.throws(() => diagnosticPayload(length));
  assert.throws(() => diagnosticPayload(1, -1));
  const random = seededRandom(3); for (let i = 0; i < 100; i++) { const value = random(); assert.ok(value >= 0 && value < 1); }
});
