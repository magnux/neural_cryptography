import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {Worker} from 'node:worker_threads';
import {encodeQr, renderQr, decodeQr, fitDecodeDimensions} from '../dist/js/qr.js';
import {QrDecoder} from '../dist/js/qr-client.js';
import {QR_BYTE_CAPACITIES} from '../dist/js/profiles.js';

const binary = Uint8Array.from({length: 256}, (_, i) => i);
const imageFor = (bytes = binary, ecc = 'Q') => renderQr(encodeQr(bytes, {ecc}), {scale: 5});

for (const ecc of ['L', 'M', 'Q', 'H']) {
  test(`all 256 byte values survive standard QR Byte mode at ECC ${ecc}`, () => {
    const image = imageFor(binary, ecc);
    const result = decodeQr(image);
    assert.deepEqual(result?.bytes, binary);
    assert.equal(result.version, encodeQr(binary, {ecc}).version);
    assert.ok(result.location.topLeftCorner.x > 0);
  });
}

for (const [ecc, capacity] of Object.entries(QR_BYTE_CAPACITIES)) {
  test(`maximum ${capacity}-byte ECC ${ecc} symbol round trips; one extra byte rejects`, () => {
    const bytes = Uint8Array.from({ length: capacity }, (_, i) => (i * 73 + (i >> 3)) & 255);
    const code = encodeQr(bytes, { ecc });
    assert.equal(code.version, 40);
    assert.equal(code.size, 177);
    assert.equal(code.ecc, ecc);
    assert.deepEqual(decodeQr(renderQr(code, { scale: 3 }))?.bytes, bytes);
    assert.throws(() => encodeQr(new Uint8Array(capacity + 1), { ecc }), /capacity/);
  });
}

test('small acknowledgment uses version 3 Q; rendering has an unbroken quiet zone', () => {
  const code = encodeQr(new Uint8Array(28));
  assert.equal(code.version, 3);
  assert.equal(code.size, 29);
  const image = renderQr(code, {scale: 3});
  assert.equal(image.width, (29 + 8) * 3);
  for (let y = 0; y < image.height; y++) for (let x = 0; x < image.width; x++) {
    if (x < 12 || y < 12 || x >= image.width - 12 || y >= image.height - 12) {
      assert.equal(image.data[(y * image.width + x) * 4], 255);
    }
  }
});

test('integer module rendering decodes at two pixels per module', () => {
  const bytes = binary.slice(0, 100);
  assert.deepEqual(decodeQr(renderQr(encodeQr(bytes), {scale: 2}))?.bytes, bytes);
});

test('moderate contrast loss, additive noise, and softening recover bytes', () => {
  const bytes = binary.slice(0, 100);
  const image = imageFor(bytes);
  const original = image.data.slice();
  let seed = 1234;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  for (let y = 1; y < image.height - 1; y++) for (let x = 1; x < image.width - 1; x++) {
    let value = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      value += original[((y + dy) * image.width + x + dx) * 4];
    }
    value = 25 + (value / 9) * 0.78 + (random() - 0.5) * 24;
    const i = (y * image.width + x) * 4;
    image.data[i] = image.data[i + 1] = image.data[i + 2] = value;
  }
  assert.deepEqual(decodeQr(image)?.bytes, bytes);
});

test('mild projective distortion and rotation remain decodable', () => {
  const bytes = binary.slice(0, 100);
  const image = imageFor(bytes);
  const [a,b,c,d,e,f,g,h,i] = [1,0.1,32,0.04,0.98,32,0.0003,0.00015,1];
  // The inverse homography maps destination pixels back to the rendered symbol.
  const inverse = [e*i-f*h,c*h-b*i,b*f-c*e,f*g-d*i,a*i-c*g,c*d-a*f,d*h-e*g,b*g-a*h,a*e-b*d];
  const width = image.width + 64;
  const data = new Uint8ClampedArray(width * width * 4).fill(255);
  for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
    const denominator = inverse[6] * x + inverse[7] * y + inverse[8];
    const u = Math.round((inverse[0]*x + inverse[1]*y + inverse[2]) / denominator);
    const v = Math.round((inverse[3]*x + inverse[4]*y + inverse[5]) / denominator);
    if (u < 0 || v < 0 || u >= image.width || v >= image.height) continue;
    const value = image.data[(v * image.width + u) * 4];
    const offset = ((width - 1 - x) * width + y) * 4;
    data[offset] = data[offset + 1] = data[offset + 2] = value;
  }
  assert.deepEqual(decodeQr({width, height: width, data})?.bytes, bytes);
});

test('blank images and malformed or oversized input are bounded', () => {
  assert.equal(decodeQr({width: 100, height: 100, data: new Uint8ClampedArray(40_000).fill(255)}), null);
  assert.throws(() => encodeQr(new Uint8Array(3000)), /capacity/);
  assert.throws(() => encodeQr(new Uint8Array(150), {maxVersion: 3}));
  assert.throws(() => encodeQr(new Uint8Array(1), {ecc: '__proto__'}), /error correction/);
  assert.throws(() => renderQr(encodeQr(new Uint8Array(1)), {quiet: 3}), /quiet zone/);
  assert.throws(() => renderQr(encodeQr(new Uint8Array(1)), {scale: 1.5}), /module scale/);
  assert.throws(() => decodeQr({width: 4000, height: 3000, data: new Uint8ClampedArray(0)}), /megapixels/);
  assert.throws(() => decodeQr({width: 10, height: 10, data: new Uint8ClampedArray(10)}), /length/);
  assert.deepEqual(fitDecodeDimensions(3840, 2160), {width: 1920, height: 1080});
});

test('bundled QR dependency files match their pinned manifest hashes', async () => {
  const root = new URL('../dist/vendor/', import.meta.url);
  const manifest = JSON.parse(await readFile(new URL('manifest.json', root), 'utf8'));
  for (const dependency of manifest.dependencies) {
    const bytes = await readFile(new URL(dependency.file, root));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), dependency.sha256);
  }
});

class NodeWorkerAdapter {
  constructor() { this.worker = new Worker(new URL('./qr-worker-harness.mjs', import.meta.url)); }
  postMessage(message, transfer) { this.worker.postMessage(message, transfer); }
  terminate() { this.worker.terminate(); }
  addEventListener(type, handler) {
    this.worker.on(type, value => handler(type === 'message' ? {data: value} : value));
  }
}

test('real worker decodes binary bytes, preserves caller input, and rejects queued frames', async () => {
  const decoder = new QrDecoder({workerFactory: () => new NodeWorkerAdapter()});
  try {
    const image = imageFor();
    const pending = decoder.decode(image);
    assert.equal(decoder.busy, true);
    await assert.rejects(decoder.decode(image), /busy/);
    assert.deepEqual((await pending).bytes, binary);
    assert.equal(decoder.busy, false);
    assert.equal(image.data.length, image.width * image.height * 4);
    assert.deepEqual((await decoder.decode(image)).bytes, binary);
  } finally { decoder.close(); }
  await assert.rejects(decoder.decode(imageFor()), /closed/);
});

test('worker timeout clears the pending frame and allows a fresh worker', async () => {
  let created = 0;
  let terminated = 0;
  const decoder = new QrDecoder({timeoutMs: 10, workerFactory: () => {
    created++;
    return {addEventListener() {}, postMessage() {}, terminate() { terminated++; }};
  }});
  try {
    await assert.rejects(decoder.decode(imageFor()), /timed out/);
    assert.equal(decoder.busy, false);
    await assert.rejects(decoder.decode(imageFor()), /timed out/);
    assert.equal(created, 2);
    assert.equal(terminated, 2);
  } finally { decoder.close(); }
});
