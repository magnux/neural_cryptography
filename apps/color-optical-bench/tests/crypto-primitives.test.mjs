import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import {
  PROTOCOL, assertBytes, concatBytes, framed, packBits, unpackBits, hex, fromHex,
  u16, u32, sameBytes, exactKeys, canonicalJSON,
} from '../dist/js/crypto/bytes.js';
import { sha256, hash, shake256, randomBytes } from '../dist/js/crypto/hashes.js';

const ascii = text => new TextEncoder().encode(text);
const text = bytes => new TextDecoder().decode(bytes);

test('SHA-256 standard vectors, framed Python fixture, and multipart separation', async () => {
  assert.equal(PROTOCOL, 'NeuralPicnic-POC-MPC3-v2');
  assert.equal(hex(await sha256(ascii(''))), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
  assert.equal(hex(await sha256(ascii('abc'))), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  const parts = [ascii('abc'), new Uint8Array(), Uint8Array.of(0, 128, 255)];
  // Generated independently by Python struct.pack('>I', ...) + hashlib.sha256.
  assert.equal(hex(framed('fixture', ...parts)), '000000184e657572616c5069636e69632d504f432d4d5043332d763200000007666978747572650000000361626300000000000000030080ff');
  assert.equal(hex(await hash('fixture', ...parts)), '7d12c29cd7462bda2c7a025714e68cf4ba06260195e01177223567f544233cf3');
  assert.notDeepEqual(await hash('fixture', ascii('ab'), ascii('c')), await hash('fixture', ascii('a'), ascii('bc')));
  assert.notDeepEqual(await hash('one', ascii('abc')), await hash('two', ascii('abc')));
  assert.throws(() => framed('é', new Uint8Array()), /ASCII/);
  assert.throws(() => framed('x', [1, 2]), /Uint8Array/);
  await assert.rejects(sha256('abc'), /Uint8Array/);
});

test('SHAKE-256 matches Python hashlib across absorb and squeeze block boundaries', () => {
  assert.equal(hex(shake256(ascii(''), 32)), '46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f');
  assert.equal(hex(shake256(ascii('abc'), 64)), '483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739d5a15bef186a5386c75744c0527e1faa9f8726e462a12a4feb06bd8801e751e4');
  const input = Uint8Array.from({ length: 256 }, (_, index) => index);
  const expected = '336c8aa7f2b08bda6bd7402cd2ea89760b7728a8b31802b80524756361165366ff8159f2f4568a2bfa286db6387895629938c2868a6421c37f988455763a75e4b9259e0a939aaa68295119ccea72c9f0ca7d048aa70eeeb4534c6bd08ecc6163217c790f33b84a89623f8e5538b734967e9490a48b7d0658afb4565364e8b234dfe6a2bceb12ce2130eec00bf2113615a276819d7815f5891d07600275f4d8fbc87b056f44bc2b141ca5ed9e4cb6e9a7bf71f520971dca1c8da6140e2af31faef5502e84991a2d9e';
  assert.equal(hex(shake256(input, 200)), expected);
  assert.deepEqual(input, Uint8Array.from({ length: 256 }, (_, index) => index));
  assert.deepEqual(shake256(input, 0), new Uint8Array());
  assert.equal(hex(shake256(input, 32)), expected.slice(0, 64));
  for (const length of [-1, 1.5, NaN, Infinity, 8 * 1024 * 1024 + 1, '32']) {
    assert.throws(() => shake256(input, length), /length/);
  }
  assert.throws(() => shake256('abc', 32), /Uint8Array/);
});

test('integer framing, byte validation and canonical hexadecimal reject ambiguous inputs', () => {
  assert.equal(hex(u16(65535)), 'ffff');
  assert.equal(hex(u32(0x89abcdef)), '89abcdef');
  assert.equal(hex(u32(0xffffffff)), 'ffffffff');
  for (const value of [-1, 1.5, NaN, Infinity, true, '1']) {
    assert.throws(() => u16(value), /uint16/);
    assert.throws(() => u32(value), /uint32/);
  }
  assert.throws(() => u16(65536));
  assert.throws(() => u32(2 ** 32));
  const bytes = Uint8Array.of(0, 15, 255);
  assert.strictEqual(assertBytes(bytes, 3), bytes);
  assert.throws(() => assertBytes(bytes, 2), /exactly 2/);
  assert.throws(() => assertBytes(bytes.buffer), /Uint8Array/);
  assert.throws(() => assertBytes(new Uint16Array(3)), /Uint8Array/);
  assert.equal(hex(bytes), '000fff');
  assert.deepEqual(fromHex('000fff', 3), bytes);
  for (const invalid of ['000FFF', ' 000fff', '000fff ', '0x00ff', '000ffg', '000ff', '000fffff', null]) {
    assert.throws(() => fromHex(invalid, 3, 'test field'), /test field/);
  }
  assert.deepEqual(fromHex('', 0), new Uint8Array());
  assert.throws(() => fromHex('', -1));
  assert.equal(sameBytes(bytes, fromHex('000fff', 3)), true);
  assert.equal(sameBytes(bytes, fromHex('000ffe', 3)), false);
  assert.equal(sameBytes(bytes, new Uint8Array()), false);
  assert.throws(() => sameBytes(bytes, [0, 15, 255]), /Uint8Array/);
  const combined = concatBytes(bytes, Uint8Array.of(3));
  combined[0] = 8;
  assert.equal(bytes[0], 0);
  assert.deepEqual(concatBytes(), new Uint8Array());
});

test('bit packing is MSB-first, zero-pads partial bytes and rejects non-bits', () => {
  const bytes = Uint8Array.from({ length: 256 }, (_, i) => i);
  assert.deepEqual(packBits(unpackBits(bytes)), bytes);
  assert.deepEqual(unpackBits(Uint8Array.of(0xa1)), Uint8Array.of(1, 0, 1, 0, 0, 0, 0, 1));
  assert.equal(hex(packBits(Uint8Array.of(1, 0, 1, 1, 0, 0, 0, 1, 1))), 'b180');
  assert.deepEqual(packBits(new Uint8Array()), new Uint8Array());
  assert.throws(() => packBits(Uint8Array.of(0, 2)), /0 or 1/);
  assert.throws(() => packBits([0, 1]), /Uint8Array/);
});

test('canonical JSON matches Python sorted ensure_ascii encoding including Unicode', () => {
  const object = {
    z: [true, false, null, 42, -3],
    'é': 'quote " slash / \\ \b\f\n\r\t\x00\x1f\x7f\x80é𝄞\ud800',
    '😀': 1, '\uffff': 2, a: { 2: 2, 10: 10 },
  };
  const expected = String.raw`{"a":{"10":10,"2":2},"z":[true,false,null,42,-3],"\u00e9":"quote \" slash / \\ \b\f\n\r\t\u0000\u001f\u007f\u0080\u00e9\ud834\udd1e\ud800","\uffff":2,"\ud83d\ude00":1}`;
  assert.equal(text(canonicalJSON(object)), expected);
  assert.ok(canonicalJSON(object).every(byte => byte < 128));
  assert.equal(text(canonicalJSON({ b: { c: 1, a: 2 }, a: 'x' })), '{"a":"x","b":{"a":2,"c":1}}');
  assert.equal(text(canonicalJSON(Object.assign(Object.create(null), { a: 1 }))), '{"a":1}');
  assert.equal(text(canonicalJSON('\udfff')), '"\\udfff"');
});

test('canonical JSON and schema keys reject values without unambiguous protocol encodings', () => {
  const cyclic = {}; cyclic.self = cyclic;
  const accessor = Object.defineProperty({}, 'a', { enumerable: true, get() { throw new Error('must not run'); } });
  for (const value of [undefined, 1.5, Infinity, NaN, 2 ** 53, 1n, Symbol('x'), new Date(), new Uint8Array(1), cyclic, [undefined], new Array(2), accessor]) {
    assert.throws(() => canonicalJSON(value));
  }
  assert.equal(exactKeys({ a: 1, b: 2 }, ['b', 'a']), true);
  for (const obj of [null, [], 'a', new Date(), Object.create({ a: 1 }), { a: 1, b: 2 }, accessor]) {
    assert.equal(exactKeys(obj, ['a']), false);
  }
  assert.equal(exactKeys({ a: 1 }, ['a', 'a']), false);
  assert.equal(exactKeys({ a: 1, [Symbol('a')]: 2 }, ['a']), false);
  assert.throws(() => canonicalJSON({ [Symbol('a')]: 2 }));
  assert.throws(() => canonicalJSON(Object.defineProperty({}, 'a', { value: 1 })), /enumerable/);
});

test('secure RNG returns fresh bytes, chunks Web Crypto requests, and has no fallback', async () => {
  const first = randomBytes(64), second = randomBytes(64);
  assert.equal(first.length, 64);
  assert.notDeepEqual(first, second);
  assert.deepEqual(randomBytes(0), new Uint8Array());
  for (const invalid of [-1, 0.5, NaN, Infinity, 8 * 1024 * 1024 + 1, '32']) assert.throws(() => randomBytes(invalid), /length/);
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  try {
    const calls = [];
    Object.defineProperty(globalThis, 'crypto', { configurable: true, value: { getRandomValues(bytes) {
      calls.push(bytes.length); bytes.fill(calls.length); return bytes;
    } } });
    const large = randomBytes(131073);
    assert.deepEqual(calls, [65536, 65536, 1]);
    assert.equal(large[0], 1); assert.equal(large[65536], 2); assert.equal(large[131072], 3);
    Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined });
    assert.throws(() => randomBytes(32), /secure randomness/);
    await assert.rejects(sha256(new Uint8Array()), /Web Crypto/);
  } finally { Object.defineProperty(globalThis, 'crypto', descriptor); }
});

test('vendored SHAKE dependency files retain upstream checksums and local-only imports', async () => {
  const expected = {
    // The provenance README records SHA-256s from the integrity-checked npm tarball.
  };
  const base = new URL('../dist/js/crypto/vendor/noble-hashes/', import.meta.url);
  const readme = await readFile(new URL('README.md', base), 'utf8');
  for (const match of readme.matchAll(/^([0-9a-f]{64})  (.+)$/gm)) expected[match[2]] = match[1];
  assert.deepEqual(Object.keys(expected).sort(), ['LICENSE', '_u64.js', 'sha3.js', 'utils.js']);
  for (const [name, digest] of Object.entries(expected)) {
    const bytes = await readFile(new URL(name, base));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), digest, name);
    if (name.endsWith('.js')) {
      for (const match of bytes.toString().matchAll(/^import .*? from ["']([^"']+)["']/gm)) assert.match(match[1], /^\.\//);
      assert.doesNotMatch(bytes.toString(), /\b(?:fetch|XMLHttpRequest|require)\s*\(/);
    }
  }
});
