import test from 'node:test';
import assert from 'node:assert/strict';
import { QR_BYTE_CAPACITIES, SPEED_PRESETS, DEFAULT_PRESET, maxChunkBytes, validateProfile } from '../dist/js/profiles.js';
import { DATA_OVERHEAD, LIMITS } from '../dist/js/protocol.js';

test('speed profiles have unique names and respect each QR correction capacity', () => {
  assert.equal(DEFAULT_PRESET, 'slow');
  assert.deepEqual(SPEED_PRESETS.map(p => p.id), ['slow', 'medium', 'fast', 'faster', 'fastest']);
  for (const preset of SPEED_PRESETS) {
    assert.deepEqual(validateProfile(preset), preset);
    assert.ok(preset.chunkBytes + DATA_OVERHEAD <= QR_BYTE_CAPACITIES[preset.ecc]);
    assert.equal(Object.isFrozen(preset), true);
  }
  assert.equal(SPEED_PRESETS.at(-1).chunkBytes, LIMITS.maxChunkBytes);
  assert.equal(LIMITS.maxChunkBytes + DATA_OVERHEAD, QR_BYTE_CAPACITIES.L);
  assert.deepEqual(Object.fromEntries(Object.keys(QR_BYTE_CAPACITIES).map(ecc => [ecc, maxChunkBytes(ecc)])),
    { L: 2935, M: 2313, Q: 1645, H: 1255 });
});

test('default live profile preserves report metadata and independent control correction', () => {
  assert.deepEqual(validateProfile(), { chunkBytes: 128, ecc: 'Q', windowSize: 1, dwellMs: 350, retryMs: 2500 });
  const extra = { presetId: 'custom', timeoutMs: 3600000, maxRetries: 20, controlEcc: 'H' };
  assert.deepEqual(validateProfile(extra), { ...extra, ...validateProfile() });
  assert.equal(validateProfile({ ecc: 'L', chunkBytes: 2935 }).chunkBytes, 2935);
  assert.equal(validateProfile({ ecc: 'Q', chunkBytes: 1645 }).chunkBytes, 1645);
});

test('invalid live profiles and impossible ECC/chunk combinations reject early', () => {
  for (const ecc of ['X', '__proto__', 'toString', 0, new String('L')]) assert.throws(() => maxChunkBytes(ecc), /error correction/);
  for (const profile of [null, [], { chunkBytes: 0 }, { chunkBytes: 2936, ecc: 'L' }, { chunkBytes: 1646, ecc: 'Q' },
    { chunkBytes: 1256, ecc: 'H' }, { chunkBytes: 2314, ecc: 'M' }, { chunkBytes: 10.1 }, { windowSize: 0 },
    { windowSize: 9 }, { dwellMs: 49 }, { dwellMs: 2001 }, { retryMs: 499 }, { retryMs: 10001 },
    { dwellMs: 1500, retryMs: 1000 }, { controlEcc: 'X' }]) assert.throws(() => validateProfile(profile));
});
