/** Live transfer targets, not guarantees of camera decoding speed. */
import { DATA_OVERHEAD } from './protocol.js';

// Maximum binary Byte-mode payloads for a version 40 QR at each correction level.
export const QR_BYTE_CAPACITIES = Object.freeze({ L: 2953, M: 2331, Q: 1663, H: 1273 });
export const DEFAULT_PRESET = 'slow';
export const SPEED_PRESETS = Object.freeze([
  { id: 'slow', label: 'Slow', description: 'Small codes; wait for each acknowledgment.', chunkBytes: 128, ecc: 'Q', windowSize: 1, dwellMs: 350, retryMs: 2500 },
  { id: 'medium', label: 'Medium', description: 'Larger codes; wait for each acknowledgment.', chunkBytes: 512, ecc: 'Q', windowSize: 1, dwellMs: 250, retryMs: 2000 },
  { id: 'fast', label: 'Fast', description: 'Four-packet windows with medium correction.', chunkBytes: 1024, ecc: 'M', windowSize: 4, dwellMs: 200, retryMs: 2000 },
  { id: 'faster', label: 'Faster', description: 'Large codes and eight-packet windows.', chunkBytes: 2048, ecc: 'M', windowSize: 8, dwellMs: 150, retryMs: 2000 },
  { id: 'fastest', label: 'Fastest', description: 'Maximum QR capacity; needs excellent alignment and resolution.', chunkBytes: 2935, ecc: 'L', windowSize: 8, dwellMs: 100, retryMs: 1500 },
].map(Object.freeze));

export function maxChunkBytes(ecc) {
  if (typeof ecc !== 'string' || !Object.hasOwn(QR_BYTE_CAPACITIES, ecc)) throw new RangeError('QR error correction must be L, M, Q, or H');
  return QR_BYTE_CAPACITIES[ecc] - DATA_OVERHEAD;
}

function integer(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer from ${min} to ${max}`);
  }
  return value;
}

/** Validate live controls, preserving caller metadata and non-profile settings. */
export function validateProfile(profile = {}) {
  if (!profile || typeof profile !== 'object' || Array.isArray(profile)) throw new TypeError('Profile must be an object');
  const defaults = SPEED_PRESETS[0];
  const ecc = profile.ecc ?? defaults.ecc;
  if (profile.controlEcc !== undefined) maxChunkBytes(profile.controlEcc);
  const chunkBytes = integer(profile.chunkBytes ?? defaults.chunkBytes, 1, maxChunkBytes(ecc), `chunkBytes at ECC ${ecc}`);
  const windowSize = integer(profile.windowSize ?? defaults.windowSize, 1, 8, 'windowSize');
  const dwellMs = integer(profile.dwellMs ?? defaults.dwellMs, 50, 2000, 'dwellMs');
  const retryMs = integer(profile.retryMs ?? defaults.retryMs, Math.max(500, dwellMs), 10000, 'retryMs');
  return { ...profile, chunkBytes, ecc, windowSize, dwellMs, retryMs };
}
