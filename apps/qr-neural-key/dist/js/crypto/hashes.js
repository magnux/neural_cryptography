import { assertBytes, framed } from './bytes.js';
import { shake256 as nobleShake256 } from './vendor/noble-hashes/sha3.js';

export async function sha256(bytes) {
  assertBytes(bytes);
  if (!globalThis.crypto?.subtle) throw new Error('Web Crypto is required; use a secure browser context');
  return new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes));
}

export async function hash(label, ...parts) {
  return sha256(framed(label, ...parts));
}

export function shake256(bytes, length) {
  assertBytes(bytes);
  if (!Number.isSafeInteger(length) || length < 0 || length > 8 * 1024 * 1024) {
    throw new RangeError('SHAKE output length must be an integer from 0 to 8 MiB');
  }
  return nobleShake256(bytes, { dkLen: length });
}

// The proof format is bounded at 8 MiB. Do not silently provide a weak RNG fallback.
export function randomBytes(length) {
  if (!Number.isSafeInteger(length) || length < 0 || length > 8 * 1024 * 1024) {
    throw new RangeError('random byte length must be an integer from 0 to 8 MiB');
  }
  if (!globalThis.crypto?.getRandomValues) throw new Error('Web Crypto secure randomness is required');
  const bytes = new Uint8Array(length);
  for (let offset = 0; offset < length; offset += 65536) {
    globalThis.crypto.getRandomValues(bytes.subarray(offset, Math.min(offset + 65536, length)));
  }
  return bytes;
}
