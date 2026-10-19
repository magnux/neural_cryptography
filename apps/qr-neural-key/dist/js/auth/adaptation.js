/** Local outgoing QR targets. The peer independently chooses its own targets. */
import { SPEED_PRESETS, validateProfile } from '../profiles.js';

export const PHONE_PROFILE = Object.freeze(validateProfile({ chunkBytes: 128, ecc: 'Q', windowSize: 1, dwellMs: 350, retryMs: 2500 }));
export const DESKTOP_PROFILE = Object.freeze(validateProfile(SPEED_PRESETS.find(profile => profile.id === 'fast')));
export const CHUNK_LADDER = Object.freeze([64, 128, 256, 512, 1024, 2048, 2935]);
export const MAX_AUTO_RESTARTS = 6;
export const CLEAN_WINDOWS_TO_GROW = 8;

export function smallerChunk(chunkBytes) {
  return [...CHUNK_LADDER].reverse().find(value => value < chunkBytes) ?? chunkBytes;
}

export function largerChunk(chunkBytes, ceiling) {
  return CHUNK_LADDER.find(value => value > chunkBytes && value <= ceiling) ?? chunkBytes;
}

export function validateAckHold(value) {
  if (!Number.isSafeInteger(value) || value < 250 || value > 5000) throw new RangeError('ACK hold must be an integer from 250 to 5000 ms');
  return value;
}
