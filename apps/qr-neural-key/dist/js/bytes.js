/** Byte utilities. CRC detects damage; SHA-256 integrity is not authentication. */
export function bytesToHex(bytes) { return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join(''); }
export function hexToBytes(hex, size) {
  if (typeof hex !== 'string' || hex.length !== size * 2 || !/^[0-9a-f]+$/.test(hex)) throw Error(`Expected ${size}-byte lowercase hex value`);
  return Uint8Array.from(hex.match(/../g), s => parseInt(s, 16));
}
export function randomId() { return bytesToHex(crypto.getRandomValues(new Uint8Array(8))); }
export async function sha256(bytes) {
  if (!(bytes instanceof Uint8Array)) throw Error('Expected Uint8Array');
  return bytesToHex(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)));
}
export function equalBytes(a, b) { return a.length === b.length && a.every((value, i) => value === b[i]); }
const TABLE = Uint32Array.from({ length: 256 }, (_, i) => {
  for (let j = 0; j < 8; j++) i = (i >>> 1) ^ ((i & 1) ? 0xedb88320 : 0);
  return i >>> 0;
});
export function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = TABLE[(crc ^ byte) & 255] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}
