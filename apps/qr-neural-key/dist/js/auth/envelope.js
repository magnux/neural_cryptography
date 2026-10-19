/** One bounded application format carried as ordinary QR transport bytes. */
import { PROTOCOL, canonicalJSON, exactKeys, fromHex, sameBytes } from '../crypto/bytes.js';
import { validateChallenge } from '../crypto/proof.js';

const MAGIC = Uint8Array.of(78, 81, 65, 49); // NQA1
const PREFIX_BYTES = 8;
export const MAX_AUTH_BYTES = 524288;
export const MAX_AUTH_HEADER_BYTES = 4096;
export const AUTH_PROTOCOL = 'neural-qr-auth-v1';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const EMPTY = new Uint8Array();
const decoder = new TextDecoder('utf-8', { fatal: true });

export function validDeviceId(value) { return typeof value === 'string' && UUID.test(value); }

export function validateAuthPublic(document) {
  if (!exactKeys(document, ['protocol', 'circuit', 'minimum_repetitions', 'anchor_outputs']) || document.protocol !== PROTOCOL ||
      !exactKeys(document.circuit, ['name', 'side', 'rounds', 'key_bits']) ||
      document.circuit.name !== 'binary-rotconv-feistel-v1' || document.circuit.side !== 8 ||
      document.circuit.rounds !== 16 || document.circuit.key_bits !== 256 ||
      !Number.isInteger(document.minimum_repetitions) || document.minimum_repetitions < 1 || document.minimum_repetitions > 512) {
    throw new TypeError('Unsupported authentication public key or circuit');
  }
  fromHex(document.anchor_outputs, 32, 'anchor outputs');
}

function validateHeader(header, bodyLength) {
  const fields = {
    HELLO: ['protocol', 'type', 'sessionId', 'deviceId', 'publicKey'],
    IDENTITY: ['protocol', 'type', 'sessionId', 'deviceId', 'publicKey'],
    CHALLENGE: ['protocol', 'type', 'sessionId', 'binding', 'challenge'],
    PROOF: ['protocol', 'type', 'sessionId', 'binding', 'nonce'],
    RESULT: ['protocol', 'type', 'sessionId', 'binding', 'nonce', 'proofDigest', 'accepted'],
  };
  if (!header || !Object.hasOwn(fields, header.type) || !exactKeys(header, fields[header.type]) || header.protocol !== AUTH_PROTOCOL) {
    throw new TypeError('Invalid authentication envelope fields');
  }
  fromHex(header.sessionId, 16, 'session ID');
  if (header.type === 'HELLO' || header.type === 'IDENTITY') {
    if (!validDeviceId(header.deviceId)) throw new TypeError('Invalid device UUID');
    validateAuthPublic(header.publicKey);
  } else fromHex(header.binding, 32, 'pairing binding');
  if (header.type === 'CHALLENGE') validateChallenge(header.challenge);
  if (header.type === 'PROOF' || header.type === 'RESULT') fromHex(header.nonce, 32, 'challenge nonce');
  if (header.type === 'RESULT') {
    fromHex(header.proofDigest, 32, 'proof digest');
    if (typeof header.accepted !== 'boolean') throw new TypeError('Invalid authentication result');
  }
  if ((header.type === 'PROOF') !== (bodyLength > 0)) throw new TypeError('Only PROOF envelopes contain a nonempty body');
}

export function encodeEnvelope(header, body = EMPTY) {
  if (!(body instanceof Uint8Array) || body.length > MAX_AUTH_BYTES - PREFIX_BYTES) throw new RangeError('Authentication body is too large');
  validateHeader(header, body.length);
  const encoded = canonicalJSON(header);
  if (encoded.length > MAX_AUTH_HEADER_BYTES || PREFIX_BYTES + encoded.length + body.length > MAX_AUTH_BYTES) {
    throw new RangeError('Authentication envelope is too large');
  }
  const output = new Uint8Array(PREFIX_BYTES + encoded.length + body.length);
  output.set(MAGIC);
  new DataView(output.buffer).setUint32(4, encoded.length, false);
  output.set(encoded, PREFIX_BYTES);
  output.set(body, PREFIX_BYTES + encoded.length);
  return output;
}

export function decodeEnvelope(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.length < PREFIX_BYTES || bytes.length > MAX_AUTH_BYTES || !sameBytes(bytes.subarray(0, 4), MAGIC)) {
    throw new TypeError('Invalid authentication envelope');
  }
  const length = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(4, false);
  if (length < 1 || length > MAX_AUTH_HEADER_BYTES || PREFIX_BYTES + length > bytes.length) throw new RangeError('Invalid authentication header length');
  const encoded = bytes.subarray(PREFIX_BYTES, PREFIX_BYTES + length);
  const header = JSON.parse(decoder.decode(encoded));
  validateHeader(header, bytes.length - PREFIX_BYTES - length);
  // One encoding only: rejects duplicate JSON keys, whitespace variants, etc.
  if (!sameBytes(encoded, canonicalJSON(header))) throw new TypeError('Noncanonical authentication header');
  return { header, body: bytes.slice(PREFIX_BYTES + length) };
}
