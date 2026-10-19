// Exact byte conventions shared with neural_zkboo_experiment.py.
export const PROTOCOL = 'NeuralPicnic-POC-MPC3-v2';
const encoder = new TextEncoder();

export function assertBytes(bytes, length, name = 'bytes') {
  if (!(bytes instanceof Uint8Array)) throw new TypeError(`${name} must be a Uint8Array`);
  if (length !== undefined && (!Number.isSafeInteger(length) || length < 0 || bytes.length !== length)) {
    throw new RangeError(`${name} must contain exactly ${length} bytes`);
  }
  return bytes;
}

export function concatBytes(...parts) {
  let length = 0;
  for (const part of parts) {
    assertBytes(part);
    length += part.length;
    if (!Number.isSafeInteger(length) || length > 0xffffffff) throw new RangeError('byte sequence is too long');
  }
  const result = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) { result.set(part, offset); offset += part.length; }
  return result;
}

function unsigned(n, bytes) {
  if (!Number.isSafeInteger(n) || n < 0 || n >= 2 ** (8 * bytes)) throw new RangeError(`expected uint${bytes * 8}`);
  const result = new Uint8Array(bytes);
  const view = new DataView(result.buffer);
  if (bytes === 4) view.setUint32(0, n, false);
  else view.setUint16(0, n, false);
  return result;
}
export const u32 = n => unsigned(n, 4);
export const u16 = n => unsigned(n, 2);

export function framed(label, ...parts) {
  if (typeof label !== 'string' || /[^\x00-\x7f]/.test(label)) throw new TypeError('framing label must be ASCII');
  const fields = [encoder.encode(PROTOCOL), encoder.encode(label), ...parts];
  const output = [];
  for (const field of fields) { assertBytes(field); output.push(u32(field.length), field); }
  return concatBytes(...output);
}

export function packBits(bits) {
  assertBytes(bits, undefined, 'bits');
  const result = new Uint8Array(Math.ceil(bits.length / 8));
  for (let i = 0; i < bits.length; i++) {
    if (bits[i] !== 0 && bits[i] !== 1) throw new RangeError('bits must contain only 0 or 1');
    result[i >> 3] |= bits[i] << (7 - (i & 7));
  }
  return result;
}

export function unpackBits(bytes) {
  assertBytes(bytes);
  const result = new Uint8Array(bytes.length * 8);
  for (let i = 0; i < result.length; i++) result[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
  return result;
}

export function hex(bytes) {
  assertBytes(bytes);
  let result = '';
  for (const byte of bytes) result += byte.toString(16).padStart(2, '0');
  return result;
}

export function fromHex(text, count, name = 'hexadecimal value') {
  if (!Number.isSafeInteger(count) || count < 0) throw new RangeError('hexadecimal byte count must be a nonnegative integer');
  if (typeof text !== 'string' || text.length !== count * 2 || !/^[0-9a-f]*$/.test(text)) {
    throw new TypeError(`${name} must encode exactly ${count} bytes as canonical lowercase hexadecimal`);
  }
  const result = new Uint8Array(count);
  for (let i = 0; i < count; i++) result[i] = Number.parseInt(text.slice(i * 2, i * 2 + 2), 16);
  return result;
}

// No early exit for differing byte contents. JS does not promise constant-time execution.
export function sameBytes(a, b) {
  assertBytes(a); assertBytes(b);
  if (a.length !== b.length) return false;
  let difference = 0;
  for (let i = 0; i < a.length; i++) difference |= a[i] ^ b[i];
  return difference === 0;
}

function isRecord(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

export function exactKeys(obj, keys) {
  if (!isRecord(obj) || !Array.isArray(keys) || keys.some(key => typeof key !== 'string')) return false;
  const names = Reflect.ownKeys(obj);
  if (names.length !== keys.length || new Set(keys).size !== keys.length) return false;
  return names.every(key => typeof key === 'string' && keys.includes(key) &&
    Object.hasOwn(Object.getOwnPropertyDescriptor(obj, key), 'value'));
}

// Python orders Unicode strings by code point, whereas JS .sort() orders UTF-16 units.
function unicodeOrder(a, b) {
  const x = Array.from(a, char => char.codePointAt(0));
  const y = Array.from(b, char => char.codePointAt(0));
  for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] - y[i];
  return x.length - y.length;
}

function asciiJSONString(text) {
  // Python ensure_ascii=True also escapes DEL (0x7f), and uses lowercase hex.
  return JSON.stringify(text).replace(/[\x7f-\uffff]/g, char =>
    `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`);
}

/** Python sorted, compact, ensure_ascii JSON for the protocol's integer-only schemas.
 * Reject floats and unsafe integers instead of inventing cross-language rounding rules.
 * Lone surrogate strings are escaped losslessly here; UTF-8 challenge validation is separate.
 */
export function canonicalJSON(value) {
  const active = new Set();
  function encode(item, depth) {
    if (depth > 256) throw new RangeError('canonical JSON nesting is too deep');
    if (item === null) return 'null';
    if (typeof item === 'boolean') return item ? 'true' : 'false';
    if (typeof item === 'string') return asciiJSONString(item);
    if (typeof item === 'number') {
      if (!Number.isSafeInteger(item)) throw new TypeError('canonical JSON numbers must be safe integers');
      return String(item);
    }
    if (typeof item !== 'object') throw new TypeError('unsupported canonical JSON value');
    if (active.has(item)) throw new TypeError('canonical JSON cannot contain cycles');
    active.add(item);
    try {
      if (Array.isArray(item)) {
        const keys = Reflect.ownKeys(item);
        if (keys.length !== item.length + 1 || keys.some(key => key !== 'length' &&
          (typeof key !== 'string' || !/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= item.length))) {
          throw new TypeError('canonical JSON arrays must be dense and have no extra properties');
        }
        const values = [];
        for (let i = 0; i < item.length; i++) {
          const descriptor = Object.getOwnPropertyDescriptor(item, String(i));
          if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError('canonical JSON does not evaluate accessors');
          values.push(encode(descriptor.value, depth + 1));
        }
        return `[${values.join(',')}]`;
      }
      if (!isRecord(item)) throw new TypeError('canonical JSON objects must be plain records');
      const keys = Reflect.ownKeys(item);
      if (keys.some(key => typeof key !== 'string')) throw new TypeError('canonical JSON keys must be strings');
      return `{${keys.sort(unicodeOrder).map(key => {
        const descriptor = Object.getOwnPropertyDescriptor(item, key);
        if (!descriptor.enumerable || !Object.hasOwn(descriptor, 'value')) throw new TypeError('canonical JSON requires enumerable data properties');
        return `${asciiJSONString(key)}:${encode(descriptor.value, depth + 1)}`;
      }).join(',')}}`;
    } finally { active.delete(item); }
  }
  return encoder.encode(encode(value, 0));
}
