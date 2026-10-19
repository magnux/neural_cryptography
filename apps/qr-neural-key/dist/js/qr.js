import qrcodegen from '../vendor/qrcodegen.js';
import jsQR from '../vendor/jsQR.js';
import { QR_BYTE_CAPACITIES } from './profiles.js';

export const MAX_QR_BYTES = QR_BYTE_CAPACITIES.L;
export const MAX_DECODE_PIXELS = 8_000_000;

function integer(value, min, max, name) {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer from ${min} to ${max}`);
  }
}

/** Standard QR Byte mode; binary bytes are never interpreted as text. */
export function encodeQr(bytes, {ecc = 'Q', minVersion = 1, maxVersion = 40} = {}) {
  if (!(bytes instanceof Uint8Array)) throw new TypeError('QR payload must be Uint8Array');
  if (typeof ecc !== 'string' || !Object.hasOwn(QR_BYTE_CAPACITIES, ecc)) throw new RangeError('QR error correction must be L, M, Q, or H');
  if (bytes.length > QR_BYTE_CAPACITIES[ecc]) throw new RangeError(`QR payload exceeds the ${QR_BYTE_CAPACITIES[ecc]}-byte capacity at ECC ${ecc}`);
  integer(minVersion, 1, 40, 'minimum version');
  integer(maxVersion, minVersion, 40, 'maximum version');
  const levels = {L: qrcodegen.QrCode.Ecc.LOW, M: qrcodegen.QrCode.Ecc.MEDIUM, Q: qrcodegen.QrCode.Ecc.QUARTILE, H: qrcodegen.QrCode.Ecc.HIGH};
  // Disable automatic ECC boosting so the selected correction level is exact.
  const qr = qrcodegen.QrCode.encodeSegments(
    [qrcodegen.QrSegment.makeBytes(bytes)], levels[ecc], minVersion, maxVersion, -1, false,
  );
  const modules = new Uint8Array(qr.size * qr.size);
  for (let y = 0; y < qr.size; y++) for (let x = 0; x < qr.size; x++) {
    modules[y * qr.size + x] = Number(qr.getModule(x, y));
  }
  return {size: qr.size, version: qr.version, ecc, modules};
}

/** Integer pixels per module and a white quiet zone of at least four modules. */
export function renderQr(code, {scale = 8, quiet = 4} = {}) {
  integer(scale, 1, 32, 'module scale');
  integer(quiet, 4, 32, 'quiet zone');
  if (!code || !(code.modules instanceof Uint8Array)) throw new TypeError('Invalid QR code');
  integer(code.size, 21, 177, 'QR module count');
  if (code.modules.length !== code.size * code.size) throw new RangeError('Invalid QR module array');
  const width = (code.size + 2 * quiet) * scale;
  if (width * width > MAX_DECODE_PIXELS) throw new RangeError('Rendered QR exceeds the image size limit');
  const data = new Uint8ClampedArray(width * width * 4);
  data.fill(255);
  for (let y = 0; y < code.size; y++) for (let x = 0; x < code.size; x++) {
    if (!code.modules[y * code.size + x]) continue;
    for (let dy = 0; dy < scale; dy++) for (let dx = 0; dx < scale; dx++) {
      const i = (((y + quiet) * scale + dy) * width + (x + quiet) * scale + dx) * 4;
      data[i] = data[i + 1] = data[i + 2] = 0;
    }
  }
  return {width, height: width, data};
}

export function validateQrImage(image) {
  if (!image || !(image.data instanceof Uint8ClampedArray)) {
    throw new TypeError('QR image must contain RGBA Uint8ClampedArray data');
  }
  integer(image.width, 1, 8192, 'image width');
  integer(image.height, 1, 8192, 'image height');
  if (image.width * image.height > MAX_DECODE_PIXELS) throw new RangeError('QR image exceeds 8 megapixels');
  if (image.data.length !== image.width * image.height * 4) throw new RangeError('Invalid RGBA data length');
}

/** Decoder shared by browser workers and deterministic Node image tests. */
export function decodeQr(image) {
  validateQrImage(image);
  const result = jsQR(image.data, image.width, image.height, {inversionAttempts: 'dontInvert'});
  if (!result || result.binaryData.length > MAX_QR_BYTES) return null;
  return {bytes: Uint8Array.from(result.binaryData), location: result.location, version: result.version};
}

/** Fit camera captures before getImageData; avoids allocating a full sensor frame. */
export function fitDecodeDimensions(width, height, {maxSide = 1920, maxPixels = 2_073_600} = {}) {
  integer(width, 1, 65536, 'source width');
  integer(height, 1, 65536, 'source height');
  integer(maxSide, 1, 8192, 'capture maximum side');
  integer(maxPixels, 1, MAX_DECODE_PIXELS, 'capture maximum pixels');
  const ratio = Math.min(1, maxSide / Math.max(width, height), Math.sqrt(maxPixels / (width * height)));
  return {width: Math.max(1, Math.floor(width * ratio)), height: Math.max(1, Math.floor(height * ratio))};
}
