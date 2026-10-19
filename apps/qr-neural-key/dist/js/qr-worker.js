import {decodeQr} from './qr.js';

self.addEventListener('message', event => {
  const {id, image} = event.data || {};
  if (!Number.isSafeInteger(id) || id < 1) return;
  try {
    const result = decodeQr(image);
    self.postMessage({id, result}, result ? [result.bytes.buffer] : []);
  } catch (error) {
    self.postMessage({id, error: String(error?.message || 'QR decoding failed').slice(0, 256)});
  }
});
