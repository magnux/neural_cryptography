import {validateQrImage} from './qr.js';

/** One decode in flight, with no queued camera frames. UI owns capture resizing. */
export class QrDecoder {
  constructor({workerFactory, timeoutMs = 8000} = {}) {
    if (!Number.isFinite(timeoutMs) || timeoutMs < 10 || timeoutMs > 60000) throw new RangeError('Invalid decode timeout');
    this.workerFactory = workerFactory || (() => new Worker(new URL('./qr-worker.js', import.meta.url), {type: 'module'}));
    this.timeoutMs = timeoutMs;
    this.worker = null;
    this.pending = null;
    this.nextId = 0;
    this.closed = false;
  }

  get busy() { return this.pending !== null; }

  _reset(error) {
    this.worker?.terminate();
    this.worker = null;
    if (this.pending) {
      clearTimeout(this.pending.timer);
      const {reject} = this.pending;
      this.pending = null;
      reject(error);
    }
  }

  _createWorker() {
    if (this.worker) return;
    const worker = this.workerFactory();
    this.worker = worker;
    worker.addEventListener('message', ({data}) => {
      if (this.worker !== worker || !this.pending || data?.id !== this.pending.id) return;
      const {resolve, reject, timer} = this.pending;
      clearTimeout(timer);
      this.pending = null;
      if (data.error) reject(new Error(data.error));
      else resolve(data.result);
    });
    worker.addEventListener('error', event => {
      if (this.worker === worker) this._reset(new Error(event.message || 'QR decoder worker failed'));
    });
    worker.addEventListener('messageerror', () => {
      if (this.worker === worker) this._reset(new Error('QR decoder worker returned an unreadable message'));
    });
  }

  async decode(image) {
    if (this.closed) throw new Error('QR decoder is closed');
    if (this.busy) throw new Error('QR decoder is busy; skip this camera frame');
    validateQrImage(image);
    this._createWorker();
    const id = ++this.nextId;
    // Transfer an owned copy so the caller's ImageData remains usable.
    const data = new Uint8ClampedArray(image.data);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this._reset(new Error('QR decoding timed out')), this.timeoutMs);
      this.pending = {id, resolve, reject, timer};
      try {
        this.worker.postMessage({id, image: {width: image.width, height: image.height, data}}, [data.buffer]);
      } catch (error) {
        this._reset(error);
      }
    });
  }

  close() {
    this.closed = true;
    this._reset(new Error('QR decoder closed'));
  }
}
