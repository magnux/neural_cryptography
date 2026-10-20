/** One request at a time. Cancellation kills the worker and releases its model. */
export class DecoderClient {
  constructor({workerFactory = () => new Worker(new URL('./worker.js', import.meta.url), {type: 'module'}), onProgress = () => {}} = {}) {
    this.workerFactory = workerFactory; this.onProgress = onProgress;
    this.worker = null; this.pending = null; this.serial = 0;
  }
  call(operation, options = {}, transfer = []) {
    if (this.pending) return Promise.reject(new Error('A decoder operation is already running.'));
    if (!this.worker) {
      this.worker = this.workerFactory();
      const worker = this.worker;
      worker.onmessage = event => {
        if (worker !== this.worker || !this.pending || event.data?.id !== this.pending.id) return;
        const data = event.data;
        if (data.type === 'progress') { this.onProgress(data.message); return; }
        const pending = this.pending; this.pending = null; clearTimeout(pending.timeout);
        if (data.type === 'result') pending.resolve(data.result);
        else pending.reject(new Error(data.error || 'Decoder worker failed.'));
      };
      worker.onerror = event => { if (worker === this.worker) this.cancel(new Error(event.message || 'Decoder worker failed.')); };
      worker.onmessageerror = () => { if (worker === this.worker) this.cancel(new Error('Decoder worker returned unreadable data.')); };
    }
    const id = ++this.serial;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => this.cancel(new Error('Decoder operation exceeded two minutes. Try WASM or the smaller grid.')), 120000);
      this.pending = {id, resolve, reject, timeout};
      try { this.worker.postMessage({id, operation, options}, transfer); }
      catch (error) { this.cancel(error); }
    });
  }
  cancel(error = new DOMException('Decoder operation canceled.', 'AbortError')) {
    const pending = this.pending; this.pending = null;
    if (this.worker) this.worker.terminate(); this.worker = null;
    if (pending) { clearTimeout(pending.timeout); pending.reject(error); }
  }
}
