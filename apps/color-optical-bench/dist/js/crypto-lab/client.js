/** Main-thread client: no key material, computation, or persistence here. */
const TYPES = new Set(['createKey', 'newChallenge', 'prove', 'verify', 'tamper', 'benchmark', 'selfTest']);
const MAX_PROOF_BYTES = 8 * 1024 * 1024;
function errorWithName(name, message) { const error = new Error(message); error.name = name; return error; }

// Copy only the visible bytes of a typed-array view, never its potentially huge
// backing buffer. Bound total copied bytes before worker serialization. Exact
// operation/public-key/challenge schemas are checked again inside the worker.
function snapshotPayload(value, depth = 0, budget = { nodes: 0, bytes: 0 }) {
  if (++budget.nodes > 64 || depth > 4) throw new TypeError('Crypto request is too complex');
  if (value instanceof Uint8Array) {
    budget.bytes += value.byteLength;
    if (budget.bytes > MAX_PROOF_BYTES) throw new RangeError('Crypto request bytes exceed 8 MiB');
    return new Uint8Array(value);
  }
  if (typeof value === 'string') { if (value.length > 2048) throw new RangeError('Crypto request text is too long'); return value; }
  if (Number.isSafeInteger(value)) return value;
  if (value === null || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError('Crypto request must contain plain records, strings, integers, and bytes');
  const keys = Reflect.ownKeys(value);
  if (keys.length > 8) throw new TypeError('Too many crypto request fields');
  const snapshot = Object.create(null);
  for (const key of keys) {
    const field = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || key.length > 64 || !field.enumerable || !Object.hasOwn(field, 'value')) throw new TypeError('Invalid crypto request field');
    snapshot[key] = snapshotPayload(field.value, depth + 1, budget);
  }
  return snapshot;
}

export class CryptoClient {
  #worker = null;
  #pending = null;
  #serial = 0;
  #onInvalidate;
  constructor({ onInvalidate } = {}) {
    if (onInvalidate !== undefined && typeof onInvalidate !== 'function') throw new TypeError('onInvalidate must be a function');
    this.#onInvalidate = onInvalidate;
  }
  get busy() { return this.#pending !== null; }
  get alive() { return this.#worker !== null; }

  #stop(error) {
    const worker = this.#worker;
    this.#worker = null;
    if (worker) {
      worker.onmessage = worker.onerror = worker.onmessageerror = null;
      worker.terminate();
    }
    const pending = this.#pending;
    this.#pending = null;
    if (pending) { clearTimeout(pending.timer); pending.reject(error); }
    if (worker) {
      try { this.#onInvalidate?.(error); } catch { /* State listeners cannot block worker cleanup. */ }
    }
  }
  cancel() { this.#stop(errorWithName('AbortError', 'Operation cancelled; the ephemeral key was discarded')); }
  destroy() { this.#stop(errorWithName('AbortError', 'Crypto worker closed; the ephemeral key was discarded')); }

  request(type, payload = {}, { onProgress, timeoutMs } = {}) {
    return new Promise((resolve, reject) => {
      if (!TYPES.has(type)) { reject(new TypeError('Unsupported Crypto Lab operation')); return; }
      if (this.busy) { reject(errorWithName('InvalidStateError', 'The crypto worker is busy')); return; }
      const maxTimeout = type === 'benchmark' ? 300000 : 120000;
      const timeout = timeoutMs ?? maxTimeout;
      if (!Number.isInteger(timeout) || timeout < 1 || timeout > maxTimeout) { reject(new RangeError(`timeoutMs must be 1..${maxTimeout}`)); return; }
      if (onProgress !== undefined && typeof onProgress !== 'function') { reject(new TypeError('onProgress must be a function')); return; }
      let snapshot;
      try { snapshot = snapshotPayload(payload); }
      catch (error) { reject(error); return; }
      try {
        if (!this.#worker) {
          const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module', name: 'neural-crypto-lab' });
          this.#worker = worker;
          worker.onmessage = ({ data }) => {
            // Worker identity and monotonically increasing job IDs both protect
            // a recreated worker from any late event from its predecessor.
            const pending = this.#pending;
            if (this.#worker !== worker || !pending || data?.id !== pending.id) return;
            if (data.kind === 'progress') {
              try { pending.onProgress?.(data.progress); } catch { /* UI listeners do not alter proof jobs. */ }
            } else if (data.kind === 'result' || data.kind === 'error') {
              clearTimeout(pending.timer); this.#pending = null;
              if (data.kind === 'result') pending.resolve(data.result);
              else pending.reject(errorWithName(data.error?.name || 'Error', data.error?.message || 'Crypto operation failed'));
            }
          };
          worker.onerror = event => {
            if (this.#worker !== worker) return;
            event.preventDefault?.();
            this.#stop(errorWithName('WorkerError', 'Crypto worker failed; the ephemeral key was discarded'));
          };
          worker.onmessageerror = () => {
            if (this.#worker === worker) this.#stop(errorWithName('WorkerError', 'Crypto worker message failed; the ephemeral key was discarded'));
          };
        }
        const id = ++this.#serial;
        const timer = setTimeout(() => {
          if (this.#pending?.id === id) this.#stop(errorWithName('TimeoutError', 'Crypto operation timed out; the ephemeral key was discarded'));
        }, timeout);
        this.#pending = { id, resolve, reject, onProgress, timer };
        // Transfer a CLONE. The proof selected in the main UI remains available.
        const transfer = snapshot.proof instanceof Uint8Array ? [snapshot.proof.buffer] : [];
        this.#worker.postMessage({ id, type, payload: snapshot }, transfer);
      } catch (error) {
        if (this.#pending) this.#stop(error);
        else { this.#worker?.terminate(); this.#worker = null; reject(error); }
      }
    });
  }
}
