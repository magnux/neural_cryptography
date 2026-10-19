/** Module-worker boundary. The secret never leaves CryptoOperations. */
import { CryptoOperations } from './operations.js';
import { exactKeys } from '../crypto/bytes.js';

const operations = new CryptoOperations();
let active = false;
self.onmessage = async ({ data }) => {
  if (!exactKeys(data, ['id', 'type', 'payload']) || !Number.isSafeInteger(data.id) || data.id < 1 || typeof data.type !== 'string') return;
  const { id, type, payload } = data;
  if (active) {
    self.postMessage({ id, kind: 'error', error: { name: 'InvalidStateError', message: 'The crypto worker is busy' } });
    return;
  }
  active = true;
  try {
    let lastProgressAt = -Infinity, lastPhase = '';
    const result = await operations.run(type, payload, progress => {
      const now = performance.now();
      // Keep progress useful on phones without flooding the UI with every
      // repetition. Always publish phase boundaries and completion.
      if (progress.phase !== lastPhase || progress.completed === 0 ||
          progress.completed === progress.total || now - lastProgressAt >= 50) {
        self.postMessage({ id, kind: 'progress', progress });
        lastProgressAt = now; lastPhase = progress.phase;
      }
    });
    const transfer = [];
    if (result.proof instanceof Uint8Array) transfer.push(result.proof.buffer);
    if (result.latest?.proof instanceof Uint8Array) transfer.push(result.latest.proof.buffer);
    self.postMessage({ id, kind: 'result', result }, transfer);
  } catch (error) {
    self.postMessage({ id, kind: 'error', error: { name: error?.name || 'Error', message: error?.message || 'Crypto operation failed' } });
  } finally { active = false; }
};
