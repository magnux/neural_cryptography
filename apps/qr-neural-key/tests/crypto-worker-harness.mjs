// Node bridge for the actual browser module worker. No browser/DOM simulation.
import { parentPort } from 'node:worker_threads';
globalThis.self = { postMessage: (message, transfer) => parentPort.postMessage(message, transfer) };
await import('../dist/js/crypto-worker/worker.js');
parentPort.on('message', data => self.onmessage({ data }));
