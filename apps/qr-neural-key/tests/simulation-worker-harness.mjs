// Run the actual browser worker module with Node's message-port adapter.
import { parentPort } from 'node:worker_threads';
globalThis.self = { postMessage: message => parentPort.postMessage(message) };
await import('../dist/js/simulation-worker.js');
parentPort.on('message', data => self.onmessage({ data }));
