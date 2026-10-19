import {parentPort} from 'node:worker_threads';

const listeners = new Map();
globalThis.self = {
  addEventListener(type, handler) { listeners.set(type, handler); },
  postMessage(message, transfer) { parentPort.postMessage(message, transfer); },
};
await import('../dist/js/qr-worker.js');
parentPort.on('message', data => listeners.get('message')?.({data}));
