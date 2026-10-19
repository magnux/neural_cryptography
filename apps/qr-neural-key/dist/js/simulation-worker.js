import { runSimulation } from './simulation.js';

let running = false;
self.onmessage = async ({ data }) => {
  const { type, id, options } = data ?? {};
  if (type !== 'run') return;
  if (running) { self.postMessage({ type: 'error', id, error: 'A simulation is already running' }); return; }
  running = true;
  try {
    const report = await runSimulation(options, {
      onProgress: report => self.postMessage({ type: 'progress', id, report }),
    });
    self.postMessage({ type: 'result', id, report });
  } catch (error) {
    self.postMessage({ type: 'error', id, error: String(error?.message ?? error).slice(0, 512) });
  } finally { running = false; }
};
