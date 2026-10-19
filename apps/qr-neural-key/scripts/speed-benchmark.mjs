/** Compare preset recovery on identical data/fault settings; not a camera speed test. */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { SPEED_PRESETS } from '../dist/js/profiles.js';
import { runSimulation } from '../dist/js/simulation.js';

const reports = [];
for (const preset of SPEED_PRESETS) {
  const report = await runSimulation({
    ...preset, presetId: preset.id, bytes: 16384, seed: 1234,
    controlEcc: 'Q', dropRate: 0.15, duplicateRate: 0.15, corruptRate: 0.05,
  });
  assert.equal(report.status, 'delivered', `${preset.id}: ${report.sender.error ?? report.error}`);
  assert.equal(report.exact, true, `${preset.id}: byte mismatch`);
  assert.equal(report.confirmationReceived, true, `${preset.id}: no final ACK`);
  assert.equal(report.stats.deliveries, 1, `${preset.id}: repeated delivery`);
  reports.push({ preset: preset.id, ...report });
  console.log(`${preset.label}: 16,384 bytes exact, ${report.stats.transmissions} packets, ` +
    `${report.stats.dropped} dropped, ${report.stats.corrupted} damaged, ` +
    `${(report.elapsedMs / 1000).toFixed(2)} s compute / ${(report.simulatedMs / 1000).toFixed(2)} s modeled`);
}
const output = new URL('../local-results/qr-speed-presets.json', import.meta.url);
await mkdir(new URL('../local-results/', import.meta.url), { recursive: true });
await writeFile(output, JSON.stringify({
  schema: 'qr-speed-validation-v1', generatedAt: new Date().toISOString(), runtime: process.version,
  note: 'Actual QR image encoding/decoding with synthetic packet faults. Preset names and modeled timings do not predict device performance; the simulator charges a dwell slot to control packets as well as data.',
  reports,
}, null, 2) + '\n');
console.log(`Saved ${output.pathname}`);
