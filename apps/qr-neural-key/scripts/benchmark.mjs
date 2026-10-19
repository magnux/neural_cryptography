/** Reproducible software-channel checks, including the complete 165 KB proof-sized payload. */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { runSimulation } from '../dist/js/simulation.js';

const cases = [
  ['default-1024', { bytes: 1024 }],
  ['window4-4096', { bytes: 4096, windowSize: 4, seed: 12 }],
  ['proof-sized-165000', { bytes: 165000 }],
];
const reports = [];
for (const [name, options] of cases) {
  const report = await runSimulation(options);
  assert.equal(report.status, 'delivered', `${name}: ${report.sender.error ?? report.error}`);
  assert.equal(report.exact, true, `${name}: byte mismatch`);
  assert.equal(report.confirmationReceived, true, `${name}: no return-path confirmation`);
  assert.equal(report.stats.deliveries, 1, `${name}: repeated delivery`);
  assert.equal(report.expectedDigest, report.receivedDigest, `${name}: digest mismatch`);
  reports.push({ name, ...report });
  console.log(`${name}: ${report.payloadBytes} bytes exact, ${report.stats.transmissions} QR transmissions, ${report.stats.dropped} dropped, ${report.stats.corrupted} damaged, ${(report.elapsedMs / 1000).toFixed(2)} s wall / ${(report.simulatedMs / 60000).toFixed(2)} min simulated`);
}
const output = new URL('../local-results/qr-simulation.json', import.meta.url);
await mkdir(new URL('../local-results/', import.meta.url), { recursive: true });
await writeFile(output, JSON.stringify({
  schema: 'qr-simulation-validation-v1', generatedAt: new Date().toISOString(), runtime: process.version,
  note: 'Real standard QR encode/render/decode on synthetic images. Packet-fault model only; these timings do not predict camera throughput or moire tolerance.',
  reports,
}, null, 2) + '\n');
console.log(`Saved ${output.pathname}`);
