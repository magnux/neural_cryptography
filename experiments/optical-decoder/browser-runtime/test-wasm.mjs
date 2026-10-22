#!/usr/bin/env node
/** Optional ONNX Runtime Web/WASM parity check in Node; not a browser benchmark. */
import {readFile, writeFile, mkdir, stat} from 'node:fs/promises';
import {resolve, dirname, sep} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {validateManifest, verifyArtifact, decodeFloat32, compareLogits, accumulateParity, summarizeTimings}
  from '../../../apps/color-optical-bench/dist/js/decoder-lab/runtime.js';

function options(args) {
  let artifacts = new URL('../artifacts/smoke/', import.meta.url), out = null;
  for (let i=0; i<args.length; i++) {
    const flag = args[i];
    if (!['--artifacts', '--out'].includes(flag) || !args[i+1] || args[i+1].startsWith('--')) {
      throw new Error('Usage: node test-wasm.mjs [--artifacts DIRECTORY] [--out REPORT.json]');
    }
    const value = resolve(args[++i]);
    if (flag === '--artifacts') artifacts = pathToFileURL(value+sep);
    else out = value;
  }
  return {artifacts, out};
}

async function main() {
  const {artifacts, out} = options(process.argv.slice(2));
  const manifestPath = new URL('manifest.json', artifacts);
  if ((await stat(manifestPath)).size > 65536) throw new Error('Decoder manifest exceeds 64 KiB.');
  const manifest = validateManifest(JSON.parse(await readFile(manifestPath, 'utf8')));
  let ort;
  try { ort = await import('./node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs'); }
  catch { throw new Error('Install the pinned browser runtime first: npm ci --prefix experiments/optical-decoder/browser-runtime'); }
  const runtimeRoot = new URL('./node_modules/onnxruntime-web/dist/', import.meta.url);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = runtimeRoot.href;
  const report = {format: 'optical-decoder-node-wasm-report', version: 1, modelStatus: manifest.status,
    environment: `Node.js ${process.version}`, runtimeVersions: {...ort.env.versions},
    execution: 'ONNX Runtime Web WebGPU bundle, WASM provider, one thread. Not a browser or phone measurement.',
    artifacts: fileURLToPath(artifacts), warmups: 2, measuredRuns: 10, cases: []};
  async function artifact(entry) {
    const url = new URL(entry.file, artifacts);
    if ((await stat(url)).size !== entry.bytes) throw new Error(`Wrong byte length for ${entry.file}.`);
    const bytes = await readFile(url), buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset+bytes.byteLength);
    return verifyArtifact(buffer, entry);
  }
  for (const entry of manifest.cases) {
    const model = await artifact(entry.model);
    const input = decodeFloat32(await artifact(entry.input));
    const expected = decodeFloat32(await artifact(entry.expected));
    const started = performance.now();
    const session = await ort.InferenceSession.create(model, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    const initMs = performance.now()-started;
    const tensor = new ort.Tensor('float32', input, entry.inputShape), samples = [];
    let parity = null, exportToleranceParity = null;
    try {
      if (session.inputNames.length !== 1 || session.inputNames[0] !== manifest.inputName ||
          session.outputNames.length !== 1 || session.outputNames[0] !== manifest.outputName) throw new Error('Export tensor names do not match the manifest.');
      for (let i=0; i<report.warmups+report.measuredRuns; i++) {
        const start = performance.now(), result = await session.run({[manifest.inputName]: tensor});
        try {
          const logits = result[manifest.outputName];
          if (logits.type !== 'float32' || logits.dims.length !== 4 || logits.dims.some((v,k)=>v!==entry.outputShape[k])) throw new Error('Unexpected output type or shape.');
          const values = new Float32Array(logits.data), elapsed = performance.now()-start;
          if (i >= report.warmups) {
            samples.push(elapsed);
            parity = accumulateParity(parity, compareLogits(values, expected));
            exportToleranceParity = accumulateParity(exportToleranceParity, compareLogits(values, expected, manifest.tolerance));
          }
        } finally { for (const value of Object.values(result)) value.dispose?.(); }
      }
    } finally { tensor.dispose(); await session.release(); }
    report.cases.push({size: entry.size, modelSha256: entry.model.sha256, initMs, parity, exportToleranceParity,
      timings: summarizeTimings(samples), timingScope: 'session.run plus CPU output copy; excludes parity comparison'});
  }
  report.passed = report.cases.every(entry=>entry.parity.passed && entry.exportToleranceParity.passed);
  const text = JSON.stringify(report, null, 2)+'\n';
  if (out) { await mkdir(dirname(out), {recursive: true}); await writeFile(out, text); }
  process.stdout.write(text);
  if (!report.passed) process.exitCode = 1;
}
main().catch(error=>{console.error(`Decoder Web/WASM check failed: ${error.message}`);process.exitCode=1;});
