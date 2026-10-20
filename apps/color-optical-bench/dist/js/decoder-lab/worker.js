import {validateManifest, validateOptions, verifyArtifact, decodeFloat32, compareLogits, accumulateParity, summarizeTimings} from './runtime.js';
import {extractLearnedInput} from '../learned-input.js';

const assetRoot = new URL('../../models/optical-decoder/', import.meta.url);
const vendorRoot = new URL('../../vendor/onnxruntime/', import.meta.url);
let ort, cached = null, busy = false;

async function fetchBounded(url, maximum) {
  const response = await fetch(url, {cache: 'no-cache'});
  if (!response.ok) throw new Error(`Missing decoder asset (${response.status}). Export the model, install its browser runtime, then run npm run prepare:decoder.`);
  if (new URL(response.url).origin !== location.origin) throw new Error('Decoder assets must be served from this app.');
  if (!response.body) throw new Error('This browser cannot stream decoder assets.');
  const reader = response.body.getReader(), chunks = []; let total = 0;
  try {
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      total += value.byteLength;
      if (total > maximum) throw new Error('Decoder asset exceeds its declared size limit.');
      chunks.push(value);
    }
  } catch (error) { await reader.cancel().catch(() => {}); throw error; }
  const result = new Uint8Array(total); let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
  return result.buffer;
}
async function artifact(entry) {
  return verifyArtifact(await fetchBounded(new URL(entry.file, assetRoot), entry.bytes), entry);
}
async function runtime() {
  if (!ort) {
    try { ort = await import(new URL('ort.webgpu.min.mjs', vendorRoot).href); }
    catch { throw new Error('ONNX Runtime assets are missing or cannot load. Run npm run prepare:decoder and serve the prepared app over HTTPS or localhost.'); }
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = vendorRoot.href;
  }
  return ort;
}
async function load(options, progress) {
  const started = performance.now();
  progress('Checking local model and fixture hashes…');
  const manifest = validateManifest(JSON.parse(new TextDecoder().decode(await fetchBounded(new URL('manifest.json', assetRoot), 65536))));
  const entry = manifest.cases.find(c => c.size === options.size);
  const key = `${entry.model.sha256}/${options.backend}`;
  // A changed fixture must never reuse a previous expected-output buffer.
  const fixtureKey = `${entry.input.sha256}/${entry.expected.sha256}`;
  if (cached?.key === key && cached.fixtureKey === fixtureKey) {
    return {...cached, loadMs: performance.now()-started, initMs: 0, cacheReused: true};
  }
  if (cached) { await cached.session.release(); cached = null; }
  const model = await artifact(entry.model), input = decodeFloat32(await artifact(entry.input)), expected = decodeFloat32(await artifact(entry.expected));
  await runtime();
  let effectiveBackend = options.backend === 'auto' ? (navigator.gpu ? 'webgpu' : 'wasm') : options.backend;
  let fallbackReason = options.backend === 'auto' && !navigator.gpu ? 'WebGPU is unavailable in this worker.' : null;
  const initialized = performance.now();
  progress(`Initializing ${effectiveBackend.toUpperCase()}…`);
  let session;
  try {
    session = await ort.InferenceSession.create(model, {executionProviders: [effectiveBackend], graphOptimizationLevel: 'all'});
  } catch (error) {
    if (options.backend !== 'auto' || effectiveBackend !== 'webgpu') throw new Error(`${effectiveBackend.toUpperCase()} initialization failed: ${error.message}`);
    fallbackReason = `WebGPU initialization failed: ${error.message}`;
    effectiveBackend = 'wasm'; progress('WebGPU unavailable; initializing WASM…');
    session = await ort.InferenceSession.create(model, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
  }
  if (session.inputNames.length !== 1 || session.inputNames[0] !== manifest.inputName ||
      session.outputNames.length !== 1 || session.outputNames[0] !== manifest.outputName) {
    await session.release(); throw new Error('Exported tensor names do not match the model contract.');
  }
  const initMs = performance.now()-initialized;
  cached = {key, fixtureKey, session, manifest, entry, input, expected, effectiveBackend, fallbackReason};
  return {...cached, loadMs: performance.now()-started, initMs, cacheReused: false};
}
async function inference(context, pixels) {
  const tensor = new ort.Tensor('float32', pixels, context.entry.inputShape), started = performance.now();
  let outputs;
  try {
    outputs = await context.session.run({[context.manifest.inputName]: tensor});
    const result = outputs[context.manifest.outputName];
    if (!result || result.type !== 'float32' || result.dims.length !== 4 ||
        result.dims.some((n, i) => n !== context.entry.outputShape[i])) throw new Error('Unexpected inference output shape or type.');
    const values = result.data;
    return {values: new Float32Array(values), ms: performance.now()-started};
  } finally {
    tensor.dispose?.();
    for (const output of Object.values(outputs || {})) output.dispose?.();
  }
}
function describe(context, options) {
  return {format: 'optical-decoder-m1-report', version: 1, modelStatus: context.manifest.status, size: options.size,
    requestedBackend: options.backend, effectiveBackend: context.effectiveBackend, fallbackReason: context.fallbackReason,
    runtimeVersions: {...(ort.env.versions || {})}, exportVersions: context.manifest.versions,
    modelSha256: context.entry.model.sha256, modelBytes: context.entry.model.bytes, parameterCount: context.manifest.parameterCount,
    inputShape: context.entry.inputShape, outputShape: context.entry.outputShape, loadMs: context.loadMs, initMs: context.initMs,
    cacheReused: context.cacheReused, wasmThreads: 1, timestamp: new Date().toISOString(),
    memory: {inputTensorBytes: context.entry.input.bytes, outputTensorBytes: context.entry.expected.bytes,
      fixtureBytes: context.entry.input.bytes+context.entry.expected.bytes, modelFileBytes: context.entry.model.bytes,
      note: 'Known buffer sizes only; runtime working memory and GPU allocations are not measured.'}};
}
async function benchmark(options, progress) {
  const context = await load(options, progress);
  progress('Warming up the model (2 runs)…');
  for (let i = 0; i < 2; i++) await inference(context, context.input);
  const times = []; let parity;
  for (let i = 0; i < options.iterations; i++) {
    progress(`Measuring inference ${i+1}/${options.iterations}…`);
    const result = await inference(context, context.input);
    times.push(result.ms);
    const current = compareLogits(result.values, context.expected);
    parity = accumulateParity(parity, current);
  }
  return {...describe(context, options), operation: 'benchmark', warmups: 2, parity, timings: summarizeTimings(times), samplesMs: times,
    timingScope: 'session.run plus CPU output copy; excludes fixture comparison and camera preprocessing'};
}
async function capture(options, progress, image) {
  progress('Validating the monochrome header and rectifying the current exposure…');
  const input = extractLearnedInput(image);
  if (!input.ok) throw new Error(input.error || 'No supported 64/128-cell matrix was found.');
  if (input.info.size !== options.size) throw new Error(`Camera sees ${input.info.size} cells; select that model size first.`);
  const context = await load(options, progress);
  progress('Running the untrained model on this camera crop…');
  const result = await inference(context, input.input);
  if (result.values.some(value => !Number.isFinite(value))) throw new Error('Camera inference produced non-finite logits.');
  return {...describe(context, options), operation: 'capture', inputStats: input.stats, inferenceMs: result.ms,
    paletteColors: input.info.colors, logitsFinite: true,
    note: 'Untrained inference only. No payload was decoded or accepted.'};
}
self.onmessage = async event => {
  const {id, operation, options = {}} = event.data || {};
  if (busy) { self.postMessage({id, type: 'error', error: 'Worker is busy.'}); return; }
  busy = true;
  const progress = message => self.postMessage({id, type: 'progress', message});
  try {
    const validated = validateOptions(options);
    let result;
    if (operation === 'benchmark') result = await benchmark(validated, progress);
    else if (operation === 'capture') result = await capture(validated, progress, options.image);
    else throw new Error('Unknown decoder operation.');
    self.postMessage({id, type: 'result', result});
  } catch (error) { self.postMessage({id, type: 'error', error: error.message || String(error)}); }
  finally { busy = false; }
};
