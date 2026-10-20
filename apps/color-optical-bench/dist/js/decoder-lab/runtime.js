/** Pure M1 contract checks and measurements, shared by preparation and browser tests. */
export const MODEL_FORMAT = 'optical-decoder-m1';
export const MAX_MODEL_BYTES = 4 * 1024 * 1024;
export const PARITY_TOLERANCE = Object.freeze({atol: 1e-4, rtol: 1e-3});
const shapeEqual = (a, b) => Array.isArray(a) && a.length === b.length && a.every((v, i) => v === b[i]);
const plain = value => value && typeof value === 'object' && !Array.isArray(value);

export function validateManifest(manifest) {
  if (!plain(manifest) || manifest.format !== MODEL_FORMAT || manifest.version !== 1 ||
      manifest.status !== 'untrained' || manifest.parameterCount !== 121792 || manifest.pixelsPerCell !== 4 ||
      manifest.channels !== 3 || manifest.classes !== 16 || manifest.inputName !== 'pixels' || manifest.outputName !== 'logits' ||
      !Number.isSafeInteger(manifest.seed) || !Number.isSafeInteger(manifest.opset) || manifest.opset < 13 || manifest.opset > 25 ||
      !plain(manifest.versions) || !Array.isArray(manifest.cases) || manifest.cases.length !== 2) {
    throw new Error('Unsupported decoder manifest. Regenerate the current M1 untrained model and fixtures.');
  }
  const sizes = new Set(), files = new Set();
  for (const entry of manifest.cases) {
    if (!plain(entry) || ![64, 128].includes(entry.size) || sizes.has(entry.size)) throw new Error('Invalid decoder grid cases.');
    sizes.add(entry.size);
    const g = entry.size;
    if (!shapeEqual(entry.inputShape, [1, 3, 4*g, 4*g]) || !shapeEqual(entry.outputShape, [1, 16, g, g])) throw new Error('Invalid decoder tensor shapes.');
    for (const [name, bytes] of [['model', null], ['input', 3*16*g*g*4], ['expected', 16*g*g*4]]) {
      const artifact = entry[name];
      if (!plain(artifact) || typeof artifact.file !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$/.test(artifact.file) ||
          artifact.file.includes('..') || files.has(artifact.file) || !/^[a-f0-9]{64}$/.test(artifact.sha256) ||
          !Number.isSafeInteger(artifact.bytes) || artifact.bytes < 1 || artifact.bytes > MAX_MODEL_BYTES ||
          (bytes !== null && artifact.bytes !== bytes)) throw new Error(`Invalid ${name} artifact in decoder manifest.`);
      files.add(artifact.file);
    }
  }
  return manifest;
}

export function validateOptions({size = 64, backend = 'auto', iterations = 10} = {}) {
  if (![64, 128].includes(size)) throw new Error('Choose a 64 or 128 cell grid.');
  if (!['auto', 'wasm', 'webgpu'].includes(backend)) throw new Error('Unknown inference backend.');
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 30) throw new Error('Benchmark runs must be 1–30.');
  return {size, backend, iterations};
}

export function decodeFloat32(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength % 4 !== 0) throw new Error('Invalid float32 artifact.');
  const values = new Float32Array(buffer.byteLength / 4), view = new DataView(buffer);
  for (let i = 0; i < values.length; i++) {
    values[i] = view.getFloat32(i*4, true);
    if (!Number.isFinite(values[i])) throw new Error('Non-finite value in numerical fixture.');
  }
  return values;
}

export async function verifyArtifact(buffer, entry) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== entry.bytes) throw new Error(`Wrong byte length for ${entry.file}.`);
  const hash = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', buffer));
  const hex = Array.from(hash, value => value.toString(16).padStart(2, '0')).join('');
  if (hex !== entry.sha256) throw new Error(`SHA-256 mismatch for ${entry.file}. Regenerate and prepare decoder assets together.`);
  return buffer;
}

export function compareLogits(actual, expected, {atol, rtol} = PARITY_TOLERANCE) {
  if (!actual || !expected || actual.length !== expected.length || !actual.length ||
      !Number.isFinite(atol) || atol < 0 || !Number.isFinite(rtol) || rtol < 0) throw new Error('Invalid parity comparison.');
  let failures = 0, maxAbsoluteError = 0, sumAbsoluteError = 0, nonFinite = 0;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) { failures++; nonFinite++; continue; }
    const error = Math.abs(actual[i]-expected[i]);
    maxAbsoluteError = Math.max(maxAbsoluteError, error); sumAbsoluteError += error;
    if (error > atol + rtol*Math.abs(expected[i])) failures++;
  }
  return {passed: failures === 0, checked: actual.length, failures, nonFinite, atol, rtol,
    maxAbsoluteError: nonFinite ? null : maxAbsoluteError,
    meanAbsoluteError: nonFinite ? null : sumAbsoluteError / actual.length};
}

export function summarizeTimings(values) {
  if (!Array.isArray(values) || !values.length || values.some(v => !Number.isFinite(v) || v < 0)) throw new Error('Invalid timing samples.');
  const sorted = [...values].sort((a, b) => a-b);
  const percentile = p => sorted[Math.min(sorted.length-1, Math.ceil(p*sorted.length)-1)];
  return {count: values.length, p50Ms: percentile(.5), p95Ms: percentile(.95), minMs: sorted[0], maxMs: sorted.at(-1),
    meanMs: values.reduce((a, b) => a+b, 0)/values.length};
}

/** Keep failure sticky across runs, even when a passing run has a larger absolute error. */
export function accumulateParity(previous, current) {
  if (!previous) return {...current, runsChecked: 1, failedRuns: current.passed ? 0 : 1};
  if (previous.checked !== current.checked || previous.atol !== current.atol || previous.rtol !== current.rtol) throw new Error('Incompatible parity checks.');
  return {...current, passed: previous.passed && current.passed,
    failures: Math.max(previous.failures, current.failures), nonFinite: Math.max(previous.nonFinite, current.nonFinite),
    maxAbsoluteError: previous.maxAbsoluteError === null || current.maxAbsoluteError === null ? null : Math.max(previous.maxAbsoluteError, current.maxAbsoluteError),
    meanAbsoluteError: previous.meanAbsoluteError === null || current.meanAbsoluteError === null ? null : Math.max(previous.meanAbsoluteError, current.meanAbsoluteError),
    runsChecked: previous.runsChecked+1, failedRuns: previous.failedRuns+(current.passed ? 0 : 1)};
}
