/** Current corrector contract, shared by preparation and browser tests. */
export const MODEL_FORMAT = 'neural-corrector';
export const MAX_MODEL_BYTES = 4 * 1024 * 1024;
export const MAX_INPUT_FRAMES = 64;
export const MAX_INPUT_BYTES = 3 * MAX_INPUT_FRAMES * 512 * 512 * 4;
export const validFrameCount = frames => Number.isInteger(frames) && frames >= 1 && frames <= MAX_INPUT_FRAMES;
export const PARITY_TOLERANCE = Object.freeze({atol: 1e-4, rtol: 1e-3});
const shapeEqual = (a, b) => Array.isArray(a) && a.length === b.length && a.every((v, i) => v === b[i]);
export const MODEL_PROFILES = Object.freeze({
  'U-tube':Object.freeze({base:140112,perFrame:864}),
  'U-tube-2x':Object.freeze({base:275472,perFrame:864}),
  'U-tube-2x-attn':Object.freeze({base:432848,perFrame:0})
});
const plain = value => value && typeof value === 'object' && !Array.isArray(value);

export function validateManifest(manifest) {
  if (!plain(manifest) || manifest.format !== MODEL_FORMAT || manifest.version !== 1 ||
      !['untrained', 'trained'].includes(manifest.status) || !validFrameCount(manifest.framesPerInput) || manifest.pixelsPerCell !== 4 ||
      manifest.classes !== 16 || manifest.inputName !== 'pixels' || manifest.outputName !== 'logits' ||
      !Number.isSafeInteger(manifest.seed) || !Number.isSafeInteger(manifest.opset) || manifest.opset < 13 || manifest.opset > 25 ||
      !plain(manifest.versions) || !Array.isArray(manifest.cases)) {
    throw new Error('Unsupported corrector manifest. Regenerate the current model and fixtures.');
  }
  if (!plain(manifest.tolerance) || Object.keys(manifest.tolerance).length !== 2 ||
      manifest.tolerance.atol !== PARITY_TOLERANCE.atol || manifest.tolerance.rtol !== PARITY_TOLERANCE.rtol) {
    throw new Error('Invalid corrector parity tolerance. Regenerate and prepare the current model bundle.');
  }
  const profile=manifest.modelProfile;
  if (!Object.hasOwn(MODEL_PROFILES,profile)||MODEL_PROFILES[profile].base+MODEL_PROFILES[profile].perFrame*manifest.framesPerInput!==manifest.parameterCount) throw new Error('Invalid corrector model profile/count.');
  const attention=profile==='U-tube-2x-attn',frames=manifest.framesPerInput;
  if (manifest.inputLayout!==(attention?'phase-paired-rgb-presence-planes':'stacked-rgb') ||
      manifest.channels!==(attention?32:3*frames) || attention&&frames!==8 ||
      manifest.minFramesPerInput!==(attention?4:frames) ||
      !shapeEqual(manifest.frameCounts,attention?[4,5,6,7,8]:[frames]) ||
      manifest.outputLabelSpace!==(attention?'logical':'physical') ||
      !shapeEqual(manifest.phaseMasks,attention?[0,15,7,8]:[]) ||
      manifest.cases.length!==(attention?1:2)) throw Error('Invalid corrector input layout/frame policy.');
  // Preserve calibrated overshoot in captures/fixtures; the exported graph
  // clips to the common real/synthetic range before its first learned layer.
  if (!plain(manifest.normalization) || manifest.normalization.divisor !== 127.5 ||
      manifest.normalization.offset !== -1 || !shapeEqual(manifest.normalization.clip, [-1.5, 1.5]) ||
      !shapeEqual(manifest.networkInputClip, [-1, 1])) {
    throw new Error('Invalid corrector normalization/clip contract. Regenerate the current model and fixtures.');
  }
  if (manifest.status === 'trained' && (!/^[a-f0-9]{64}$/.test(manifest.checkpointSha256) || !plain(manifest.training))) {
    throw new Error('Trained exports require checkpoint provenance.');
  }
  const sizes = new Set(), files = new Set();
  for (const entry of manifest.cases) {
    if (!plain(entry) || !(attention?[64]:[64, 128]).includes(entry.size) || sizes.has(entry.size)) throw new Error('Invalid corrector grid cases.');
    sizes.add(entry.size);
    const g = entry.size;
    if (!manifest.frameCounts.includes(entry.framesAvailable)) throw Error('Invalid fixture observation count.');
    if (!shapeEqual(entry.inputShape, [1, manifest.channels, 4*g, 4*g]) || !shapeEqual(entry.outputShape, [1, 16, g, g])) throw new Error('Invalid corrector tensor shapes.');
    for (const [name, bytes] of [['model', null], ['input', manifest.channels*16*g*g*4], ['expected', 16*g*g*4]]) {
      const artifact = entry[name];
      if (!plain(artifact) || typeof artifact.file !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$/.test(artifact.file) ||
          artifact.file.includes('..') || files.has(artifact.file) || !/^[a-f0-9]{64}$/.test(artifact.sha256) ||
          !Number.isSafeInteger(artifact.bytes) || artifact.bytes < 1 || artifact.bytes > (name==='model'?MAX_MODEL_BYTES:MAX_INPUT_BYTES) ||
          (bytes !== null && artifact.bytes !== bytes)) throw new Error(`Invalid ${name} artifact in corrector manifest.`);
      files.add(artifact.file);
    }
  }
  return manifest;
}

export function validateOptions({size = 64, backend = 'auto', iterations = 10, captureFrames = 0} = {}) {
  if (![64, 128].includes(size)) throw new Error('Choose a 64 or 128 cell grid.');
  if (!['auto', 'wasm', 'webgpu'].includes(backend)) throw new Error('Unknown inference backend.');
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 30) throw new Error('Benchmark runs must be 1–30.');
  if (![0,4,5,6,7,8].includes(captureFrames)) throw Error('Choose model-default camera exposures, or 4–8 for a phase-grouped model.');
  return {size, backend, iterations, captureFrames};
}

/** Inspect the exact current tensor layout, including an explicit absence mask.
 * RGB zero means mid-gray and is never interpreted as a missing observation.
 */
export function inputPresence(pixels, manifest, size) {
  const plane=(4*size)**2,frames=manifest.framesPerInput;
  if (!(pixels instanceof Float32Array)||pixels.length!==manifest.channels*plane||
      pixels.some(value=>!Number.isFinite(value))) throw Error('Input does not match the prepared frame-stack contract.');
  if (manifest.inputLayout==='stacked-rgb') return Array.from({length:frames},(_,i)=>i);
  if (manifest.inputLayout!=='phase-paired-rgb-presence-planes'||frames!==8||size!==64||manifest.channels!==32)
    throw Error('Unsupported packed corrector input.');
  const active=[];
  for(let slot=0;slot<frames;slot++) {
    const start=(3*frames+slot)*plane,flag=pixels[start];
    if (flag!==0&&flag!==1) throw Error('Presence planes must contain binary flags.');
    for(let i=start+1;i<start+plane;i++)if(pixels[i]!==flag)throw Error('Presence flags must be spatially constant.');
    if(flag===1)active.push(slot);
  }
  if(!manifest.frameCounts.includes(active.length))throw Error('Input has an unsupported number of available captures.');
  if([0,1,2,3].some(phase=>!active.some(slot=>Math.floor(slot/2)===phase)))throw Error('Phase-grouped inputs require at least one capture of every phase.');
  return active;
}

/** Pack genuine camera observations into their declared phase pairs.
 * The phase ID is public header metadata, never inferred from RGB or targets.
 */
export function packFrameStack(stack, manifest, size, placement) {
  const plane=(4*size)**2,one=3*plane,capacity=manifest.framesPerInput;
  if(!(stack instanceof Float32Array)||!stack.length||stack.length%one||
      stack.some(value=>!Number.isFinite(value)))throw Error('Invalid RGB frame stack.');
  const available=stack.length/one;
  if(!manifest.frameCounts.includes(available))throw Error('Prepared model does not support this capture count.');
  if(manifest.inputLayout==='stacked-rgb') {
    if(available!==capacity||placement!==undefined)throw Error('Fixed RGB models require their complete ordered frame stack.');
    return stack;
  }
  if(manifest.inputLayout!=='phase-paired-rgb-presence-planes'||capacity!==8||size!==64)throw Error('Unsupported packed corrector input.');
  if(!plain(placement)||!Array.isArray(placement.phases)||placement.phases.length!==available||
      placement.phases.some(phase=>!Number.isInteger(phase)||phase<0||phase>3))throw Error('Phase-grouped packing requires a phase ID for every capture.');
  const counts=[0,0,0,0],defaults=placement.phases.map(phase=>2*phase+counts[phase]++);
  if(counts.some(count=>count<1||count>2))throw Error('Phase-grouped packing requires one or two captures of every phase.');
  const slots=placement.slots??defaults;
  if(!Array.isArray(slots)||slots.length!==available||new Set(slots).size!==available||
      slots.some((slot,i)=>!Number.isInteger(slot)||slot<0||slot>=capacity||Math.floor(slot/2)!==placement.phases[i]))throw Error('Invalid available-frame phase slots.');
  const pixels=new Float32Array(manifest.channels*plane);
  slots.forEach((slot,index)=>{
    pixels.set(stack.subarray(index*one,(index+1)*one),slot*one);
    pixels.fill(1,(3*capacity+slot)*plane,(3*capacity+slot+1)*plane);
  });
  return pixels;
}

export function clipNetworkInput(pixels, manifest, size) {
  inputPresence(pixels,manifest,size);
  const output=pixels.slice(),end=3*manifest.framesPerInput*(4*size)**2;
  const [low,high]=manifest.networkInputClip;
  for(let i=0;i<end;i++)output[i]=Math.max(low,Math.min(high,output[i]));
  return output;
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
  if (hex !== entry.sha256) throw new Error(`SHA-256 mismatch for ${entry.file}. Regenerate and prepare corrector assets together.`);
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
