import {validateManifest, validateOptions, verifyArtifact, decodeFloat32, compareLogits, accumulateParity, summarizeTimings, inputPresence, packFrameStack} from './runtime.js';
import {extractLearnedInput} from '../learned-input.js';
import {CameraBurst} from './burst.js';
import {TransportCapture} from '../neural-auth/capture.js';
import {LivePhaseBurst,requireTransportModel,isTransportData,recoverCorrectedPayload} from './live-burst.js';

const assetRoot = new URL('../../models/neural-corrector/', import.meta.url);
const vendorRoot = new URL('../../vendor/onnxruntime/', import.meta.url);
let ort, cached = null, busy = false, cameraBurst=null;
let transportContext=null,transportMode=null;
const liveBurst=new LivePhaseBurst();
const transportCapture=new TransportCapture();

async function fetchBounded(url, maximum) {
  const response = await fetch(url, {cache: 'no-cache'});
  if (!response.ok) throw new Error(`Missing corrector asset (${response.status}). Export the model, install its browser runtime, then run npm run prepare:corrector.`);
  if (new URL(response.url).origin !== location.origin) throw new Error('Corrector assets must be served from this app.');
  if (!response.body) throw new Error('This browser cannot stream corrector assets.');
  const reader = response.body.getReader(), chunks = []; let total = 0;
  try {
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      total += value.byteLength;
      if (total > maximum) throw new Error('Corrector asset exceeds its declared size limit.');
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
    catch { throw new Error('ONNX Runtime assets are missing or cannot load. Run npm run prepare:corrector and serve the prepared app over HTTPS or localhost.'); }
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = vendorRoot.href;
  }
  return ort;
}
async function load(options, progress, {transport=false}={}) {
  const started = performance.now();
  progress('Checking local model and fixture hashes…');
  const manifest = validateManifest(JSON.parse(new TextDecoder().decode(await fetchBounded(new URL('manifest.json', assetRoot), 65536))));
  if(transport)requireTransportModel(manifest);
  const entry = manifest.cases.find(c => c.size === options.size);
  if(!entry)throw Error(`The prepared ${manifest.modelProfile} model supports ${manifest.cases.map(c=>c.size).join('/')} cells; choose a supported matrix size.`);
  const key = `${entry.model.sha256}/${options.backend}`;
  // A changed fixture must never reuse a previous expected-output buffer.
  const fixtureKey = `${entry.input.sha256}/${entry.expected.sha256}`;
  if (cached?.key === key && cached.fixtureKey === fixtureKey) {
    cached.manifest = manifest;
    return {...cached, loadMs: performance.now()-started, initMs: 0, cacheReused: true};
  }
  if (cached) { await cached.session.release(); cached = null; }
  const model = await artifact(entry.model), input = decodeFloat32(await artifact(entry.input)), expected = decodeFloat32(await artifact(entry.expected));
  if(inputPresence(input,manifest,entry.size).length!==entry.framesAvailable)throw Error('Fixture presence mask disagrees with its declared capture count.');
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
  inputPresence(pixels,context.manifest,context.entry.size);
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
  return {format: 'neural-corrector-report', version: 1, modelStatus: context.manifest.status,
    modelProfile: context.manifest.modelProfile, inputLayout:context.manifest.inputLayout,
    outputLabelSpace:context.manifest.outputLabelSpace,phaseMasks:context.manifest.phaseMasks,
    frameCounts:context.manifest.frameCounts, minFramesPerInput:context.manifest.minFramesPerInput, networkInputClip: context.manifest.networkInputClip,
    checkpointSha256: context.manifest.checkpointSha256 ?? null, size: options.size, framesPerInput:context.manifest.framesPerInput,
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
  transportContext=null;transportMode=null;liveBurst.reset();transportCapture.reset();
  cameraBurst=null;
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
async function capture(options, progress, image, runId, mediaTime, presentedFrames) {
  transportContext=null;transportMode=null;liveBurst.reset();
  if (!Number.isSafeInteger(runId)||runId<1||!Number.isFinite(mediaTime)||mediaTime<0||!Number.isSafeInteger(presentedFrames)||presentedFrames<1) throw Error('Capture needs a fresh run and video frame identity.');
  const context = await load(options, progress), frames=options.captureFrames||context.manifest.framesPerInput;
  if(!context.manifest.frameCounts.includes(frames))throw Error(`The prepared model supports ${context.manifest.frameCounts.join('/')} captures; choose model default or a supported count.`);
  const phaseGrouped=context.manifest.inputLayout==='phase-paired-rgb-presence-planes';
  if (!cameraBurst||cameraBurst.frames!==frames||cameraBurst.phaseGrouped!==phaseGrouped) cameraBurst=new CameraBurst(frames,{phaseGrouped});
  const identity={context:`${context.key}/${runId}/${frames}`,mediaTime,presentedFrames};
  progress('Validating the monochrome header and rectifying the current exposure…');
  const input = extractLearnedInput(image);
  if (!input.ok) {
    if(!phaseGrouped){cameraBurst=null;throw new Error(input.error || 'No supported 64/128-cell matrix was found.');}
    // A display transition can temporarily invalidate the header. Keep prior
    // phases from this run, bounded by collection attempts and a 60 s age limit.
    const waiting=cameraBurst.miss(identity);
    progress('Waiting for a clear phase header; previous captures are retained.');
    return {operation:'capture',waiting:true,collected:waiting.collected,framesPerInput:context.manifest.framesPerInput,
      requiredCaptures:frames,phaseGrouped:true,phaseCounts:waiting.phaseCounts,skippedExposure:true};
  }
  if (input.info.size !== options.size) { cameraBurst=null; throw new Error(`Camera sees ${input.info.size} cells; select that model size first.`); }
  const stacked=cameraBurst.add(input,identity);
  if (!stacked.ready) {
    progress(phaseGrouped?`Collected phases ${stacked.phaseCounts.join('/')}; show all four phases of this matrix.`:`Collected ${stacked.collected}/${frames} same-phase exposures. Keep the matrix steady.`);
    return {operation:'capture',waiting:true,collected:stacked.collected,framesPerInput:context.manifest.framesPerInput,requiredCaptures:frames,phaseGrouped,phaseCounts:stacked.phaseCounts};
  }
  progress(`Running the selected model on ${frames} camera exposure${frames===1?'':'s'}…`);
  try {
    const packed=packFrameStack(stacked.input,context.manifest,options.size,phaseGrouped?{phases:stacked.phases,slots:stacked.slots}:undefined);
    const result = await inference(context, packed);
    if (result.values.some(value => !Number.isFinite(value))) throw new Error('Camera inference produced non-finite logits.');
    const description=describe(context, options);
    return {...description, operation: 'capture', inputStats:{...input.stats,preprocessMs:stacked.preprocessMs}, inferenceMs: result.ms,
      requiredCaptures:frames,phaseGrouped,phaseCounts:stacked.phaseCounts, memory:{...description.memory,cameraStackBytes:stacked.input.byteLength,cameraCollectionBytes:stacked.bufferBytes??0,
        packedInputBytes:packed===stacked.input?0:packed.byteLength,
        note:'The camera stack and optional packed input are additional to cached numerical fixtures. These are known buffer sizes, not peak memory; camera preprocessing, runtime working memory and GPU allocations are not measured.'},
      paletteColors: input.info.colors, logitsFinite: true, inputCaptures:frames, mediaTimes:stacked.mediaTimes,presentedFrames:stacked.presentedFrames,
      note: `${phaseGrouped?'Four-phase canonical-output':'Same-phase'} crop inference only. Assess symbol and payload recovery with the offline evaluation tools.`};
  } finally { cameraBurst=null; }
}

async function initializeTransport(options,progress) {
  cameraBurst=null;liveBurst.reset();transportCapture.reset();transportMode=null;transportContext=null;
  const context=await load(options,progress,{transport:true});
  requireTransportModel(context.manifest);
  if(options.size!==64)throw Error('Neural-ZKBoo transport requires the G64 model.');
  transportContext=context;
  return {...describe(context,options),operation:'transport-init',ready:true};
}

async function scanTransport(options,progress,raw) {
  const context=transportContext;
  // load() returns a description copy; compare the underlying session identity.
  if(!context||context.session!==cached?.session||context.entry.size!==options.size||context.key!==`${context.entry.model.sha256}/${options.backend}`)
    throw Error('Initialize the trained corrector before starting the camera receiver.');
  requireTransportModel(context.manifest);
  const {image,runId,mediaTime,presentedFrames,expected}=raw;
  if(!['data','ack'].includes(expected)||!Number.isSafeInteger(runId)||runId<1)throw Error('Invalid Neural-ZKBoo capture request.');
  const identity={context:`${context.key}/${runId}/${expected}`,mediaTime,presentedFrames};
  if(transportMode!==expected){liveBurst.reset();transportCapture.reset();transportMode=expected;}
  const before=liveBurst.status();
  if(expected==='ack') {
    const state=liveBurst.miss(identity);
    if(state.duplicateExposure)return {operation:'transport-scan',expected,...state,ok:false,waiting:true,stage:'duplicate',error:'Waiting for a new camera exposure.',method:'qr-ack'};
    const decoded=transportCapture.scan(image,{expected,context:identity.context,mediaTime});
    return {operation:'transport-scan',expected,...decoded,waiting:!decoded.ok,inputCaptures:1,inferenceMs:0};
  }
  const capture=transportCapture.scan(image,{expected,context:identity.context,mediaTime});
  const base={operation:'transport-scan',expected,info:capture.info??null,corners:capture.corners??[],
    tentativeCorners:capture.tentativeCorners??[],markers:capture.markers??[],phaseCounts:before.phaseCounts,collected:before.collected,
    effectiveBackend:context.effectiveBackend,stats:capture.stats??{}};
  if(!capture.transform||!capture.samples)return {...base,...liveBurst.miss(identity),ok:false,waiting:true,stage:capture.stage,error:capture.error};
  if(!isTransportData(capture.info))return {...base,...liveBurst.miss(identity),ok:false,waiting:true,stage:'profile',error:'Waiting for a 64×64, 16-color, four-phase data matrix with grid 8 and RS32.'};
  const crop=extractLearnedInput(image,capture);
  if(!crop.ok)return {...base,...liveBurst.miss(identity),ok:false,waiting:true,stage:crop.stage,error:crop.error};
  const state=liveBurst.add(crop,identity);
  if(!state.ready||state.duplicateExposure)return {...base,...state,ok:false,waiting:true,stage:'collect',error:null};
  const burst=liveBurst.snapshot();
  progress(`Correcting ${burst.inputCaptures} captures (${burst.phaseCounts.join('/')})…`);
  const packed=packFrameStack(burst.input,context.manifest,64,{phases:burst.phases,slots:burst.slots});
  const result=await inference(context,packed);
  const decoded=recoverCorrectedPayload(result.values,burst.info);
  if(!decoded.ok)transportCapture.invalidate();
  const response={...base,...decoded,waiting:!decoded.ok,info:burst.info,inputCaptures:burst.inputCaptures,
    phaseCounts:burst.phaseCounts,collected:state.collected,replacements:state.replacements,
    bufferBytes:state.bufferBytes,inferenceMs:result.ms,preprocessMs:crop.stats.preprocessMs,
    stats:{...base.stats,...decoded.stats},method:'neural-corrector'};
  if(decoded.ok)liveBurst.reset();
  return response;
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
    else if (operation === 'capture') result = await capture(validated, progress, options.image, options.runId, options.mediaTime, options.presentedFrames);
    else if (operation === 'transport-init') result = await initializeTransport(validated,progress);
    else if (operation === 'transport-scan') result = await scanTransport(validated,progress,options);
    else if (operation === 'transport-reset') {liveBurst.reset();transportCapture.reset();transportMode=null;result={operation,ready:Boolean(transportContext)};}
    else throw new Error('Unknown corrector operation.');
    self.postMessage({id, type: 'result', result});
  } catch (error) { cameraBurst=null;liveBurst.reset();transportCapture.reset(); self.postMessage({id, type: 'error', error: error.message || String(error)}); }
  finally { busy = false; }
};
