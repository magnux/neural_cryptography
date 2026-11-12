import {validateManifest, validateOptions, verifyArtifact, decodeFloat32, compareLogits, accumulateParity, summarizeTimings, inputPresence, packFrameStack} from './runtime.js';
import {extractLearnedInput} from '../learned-input.js';
import {CameraBurst} from './burst.js';
import {TransportCapture} from '../neural-auth/capture.js';
import {LivePhaseBurst,requireTransportModel,isTransportData,recoverCorrectedPayload} from './live-burst.js';
import {CONTROL_SIZE} from '../neural-qr/packet.js';
import {geometryHintMaxAgeMs} from '../neural-qr/vision.js';

const assetRoot = new URL('../../models/neural-corrector/', import.meta.url);
const vendorRoot = new URL('../../vendor/onnxruntime/', import.meta.url);
let ort, busy = false, cameraBurst=null;
// Keep only two compiled models; their larger working allocations are owned by
// ONNX Runtime. Three verified artifact sets cover control and both data sizes,
// including offline fallback after transport initialization.
const sessions=new Map(),assets=new Map();
const MAX_SESSIONS=2,MAX_ASSET_SETS=3;
let transportContext=null,transportMode=null;
const liveBurst=new LivePhaseBurst();
const transportCapture=new TransportCapture(),labCapture=new TransportCapture();

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
async function load(options, progress, {transport=false, manifest:providedManifest=null}={}) {
  const started=performance.now();
  const manifest=providedManifest??validateManifest(JSON.parse(new TextDecoder().decode(
    await fetchBounded(new URL('manifest.json',assetRoot),65536))));
  if(transport)requireTransportModel(manifest);
  const entry=manifest.cases.find(c=>c.size===options.size);
  if(!entry)throw Error(`The prepared ${manifest.modelProfile} model supports ${manifest.cases.map(c=>c.size).join('/')} cells; choose a supported matrix size.`);
  const artifactKey=`${entry.model.sha256}/${entry.input.sha256}/${entry.expected.sha256}`;
  const key=`${artifactKey}/${options.backend}`;
  if(sessions.has(key)) {
    const context=sessions.get(key);sessions.delete(key);sessions.set(key,context);
    Object.assign(context,{manifest,loadMs:performance.now()-started,initMs:0,cacheReused:true});
    if(transport)await verifyContextParity(context,options,progress);
    return context;
  }
  progress(`Checking G${entry.size} model and fixture hashes…`);
  let bytes=assets.get(artifactKey);
  if(bytes){assets.delete(artifactKey);assets.set(artifactKey,bytes);}
  else {
    bytes={model:await artifact(entry.model),input:decodeFloat32(await artifact(entry.input)),
      expected:decodeFloat32(await artifact(entry.expected))};
    if(inputPresence(bytes.input,manifest,entry.size).length!==entry.framesAvailable)
      throw Error('Fixture presence mask disagrees with its declared capture count.');
    assets.set(artifactKey,bytes);
    while(assets.size>MAX_ASSET_SETS){
      const protectedKeys=new Set([...sessions.values()].map(context=>context.artifactKey));
      const oldest=[...assets.keys()].find(key=>!protectedKeys.has(key));
      assets.delete(oldest);
    }
  }
  await runtime();
  // Release before allocating a replacement, including GPU working memory.
  while(sessions.size>=MAX_SESSIONS){const first=sessions.keys().next().value;await sessions.get(first).session.release();sessions.delete(first);}
  let effectiveBackend=options.backend==='auto'?(navigator.gpu?'webgpu':'wasm'):options.backend;
  let fallbackReason=options.backend==='auto'&&!navigator.gpu?'WebGPU is unavailable in this worker.':null;
  const initialized=performance.now();
  progress(`Initializing G${entry.size} ${effectiveBackend.toUpperCase()}…`);
  let session;
  try{session=await ort.InferenceSession.create(bytes.model,{executionProviders:[effectiveBackend],graphOptimizationLevel:'all'});}
  catch(error){
    if(options.backend!=='auto'||effectiveBackend!=='webgpu')throw Error(`${effectiveBackend.toUpperCase()} initialization failed: ${error.message}`);
    fallbackReason=`WebGPU initialization failed: ${error.message}`;effectiveBackend='wasm';
    progress(`Initializing G${entry.size} WASM fallback…`);
    session=await ort.InferenceSession.create(bytes.model,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  }
  try{validateTensorNames(session,manifest);}catch(error){await session.release();throw error;}
  const context={key,artifactKey,session,manifest,entry,...bytes,effectiveBackend,fallbackReason,
    loadMs:performance.now()-started,initMs:performance.now()-initialized,cacheReused:false,parity:null};
  sessions.set(key,context);
  if(transport)await verifyContextParity(context,options,progress);
  return context;
}
function validateTensorNames(session,manifest){
  if(session.inputNames.length!==1||session.inputNames[0]!==manifest.inputName||
    session.outputNames.length!==1||session.outputNames[0]!==manifest.outputName)
    throw Error('Exported tensor names do not match the model contract.');
}
async function verifyContextParity(context,options,progress){
  if(context.parity?.passed)return context.parity;
  const check=async()=>{
    const result=await inference(context,context.input),parity=compareLogits(result.values,context.expected);
    if(!parity.passed)throw Error(`G${context.entry.size} numerical parity failed (${parity.failures}/${parity.checked} values).`);
    context.parity={...parity,inferenceMs:result.ms};return context.parity;
  };
  progress(`Checking G${context.entry.size} browser numerical parity…`);
  try{return await check();}
  catch(error){
    if(options.backend==='auto'&&context.effectiveBackend==='webgpu'){
      context.fallbackReason=`WebGPU inference/parity failed: ${error.message}`;
      await context.session.release();sessions.delete(context.key);
      context.effectiveBackend='wasm';context.parity=null;
      progress(`Rechecking G${context.entry.size} using WASM…`);
      try{
        context.session=await ort.InferenceSession.create(context.model,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
        validateTensorNames(context.session,context.manifest);
        const parity=await check();sessions.set(context.key,context);return parity;
      }catch(fallback){await context.session.release().catch(()=>{});throw Error(`WASM fallback failed: ${fallback.message}`);}
    }
    sessions.delete(context.key);await context.session.release().catch(()=>{});throw error;
  }
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
async function capture(options, progress, image, runId, mediaTime, presentedFrames, captureIntervalMs=100) {
  geometryHintMaxAgeMs(captureIntervalMs);
  transportContext=null;transportMode=null;liveBurst.reset();
  if (!Number.isSafeInteger(runId)||runId<1||!Number.isFinite(mediaTime)||mediaTime<0||!Number.isSafeInteger(presentedFrames)||presentedFrames<1) throw Error('Capture needs a fresh run and video frame identity.');
  const context = await load(options, progress), frames=options.captureFrames||context.manifest.framesPerInput;
  if(!context.manifest.frameCounts.includes(frames))throw Error(`The prepared model supports ${context.manifest.frameCounts.join('/')} captures; choose model default or a supported count.`);
  const phaseGrouped=context.manifest.inputLayout==='phase-paired-rgb-presence-planes';
  if(context.manifest.datasetFormat==='neural-qr-dataset')
    return captureNeuralQR(options,progress,context,image,{runId,mediaTime,presentedFrames,captureIntervalMs},frames);
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

async function captureNeuralQR(options,progress,context,image,identity,frames){
  if(!(cameraBurst instanceof LivePhaseBurst)||cameraBurst.frames!==frames){
    cameraBurst=new LivePhaseBurst();cameraBurst.frames=frames;
  }
  const frameIdentity={context:`lab/${context.key}/${identity.runId}/${frames}`,
    mediaTime:identity.mediaTime,presentedFrames:identity.presentedFrames};
  progress('Reading Neural-QR grid and phase cues…');
  const crop=labCapture.scan(image,{expected:'data',context:frameIdentity.context,mediaTime:identity.mediaTime,
    captureIntervalMs:identity.captureIntervalMs});
  if(!crop.ok){
    const state=cameraBurst.miss(frameIdentity);
    return {operation:'capture',waiting:true,collected:state.collected,phaseCounts:state.phaseCounts,
      framesPerInput:8,requiredCaptures:frames,phaseGrouped:true,skippedExposure:true,error:crop.error,stage:crop.stage,stats:crop.stats};
  }
  if(crop.info.size!==options.size){cameraBurst=null;throw Error(`Camera sees G${crop.info.size}; select that model size first.`);}
  const state=cameraBurst.add(crop,frameIdentity),burst=cameraBurst.snapshot({frames});
  if(!burst||state.duplicateExposure)return {operation:'capture',waiting:true,collected:state.collected,
    phaseCounts:state.phaseCounts,framesPerInput:8,requiredCaptures:frames,phaseGrouped:true};
  progress(`Running G${options.size} on ${frames} Neural-QR exposures…`);
  try{
    const packed=packFrameStack(burst.input,context.manifest,options.size,{phases:burst.phases,slots:burst.slots});
    const result=await inference(context,packed);
    if(result.values.some(value=>!Number.isFinite(value)))throw Error('Camera inference produced non-finite logits.');
    return {...describe(context,options),operation:'capture',inputStats:{...crop.stats,preprocessMs:burst.preprocessMs},inferenceMs:result.ms,
      requiredCaptures:frames,phaseGrouped:true,phaseCounts:burst.phaseCounts,inputCaptures:frames,
      paletteColors:16,logitsFinite:true,mediaTimes:burst.mediaTimes,presentedFrames:burst.presentedFrames,
      note:'Header-free Neural-QR crop inference only. Live authentication additionally requires RS/CRC and transfer-context checks.'};
  }finally{cameraBurst=null;}
}

async function initializeTransport(options,progress) {
  cameraBurst=null;liveBurst.reset();transportCapture.reset();labCapture.reset();transportMode=null;transportContext=null;
  if(![32,64,128].includes(options.size))throw Error('Neural-QR transport reception requires a G32, G64 or G128 cap.');
  const manifest=validateManifest(JSON.parse(new TextDecoder().decode(await fetchBounded(new URL('manifest.json',assetRoot),65536))));
  requireTransportModel(manifest);
  const required=[32,64,128].filter(size=>size<=options.size);
  if(required.some(size=>!manifest.cases.some(entry=>entry.size===size)))
    throw Error(`Export all receiving geometries through G${options.size}, including G32 controls.`);
  const checked=[];let capContext;
  // Preload the fallback engine too. A later WebGPU failure while disconnected
  // must not trigger the first network request for its WASM binary.
  let wasmFallbackReady=options.backend==='wasm'||options.backend==='auto'&&!navigator.gpu;
  if(options.backend==='auto'&&navigator.gpu){
    progress('Preparing the offline WASM fallback engine…');
    await load({...options,size:CONTROL_SIZE,backend:'wasm'},progress,{transport:true,manifest});
    wasmFallbackReady=true;
  }
  // Leave the requested data profile and G32 compiled. The intermediate fallback
  // remains in the verified byte cache and can be instantiated without a server.
  for(const size of [...required.filter(size=>size!==CONTROL_SIZE),CONTROL_SIZE]){
    const context=await load({...options,size},progress,{transport:true,manifest});
    checked.push({size,effectiveBackend:context.effectiveBackend,fallbackReason:context.fallbackReason,
      parity:context.parity,modelBytes:context.entry.model.bytes,inputBytes:context.entry.input.bytes,
      loadMs:context.loadMs,initMs:context.initMs});
    if(size===options.size)capContext=context;
  }
  transportContext={manifest,maxSize:options.size,backend:options.backend};
  return {...describe(capContext,options),operation:'transport-init',transport:'neural-qr',ready:true,
    incomingMax:options.size,controlSize:CONTROL_SIZE,availableSizes:required,offlineReady:true,wasmFallbackReady,
    parity:{passed:checked.every(item=>item.parity?.passed),cases:checked.sort((a,b)=>a.size-b.size)},
    cache:{maximumSessions:MAX_SESSIONS,maximumArtifactSets:MAX_ASSET_SETS,
      artifactBytes:[...assets.values()].reduce((sum,item)=>sum+item.model.byteLength+item.input.byteLength+item.expected.byteLength,0)}};
}

async function scanTransport(options,progress,raw) {
  const settings=transportContext;
  if(!settings||settings.maxSize!==options.size||settings.backend!==options.backend)
    throw Error('Initialize the trained corrector before starting the camera receiver.');
  const {image,runId,mediaTime,presentedFrames,expected,captureIntervalMs=100}=raw;
  geometryHintMaxAgeMs(captureIntervalMs);
  if(!['data','ack'].includes(expected)||!Number.isSafeInteger(runId)||runId<1)throw Error('Invalid Neural-ZKBoo capture request.');
  const identity={context:`${runId}/${expected}`,mediaTime,presentedFrames};
  if(transportMode!==expected){liveBurst.reset();transportCapture.reset();transportMode=expected;}
  const before=liveBurst.status();
  const capture=transportCapture.scan(image,{expected,context:identity.context,mediaTime,captureIntervalMs});
  const base={operation:'transport-scan',transport:'neural-qr',expected,info:capture.info??null,
    corners:capture.corners??[],tentativeCorners:capture.tentativeCorners??[],markers:capture.markers??[],
    phaseCounts:before.phaseCounts,collected:before.collected,stats:capture.stats??{},method:'neural-corrector'};
  if(!capture.ok)return {...base,...liveBurst.miss(identity),ok:false,waiting:true,stage:capture.stage,error:capture.error};
  const size=capture.info.size;
  if(!isTransportData(capture.info)||size<CONTROL_SIZE||size>settings.maxSize||expected==='ack'&&size!==CONTROL_SIZE){
    liveBurst.reset();
    return {...base,...liveBurst.miss(identity),ok:false,waiting:true,stage:'profile',
      error:expected==='ack'?`Waiting for a G${CONTROL_SIZE} Neural-QR ACK.`:`G${size} exceeds the receiving profile; maximum is G${settings.maxSize}.`};
  }
  const state=liveBurst.add(capture,identity);
  if(!state.ready||state.duplicateExposure)return {...base,...state,ok:false,waiting:true,stage:'collect',error:null};
  const burst=liveBurst.nextTransportSnapshot();
  if(!burst)return {...base,...state,ok:false,waiting:true,stage:'collect',error:null};
  const context=await load({...options,size},progress,{transport:true,manifest:settings.manifest});
  progress(`Correcting G${size}, ${burst.inputCaptures} captures (${burst.phaseCounts.join('/')})…`);
  const packed=packFrameStack(burst.input,context.manifest,size,{phases:burst.phases,slots:burst.slots});
  const result=await inference(context,packed);
  const decoded=recoverCorrectedPayload(result.values,burst.info);
  const response={...base,...decoded,waiting:!decoded.ok,info:burst.info,inputCaptures:burst.inputCaptures,
    phaseCounts:burst.phaseCounts,collected:state.collected,replacements:state.replacements,
    bufferBytes:state.bufferBytes,inferenceMs:result.ms,preprocessMs:capture.stats.ms,
    effectiveBackend:context.effectiveBackend,stats:{...base.stats,...decoded.stats}};
  // Packet identity and expected kind are enforced by unpackOptical/TurnChannel.
  // Old/new captures can coexist across a transition; CRC failure never counts
  // as a received packet and newer captures continue replacing old samples.
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
    else if (operation === 'capture') result = await capture(validated, progress, options.image, options.runId, options.mediaTime, options.presentedFrames, options.captureIntervalMs);
    else if (operation === 'transport-init') result = await initializeTransport(validated,progress);
    else if (operation === 'transport-scan') result = await scanTransport(validated,progress,options);
    else if (operation === 'transport-reset') {liveBurst.reset();transportCapture.reset();transportMode=null;result={operation,ready:Boolean(transportContext)};}
    else throw new Error('Unknown corrector operation.');
    self.postMessage({id, type: 'result', result});
  } catch (error) { cameraBurst=null;liveBurst.reset();transportCapture.reset();labCapture.reset(); self.postMessage({id, type: 'error', error: error.message || String(error)}); }
  finally { busy = false; }
};
