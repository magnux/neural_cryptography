/** Untrained corrector execution for the explicitly target-assisted simulator.
 * Live receive code never imports this module or the simulation model path.
 */
import {validateManifest,verifyArtifact,decodeFloat32,compareLogits,inputPresence} from './js/corrector-lab/runtime.js';

export async function createSimulationCorrector({
  modelRoot=new URL('./models/simulation-corrector/',import.meta.url),
  runtimeRoot=new URL('./vendor/onnxruntime/',import.meta.url),
  read=async url=>{
    const response=await fetch(url,{cache:'no-cache'});
    if(!response.ok)throw Error(`Missing simulation asset (${response.status}): ${new URL(url).pathname}. Export an untrained VW model and prepare the simulation assets first.`);
    return response.arrayBuffer();
  },
  onProgress=()=>{},statistics=true,
}={}) {
  const started=statistics?performance.now():null,manifestBytes=await read(new URL('manifest.json',modelRoot));
  if(manifestBytes.byteLength>65536)throw Error('Simulation manifest exceeds 64 KiB.');
  const manifest=validateManifest(JSON.parse(new TextDecoder().decode(manifestBytes)));
  if(manifest.status!=='untrained'||manifest.modelProfile!=='VW'||![32,64].every(size=>manifest.cases.some(entry=>entry.size===size)))
    throw Error('This diagnostic requires a separate untrained VW export with G32 and G64. Live trained models are not substituted.');
  let ort;
  try{ort=await import(new URL('ort.webgpu.min.mjs',runtimeRoot).href);}
  catch(error){throw Error(`Simulation ONNX Runtime is unavailable. Prepare the pinned browser runtime. ${error.message}`);}
  ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=runtimeRoot.href;
  const contexts=new Map(),models=[];
  const readArtifact=async artifact=>verifyArtifact(await read(new URL(artifact.file,modelRoot)),artifact);
  async function infer(context,pixels,{statistics:measure=statistics}={}) {
    inputPresence(pixels,manifest,context.entry.size);
    const tensor=new ort.Tensor('float32',pixels,context.entry.inputShape),tick=measure?performance.now():null;
    let outputs;
    try{
      outputs=await context.session.run({[manifest.inputName]:tensor});
      const output=outputs[manifest.outputName];
      if(!output||output.type!=='float32'||output.dims.length!==4||output.dims.some((n,i)=>n!==context.entry.outputShape[i]))
        throw Error('Simulation model returned the wrong tensor shape.');
      const values=new Float32Array(output.data);
      if(values.some(value=>!Number.isFinite(value)))throw Error('Simulation model returned non-finite logits.');
      return {values,...(measure?{ms:performance.now()-tick}:{})};
    }finally{tensor.dispose?.();for(const output of Object.values(outputs??{}))output.dispose?.();}
  }
  try{
    for(const size of [32,64]){
      onProgress(`Initializing actual untrained VW G${size} inference…`);
      const entry=manifest.cases.find(item=>item.size===size),model=await readArtifact(entry.model),init=statistics?performance.now():null;
      const session=await ort.InferenceSession.create(model,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
      const context={manifest,entry,session,effectiveBackend:'wasm'};contexts.set(size,context);
      if(session.inputNames.length!==1||session.inputNames[0]!==manifest.inputName||session.outputNames.length!==1||session.outputNames[0]!==manifest.outputName)
        throw Error('Simulation model tensor names differ from its manifest.');
      const initMs=statistics?performance.now()-init:undefined,input=decodeFloat32(await readArtifact(entry.input)),expected=decodeFloat32(await readArtifact(entry.expected));
      const actual=await infer(context,input),parity=compareLogits(actual.values,expected,manifest.tolerance);
      if(!parity.passed)throw Error(`Untrained VW G${size} numerical parity failed (${parity.failures} values).`);
      // Parity is the warmup, excluded from handshake timing. This verifies
      // arithmetic agreement only, never learned optical correction accuracy.
      models.push({size,sha256:entry.model.sha256,modelBytes:entry.model.bytes,initMs,warmupMs:actual.ms,parity});
    }
  }catch(error){for(const context of contexts.values())await context.session.release().catch(()=>{});throw error;}
  return {
    manifest,
    description:{measured:statistics,executed:true,kind:'actual-untrained-vw',backend:'wasm',wasmThreads:1,modelStatus:manifest.status,
      modelProfile:manifest.modelProfile,parameterCount:manifest.parameterCount,runtimeVersions:{...ort.env.versions},
      initializationMs:statistics?performance.now()-started:undefined,models,
      timingScope:statistics?'Actual ONNX session.run plus CPU output copy; inference output is discarded only after measurement.':'Detailed timing disabled; actual ONNX inference still executes before output substitution.'},
    loadContext:async size=>{const context=contexts.get(size);if(!context)throw Error(`Unprepared simulation model G${size}.`);return context;},
    infer,
    close:async()=>{for(const context of contexts.values())await context.session.release();contexts.clear();},
  };
}
