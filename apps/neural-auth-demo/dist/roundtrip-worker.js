// Dedicated timing-tab worker. No authentication keys, expected test payloads,
// trained model, target logits or four-exposure burst are used by this reader.
import {TimingScanner} from './roundtrip-core.js';
import {MAX_CAPTURE_PIXELS} from './js/dataset/camera.js';
import {packFrameStack} from './js/corrector-lab/runtime.js';

const loadUntrained=async options=>{
  const {createSimulationCorrector}=await import('./simulation-runtime.js');
  return createSimulationCorrector(options);
};

function rgba(image){
  const {width,height}=image??{};
  if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<32||height<32||width*height>MAX_CAPTURE_PIXELS)
    throw Error('Timing capture dimensions exceed the camera bounds.');
  const data=image.data instanceof ArrayBuffer?new Uint8ClampedArray(image.data):image.data;
  if(!(data instanceof Uint8ClampedArray||data instanceof Uint8Array)||data.length!==4*width*height)
    throw Error('Expected one RGBA camera capture.');
  return {width,height,data};
}

export function timingWorkerHandler(post,{createCorrector=loadUntrained}={}){
  let busy=false,scanner=null,model=null,context=null,dummy=null;
  return async({data:m})=>{
    const id=m?.id;
    if(!Number.isSafeInteger(id)||id<0)return;
    if(busy){post({id,error:'Timing worker is busy.'});return;}
    busy=true;
    try{
      let value;
      if(m.op==='init'){
        if(scanner)throw Error('Timing worker is already initialized.');
        if(![16,32,64,128].includes(m.size)||typeof m.inference!=='boolean')throw Error('Invalid timing settings.');
        if(m.inference){
          if(![32,64].includes(m.size))throw Error('Optional dummy inference requires a G32 or G64 untrained VW export.');
          // Hash/parity checks and warmup finish before the first matrix is drawn.
          model=await createCorrector({onProgress:progress=>post({id,progress}),statistics:true});
          try{
            context=await model.loadContext(m.size);
            // Grey RGB for a dummy K4 input, explicitly one occupied slot per
            // phase. The pixels never come from (or modify) the camera decode.
            dummy=packFrameStack(new Float32Array(4*3*(4*m.size)**2),model.manifest,m.size,{phases:[0,1,2,3]});
          }catch(error){await model.close();model=null;context=null;throw error;}
        }
        scanner=new TimingScanner();
        value={size:m.size,inference:m.inference,backend:m.inference?'wasm':null};
      }else if(!scanner)throw Error('Initialize the timing worker first.');
      else if(m.op==='scan'){
        if(!Number.isFinite(m.captureIntervalMs)||m.captureIntervalMs<1||m.captureIntervalMs>10000)
          throw Error('Timing capture interval is outside the supported range.');
        value=scanner.scan(rgba(m.image),{now:performance.now(),captureIntervalMs:m.captureIntervalMs});
      }else if(m.op==='infer'){
        if(!model)throw Error('Dummy inference is disabled.');
        const {ms}=await model.infer(context,dummy);
        // Never send the estimated symbols/logits to the classical echo path.
        value={inferenceMs:ms,backend:'wasm',size:context.entry.size};
      }else throw Error('Unknown timing operation.');
      post({id,value},value?.frame?.symbols?[value.frame.symbols.buffer]:[]);
    }catch(error){post({id,error:error?.message||String(error)});}
    finally{busy=false;}
  };
}
if(typeof WorkerGlobalScope!=='undefined'&&self instanceof WorkerGlobalScope)
  self.onmessage=timingWorkerHandler((reply,transfer=[])=>self.postMessage(reply,transfer));
