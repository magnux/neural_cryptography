// Timing lab only: bounded, classical read of one G16 camera exposure.
// The worker receives neither a probe nor the expected digest.
import {MAX_CAPTURE_PIXELS} from './js/dataset/camera.js';
import {decodeReceipt16Capture} from './js/neural-qr/receipt16.js';
import {StageProfiler} from './js/neural-auth/timing.js';

export function receiptWorkerHandler(post){
  let ready=false,step=0,previousGeometry=null;
  return ({data:{id,operation,options={}}})=>{
    try{
      if(operation==='receipt-init'){
        ready=true;
        post({id,type:'result',result:{ready,modelStatus:'not-used',effectiveBackend:'classical-js'}});return;
      }
      if(!ready||operation!=='receipt-scan')throw Error('Initialize the G16 receipt reader first.');
      const {image,runId,captureIntervalMs}=options,{width,height,data}=image??{};
      if(!Number.isSafeInteger(runId)||runId<1||runId<step||!Number.isFinite(captureIntervalMs)||captureIntervalMs<50||captureIntervalMs>2000)
        throw Error('Invalid G16 receipt step or capture interval.');
      if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<32||height<32||width*height>MAX_CAPTURE_PIXELS||
        !(data instanceof Uint8ClampedArray||data instanceof Uint8Array)||data.length!==4*width*height)
        throw Error('Expected one bounded RGBA exposure.');
      if(runId!==step){step=runId;previousGeometry=null;}
      const profiler=new StageProfiler(true);
      const r=decodeReceipt16Capture(image,{previousGeometry,now:performance.now(),captureIntervalMs,profiler});
      previousGeometry=r.geometry??null;
      // Exclude rasters, palette samples, and any other acquisition intermediates.
      const {ok,stage,error,info,corners,tentativeCorners,stats,digest}=r;
      post({id,type:'result',result:{ok,stage,error,info,corners,tentativeCorners,stats,digest,
        inputCaptures:info?.size===16?1:null,timings:profiler.snapshot()}});
    }catch(e){post({id,type:'error',error:e.message});}
  };
}

if(typeof WorkerGlobalScope!=='undefined'&&self instanceof WorkerGlobalScope)
  self.onmessage=receiptWorkerHandler(reply=>self.postMessage(reply));
