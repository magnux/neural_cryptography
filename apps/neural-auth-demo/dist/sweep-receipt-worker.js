// Timing lab only: bounded, classical read of one G16 camera exposure.
// The worker receives neither a probe nor the expected digest.
import {ReceiptCaptureReader} from './js/neural-qr/receipt-reader.js';

export function receiptWorkerHandler(post){
  let ready=false;
  const reader=new ReceiptCaptureReader();
  return ({data:{id,operation,options={}}})=>{
    try{
      if(operation==='receipt-init'){
        reader.reset();ready=true;
        post({id,type:'result',result:{ready,modelStatus:'not-used',effectiveBackend:'classical-js'}});return;
      }
      if(!ready||operation!=='receipt-scan')throw Error('Initialize the G16 receipt reader first.');
      post({id,type:'result',result:reader.scan({...options,statistics:true})});
    }catch(e){post({id,type:'error',error:e.message});}
  };
}

if(typeof WorkerGlobalScope!=='undefined'&&self instanceof WorkerGlobalScope)
  self.onmessage=receiptWorkerHandler(reply=>self.postMessage(reply));
