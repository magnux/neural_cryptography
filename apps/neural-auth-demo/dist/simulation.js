/** Camera-free diagnostic, always isolated in its own worker. */
export function runSimulation({ackSize=16,onProgress=()=>{},statistics=true,optimized=true,compare=true,signal}={}) {
  return new Promise((resolve,reject)=>{
    if(signal?.aborted){reject(new DOMException('Simulation cancelled','AbortError'));return;}
    const worker=new Worker(new URL('./simulation-worker.js',import.meta.url),{type:'module'});
    let finished=false;
    const finish=(error,report)=>{if(finished)return;finished=true;worker.terminate();signal?.removeEventListener('abort',cancel);error?reject(error):resolve(report);};
    const cancel=()=>finish(new DOMException('Simulation cancelled','AbortError'));
    signal?.addEventListener('abort',cancel,{once:true});
    worker.onmessage=({data})=>{
      if(data.type==='progress')onProgress(data.message);
      else if(data.type==='result')finish(null,data.report);
      else if(data.type==='error')finish(Error(data.message));
    };
    worker.onerror=event=>finish(Error(event.message||'Simulation worker failed.'));
    worker.postMessage({ackSize,statistics:Boolean(statistics),optimized:Boolean(optimized),compare:Boolean(compare)});
  });
}
