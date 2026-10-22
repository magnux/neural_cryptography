/** Stateless extraction worker. Collected samples stay with the page on cancel. */
export class CaptureClient {
  constructor({workerFactory = () => new Worker(new URL('./worker.js', import.meta.url), {type:'module'}), timeoutMs = 30000} = {}) {
    this.workerFactory=workerFactory; this.timeoutMs=timeoutMs; this.worker=null; this.pending=null; this.serial=0;
  }
  call(operation, options={}, transfer=[]) {
    if(this.pending) return Promise.reject(new Error('A capture operation is already running.'));
    if(!this.worker){
      const worker=this.worker=this.workerFactory();
      worker.onmessage=event=>{
        if(worker!==this.worker||!this.pending||event.data?.id!==this.pending.id)return;
        const pending=this.pending;this.pending=null;clearTimeout(pending.timer);
        if(event.data.type==='result')pending.resolve(event.data.result);
        else pending.reject(new Error(event.data.error||'Capture worker failed.'));
      };
      worker.onerror=event=>{if(worker===this.worker)this.cancel(new Error(event.message||'Capture worker failed.'));};
      worker.onmessageerror=()=>{if(worker===this.worker)this.cancel(new Error('Unreadable capture result.'));};
    }
    const id=++this.serial;
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>this.cancel(new Error('Capture processing timed out. Try a lower camera resolution.')),this.timeoutMs);
      this.pending={id,resolve,reject,timer};
      try{this.worker.postMessage({id,operation,options},transfer);}catch(error){this.cancel(error);}
    });
  }
  cancel(error=new DOMException('Capture canceled. Previously collected samples are retained.','AbortError')){
    const pending=this.pending;this.pending=null;
    this.worker?.terminate();this.worker=null;
    if(pending){clearTimeout(pending.timer);pending.reject(error);}
  }
}
