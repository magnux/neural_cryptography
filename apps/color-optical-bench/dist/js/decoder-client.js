/** A bounded, single-job worker client. No camera/test state is shared. */
export class DecoderClient {
  constructor(){this.worker=null;this.pending=null;this.nextId=0;this.start();}
  get busy(){return this.pending!==null;}
  start(){
    this.worker=new Worker(new URL('./decoder-worker.js',import.meta.url),{type:'module'});
    this.worker.onmessage=({data})=>{
      const job=this.pending;if(!job||data.id!==job.id)return;
      this.pending=null;clearTimeout(job.timer);job.resolve(data.result);
    };
    this.worker.onerror=event=>this.fail(Error(event.message||'Decoder worker unavailable. Reload the app over HTTP(S).'));
  }
  fail(error){
    this.worker?.terminate();this.worker=null;
    const job=this.pending;this.pending=null;
    if(job){clearTimeout(job.timer);job.reject(error);}
  }
  reset(){this.worker?.postMessage({type:'reset'});}
  decode(image,options={}){
    return this.request({image,options},[image.data.buffer]);
  }
  simulate(frame,renderPixels,simulation,options={}){
    return this.request({type:'simulate',frame,renderPixels,simulation,options});
  }
  request(message,transfers=[]){
    if(this.busy)return Promise.reject(Error('A capture is already being decoded.'));
    if(!this.worker){try{this.start();}catch(error){return Promise.reject(error);}}
    return new Promise((resolve,reject)=>{
      const id=++this.nextId,timer=setTimeout(()=>this.fail(Error('Decode timed out. Try a smaller image.')),20000);
      this.pending={id,resolve,reject,timer};
      try{this.worker.postMessage({...message,id,epoch:0},transfers);}catch(error){this.fail(error);}
    });
  }
}
