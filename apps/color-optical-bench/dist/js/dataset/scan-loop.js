/** One worker job at a time. Stop invalidates results before canceling the worker. */
export class ScanLoop {
  constructor({sample,receive,onError=()=>{},cancel=()=>{},interval=()=>100,
    schedule=(fn,ms)=>setTimeout(fn,ms),unschedule=id=>clearTimeout(id)}){
    Object.assign(this,{sample,receive,onError,cancel,interval,schedule,unschedule});
    this.generation=0;this.running=false;this.timer=null;
  }
  start(){if(this.running)return;this.running=true;const token=++this.generation;this.tick(token);}
  stop(){this.running=false;this.generation++;this.unschedule(this.timer);this.timer=null;this.cancel();}
  async tick(token){
    const started=performance.now();
    try{const result=await this.sample();if(this.running&&token===this.generation)await this.receive(result);}
    catch(error){if(this.running&&token===this.generation)this.onError(error);}
    finally{if(this.running&&token===this.generation)this.timer=this.schedule(()=>this.tick(token),Math.max(0,this.interval()-(performance.now()-started)));}
  }
}
