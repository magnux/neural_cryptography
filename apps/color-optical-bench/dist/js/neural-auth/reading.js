/** Provisional camera activity, independent of transport notices or acceptance. */
export const READING_DECAY_MS=3000;

export class ReadingCue {
  constructor({onChange=()=>{},setTimer=(callback,delay)=>globalThis.setTimeout(callback,delay),
    clearTimer=id=>globalThis.clearTimeout(id)}={}){
    this.onChange=onChange;this.setTimer=setTimer;this.clearTimer=clearTimer;
    this.state='idle';this.key=null;this.timer=null;this.revision=0;
  }
  setState(state){
    if(state===this.state)return;
    this.state=state;this.onChange(state);
  }
  cancelDecay(){
    this.revision++;
    if(this.timer!==null)this.clearTimer(this.timer);
    this.timer=null;
  }
  decay(){
    // Repeated misses cannot extend the fade or flicker it back to full opacity.
    if(this.state!=='reading')return;
    this.setState('decaying');
    const revision=++this.revision;
    this.timer=this.setTimer(()=>{
      if(revision!==this.revision)return;
      this.timer=null;this.setState('idle');
    },READING_DECAY_MS);
  }
  displayed(key){
    if(key===this.key)return;
    this.key=key;this.settle();
  }
  observe(result={}){
    const {stage,info}=result;
    if(result.ok||stage==='profile')this.settle();
    else if(result.updated===true&&info&&!result.duplicateExposure){
      this.cancelDecay();this.setState('reading');
    }else if(stage==='collect'&&result.duplicateExposure||info&&['fec','crc'].includes(stage)){
      // Repeated exposures and unsuccessful recovery do not imply lost geometry.
    }else this.decay();
    return this.state;
  }
  settle(){this.cancelDecay();this.setState('idle');}
  reset(){this.key=null;this.settle();}
}
