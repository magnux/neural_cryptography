/** Provisional camera activity only; samples do not establish packet identity.
 * Use this to mute the surrounding UI, never the transmitted matrix pixels.
 */
export class ReadingCue {
  constructor({onChange=()=>{},onReadingStarted=()=>{}}={}){
    this.onChange=onChange;this.onReadingStarted=onReadingStarted;
    this.active=false;this.key=null;this.started=false;
  }
  setActive(active){
    if(active===this.active)return;
    this.active=active;this.onChange(active);
  }
  displayed(key){
    if(key===this.key)return;
    this.key=key;this.started=false;this.setActive(false);
  }
  observe(result={}){
    const {stage,info}=result;
    if(result.ok||stage==='profile')this.settle();
    else if(result.updated===true&&info&&!result.duplicateExposure){
      this.setActive(true);
      // A repeated old peer packet can start another candidate after recovery.
      // Dismiss a stale handoff notice at most once for our displayed packet.
      if(!this.started){this.started=true;this.onReadingStarted();}
    }else if(stage==='collect'&&result.duplicateExposure||info&&['fec','crc'].includes(stage)){
      // Repeated exposures and unsuccessful recovery do not imply lost geometry.
    }else this.settle();
    return this.active;
  }
  settle(){this.setActive(false);}
  reset(){this.key=null;this.started=false;this.setActive(false);}
}
