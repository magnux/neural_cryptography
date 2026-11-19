// Display scheduler for a single outgoing packet. A timer becoming due does
// not mean any pixels were presented: submit each phase on a refresh callback.
export class FirstPresentation {
  constructor({now=()=>performance.now(),raf=fn=>requestAnimationFrame(fn),cancelRAF=id=>cancelAnimationFrame(id),
    defer=(fn,ms)=>setTimeout(fn,ms),cancelDefer=id=>clearTimeout(id)}={}) {
    Object.assign(this,{now,raf,cancelRAF,defer,cancelDefer});
    this.revision=0;this.frame=null;this.timer=null;this.holding=false;
  }
  clear(){
    ++this.revision;this.holding=false;
    if(this.frame!==null){this.cancelRAF(this.frame);this.frame=null;}
    if(this.timer!==null){this.cancelDefer(this.timer);this.timer=null;}
  }
  // Call once when the logical packet changes (not on same-packet updates).
  // paint(index,first,dwellMs,round) submits the actual pixels. A blank listener calls clear().
  start({phaseMs,incrementMs=0,maxPhaseMs=phaseMs,paint,onReady=()=>{},onError=error=>{throw error;}}){
    if(!Number.isFinite(phaseMs)||phaseMs<=0||!Number.isFinite(incrementMs)||incrementMs<0||
      !Number.isFinite(maxPhaseMs)||maxPhaseMs<phaseMs||typeof paint!=='function')throw Error('Invalid phase presentation.');
    this.clear();const ticket=this.revision;
    this.holding=true;let index=0,first=true,due=0,round=1,dwellMs=phaseMs;
    const current=()=>this.revision===ticket;
    // A callback can synchronously replace or clear its own packet. A queued
    // callback from the old packet must not cancel or release the replacement.
    const enqueue=fn=>{
      if(!current())return;
      this.frame=this.raf(()=>{
        if(!current())return;
        this.frame=null;
        try{fn();}catch(error){if(current()){this.clear();onError(error);}}
      });
    };
    const next=()=>{
      if(!current())return;
      const remaining=due-this.now();
      // Wake just before the deadline, then use the first eligible refresh.
      // Avoid polling on every display frame during a long phase.
      if(remaining>16){
        this.timer=this.defer(()=>{if(current()){this.timer=null;enqueue(draw);}},remaining-16);
      }else enqueue(draw);
    };
    const draw=()=>{
      if(!first&&this.now()<due){next();return;}
      const phase=index;paint(phase,first,dwellMs,round);
      if(!current())return;
      // Use the dwell of the phase just drawn, including phase 3. Only the
      // next round of the same packet gets longer (never the initial pass).
      first=false;index=(phase+1)%4;due=this.now()+dwellMs;
      if(phase===3){round++;dwellMs=Math.min(maxPhaseMs,dwellMs+incrementMs);}
      if(phase===3&&this.holding){
        // The next refresh gives phase 3 a rendering opportunity, then lets
        // the camera start without waiting out an extra full phase.
        enqueue(()=>{this.holding=false;onReady();next();});
      }else next();
    };
    enqueue(draw);
  }
}
