// One manually armed step at a time. Each device uses its OWN monotonic clock.
// Sender RTT starts at DATA paint; receiver arm-to-ACK can include user alignment.
import {StageProfiler} from './js/neural-auth/timing.js';
import {chunkBytesForSize,CONTROL_SIZE} from './js/neural-auth/optical.js';

const marks=['probeDraw','firstLock','allPhases','packetDecoded','ackDraw','ackDecoded'];
const ms=v=>Math.round(v*10)/10;
const delta=(end,start)=>end==null||start==null||end<start?null:ms(end-start);
function integer(n,min,max,label){
  if(!Number.isSafeInteger(n)||n<min||n>max)throw Error(label+' must be '+min+'–'+max+'.');
  return n;
}
// Draw one local timing offset per admitted exposure, never per camera
// callback. Late callbacks (e.g. after inference) don't trigger catch-up scans.
export class CapturePacer{
  constructor(baseMs,vary=false,random=Math.random){
    this.baseMs=baseMs;this.vary=vary;this.random=random;this.last=null;this.waitMs=0;
  }
  take(stamp){
    if(!Number.isFinite(stamp)||(this.last!==null&&stamp-this.last<this.waitMs))return null;
    const interval=this.last===null?null:{requested:this.waitMs,actual:stamp-this.last};
    this.last=stamp;
    this.waitMs=this.vary?this.baseMs*(.8+.4*this.random()):this.baseMs;
    return {interval};
  }
}

export class SweepRun{
  constructor({role,size=64,ackSize=CONTROL_SIZE,localColorCorrection=false,startPhaseMs=500,startCaptureMs=450,decrementMs=50,timeoutMs=30000,varyCapture=false,phaseWatch=true,progressive=false,incrementMs=50,maxPhaseMs=500}){
    if(![16,CONTROL_SIZE].includes(ackSize)||!['sender','receiver'].includes(role)||![32,64,128].includes(size))throw Error('Choose a sweep role and supported grid.');
    integer(startPhaseMs,50,2000,'Initial display interval');integer(startCaptureMs,50,2000,'Initial capture interval');
    integer(decrementMs,0,500,'Interval decrement');integer(timeoutMs,1000,120000,'Step timeout');
    if(progressive){integer(incrementMs,1,2000,'Increase per pass');integer(maxPhaseMs,startPhaseMs,2000,'Maximum phase interval');}
    this.config=Object.freeze({role,size,ackSize,localColorCorrection:localColorCorrection===true&&!(role==='sender'&&ackSize===16),
      startPhaseMs,startCaptureMs,decrementMs,timeoutMs,varyCapture:varyCapture===true,phaseWatch:phaseWatch===true,
      progressive:progressive===true,incrementMs:progressive?incrementMs:null,maxPhaseMs:progressive?maxPhaseMs:null});
    this.count=decrementMs===0?5:Math.min(64,1+Math.floor((Math.min(startPhaseMs,startCaptureMs)-50)/decrementMs));
    this.started=new Date().toISOString();this.rows=[];this.active=null;this.profiler=null;this.displays=new Map();
  }
  get next(){
    if(this.active||this.rows.length>=this.count)return null;
    const i=this.rows.length,{startPhaseMs,startCaptureMs,decrementMs}=this.config;
    return {step:i+1,phaseMs:startPhaseMs-i*decrementMs,captureMs:startCaptureMs-i*decrementMs};
  }
  begin(now){
    const next=this.next;
    if(!next||!Number.isFinite(now))throw Error('No sweep step is ready to start.');
    this.profiler=new StageProfiler(true);
    return this.active={...next,startedAt:now,checkpointsMs:{}};
  }
  mark(name,now){
    const a=this.active;
    if(!a||!marks.includes(name)||Object.hasOwn(a.checkpointsMs,name)||!Number.isFinite(now)||now<a.startedAt)return;
    a.checkpointsMs[name]=ms(now-a.startedAt);
  }
  sample(name,value){this.profiler?.record(name,value);}
  // ACKs continue cycling after the receiver's step is over. Keep their
  // cadence on that row until Next clears the display or the run stops.
  painted(step,kind,tick,at,dwellMs){
    if(kind!==(this.config.role==='sender'?'data':'ack')||!Number.isSafeInteger(step)||step<1||
      step!==(this.active?.step??this.rows.length)||!Number.isSafeInteger(tick)||tick<0||!Number.isFinite(at))return;
    let d=this.displays.get(step);
    if(!d){d={kind,changes:0,skippedSlots:0,phases:[0,0,0,0],lastTick:null,lastAt:null,cycle:-1,phaseMask:0,completeCycles:0,peakDwellMs:null,intervals:new StageProfiler(true)};this.displays.set(step,d);}
    if(d.lastTick!==null){
      if(tick<=d.lastTick||at<d.lastAt)return;
      d.skippedSlots+=tick-d.lastTick-1;
      // A whole missed cycle can repaint the same phase without changing pixels.
      if(tick%4===d.lastTick%4){d.lastTick=tick;return;}
      d.intervals.record('spacingMs',at-d.lastAt);
    }
    const cycle=Math.floor(tick/4);
    if(d.cycle!==cycle){d.cycle=cycle;d.phaseMask=0;}
    d.phaseMask|=1<<(tick%4);
    if(d.phaseMask===15)d.completeCycles++;
    if(Number.isFinite(dwellMs)&&dwellMs>0)d.peakDwellMs=Math.max(d.peakDwellMs??0,dwellMs);
    d.changes++;d.phases[tick%4]++;d.lastAt=at;d.lastTick=tick;
  }
  display(step){
    const d=this.displays.get(step);
    return d?{kind:d.kind,phasePresentations:d.changes,byPhase:[...d.phases],skippedSlots:d.skippedSlots,completeCycles:d.completeCycles,peakDwellMs:d.peakDwellMs,
      spacingMs:d.intervals.snapshot().spacingMs??null}:null;
  }
  finish(status,now,reader=null,phaseWatch=null){
    const a=this.active;if(!a)return null;
    const sender=this.config.role==='sender',ok=sender?'ack-read':'ack-shown';
    if(![ok,'timeout','cancelled','error'].includes(status)||!Number.isFinite(now)||now<a.startedAt)throw Error('Invalid step outcome.');
    const {startedAt,checkpointsMs:c,...step}=a;
    if(status===ok&&(sender?(c.probeDraw==null||c.ackDecoded==null):c.ackDraw==null))throw Error('Missing optical timing boundary.');
    const row={...step,status,elapsedMs:ms(now-startedAt),rttMs:status==='ack-read'?delta(c.ackDecoded,c.probeDraw):null,
      armToAckMs:status==='ack-shown'?c.ackDraw:null,lockToPacketMs:delta(c.packetDecoded,c.firstLock),
      checkpointsMs:{...c},stages:this.profiler.snapshot(),reader,phaseWatch};
    this.rows.push(row);this.active=null;this.profiler=null;return row;
  }
  quality(){
    let successfulSteps=0,timeouts=0,inferences=0,fecOrCrcFailures=0,matchedPackets=0,errors=0,cells=0;
    for(const r of this.rows){
      if(r.status==='ack-read'||r.status==='ack-shown')successfulSteps++;
      if(r.status==='timeout')timeouts++;
      inferences+=(r.reader?.inferences?.K4??0)+(r.reader?.inferences?.K8??0);
      fecOrCrcFailures+=r.reader?.fecFailures??0;
      if(r.matchedWireCER){matchedPackets++;errors+=r.matchedWireCER.errors;cells+=r.matchedWireCER.cells;}
    }
    return {steps:this.rows.length,successfulSteps,timeouts,inferences,fecOrCrcFailures,
      matchedWireCER:{matchedPackets,errors,cells,rate:cells?errors/cells:null}};
  }
  report({runId=null,model=null,camera=null,...details}={}){
    const quality=this.quality();
    return {format:'neural-qr-corrected-timing-sweep',version:1,started:this.started,runId,
      configuration:this.config,plannedSteps:this.count,recordedSteps:this.rows.length,
      successfulSteps:quality.successfulSteps,quality,
      optical:{phases:4,dataPhases:4,ackSize:this.config.ackSize,ackPhases:this.config.ackSize===16?1:4,probePayloadBytes:chunkBytesForSize(this.config.size)},
      model,camera,...details,
      timingNotes:{clock:'Local monotonic milliseconds; do not subtract timestamps between phones.',
        quality:'matchedWireCER compares predicted nibbles with the RS-corrected wire on CRC-verified, step-matched trained packets only, excluding pilots and any spare nibble. Unrecovered packets and classical G16 receipts have no such score. DATA cellSample sketches can be compared offline with the other report, including eligible unsuccessful inferences.',
        checkpoints:'Offsets from manual arm. firstLock and allPhases mark submission of qualifying camera captures, not the return of the inference.',
        rtt:'Sender: first submitted DATA draw to matching ACK checked. Receiver armToAckMs ends at its first ACK draw, and includes waiting for DATA.',
        display:'Painted phase changes, including held ACK until the receiver arms the next step, stops or exports. Display cadence does not extend step elapsed time. completeCycles counts four distinct phase draws in the same pass. peakDwellMs is the largest requested hold actually drawn; spacingMs measures time between draws.',
        capture:'captureSpacing is actual time between full captures; captureTargetSpacing records requested intervals only in timed capture/search. A busy worker can delay either.',
        phaseWatch:'When enabled after lock, a cheap ROI probe on fresh camera frames detects pixel changes. settleMs is the requested minimum; settleToCaptureMs is observed between the change callback and admitted capture callback, not between physical display and camera exposure. Watchdog reads check again even without a detected change.',
        stages:'Stage distributions include only operations actually run; worker round trip is an envelope, not an additional stage.'},
      steps:this.rows.map(row=>({...row,display:this.display(row.step)}))};
  }
}
