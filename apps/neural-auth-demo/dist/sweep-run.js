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
export class SweepRun{
  constructor({role,size=64,startPhaseMs=500,startCaptureMs=450,decrementMs=50,timeoutMs=30000}){
    if(!['sender','receiver'].includes(role)||![32,64,128].includes(size))throw Error('Choose a sweep role and supported grid.');
    integer(startPhaseMs,50,2000,'Initial display interval');integer(startCaptureMs,50,2000,'Initial capture interval');
    integer(decrementMs,1,500,'Interval decrement');integer(timeoutMs,1000,120000,'Step timeout');
    this.config=Object.freeze({role,size,startPhaseMs,startCaptureMs,decrementMs,timeoutMs});
    this.count=Math.min(64,1+Math.floor((Math.min(startPhaseMs,startCaptureMs)-50)/decrementMs));
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
  // cadence on the originating row even if the receiver has armed a new step.
  painted(step,kind,tick,at){
    if(kind!==(this.config.role==='sender'?'data':'ack')||!Number.isSafeInteger(step)||step<1||
      step>(this.active?.step??this.rows.length)||!Number.isSafeInteger(tick)||tick<0||!Number.isFinite(at))return;
    let d=this.displays.get(step);
    if(!d){d={kind,changes:0,skippedSlots:0,phases:[0,0,0,0],lastTick:null,lastAt:null,intervals:new StageProfiler(true)};this.displays.set(step,d);}
    if(d.lastTick!==null){
      if(tick<=d.lastTick||at<d.lastAt)return;
      d.skippedSlots+=tick-d.lastTick-1;
      // A whole missed cycle can repaint the same phase without changing pixels.
      if(tick%4===d.lastTick%4){d.lastTick=tick;return;}
      d.intervals.record('spacingMs',at-d.lastAt);
    }
    d.changes++;d.phases[tick%4]++;d.lastAt=at;d.lastTick=tick;
  }
  display(step){
    const d=this.displays.get(step);
    return d?{kind:d.kind,phasePresentations:d.changes,byPhase:[...d.phases],skippedSlots:d.skippedSlots,
      spacingMs:d.intervals.snapshot().spacingMs??null}:null;
  }
  finish(status,now,reader=null){
    const a=this.active;if(!a)return null;
    const sender=this.config.role==='sender',ok=sender?'ack-read':'ack-shown';
    if(![ok,'timeout','cancelled','error'].includes(status)||!Number.isFinite(now)||now<a.startedAt)throw Error('Invalid step outcome.');
    const {startedAt,checkpointsMs:c,...step}=a;
    if(status===ok&&(sender?(c.probeDraw==null||c.ackDecoded==null):c.ackDraw==null))throw Error('Missing optical timing boundary.');
    const row={...step,status,elapsedMs:ms(now-startedAt),rttMs:status==='ack-read'?delta(c.ackDecoded,c.probeDraw):null,
      armToAckMs:status==='ack-shown'?c.ackDraw:null,lockToPacketMs:delta(c.packetDecoded,c.firstLock),
      checkpointsMs:{...c},stages:this.profiler.snapshot(),reader};
    this.rows.push(row);this.active=null;this.profiler=null;return row;
  }
  report({runId=null,model=null,camera=null,...details}={}){
    return {format:'neural-qr-corrected-timing-sweep',version:1,started:this.started,runId,
      configuration:this.config,plannedSteps:this.count,recordedSteps:this.rows.length,
      successfulSteps:this.rows.filter(r=>r.status==='ack-read'||r.status==='ack-shown').length,
      optical:{phases:4,ackSize:CONTROL_SIZE,probePayloadBytes:chunkBytesForSize(this.config.size)},
      model,camera,...details,
      timingNotes:{clock:'Local monotonic milliseconds; do not subtract timestamps between phones.',
        checkpoints:'Offsets from manual arm. firstLock and allPhases mark submission of qualifying camera captures, not the return of the inference.',
        rtt:'Sender: first submitted DATA draw to matching ACK checked. Receiver armToAckMs ends at its first ACK draw, and includes waiting for DATA.',
        display:'Painted phase changes, including held ACK until superseded, stop or export. Display cadence does not extend step elapsed time.',
        stages:'Stage distributions include only operations actually run; worker round trip is an envelope, not an additional stage.'},
      steps:this.rows.map(row=>({...row,display:this.display(row.step)}))};
  }
}
