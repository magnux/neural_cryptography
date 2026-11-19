// Short, shareable report for a completed live exchange. No frame, packet,
// protocol binding, key identifier or raw auth/worker snapshot is serialized.
import {STEPS} from './auth-progress.js';

const count=n=>Number.isSafeInteger(n)&&n>=0?n:null;
const ms=n=>Number.isFinite(n)&&n>=0?Math.round(n*10)/10:null;
const difference=(end,start)=>end===null||start===null?null:end-start;

/** Fixed-size boundary clock for the same steps displayed on this device.
 * Times are monotonic, independent of wall-clock changes or a delayed Finish
 * tap. A missed boundary is unknown, never an invented zero-duration step. */
export class ExchangeStepClock{
  constructor(startedAt){
    if(!Number.isFinite(startedAt))throw Error('A monotonic exchange start time is required.');
    this.origin=startedAt;
    this.starts=Array(STEPS.length).fill(null);
    this.starts[0]=0;this.setupRecorded=false;this.current=1;this.end=null;
  }
  offset(now){
    if(!Number.isFinite(now)||now<this.origin)return null;
    return Math.round(now-this.origin);
  }
  ready(now){
    const at=this.offset(now);
    if(at===null||this.setupRecorded||this.current!==1||this.end!==null)return;
    this.starts[0]=at;this.setupRecorded=true;
  }
  observe(number,now){
    const at=this.offset(now);
    if(this.end!==null||at===null||!Number.isInteger(number)||number<=this.current||
      number>STEPS.length||at<this.starts[this.current-1])return;
    this.starts[number-1]=at;this.current=number;
  }
  finish(now){
    const at=this.offset(now);
    if(this.end===null&&at!==null&&at>=this.starts[this.current-1])this.end=at;
  }
  snapshot(){
    return {elapsedMs:this.end,setupMs:this.setupRecorded?this.starts[0]:null,
      steps:STEPS.map((label,i)=>({number:i+1,label,
        durationMs:difference(i+1<STEPS.length?this.starts[i+1]:this.end,this.starts[i])}))};
  }
}

/** Whitelist the settings and aggregates needed to compare device runs. The
 * caller provides counts from the live run (not the simulation or timing tab).
 * packetRtt is the overall timing for this device's own START/DATA → ACK. */
export function buildExchangeSummary({settings,auth,started,clock,packetRtt,captureAttempts,acceptedMatrices,inference,model,camera}){
  const role=settings?.role;
  if(typeof auth?.result!=='boolean'||!(role==='verifier'?auth.resultDelivered===true:role==='prover'&&auth.peerReported===true))return null;
  const statisticsEnabled=settings.statistics===true;
  const timed=statisticsEnabled&&clock?clock.snapshot():null;
  const optical=statisticsEnabled&&count(packetRtt?.count)>0?{count:packetRtt.count,minMs:ms(packetRtt.minMs),
    meanMs:ms(packetRtt.meanMs),maxMs:ms(packetRtt.maxMs)}:null;
  const validDimension=n=>Number.isSafeInteger(n)&&n>0&&n<=32768;
  const actualCamera=statisticsEnabled&&validDimension(camera?.width)&&validDimension(camera.height)
    ?{width:camera.width,height:camera.height,frameRate:Number.isFinite(camera.frameRate)&&camera.frameRate>0&&camera.frameRate<=1000
      ?Math.round(camera.frameRate*10)/10:null}:null;
  return {format:'neural-auth-exchange-summary',version:1,started,role,
    deviceName:auth.localDisplayName??settings.displayName??null,peerName:auth.peerDisplayName??null,
    outcome:role==='verifier'?(auth.result?'verified-locally':'rejected-locally'):(auth.result?'peer-reported-accepted':'peer-reported-rejected'),
    repetitions:count(auth.repetitions??settings.repetitions),proofBytes:count(auth.proofBytes),
    settings:{phaseCadence:settings.phaseCadence,phaseMs:count(settings.phaseMs),phaseStepMs:count(settings.phaseStepMs),
      phaseMaxMs:count(settings.phaseMaxMs),firstCycle:settings.firstCycle===true,captureMs:count(settings.captureMs),outgoingSize:count(settings.outgoingSize),
      incomingMaxSize:count(settings.incomingMaxSize),optimized:settings.optimized===true,phaseWatch:settings.phaseWatch===true},
    statisticsEnabled,elapsedMs:timed?.elapsedMs??null,setupMs:timed?.setupMs??null,steps:timed?.steps??[],
    captureAttempts:statisticsEnabled?count(captureAttempts):null,acceptedMatrices:statisticsEnabled?count(acceptedMatrices):null,
    camera:actualCamera,packetRtt:optical,inference:statisticsEnabled&&count(inference?.count)>0?{count:inference.count,meanMs:ms(inference.meanMs)}:null,
    backend:model?.effectiveBackend??null,modelProfile:model?.modelProfile??null};
}
