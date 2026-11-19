// One local measurement direction per device. No trial pattern, expected cells,
// dummy tensor, or simulation model enters this path.
import {CorrectorClient} from './js/corrector-lab/client.js';
import {DATA_SIZES,CONTROL_SIZE} from './js/neural-qr/packet.js';
import {StageProfiler} from './js/neural-auth/timing.js';

import {PHASE_REJECTIONS} from './js/neural-qr/vision.js';

const TRACE_HALF=32;
const round=ms=>Math.round(ms*10)/10;
function abort(){return new DOMException('Timing step stopped.','AbortError');}
function state(start){
  return {start,scans:0,submitted:0,stages:new StageProfiler(true),envelopes:new StageProfiler(true),stageScans:{},outcomes:{},
    phaseRejections:{},rejectedCueNearest:[[0,0,0,0],[0,0,0,0]],
    phases:[0,0,0,0],firstPhaseMs:[null,null,null,null],firstAllPhasesMs:null,previousMedia:null,
    backends:{},geometry:{hintAttempts:0,detected:0,previous:0,none:0},inferences:{K4:0,K8:0},
    fecFailures:0,recovered:0,duplicates:0,head:[],tail:[]};
}
function keep(s,result,mediaTime,begin,end){
  s.scans++;
  const stage=result.stage??(result.ok?'decoded':'other'),phase=result.info?.phase??null;
  const timings=result.timings??{},source=result.stats?.geometrySource??'none';
  if(result.stats?.previousGeometryTried)s.geometry.hintAttempts++;
  if(source==='detected'||source==='previous')s.geometry[source]++;else s.geometry.none++;
  s.outcomes[stage]=(s.outcomes[stage]??0)+1;
  s.stages.merge(timings);for(const name of Object.keys(timings))s.stageScans[name]=(s.stageScans[name]??0)+1;
  s.envelopes.record('workerRoundTripMs',end-begin);
  if(s.previousMedia!==null&&mediaTime>s.previousMedia)s.envelopes.record('exposureIntervalMs',1000*(mediaTime-s.previousMedia));
  s.previousMedia=mediaTime;
  // Timestamp the submitted exposure, not its reply: on the fourth phase the
  // reply can include several hundred milliseconds of model inference.
  const offset=round(begin-s.start),k=result.inputCaptures??null;
  if(result.duplicateExposure)s.duplicates++;
  if(timings['burst-collection']&&!result.duplicateExposure&&Number.isInteger(phase)&&phase>=0&&phase<4){
    s.phases[phase]++;s.firstPhaseMs[phase]??=offset;
    if(s.firstAllPhasesMs===null&&s.phases.every(n=>n>0))s.firstAllPhasesMs=offset;
  }
  if(timings.inference&&(k===4||k===8)){
    s.inferences['K'+k]++;
    const backend=['wasm','webgpu'].includes(result.effectiveBackend)?result.effectiveBackend:'other';
    s.backends[backend]=(s.backends[backend]??0)+1;
  }
  if(stage==='fec'||stage==='crc')s.fecFailures++;
  if(result.ok)s.recovered++; // successful RS/CRC; caller must still validate run and trial
  const row={startMs:offset,endMs:round(end-s.start),phase,stage,geometry:source,k};
  if(stage==='phase'){
    const rejection=result.stats?.phaseRejection;
    const reason=PHASE_REJECTIONS.includes(rejection?.reason)?rejection.reason:'other';
    s.phaseRejections[reason]=(s.phaseRejections[reason]??0)+1;row.phaseRejection=reason;
    const nearest=rejection?.nearest,passes=rejection?.passes;
    if(nearest?.length===2&&nearest.every(n=>Number.isInteger(n)&&n>=0&&n<4)){
      row.nearestCuePhases=[...nearest];nearest.forEach((n,i)=>s.rejectedCueNearest[i][n]++);
      if(passes?.length===2&&passes.every(v=>typeof v==='boolean'))row.cuePassed=[...passes];
    }
  }
  if(s.head.length<TRACE_HALF)s.head.push(row);
  else{if(s.tail.length===TRACE_HALF)s.tail.shift();s.tail.push(row);}
}
export class SweepReader{
  constructor({size,role,backend='auto',onProgress,client,now=()=>performance.now()}){
    if(!DATA_SIZES.includes(size)||!['sender','receiver'].includes(role)||!['auto','wasm','webgpu'].includes(backend))
      throw Error('Invalid trained sweep role, grid or backend.');
    this.expected=role==='sender'?'ack':'data';
    this.size=role==='sender'?CONTROL_SIZE:size;
    this.backend=backend;this.client=client??new CorrectorClient({onProgress});this.now=now;
    this.runId=0;this.current=null;this.ready=false;this.closed=false;
  }
  async init(){
    if(this.closed||this.ready)throw Error('Start a fresh sweep reader.');
    const begin=this.now();
    const r=await this.client.call('transport-init',{size:this.size,backend:this.backend,statistics:true});
    if(this.closed)throw abort();
    if(!r.ready||r.modelStatus!=='trained'||r.parity?.passed!==true)
      throw Error('The timing sweep requires a parity-checked, trained Neural-QR corrector.');
    this.ready=true;
    return {modelProfile:r.modelProfile,modelStatus:r.modelStatus,effectiveBackend:r.effectiveBackend,
      requestedBackend:this.backend,modelSha256:r.modelSha256??null,incomingMax:this.size,
      initializationMs:round(this.now()-begin),parityPassed:true};
  }
  beginStep(){
    if(!this.ready||this.closed)throw Error('Initialize the trained corrector first.');
    // ReceivePipeline.reset() on the first scan of a new runId discards all
    // previous geometry, accepted gates and burst samples but retains the model.
    this.runId++;this.current=state(this.now());
  }
  async scan(image,{mediaTime,presentedFrames,captureIntervalMs}){
    const s=this.current;if(!s||this.closed)throw Error('Start a sweep step first.');
    // Canonical live optimized path: dense calibrated crop, no duplicate
    // cell-sampling pass. The first corrected packet ends this step, so we
    // need neither a post-acceptance frame gate nor a second worker RPC.
    const begin=this.now();s.submitted++;
    const result=await this.client.call('transport-scan',{image,runId:this.runId,mediaTime,presentedFrames,
      captureIntervalMs,size:this.size,backend:this.backend,expected:this.expected,
      optimized:true,forceAcquisition:true,statistics:true},[image.data.buffer]);
    if(this.closed||this.current!==s)throw abort();
    keep(s,result,mediaTime,begin,this.now());return result;
  }
  snapshot(){
    const s=this.current;if(!s)return null;
    const stages=s.stages.snapshot();
    for(const [name,row] of Object.entries(stages)){
      row.measuredScans=s.stageScans[name];row.avgMsPerMeasuredScan=row.totalMs/row.measuredScans;
    }
    const {workerRoundTripMs=null,exposureIntervalMs=null}=s.envelopes.snapshot();
    return {scans:s.scans,submitted:s.submitted,withoutResult:s.submitted-s.scans,stages,workerRoundTripMs,exposureIntervalMs,geometry:{...s.geometry},
      phaseRejections:{...s.phaseRejections},rejectedCueNearest:s.rejectedCueNearest.map(r=>[...r]),
      phases:[...s.phases],firstPhaseMs:[...s.firstPhaseMs],firstAllPhasesMs:s.firstAllPhasesMs,
      inferences:{...s.inferences},inferencesByBackend:{...s.backends},fecFailures:s.fecFailures,recovered:s.recovered,duplicates:s.duplicates,outcomes:{...s.outcomes},
      trace:[...s.head,...s.tail],traceLimit:2*TRACE_HALF,omittedScans:Math.max(0,s.scans-2*TRACE_HALF)};
  }
  close(){this.closed=true;this.client.cancel();}
}
