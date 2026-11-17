import {TransportCapture} from './capture.js';
import {AcceptedFrameGate} from './frame-gate.js';
import {StageProfiler} from './timing.js';
import {LivePhaseBurst,isTransportData,recoverCorrectedPayload} from '../corrector-lab/live-burst.js';
import {packFrameStack} from '../corrector-lab/runtime.js';
import {CONTROL_SIZE} from '../neural-qr/packet.js';
import {geometryHintMaxAgeMs} from '../neural-qr/vision.js';

/** Shared real receive path. The browser worker supplies the trained model
 * loader/inference. Simulation may supply its own inference function; this
 * module has no clean symbols, targets, or way to bypass RS/CRC recovery. */
export class ReceivePipeline {
  constructor({loadContext,infer}) {
    if(typeof loadContext!=='function'||typeof infer!=='function')throw Error('Receive pipeline requires model loading and inference.');
    this.loadContext=loadContext;this.infer=infer;this.capture=new TransportCapture();this.burst=new LivePhaseBurst();
    this.gate=new AcceptedFrameGate();this.serial=0;this.reset();
  }
  reset(){
    this.capture.reset();this.burst.reset();this.gate.reset();this.context=null;this.runId=null;this.lastTime=-Infinity;
    this.records=[];this.pending=null;this.profiler=new StageProfiler(false);this.counters={scans:0,gated:0,inferences:0,decoded:0,accepted:0};
  }
  snapshot(){return {...this.counters,timings:this.profiler.snapshot(),bufferBytes:this.burst.status().bufferBytes};}
  accept({token,runId,expected}={}) {
    const pending=this.pending;
    if(!pending||pending.token!==token||pending.identity.runId!==runId||pending.identity.expected!==expected)
      return {operation:'transport-accept',accepted:false,reason:'stale-candidate'};
    this.pending=null;if(this.profiler.enabled)this.counters.accepted++;
    const probeGate=this.gate.accept(pending.records,pending.identity);
    return {operation:'transport-accept',accepted:true,probeGate};
  }
  async scan(raw) {
    const {image,runId,mediaTime,presentedFrames,expected,captureIntervalMs=100,maxSize,optimized=false,statistics=false,probe,forceFullScan=false,forceAcquisition=false}=raw;
    geometryHintMaxAgeMs(captureIntervalMs);
    if(!['data','ack','request'].includes(expected)||!Number.isSafeInteger(runId)||runId<1||!Number.isFinite(mediaTime)||mediaTime<0||
      !Number.isSafeInteger(presentedFrames)||presentedFrames<1||![32,64,128].includes(maxSize))throw Error('Invalid Neural-ZKBoo capture request.');
    const key=`${runId}/${expected}`;
    if(this.runId!==runId||mediaTime<this.lastTime){this.reset();this.runId=runId;}
    if(this.context!==key){this.capture.reset();this.burst.reset();this.gate.reset();this.records=[];this.pending=null;this.context=key;}
    this.lastTime=mediaTime;this.profiler.enabled=statistics===true;if(statistics)this.counters.scans++;
    // The consumer must acknowledge the most recently returned candidate before
    // requesting another exposure. A delayed old confirmation cannot arm a gate.
    this.pending=null;
    const timing=new StageProfiler(statistics),identity={context:key,runId,expected,mediaTime,presentedFrames};
    const finish=result=>{const timings=timing.snapshot();if(timings){this.profiler.merge(timings);result.timings=timings;result.statistics={...this.counters};}return result;};
    const unchanged=(capture={})=>{
      if(statistics)this.counters.gated++;
      return finish({operation:'transport-scan',transport:'neural-qr',expected,...this.burst.miss(identity),ok:false,waiting:true,
        stage:'unchanged',skipped:true,gateReason:'accepted-matrix',error:null,method:'neural-corrector',
        info:capture.info??null,corners:capture.corners??[],stats:{...capture.stats,acceptedMatrixUnchanged:true}});
    };
    if(optimized&&!forceFullScan&&timing.measure('frame-gate',()=>this.gate.check(image,identity,{forceFullScan:forceAcquisition})))return unchanged();
    const before=this.burst.status();
    const profileAllowed=info=>isTransportData(info)&&info.size>=CONTROL_SIZE&&info.size<=maxSize&&
      (expected==='data'||info.size===CONTROL_SIZE);
    let rejectedProfile=false;
    const capture=this.capture.scan(image,{expected,context:key,mediaTime,captureIntervalMs,profiler:timing,collectTimings:raw.statistics!==false,includeSamples:!optimized,
      beforeRaster:current=>{
        // A held DATA image cannot be the G32 control we are waiting for. Its
        // size cues suffice; do not allocate a crop or run the neural corrector.
        if(!profileAllowed(current.info)){rejectedProfile=true;return false;}
        return !optimized||forceFullScan||!timing.measure('rectified-gate',()=>this.gate.matchesCapture(image,current,identity));
      }});
    if(capture.skippedRaster&&!rejectedProfile)return unchanged(capture);
    const base={operation:'transport-scan',transport:'neural-qr',expected,info:capture.info??null,
      corners:capture.corners??[],tentativeCorners:capture.tentativeCorners??[],markers:capture.markers??[],
      phaseCounts:before.phaseCounts,collected:before.collected,stats:capture.stats??{},method:'neural-corrector'};
    if(!capture.ok)return finish({...base,...this.burst.miss(identity),ok:false,waiting:true,stage:capture.stage,error:capture.error});
    const size=capture.info.size;
    if(rejectedProfile||!profileAllowed(capture.info)){
      this.burst.reset();this.records=[];
      return finish({...base,...this.burst.miss(identity),ok:false,waiting:true,stage:'profile',
        error:expected!=='data'?`Waiting for a G${CONTROL_SIZE} Neural-QR ${expected==='ack'?'ACK':'REQUEST'}.`:`G${size} exceeds the receiving profile; maximum is G${maxSize}.`});
    }
    const state=timing.measure('burst-collection',()=>this.burst.add(capture,identity));
    if(optimized&&state.updated){
      this.records=this.records.filter(row=>row.size===size);
      this.records.push(timing.measure('fingerprint',()=>this.gate.record(image,capture,identity,probe)));
      // Keep the same two most recent genuine exposures per phase as the burst.
      for(let phase=0;phase<4;phase++)while(this.records.filter(row=>row.phase===phase).length>2)this.records.splice(this.records.findIndex(row=>row.phase===phase),1);
    }
    if(!state.ready||state.duplicateExposure)return finish({...base,...state,ok:false,waiting:true,stage:'collect',error:null});
    const burst=timing.measure('burst-snapshot',()=>this.burst.nextTransportSnapshot());
    if(!burst)return finish({...base,...state,ok:false,waiting:true,stage:'collect',error:null});
    const context=await timing.measure('model-load',()=>this.loadContext(size,{statistics:raw.statistics!==false}));
    const packed=timing.measure('burst-pack',()=>packFrameStack(burst.input,context.manifest,size,{phases:burst.phases,slots:burst.slots}));
    if(statistics)this.counters.inferences++;
    const result=await timing.measure('inference',()=>this.infer(context,packed,{statistics:raw.statistics!==false}));
    const decoded=timing.measure('rs-crc-recovery',()=>recoverCorrectedPayload(result.values,burst.info));
    const response={...base,...decoded,waiting:!decoded.ok,info:burst.info,inputCaptures:burst.inputCaptures,
      phaseCounts:burst.phaseCounts,collected:state.collected,replacements:state.replacements,
      bufferBytes:state.bufferBytes,...(raw.statistics===false?{}:{inferenceMs:result.ms,preprocessMs:capture.stats.ms}),
      effectiveBackend:context.effectiveBackend,stats:{...base.stats,...decoded.stats}};
    if(decoded.ok){
      if(statistics)this.counters.decoded++;
      if(optimized){
        const token=`${key}/${++this.serial}`;
        this.pending={token,identity,records:this.records.filter(row=>burst.presentedFrames.includes(row.presentedFrames))};
        response.acceptToken=token;
      }
      this.burst.reset();this.records=[];
    }
    return finish(response);
  }
}
