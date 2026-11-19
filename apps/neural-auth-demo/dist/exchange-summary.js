// Short, shareable report for a completed live exchange. No frame, packet,
// protocol binding, key identifier or raw auth/worker snapshot is serialized.
import {STEPS} from './auth-progress.js';

const count=n=>Number.isSafeInteger(n)&&n>=0?n:null;
const ms=n=>Number.isFinite(n)&&n>=0?Math.round(n*10)/10:null;
const difference=(end,start)=>end===null||start===null?null:end-start;

// Per-inference recovery counts for live inputs. Before RS/CRC a failed read
// has no verified packet identity; group by expected turn and observed size.
// Only 5 possible groups (DATA G32/64/128 and G32 ACK/REQUEST). No cell arrays.
const validGroup=(expected,grid)=>expected==='data'?[32,64,128].includes(grid):['request','ack'].includes(expected)&&grid===32;
const recoveryCounts=['inferences','rawCRCPassed','rsCRCPassed','rsFailures','crcFailures','correctedBytesOnCRCPass','transportAccepted','duplicates','acceptedWireCellErrors','acceptedWireCells','acceptedPacketsWithCER'];
export function recordTrainedInference(rows,expected,{info,inputCaptures,stats}){
  const grid=info?.size;
  if(!validGroup(expected,grid)||![4,8].includes(inputCaptures)||![0,1].includes(stats?.rawCRCValid))return null;
  let row=rows.find(r=>r.expected===expected&&r.grid===grid);
  if(!row){row={expected,grid,...Object.fromEntries(recoveryCounts.map(k=>[k,0]))};rows.push(row);}
  row.inferences++;
  if(stats.rawCRCValid===1)row.rawCRCPassed++;
  if(stats.rsDecoded===0)row.rsFailures++;
  else if(stats.crcAccepted===0)row.crcFailures++;
  else if(stats.crcAccepted===1){row.rsCRCPassed++;row.correctedBytesOnCRCPass+=count(stats.correctedBytes)??0;}
  return row;
}
// A repeated valid old image may recover optically without advancing a turn.
export function recordTrainedAcceptance(row,acceptance,stats){
  if(!acceptance?.accepted)return;
  if(['duplicate','duplicate-start','already-delivered','duplicate-request'].includes(acceptance.reason)){row.duplicates++;return;}
  row.transportAccepted++;
  // An intact RS codeword is a reference only after CRC and a matching
  // transport acceptance. Failed inferences cannot be assigned a CER here.
  const errors=count(stats?.wireCellErrors),cells=count(stats?.wireCells);
  if(stats?.crcAccepted===1&&errors!==null&&cells>0&&errors<=cells){
    row.acceptedPacketsWithCER++;row.acceptedWireCellErrors+=errors;row.acceptedWireCells+=cells;
  }
}
function summarizeRecovery(rows){
  return rows.filter(row=>validGroup(row.expected,row.grid)).map(({expected,grid,...counters})=>({
    expected,grid,...Object.fromEntries(recoveryCounts.map(k=>[k,count(counters[k])]))
  })).sort((a,b)=>a.grid-b.grid||a.expected.localeCompare(b.expected));
}

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
export function buildExchangeSummary({settings,auth,started,clock,packetRtt,captureAttempts,acceptedMatrices,inference,model,camera,receiptReads,trainedRecovery}){
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
    settings:{localColorCorrection:settings.localColorCorrection===true,phaseCadence:settings.phaseCadence,phaseMs:count(settings.phaseMs),phaseStepMs:count(settings.phaseStepMs),
      phaseMaxMs:count(settings.phaseMaxMs),ackSize:count(settings.ackSize),firstCycle:settings.firstCycle===true,captureMs:count(settings.captureMs),outgoingSize:count(settings.outgoingSize),
      incomingMaxSize:count(settings.incomingMaxSize),optimized:settings.optimized===true,phaseWatch:settings.phaseWatch===true},
    statisticsEnabled,elapsedMs:timed?.elapsedMs??null,setupMs:timed?.setupMs??null,steps:timed?.steps??[],
    captureAttempts:statisticsEnabled?count(captureAttempts):null,acceptedMatrices:statisticsEnabled?count(acceptedMatrices):null,
    trainedRecovery:statisticsEnabled&&trainedRecovery?summarizeRecovery(trainedRecovery):null,
    receiptReads:statisticsEnabled&&settings.ackSize===16&&receiptReads?{scans:count(receiptReads.scans),decoded:count(receiptReads.decoded),
      matched:count(receiptReads.matched),unmatched:count(receiptReads.unmatched)}:null,
    camera:actualCamera,packetRtt:optical,inference:statisticsEnabled&&count(inference?.count)>0?{count:inference.count,meanMs:ms(inference.meanMs)}:null,
    backend:model?.effectiveBackend??null,modelProfile:model?.modelProfile??null};
}
