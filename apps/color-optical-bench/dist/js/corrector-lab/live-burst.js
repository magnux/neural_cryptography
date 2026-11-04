import {frameKey} from '../format.js';
import {logitsToLogicalScores} from '../learned-input.js';
import {decodeScores} from '../vision.js';

const RGB_LENGTH=3*256*256;

/** Live authentication deliberately accepts only the trained phase model. */
export function requireTransportModel(manifest) {
  if(manifest?.status!=='trained'||manifest.modelProfile!=='U-tube-2x-attn'||
    manifest.inputLayout!=='phase-paired-rgb-presence-planes'||manifest.outputLabelSpace!=='logical'||
    manifest.framesPerInput!==8||manifest.minFramesPerInput!==4||manifest.cases?.length!==1||manifest.cases[0].size!==64) {
    throw Error('Neural-ZKBoo requires a trained U-tube-2x-attn G64 export. Export your trained checkpoint and run npm run prepare:corrector.');
  }
}

export const isTransportData = info => Boolean(info&&info.size===64&&info.colors===16&&
  info.gridBlock===8&&info.parity===32&&info.phases===4&&info.permute===true&&
  Number.isInteger(info.phase)&&info.phase>=0&&info.phase<4&&info.count===1&&info.index===0&&
  info.content?.type==='raw'&&info.probePattern===0);

/** The corrector already emits canonical logical labels. Never invert twice. */
export function recoverCorrectedPayload(logits,info) {
  if(!isTransportData(info))throw Error('Unexpected Neural-ZKBoo data matrix profile.');
  const scores=logitsToLogicalScores(logits,info,{labelSpace:'logical'});
  const decoded=decodeScores(info,scores,{}, {recovery:'erasures'});
  return {ok:decoded.ok,stage:decoded.stage,error:decoded.error,stats:decoded.stats,
    ...(decoded.ok?{chunk:decoded.chunk}: {})};
}

/** Two genuine exposures per phase, one current logical matrix, at most 6 MiB.
 * First attempt uses one frame per phase. Later attempts grow K through eight,
 * then replace the oldest observation of the newly observed phase indefinitely.
 * Failed detection never fabricates a capture or consumes a phase slot.
 */
export class LivePhaseBurst {
  constructor({maxAgeSeconds=60}={}) {
    if(!Number.isFinite(maxAgeSeconds)||maxAgeSeconds<=0||maxAgeSeconds>60)throw Error('Invalid live burst age limit.');
    this.maxAgeSeconds=maxAgeSeconds;this.reset();
  }
  reset() {
    this.context=null;this.header=null;this.info=null;this.groups=[[],[],[],[]];
    this.lastTime=-Infinity;this.lastFrame=0;this.attemptK=0;this.replacements=0;
  }
  observe({context,mediaTime,presentedFrames}={}) {
    if(typeof context!=='string'||!context||context.length>256||!Number.isFinite(mediaTime)||mediaTime<0||
      !Number.isSafeInteger(presentedFrames)||presentedFrames<1)throw Error('Invalid camera exposure identity.');
    if(context!==this.context){this.reset();this.context=context;}
    if(mediaTime<=this.lastTime||presentedFrames<=this.lastFrame)return false;
    this.lastTime=mediaTime;this.lastFrame=presentedFrames;
    if(this.groups.some(rows=>rows.some(row=>mediaTime-row.mediaTime>this.maxAgeSeconds))) {
      this.groups=[[],[],[],[]];this.header=null;this.info=null;this.attemptK=0;this.replacements=0;
    }
    return true;
  }
  status(extra={}) {
    const phaseCounts=this.groups.map(rows=>rows.length),collected=phaseCounts.reduce((sum,count)=>sum+count,0);
    return {ready:phaseCounts.every(Boolean),phaseCounts,collected,bufferBytes:collected*RGB_LENGTH*4,
      replacements:this.replacements,...extra};
  }
  miss(identity) {return this.status({duplicateExposure:!this.observe(identity)});}
  add(crop,identity) {
    if(!isTransportData(crop?.info)||!(crop.input instanceof Float32Array)||crop.input.length!==RGB_LENGTH||
      crop.input.some(value=>!Number.isFinite(value)||Math.abs(value)>1.5)) {
      throw Error('Invalid phase-grouped transport capture.');
    }
    if(!this.observe(identity))return this.status({duplicateExposure:true,updated:false});
    const header=frameKey(crop.info);
    if(header!==this.header) {
      this.groups=[[],[],[],[]];this.header=header;this.attemptK=0;this.replacements=0;
    }
    this.info={...crop.info};
    const rows=this.groups[crop.info.phase];
    if(rows.length===2){rows.shift();this.replacements++;}
    // Own the crop so a transferred/reused external buffer cannot mutate history.
    rows.push({input:crop.input.slice(),mediaTime:identity.mediaTime,presentedFrames:identity.presentedFrames,
      preprocessMs:Number(crop.stats?.preprocessMs)||0});
    return this.status({updated:true,duplicateExposure:false});
  }
  snapshot() {
    const status=this.status();
    if(!status.ready)return null;
    const count=this.attemptK===0?4:Math.min(status.collected,this.attemptK+1,8);
    const selected=this.groups.map((rows,phase)=>({...rows[rows.length-1],phase}));
    const extras=this.groups.flatMap((rows,phase)=>rows.length===2?[{...rows[0],phase}]:[])
      .sort((a,b)=>b.mediaTime-a.mediaTime);
    selected.push(...extras.slice(0,count-4));
    selected.sort((a,b)=>a.phase-b.phase||a.mediaTime-b.mediaTime);
    const input=new Float32Array(count*RGB_LENGTH),counts=[0,0,0,0],slots=[];
    selected.forEach((row,index)=>{input.set(row.input,index*RGB_LENGTH);slots.push(2*row.phase+counts[row.phase]++);});
    this.attemptK=count;
    return {...status,input,info:this.info,inputCaptures:count,phaseCounts:counts,
      phases:selected.map(row=>row.phase),slots,
      mediaTimes:selected.map(row=>row.mediaTime),presentedFrames:selected.map(row=>row.presentedFrames),
      preprocessMs:selected.reduce((sum,row)=>sum+row.preprocessMs,0)};
  }
}
