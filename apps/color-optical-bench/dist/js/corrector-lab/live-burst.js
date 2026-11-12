import {decodeNeuralQR,CONTROL_SIZE} from '../neural-qr/packet.js';

const rgbLength = size => 3*(4*size)**2;

/** Live Neural-QR accepts trained four-pixel, four-phase W-tube/VW only. */
export function requireTransportModel(manifest) {
  if(manifest?.status!=='trained'||!['W-tube','VW'].includes(manifest.modelProfile)||
    manifest.datasetFormat!=='neural-qr-dataset'||manifest.datasetVersion!==2||manifest.pixelsPerCell!==4||
    manifest.inputLayout!=='phase-paired-rgb-presence-planes'||manifest.outputLabelSpace!=='logical'||
    manifest.framesPerInput!==8||manifest.minFramesPerInput!==4||
    !manifest.cases?.some(entry=>entry.size===CONTROL_SIZE)) {
    throw Error('Neural-ZKBoo requires a trained W-tube or VW Neural-QR export including G32 controls. Export your checkpoint and run npm run prepare:corrector.');
  }
}

export const isTransportData = info => Boolean(info&&[16,32,64,128].includes(info.size)&&info.colors===16&&
  info.gridBlock===8&&info.phases===4&&info.permute===true&&info.pixelsPerCell===4&&
  Number.isInteger(info.phase)&&info.phase>=0&&info.phase<4);

/** Logits already use canonical logical colors; never reverse the phase twice. */
export function recoverCorrectedPayload(logits,info) {
  if(!isTransportData(info))throw Error('Unexpected Neural-QR transport matrix profile.');
  return decodeNeuralQR(logits,info.size);
}

/** Two genuine exposures per phase, one stop-and-wait candidate, at most 24 MiB at G128.
 * Phase/geometry are not packet identity. During packet transitions, mixed
 * observations may fail recovery; new exposures replace old ones until a valid
 * in-band packet is recovered. No pool similarity decision advances transfer.
 * Transport first attempts K4, then K8. A failed K8 needs four fresh captures
 * before another attempt; each replaces the oldest observation of its phase.
 * Failed detection never fabricates a capture or consumes a phase slot.
 */
export class LivePhaseBurst {
  constructor({maxAgeSeconds=60}={}) {
    if(!Number.isFinite(maxAgeSeconds)||maxAgeSeconds<=0||maxAgeSeconds>60)throw Error('Invalid live burst age limit.');
    this.maxAgeSeconds=maxAgeSeconds;this.reset();
  }
  reset() {
    this.context=null;this.geometryKey=null;this.info=null;this.groups=[[],[],[],[]];
    this.lastTime=-Infinity;this.lastFrame=0;this.attemptK=0;this.replacements=0;
    this.revision=0;this.attemptRevision=0;
  }
  observe({context,mediaTime,presentedFrames}={}) {
    if(typeof context!=='string'||!context||context.length>256||!Number.isFinite(mediaTime)||mediaTime<0||
      !Number.isSafeInteger(presentedFrames)||presentedFrames<1)throw Error('Invalid camera exposure identity.');
    if(context!==this.context){this.reset();this.context=context;}
    if(mediaTime<=this.lastTime||presentedFrames<=this.lastFrame)return false;
    this.lastTime=mediaTime;this.lastFrame=presentedFrames;
    if(this.groups.some(rows=>rows.some(row=>mediaTime-row.mediaTime>this.maxAgeSeconds))) {
      this.groups=[[],[],[],[]];this.geometryKey=null;this.info=null;this.attemptK=0;this.replacements=0;
      this.revision=0;this.attemptRevision=0;
    }
    return true;
  }
  status(extra={}) {
    const phaseCounts=this.groups.map(rows=>rows.length),collected=phaseCounts.reduce((sum,count)=>sum+count,0);
    return {ready:phaseCounts.every(Boolean),phaseCounts,collected,bufferBytes:collected*(this.info?rgbLength(this.info.size):0)*4,
      replacements:this.replacements,freshCaptures:this.revision-this.attemptRevision,...extra};
  }
  miss(identity) {return this.status({duplicateExposure:!this.observe(identity)});}
  add(crop,identity) {
    if(!isTransportData(crop?.info)||!(crop.input instanceof Float32Array)||crop.input.length!==rgbLength(crop.info.size)||
      crop.input.some(value=>!Number.isFinite(value)||Math.abs(value)>1.5)) {
      throw Error('Invalid phase-grouped transport capture.');
    }
    if(!this.observe(identity))return this.status({duplicateExposure:true,updated:false});
    const geometryKey=String(crop.info.size);
    if(geometryKey!==this.geometryKey) {
      this.groups=[[],[],[],[]];this.geometryKey=geometryKey;this.attemptK=0;this.replacements=0;
      this.revision=0;this.attemptRevision=0;
    }
    this.info={...crop.info};
    const rows=this.groups[crop.info.phase];
    if(rows.length===2){rows.shift();this.replacements++;}
    // Own the crop so a transferred/reused external buffer cannot mutate history.
    rows.push({input:crop.input.slice(),mediaTime:identity.mediaTime,presentedFrames:identity.presentedFrames,
      preprocessMs:Number(crop.stats?.ms??crop.stats?.preprocessMs)||0});
    this.revision++;
    return this.status({updated:true,duplicateExposure:false});
  }
  nextTransportSnapshot() {
    const status=this.status();
    if(!status.ready)return null;
    if(this.attemptK===0)return this.snapshot({frames:4});
    if(status.collected<8||status.freshCaptures<4)return null;
    return this.snapshot({frames:8});
  }
  snapshot({frames}={}) {
    const status=this.status();
    if(frames!==undefined&&![4,5,6,7,8].includes(frames))throw Error('Choose four through eight burst captures.');
    if(!status.ready||frames!==undefined&&status.collected<frames)return null;
    const count=frames??(this.attemptK===0?4:Math.min(status.collected,this.attemptK+1,8));
    const selected=this.groups.map((rows,phase)=>({...rows[rows.length-1],phase}));
    const extras=this.groups.flatMap((rows,phase)=>rows.length===2?[{...rows[0],phase}]:[])
      .sort((a,b)=>b.mediaTime-a.mediaTime);
    selected.push(...extras.slice(0,count-4));
    selected.sort((a,b)=>a.phase-b.phase||a.mediaTime-b.mediaTime);
    const length=rgbLength(this.info.size),input=new Float32Array(count*length),counts=[0,0,0,0],slots=[];
    selected.forEach((row,index)=>{input.set(row.input,index*length);slots.push(2*row.phase+counts[row.phase]++);});
    this.attemptK=count;this.attemptRevision=this.revision;
    return {...status,input,info:this.info,inputCaptures:count,phaseCounts:counts,
      phases:selected.map(row=>row.phase),slots,
      mediaTimes:selected.map(row=>row.mediaTime),presentedFrames:selected.map(row=>row.presentedFrames),
      preprocessMs:selected.reduce((sum,row)=>sum+row.preprocessMs,0)};
  }
}
