import {canonicalJSON} from '../dataset/contract.js';
import {validFrameCount, MAX_INPUT_FRAMES} from './runtime.js';

/** A bounded inference-only stack of distinct, already rectified exposures.
 * Header identity comes from each exposure; no sender labels or payload truth.
 */
export class CameraBurst {
  constructor(frames,{phaseGrouped=false}={}) {
    if (!validFrameCount(frames)) throw Error(`Camera burst requires 1–${MAX_INPUT_FRAMES} frames.`);
    if(typeof phaseGrouped!=='boolean'||phaseGrouped&&![4,5,6,7,8].includes(frames))throw Error('Phase-grouped camera bursts require four to eight captures.');
    this.frames=frames; this.phaseGrouped=phaseGrouped; this.reset();
  }
  reset() { this.context=null; this.header=null; this.rows=[]; this.input=null; this.lastTime=-Infinity; this.lastFrame=0; this.complete=false; }
  add(crop,{context,mediaTime,presentedFrames}={}) {
    const g=crop?.info?.size, length=3*(4*g)**2;
    if (typeof context!=='string'||!context||context.length>256||!Number.isFinite(mediaTime)||mediaTime<0 ||
        !Number.isSafeInteger(presentedFrames)||presentedFrames<1||
        ![64,128].includes(g)||![4,16].includes(crop.info.colors)||
        !/^[a-f0-9]{16}$/.test(crop.info.id)||![1,2,4,8,16].includes(crop.info.phases)||
        !Number.isInteger(crop.info.phase)||crop.info.phase<0||crop.info.phase>=crop.info.phases||
        !(crop.input instanceof Float32Array)||crop.input.length!==length||
        crop.input.some(value=>!Number.isFinite(value)||Math.abs(value)>1.5)) {
      this.reset(); throw Error('Invalid camera burst exposure.');
    }
    this.observe({context,mediaTime,presentedFrames});
    if(this.phaseGrouped&&(g!==64||crop.info.colors!==16||crop.info.phases!==4||crop.info.permute!==true)){
      this.reset();throw Error('Phase-grouped camera input requires a 64-cell, 16-color, four-phase permuted matrix.');
    }
    const identity={...crop.info};if(this.phaseGrouped)delete identity.phase;
    const header=canonicalJSON(identity);
    if (header!==this.header) {
      this.rows=[]; this.header=header; this.complete=false;
      // Assemble directly into one bounded tensor rather than retaining every
      // crop and then allocating a second full stack when collection completes.
      const capacity=this.phaseGrouped?8:this.frames;
      if (this.input?.length!==capacity*length) this.input=new Float32Array(capacity*length);
    }
    if(this.phaseGrouped)return this.addPhase(crop,{mediaTime,presentedFrames,length});
    if (this.rows.length>=this.frames) throw Error('Camera burst is complete; reset before collecting another.');
    this.input.set(crop.input,this.rows.length*length);
    this.rows.push({mediaTime,presentedFrames,stats:crop.stats??{}});
    const ready=this.rows.length===this.frames;
    return {ready,collected:this.rows.length,required:this.frames,input:ready?this.input:null,
      mediaTimes:this.rows.map(row=>row.mediaTime),presentedFrames:this.rows.map(row=>row.presentedFrames),
      preprocessMs:this.rows.reduce((sum,row)=>sum+(Number(row.stats.preprocessMs)||0),0)};
  }
  observe({context,mediaTime,presentedFrames}={}) {
    if(typeof context!=='string'||!context||context.length>256||!Number.isFinite(mediaTime)||mediaTime<0||
        !Number.isSafeInteger(presentedFrames)||presentedFrames<1)throw Error('Invalid camera frame identity.');
    if(context!==this.context){this.reset();this.context=context;}
    if(mediaTime<=this.lastTime||presentedFrames<=this.lastFrame)throw Error('A camera burst requires distinct increasing video frames.');
    this.lastTime=mediaTime;this.lastFrame=presentedFrames;
    if(this.phaseGrouped&&this.rows.length&&mediaTime-this.rows[0].mediaTime>60){
      this.rows=[];this.header=null;this.complete=false;
    }
  }
  miss(identity) {
    if(!this.phaseGrouped)throw Error('Transient capture retention requires phase grouping.');
    this.observe(identity);
    return {ready:false,collected:Math.min(this.frames,this.rows.length),required:this.frames,input:null,
      phaseGrouped:true,phaseCounts:[0,1,2,3].map(phase=>this.rows.filter(row=>row.phase===phase).length)};
  }
  addPhase(crop,{mediaTime,presentedFrames,length}) {
    if(this.complete)throw Error('Camera burst is complete; reset before collecting another.');
    const phase=crop.info.phase,quota=this.frames===4?1:2;
    // Ignore excess exposures from a held phase; retain a bounded eight-capture
    // buffer while waiting for the other phases of this same logical matrix.
    if(this.rows.filter(row=>row.phase===phase).length<quota) {
      this.input.set(crop.input,this.rows.length*length);
      this.rows.push({phase,mediaTime,presentedFrames,stats:crop.stats??{},index:this.rows.length});
    }
    const groups=[0,1,2,3].map(p=>this.rows.filter(row=>row.phase===p));
    const ready=groups.every(rows=>rows.length)&&this.rows.length>=this.frames;
    if(!ready)return {ready:false,collected:Math.min(this.frames,this.rows.length),required:this.frames,
      input:null,phaseGrouped:true,phaseCounts:groups.map(rows=>rows.length)};
    const selected=groups.map(rows=>rows[0]);
    const extras=groups.flatMap(rows=>rows.slice(1)).sort((a,b)=>a.mediaTime-b.mediaTime);
    selected.push(...extras.slice(0,this.frames-4));
    selected.sort((a,b)=>a.phase-b.phase||a.mediaTime-b.mediaTime);
    const input=new Float32Array(this.frames*length),counts=[0,0,0,0],slots=[];
    selected.forEach((row,i)=>{input.set(this.input.subarray(row.index*length,(row.index+1)*length),i*length);slots.push(2*row.phase+counts[row.phase]++);});
    this.complete=true;
    return {ready:true,collected:selected.length,required:this.frames,input,phaseGrouped:true,
      phaseCounts:counts,phases:selected.map(row=>row.phase),slots,bufferBytes:this.input.byteLength,
      mediaTimes:selected.map(row=>row.mediaTime),presentedFrames:selected.map(row=>row.presentedFrames),
      preprocessMs:selected.reduce((sum,row)=>sum+(Number(row.stats.preprocessMs)||0),0)};
  }
}
