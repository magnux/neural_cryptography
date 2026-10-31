import {canonicalJSON} from '../dataset/contract.js';

/** A bounded inference-only stack of distinct, already rectified exposures.
 * Header identity comes from each exposure; no sender labels or payload truth.
 */
export class CameraBurst {
  constructor(frames) {
    if (![1,4].includes(frames)) throw Error('Camera burst requires one or four frames.');
    this.frames=frames; this.reset();
  }
  reset() { this.context=null; this.header=null; this.rows=[]; this.lastTime=-Infinity; this.lastFrame=0; }
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
    if (context!==this.context) { this.reset(); this.context=context; }
    if (mediaTime<=this.lastTime||presentedFrames<=this.lastFrame) throw Error('A camera burst requires distinct increasing video frames.');
    this.lastTime=mediaTime;this.lastFrame=presentedFrames;
    const header=canonicalJSON(crop.info);
    if (header!==this.header) { this.rows=[]; this.header=header; }
    if (this.rows.length>=this.frames) throw Error('Camera burst is complete; reset before collecting another.');
    this.rows.push({input:crop.input,mediaTime,presentedFrames,stats:crop.stats??{}});
    const ready=this.rows.length===this.frames;
    let input=null;
    if (ready) {
      input=new Float32Array(this.frames*length);
      this.rows.forEach((row,index)=>input.set(row.input,index*length));
    }
    return {ready,collected:this.rows.length,required:this.frames,input,
      mediaTimes:this.rows.map(row=>row.mediaTime),presentedFrames:this.rows.map(row=>row.presentedFrames),
      preprocessMs:this.rows.reduce((sum,row)=>sum+(Number(row.stats.preprocessMs)||0),0)};
  }
}
