// Scheduling hints only. Only the worker can identify a phase or accept bytes.
import {NQR_LAYOUT} from './js/neural-qr/layout.js';
import {StageProfiler} from './js/neural-auth/timing.js';

const SIDE=12, SETTLE=10;
const cross=(ax,ay,bx,by)=>ax*by-ay*bx;
function centerPatch(result,width,height){
  const q=result?.corners,g=result?.info?.size,p=result?.info?.phase;
  if(!NQR_LAYOUT.sizes.includes(g)||!Number.isInteger(p)||p<0||p>3||
    !Number.isFinite(width)||!Number.isFinite(height)||width<1||height<1||q?.length!==4||
    !q.every(v=>Array.isArray(v)&&v.length===2&&v.every(Number.isFinite)))return null;
  // Opposite diagonals intersect at the projective center (including mirrored
  // quads). A normalized ROI can be used on either the acquired canvas or the
  // raw video, even when acquisition downscales the source.
  const [[ax,ay],[bx,by],[cx,cy],[dx,dy]]=q;
  const rx=cx-ax,ry=cy-ay,sx=dx-bx,sy=dy-by,det=cross(rx,ry,sx,sy);
  if(Math.abs(det)<1e-6)return null;
  const t=cross(bx-ax,by-ay,sx,sy)/det,u=cross(bx-ax,by-ay,rx,ry)/det;
  if(t<=0||t>=1||u<=0||u>=1)return null;
  const x=ax+t*rx,y=ay+t*ry;
  let radius=Infinity;
  for(let i=0;i<4;i++){
    const [a,b]=q[i],[c,d]=q[(i+1)%4],length=Math.hypot(c-a,d-b);
    if(!length)return null;
    radius=Math.min(radius,Math.abs(cross(c-a,d-b,x-a,y-b))/length);
  }
  // Six cells across, rather than a fixed fraction of the matrix: averaging
  // dozens of random G128 cells would make successive phases look identical.
  const side=Math.min(64,2*radius*(NQR_LAYOUT.payload.side/NQR_LAYOUT.units)*6/g);
  if(side<2||x-side/2<0||y-side/2<0||x+side/2>width||y+side/2>height)return null;
  return {x:(x-side/2)/width,y:(y-side/2)/height,w:side/width,h:side/height,aspect:width/height};
}

export function createPhaseSampler(makeCanvas=()=>document.createElement('canvas')){
  let ctx;
  return {read(source,roi){
    if(!source||!roi)return null;
    const width=source.videoWidth??source.width,height=source.videoHeight??source.height;
    if(!(width>0&&height>0)||Math.abs(width/height/roi.aspect-1)>.01)return null;
    if(!ctx){
      const canvas=makeCanvas();canvas.width=canvas.height=SIDE;
      ctx=canvas.getContext('2d',{willReadFrequently:true});if(!ctx)return null;
    }
    ctx.drawImage(source,roi.x*width,roi.y*height,roi.w*width,roi.h*height,0,0,SIDE,SIDE);
    return ctx.getImageData(0,0,SIDE,SIDE).data;
  }};
}
function sample(fn,roi){
  // A dropped camera/context or unsupported readback must not wedge capture.
  try{const a=fn?.(roi);return a?.length===SIDE*SIDE*4?a:null;}catch{return null;}
}
function changed(a,b){
  // Ignore mild global exposure drift. Keep changes of hue (even at equal
  // brightness); require distributed differences rather than a single outlier.
  let drift=0;
  for(let i=0;i<a.length;i+=4)drift+=b[i]-a[i]+b[i+1]-a[i+1]+b[i+2]-a[i+2];
  const pixels=SIDE*SIDE,bias=Math.max(-18,Math.min(18,drift/(3*pixels)));
  let score=0,votes=0;
  for(let i=0;i<a.length;i+=4){
    const r=Math.abs(b[i]-a[i]-bias),g=Math.abs(b[i+1]-a[i+1]-bias),v=Math.abs(b[i+2]-a[i+2]-bias);
    score+=Math.min(r,72)+Math.min(g,72)+Math.min(v,72);if(Math.max(r,g,v)>=18)votes++;
  }
  return score>=12*3*pixels&&votes>=pixels*.25;
}

export class PhaseWatch{
  constructor({captureMs,statistics=false}){
    if(!Number.isFinite(captureMs)||captureMs<=0)throw Error('Invalid phase watch capture interval.');
    this.watchdogMs=Math.min(1000,Math.max(250,2*captureMs));
    this.stats=statistics?{probes:0,transitions:0,searchReads:0,changeReads:0,watchdogReads:0,locks:0,lostLocks:0,probeFailures:0}:null;
    this.timings=new StageProfiler(Boolean(this.stats));this.reset();
  }
  get locked(){return this.roi!==null;}
  clearLock(){this.roi=null;this.base=null;this.edge=null;this.phaseMiss=false;}
  reset(){this.clearLock();this.lastFrame=-1;this.lastRead=-Infinity;}
  lost(){if(this.locked&&this.stats)this.stats.lostLocks++;this.clearLock();}
  observe(result,width,height,readSubmitted){
    // Do not start a lock on tentative corners or even on validated corners of
    // an unwanted profile. A transient phase-cue failure can keep one existing
    // watch alive without promoting rejected geometry.
    if(result?.skipped&&!result.info)return;
    if(result?.stage==='phase'&&this.locked&&!this.phaseMiss&&['previous','detected'].includes(result.stats?.geometrySource)){
      this.phaseMiss=true;return;
    }
    const trusted=result?.ok===true||['collect','fec','crc','unchanged'].includes(result?.stage);
    const roi=trusted&&centerPatch(result,width,height);
    if(!roi){this.lost();return;}
    // Seed from the EXACT submitted exposure; busy inference may have spanned
    // multiple screen updates by the time this worker result arrives.
    const base=sample(readSubmitted,roi);
    if(!base){if(this.stats)this.stats.probeFailures++;this.lost();return;}
    if(!this.locked&&this.stats)this.stats.locks++;
    this.roi=roi;this.base=base;this.edge=null;this.phaseMiss=false;
  }
  decide(stamp,frame,readPatch,searchDue){
    if(!Number.isFinite(stamp)||!Number.isSafeInteger(frame)||frame<=this.lastFrame)return null;
    this.lastFrame=frame;
    const admit=(reason,patch)=>{
      this.lastRead=stamp;this.edge=null;if(patch)this.base=patch;
      if(this.stats)this.stats[reason+'Reads']++;
      return reason;
    };
    if(!this.locked)return searchDue()?admit('search'):null;
    if(this.stats)this.stats.probes++;
    const patch=this.timings.measure('probeMs',()=>sample(readPatch,this.roi));
    if(!patch){if(this.stats)this.stats.probeFailures++;this.lost();return searchDue()?admit('search'):null;}
    const overdue=stamp-this.lastRead>=this.watchdogMs;
    if(changed(this.base,patch)){
      if(!this.edge){
        this.edge={stamp,patch};if(this.stats)this.stats.transitions++;
        // When worker latency exceeds the watchdog, still allow this *newly
        // observed* edge its settling period before a new expensive scan.
        return null;
      }
      if(stamp-this.edge.stamp<SETTLE)return null;
      if(!changed(this.edge.patch,patch)){
        this.timings.record('settleToCaptureMs',stamp-this.edge.stamp);
        return admit('change',patch);
      }
      if(!overdue){this.edge={stamp,patch};return null;} // another moving exposure
    }else this.edge=null; // transient spike; still the already acquired phase
    return overdue?admit('watchdog',patch):null;
  }
  snapshot(){
    if(!this.stats)return null;
    const times=this.timings.snapshot();
    return {settleMs:SETTLE,watchdogMs:this.watchdogMs,sampleSide:SIDE,...this.stats,
      probeMs:times.probeMs??null,settleToCaptureMs:times.settleToCaptureMs??null};
  }
}
