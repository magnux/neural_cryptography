import {project,sample} from '../vision.js';
import {NQR_LAYOUT} from '../neural-qr/layout.js';

const TTL_MS=4000,FULL_SCAN_MS=1000,POINTS=16;
const validImage=image=>image&&Number.isInteger(image.width)&&Number.isInteger(image.height)&&image.width>0&&image.height>0&&
  (image.data instanceof Uint8Array||image.data instanceof Uint8ClampedArray)&&image.data.length===image.width*image.height*4;
const contextKey=({runId,expected})=>`${runId}/${expected}`;

/** Sparse camera-space ROI samples, with one template per optical phase. These
 * are only a work gate: they never identify, decode or accept a packet. */
export function payloadFingerprint(image,geometry) {
  if(!validImage(image)||geometry?.width!==image.width||geometry?.height!==image.height||!Array.isArray(geometry.transform))return null;
  const p=NQR_LAYOUT.payload,values=new Uint8Array(POINTS*POINTS*3);let at=0;
  for(let y=0;y<POINTS;y++)for(let x=0;x<POINTS;x++){
    const rgb=sample(image,...project(geometry.transform,p.x+(x+.5)*p.side/POINTS,p.y+(y+.5)*p.side/POINTS));
    if(!rgb)return null;
    for(const v of rgb)values[at++]=Math.max(0,Math.min(255,Math.round(v)));
  }
  return values;
}

/** Conservative noise tolerance; uncertain or moving images always rescan. */
export function similarFingerprint(a,b) {
  if(!a||!b||a.length!==b.length||!a.length)return false;
  let total=0,changed=0;
  for(let i=0;i<a.length;i++){const delta=Math.abs(a[i]-b[i]);total+=delta;if(delta>12)changed++;}
  return total/a.length<=3&&changed/a.length<=.02;
}

function probePixels(image) {
  if(!validImage(image)||image.width!==32||image.height!==32)return null;
  const values=new Uint8Array(32*32*3);let at=0;
  for(let i=0;i<image.data.length;i+=4){values[at++]=image.data[i];values[at++]=image.data[i+1];values[at++]=image.data[i+2];}
  return values;
}

export class AcceptedFrameGate {
  constructor(){this.reset();}
  reset(){this.context=null;this.acceptedAt=-Infinity;this.lastFull=-Infinity;this.templates=[];}
  valid(identity){return this.context===contextKey(identity)&&identity.mediaTime*1000>=this.acceptedAt&&identity.mediaTime*1000-this.acceptedAt<TTL_MS;}
  accept(records,identity) {
    this.reset();this.context=contextKey(identity);this.acceptedAt=identity.mediaTime*1000;this.lastFull=this.acceptedAt;
    this.templates=records.filter(row=>row.fingerprint).slice(-8);
    return {context:this.context,acceptedAt:this.acceptedAt,ttlMs:TTL_MS,fullScanMs:FULL_SCAN_MS,
      templates:this.templates.filter(row=>row.probe).map(row=>({phase:row.phase,values:row.probe.slice()}))};
  }
  check(image,identity,{forceFullScan=false}={}) {
    if(!this.valid(identity))return false;
    const now=identity.mediaTime*1000;
    if(forceFullScan||now-this.lastFull>=FULL_SCAN_MS){this.lastFull=now;return false;}
    const fingerprints=new Map();
    return this.templates.some(row=>{
      const key=JSON.stringify(row.geometry);
      if(!fingerprints.has(key))fingerprints.set(key,payloadFingerprint(image,row.geometry));
      return similarFingerprint(row.fingerprint,fingerprints.get(key));
    });
  }
  matchesCapture(image,capture,identity) {
    if(!this.valid(identity))return false;
    const current=payloadFingerprint(image,capture.geometry);
    return this.templates.some(row=>row.phase===capture.info.phase&&row.size===capture.info.size&&similarFingerprint(row.fingerprint,current));
  }
  record(image,capture,identity,probe) {
    return {phase:capture.info.phase,size:capture.info.size,geometry:capture.geometry,
      fingerprint:payloadFingerprint(image,capture.geometry),probe:probePixels(probe),
      mediaTime:identity.mediaTime,presentedFrames:identity.presentedFrames};
  }
}

/** Optional main-thread 32×32 canvas probe, installed ONLY from a successful
 * transport-accept response. It can avoid full-resolution readback and transfer.
 * A false result requests fresh optical acquisition with forceAcquisition:true.
 * Callers can explicitly bypass every gate with forceFullScan:true. */
export class AcceptedProbeGate {
  constructor(){this.reset();}
  reset(){this.state=null;this.lastFull=-Infinity;}
  accept(state){this.state=state?.templates?.length?state:null;this.lastFull=state?.acceptedAt??-Infinity;}
  shouldSkip(probe,identity) {
    const state=this.state,now=identity.mediaTime*1000;
    if(!state||state.context!==contextKey(identity)||now<state.acceptedAt||now-state.acceptedAt>=state.ttlMs)return false;
    if(now-this.lastFull>=state.fullScanMs){this.lastFull=now;return false;}
    const current=probePixels(probe);
    return state.templates.some(row=>similarFingerprint(row.values,current));
  }
}
