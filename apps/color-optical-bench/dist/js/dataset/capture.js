/** Receiver-only evidence collection. No sender seed, payload, or labels accepted. */
import {extractCapture,classifySamples,decodeScores} from '../vision.js';
import {extractLearnedInput} from '../learned-input.js';
import {captureKey,bytesToBase64,floatBytes,base64ToBytes,floatsFromBytes,validateSessionId,MAX_CAPTURE_BYTES,MAX_RECORDS} from './contract.js';

export function captureFrame(image,{timestamp=Date.now(),camera={}}={}) {
  if(!Number.isFinite(timestamp)||timestamp<0) throw Error('Invalid capture timestamp.');
  const capture=extractCapture(image,{preview:false}),base={timestamp,camera};
  if(!capture.samples) return {...base,ok:false,stage:capture.stage,error:capture.error,stats:capture.stats};
  const learned=extractLearnedInput(image,capture);
  if(!learned.ok) return {...base,ok:false,stage:learned.stage,error:learned.error,stats:learned.stats};
  const classified=classifySamples(learned.info,capture.samples,{local:true});
  const decoded=decodeScores(learned.info,classified.scores,{}, {recovery:'erasures'});
  return {...base,ok:true,info:learned.info,key:captureKey(learned.info),
    inputBase64:bytesToBase64(floatBytes(learned.input)),samplesBase64:bytesToBase64(floatBytes(capture.samples)),
    stats:{...learned.stats,...capture.stats},colorTransform:learned.colorTransform,transform:capture.transform,
    observedPalette:classified.observedPalette,
    baseline:{accepted:decoded.ok,error:decoded.error,...decoded.stats}};
}

export function validateCapture(record) {
  if(!record||record.ok!==true||!record.info||![64,128].includes(record.info.size)||![4,16].includes(record.info.colors)||
    record.key!==captureKey(record.info)||!Number.isFinite(record.timestamp)||record.timestamp<0) throw Error('Invalid capture record.');
  const g=record.info.size;
  const input=floatsFromBytes(base64ToBytes(record.inputBase64,3*16*g*g*4),3*16*g*g);
  for(const value of input) if(value < -1.5||value > 1.5) throw Error('Input tensor outside normalization bounds.');
  floatsFromBytes(base64ToBytes(record.samplesBase64,(g*g*3+96)*4),g*g*3+96);
  if(record.original!==undefined) {
    if(record.original.mime!=='image/png') throw Error('Original capture must be PNG.');
    base64ToBytes(record.original.base64,1024*1024);
  }
  return record;
}

/** Bound raw JSON size, retained records, diagnostics, and original PNGs. */
export class CaptureBuffer {
  constructor({maxRecords=MAX_RECORDS,maxBytes=MAX_CAPTURE_BYTES,stableMs=500}={}) {
    if(!Number.isInteger(maxRecords)||maxRecords<1||maxRecords>MAX_RECORDS||!Number.isInteger(maxBytes)||maxBytes<1024||maxBytes>MAX_CAPTURE_BYTES||!Number.isInteger(stableMs)||stableMs<500||stableMs>5000) throw Error('Invalid capture limits.');
    this.maxRecords=maxRecords;this.maxBytes=maxBytes;this.stableMs=stableMs;this.reset();
  }
  reset(){this.records=[];this.diagnostics=[];this.bytes=0;this.active=null;this.originals=0;}
  resetStability(){this.active=null;}
  summary(){return {records:this.records.length,diagnostics:this.diagnostics.length,bytes:this.bytes,maxRecords:this.maxRecords,maxBytes:this.maxBytes};}
  // alreadyValidated is reserved for records produced/validated by our worker.
  // Imported JSON must always take the full validation path (including offline join).
  add(record,{alreadyValidated=false}={}) {
    const byteSize=value=>new TextEncoder().encode(JSON.stringify(value)).length;
    const diagnostic=(stage,error)=>{
      const item={timestamp:record.timestamp,stage,error:String(error).slice(0,1000),stats:record.stats??{}};
      if(record.original&&this.originals<2&&record.original.mime==='image/png'){
        base64ToBytes(record.original.base64,1024*1024);item.original=record.original;
        if(this.bytes+byteSize(item)+65536>this.maxBytes)delete item.original;
      }
      if(this.diagnostics.length>=100){const old=this.diagnostics.shift();this.bytes-=byteSize(old);if(old.original)this.originals--;}
      const n=byteSize(item);if(this.bytes+n+65536<=this.maxBytes){this.diagnostics.push(item);this.bytes+=n;if(item.original)this.originals++;}
      return {accepted:false,stage,error,reason:error,count:this.records.length,bytes:this.bytes};
    };
    if(!record.ok){this.active=null;return diagnostic(record.stage??'capture',record.error??'Capture failed.');}
    if(!alreadyValidated)validateCapture(record);
    if(this.active&&record.timestamp<=this.active.last) return diagnostic('duplicate','Capture timestamps must increase.');
    if(!this.active||record.key!==this.active.key) this.active={key:record.key,since:record.timestamp,last:record.timestamp,observations:1};
    else {this.active.last=record.timestamp;this.active.observations++;}
    if(record.timestamp-this.active.since<this.stableMs||this.active.observations<2) return diagnostic('settling','Waiting for a stable header; keep a static matrix visible.');
    if(this.records.length>=this.maxRecords) return diagnostic('full','Capture count limit reached; export and start a new batch.');
    const saved={...record,stability:{durationMs:record.timestamp-this.active.since,observations:this.active.observations,guardMs:this.stableMs}};
    if(saved.original&&this.originals>=2) delete saved.original;
    const size=new TextEncoder().encode(JSON.stringify(saved)).length;
    if(this.bytes+size+65536>this.maxBytes) return diagnostic('full','Capture memory limit reached; export and start a new batch.');
    this.records.push(saved);this.bytes+=size;if(saved.original)this.originals++;
    return {accepted:true,count:this.records.length,bytes:this.bytes,stage:'saved'};
  }
  export({notes='',sessionId=crypto.randomUUID(),camera={}}={}) {
    validateSessionId(sessionId);
    return {format:'optical-decoder-captures',version:1,created:new Date().toISOString(),sessionId,
      notes:String(notes).slice(0,2000),camera,limits:{maxRecords:this.maxRecords,maxBytes:this.maxBytes,stableMs:this.stableMs},
      transitionPolicy:'Manual static sender; consecutive checksum-valid identical headers after guard. Review and exclude mixed/settling captures offline.',
      records:this.records,diagnostics:this.diagnostics};
  }
}
