/** Receiver evidence; a public collection assignment contains no sender labels. */
import {extractNeuralQR} from '../neural-qr/vision.js';
import {captureSignature} from './diversity.js';
import {captureKey,bytesToBase64,floatBytes,base64ToBytes,floatsFromBytes,validateSessionId,validateCollectionCode,validateProfile,captureProfile,NEURAL_QR_VERSION,CAPTURE_PIXELS_PER_CELL,MAX_CAPTURE_BYTES,DEFAULT_CAPTURE_BYTES,MAX_RECORDS} from './contract.js';

export function captureFrame(image,{timestamp=Date.now(),camera={},signature=false,profile,collectionCode,previousGeometry=null,probe=false}={}) {
  if(!Number.isFinite(timestamp)||timestamp<0)throw Error('Invalid capture timestamp.');
  validateProfile(profile);validateCollectionCode(collectionCode);
  const capture=extractNeuralQR(image,{profile,previousGeometry,now:timestamp,dense:!probe});
  const base={timestamp,camera,probe,ok:capture.ok,stage:capture.stage,error:capture.error,
    stats:capture.stats,transform:capture.transform,geometry:capture.geometry,corners:capture.corners,tentativeCorners:capture.tentativeCorners};
  if(!capture.ok)return base;
  // The expected assignment never supplies optical size evidence. Acquisition
  // must identify the same profile before it can enter a retained phase pool.
  if(!capture.info||['size','colors','phases','gridBlock','permute','pixelsPerCell'].some(key=>capture.info[key]!==profile[key]))
    return {...base,ok:false,stage:'profile',error:'Optical matrix size/profile does not match the collection assignment.'};
  const info={...profile,...capture.info,id:collectionCode,collectionCode,phase:capture.info.phase,bits:4};
  const common={...base,info,key:captureKey(info),pixelsPerCell:CAPTURE_PIXELS_PER_CELL,
    acquisition:{format:'neural-qr',version:NEURAL_QR_VERSION,association:'manual-public-code',phase:'repeated-color-cues',geometry:'repeated-binary-size-cues'}};
  if(probe)return common;
  return {...common,...(signature?{signature:captureSignature(capture.input,info.size,CAPTURE_PIXELS_PER_CELL)}:{}),
    inputBase64:bytesToBase64(floatBytes(capture.input)),samplesBase64:bytesToBase64(floatBytes(capture.samples)),
    colorTransform:capture.colorTransform};
}
export function validateCapture(record){
  if(!record||record.ok!==true||!record.info||!Number.isFinite(record.timestamp)||record.timestamp<0)throw Error('Invalid capture record.');
  validateProfile(record.info);validateCollectionCode(record.info.collectionCode);
  if(record.info.id!==record.info.collectionCode||record.key!==captureKey(record.info)||record.pixelsPerCell!==CAPTURE_PIXELS_PER_CELL||
    !Number.isInteger(record.info.phase)||record.info.phase<0||record.info.phase>=4||record.info.bits!==4||
    record.acquisition?.format!=='neural-qr'||record.acquisition.version!==NEURAL_QR_VERSION||record.acquisition.geometry!=='repeated-binary-size-cues'||record.acquisition.association!=='manual-public-code'||record.acquisition.phase!=='repeated-color-cues')throw Error('Invalid Neural-QR acquisition record.');
  const g=record.info.size,values=3*(CAPTURE_PIXELS_PER_CELL*g)**2;
  const input=floatsFromBytes(base64ToBytes(record.inputBase64,values*4),values);
  for(const value of input)if(value < -1.5||value > 1.5)throw Error('Input tensor outside normalization bounds.');
  floatsFromBytes(base64ToBytes(record.samplesBase64,(g*g*3+96)*4),g*g*3+96);
  if(!Array.isArray(record.transform)||record.transform.length!==9||record.transform.some(v=>!Number.isFinite(v)))throw Error('Invalid capture geometry.');
  if(record.original!==undefined){
    if(record.original.mime!=='image/png')throw Error('Original capture must be PNG.');
    base64ToBytes(record.original.base64,1024*1024);
  }
  return record;
}

/** Bound raw JSON size, retained records, diagnostics, and original PNGs. */
export class CaptureBuffer {
  constructor({maxRecords=MAX_RECORDS,maxBytes=DEFAULT_CAPTURE_BYTES,stableMs=500}={}) {
    if(!Number.isInteger(maxRecords)||maxRecords<1||maxRecords>MAX_RECORDS||!Number.isInteger(maxBytes)||maxBytes<1024||maxBytes>MAX_CAPTURE_BYTES||!Number.isInteger(stableMs)||stableMs<500||stableMs>5000) throw Error('Invalid capture limits.');
    this.maxRecords=maxRecords;this.maxBytes=maxBytes;this.stableMs=stableMs;this.reset();
  }
  reset(){this.records=[];this.diagnostics=[];this.bytes=0;this.active=null;this.originals=0;this.signatures=new WeakMap();this.sizes=new WeakMap();this.revisions=new Map();this.thinning=[];}
  setBudget(maxBytes){
    if(!Number.isSafeInteger(maxBytes)||maxBytes<1024||maxBytes>MAX_CAPTURE_BYTES)throw Error('Invalid capture byte budget.');
    if(this.records.length||this.diagnostics.length)throw Error('Save and clear the batch before changing its byte budget.');
    this.maxBytes=maxBytes;
  }
  phaseRecords(key){return this.records.filter(record=>record.key===key);}
  revision(key){return this.revisions.get(key)??0;}
  phaseSignatures(key){return this.phaseRecords(key).map(record=>this.signatures.get(record));}
  resetStability(){this.active=null;}
  summary(){return {records:this.records.length,diagnostics:this.diagnostics.length,bytes:this.bytes,maxRecords:this.maxRecords,maxBytes:this.maxBytes};}
  // alreadyValidated is reserved for records produced/validated by our worker.
  // Imported JSON must always take the full validation path (including offline join).
  add(record,{alreadyValidated=false,maxPerPhase=64,replaceIndex=null}={}) {
    if(!Number.isInteger(maxPerPhase)||maxPerPhase<1||maxPerPhase>64)throw Error('Invalid per-phase limit.');
    const {signature,...evidence}=record;record=evidence;
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
    const corners=record.corners;
    let moved=false;
    if(this.active?.corners&&Array.isArray(corners)&&corners.length===4){
      const diagonal=Math.hypot(corners[2][0]-corners[0][0],corners[2][1]-corners[0][1]);
      moved=diagonal>0&&corners.some((point,i)=>Math.hypot(point[0]-this.active.corners[i][0],point[1]-this.active.corners[i][1])>.04*diagonal);
    }
    if(!this.active||record.key!==this.active.key||moved)this.active={key:record.key,since:record.timestamp,last:record.timestamp,observations:1,corners};
    else {this.active.last=record.timestamp;this.active.observations++;this.active.corners=corners;}
    if(record.timestamp-this.active.since<this.stableMs||this.active.observations<2) return diagnostic('settling','Waiting for stable geometry and phase cues; keep a static matrix visible.');
    const phaseRows=this.phaseRecords(record.key),phaseFull=phaseRows.length>=maxPerPhase;
    if(replaceIndex!==null&&(!Number.isInteger(replaceIndex)||replaceIndex<0||replaceIndex>=phaseRows.length))throw Error('Invalid phase replacement index.');
    if(phaseFull&&replaceIndex===null)return {accepted:false,stage:'kept',count:this.records.length,bytes:this.bytes};
    const replaced=phaseFull?phaseRows[replaceIndex]:null;
    if(!replaced&&this.records.length>=this.maxRecords) return diagnostic('full','Capture count limit reached; export and start a new batch.');
    const saved={...record,stability:{durationMs:record.timestamp-this.active.since,observations:this.active.observations,guardMs:this.stableMs}};
    const removedSize=replaced?this.sizes.get(replaced):0;
    const removedOriginal=replaced?.original?1:0;
    if(saved.original&&this.originals-removedOriginal>=2) delete saved.original;
    const size=new TextEncoder().encode(JSON.stringify(saved)).length;
    if(this.bytes-removedSize+size+65536>this.maxBytes){
      // A larger replacement must not stop an otherwise usable full pool.
      if(replaced)return {accepted:false,stage:'kept',reason:'Replacement exceeds byte budget.',count:this.records.length,bytes:this.bytes};
      return diagnostic('full','Capture memory limit reached; export and start a new batch.');
    }
    if(replaced)this.records[this.records.indexOf(replaced)]=saved;else this.records.push(saved);
    this.sizes.set(saved,size);if(signature)this.signatures.set(saved,signature);this.revisions.set(record.key,this.revision(record.key)+1);
    this.bytes+=size-removedSize;this.originals+=Number(Boolean(saved.original))-removedOriginal;
    return {accepted:true,count:this.records.length,bytes:this.bytes,stage:replaced?'replaced':'saved'};
  }
  export({notes='',sessionId=crypto.randomUUID(),camera={}}={}) {
    validateSessionId(sessionId);
    return {format:'neural-qr-captures',version:NEURAL_QR_VERSION,created:new Date().toISOString(),sessionId,
      ...(this.records.length?{collectionCode:this.records[0].info.collectionCode,profile:{...captureProfile(this.records[0].info.size),holdMs:this.records[0].info.holdMs}}:{}),
      notes:String(notes).slice(0,2000),camera,limits:{maxRecords:this.maxRecords,maxBytes:this.maxBytes,stableMs:this.stableMs},
      transitionPolicy:'Manual static sender and public collection assignment; consecutive agreeing optical size, geometry and phase cues after guard. Association is not optically authenticated. Review mixed/wrong-matrix captures offline.',
      selection:{method:'centered-band-reservoir',recentThinning:this.thinning},
      records:this.records,diagnostics:this.diagnostics};
  }
}
