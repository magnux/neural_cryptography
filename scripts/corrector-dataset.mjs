#!/usr/bin/env node
/** Neural-QR recording joins; independent OBM5 synthetic/recovery experiment tools. */
import {readFile,writeFile,mkdir,stat,lstat,realpath,readdir} from 'node:fs/promises';
import {resolve,dirname,join,sep,isAbsolute,relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {readRecordingJSON} from './lib/recording-json.mjs';
import {senderFrames,captureKey,base64ToBytes,floatBytes,floatsFromBytes,DATASET_FORMAT,validateSessionId,canonicalJSON,MAX_RECORDS,MAX_CAPTURE_BYTES,validateProfile} from '../apps/color-optical-bench/dist/js/dataset/contract.js';
import {validateCapture} from '../apps/color-optical-bench/dist/js/dataset/capture.js';
import {renderFrame,parseHeader,encodeSequence,phaseFrames,PALETTE,sameBytes,capacity,frameKey} from '../apps/color-optical-bench/dist/js/format.js';
import {gridLayout} from '../apps/color-optical-bench/dist/js/grid.js';
import {extractLearnedInput,logitsToLogicalScores} from '../apps/color-optical-bench/dist/js/learned-input.js';
import {classifySamples,decodeScores} from '../apps/color-optical-bench/dist/js/vision.js';
import {validFrameCount} from '../apps/color-optical-bench/dist/js/corrector-lab/runtime.js';

const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const NEURAL_QR_DATASET_FORMAT='neural-qr-dataset';
const opticalCaptureKey=info=>`${frameKey(info)}/${info.phase}`;
// Offline corpora and browser capture bundles have different resource budgets.
const MAX_MANIFEST_BYTES=256*1024*1024;
const MAX_CAPTURE_JSON=MAX_CAPTURE_BYTES+4*1024*1024;
export const MAX_GENERATION_COUNT=2**20;
export function generationCount(value){return int(value,8,1,MAX_GENERATION_COUNT);}
function int(value,def,min,max){const n=value===undefined?def:Number(value);if(!Number.isInteger(n)||n<min||n>max)throw Error('Invalid numeric argument.');return n;}
function splitName(value='train'){if(!['train','validation','test'].includes(value))throw Error('Use split train, validation, or test.');return value;}
function recordingSplit(bundle,split,path) {
  if(bundle.datasetSplit!==undefined&&(splitName(bundle.datasetSplit)!==split))throw Error(`Recording split disagrees with selected split: ${path}`);
}
function validateCaptureBundle(bundle) {
  if(bundle?.format!=='neural-qr-captures'||bundle.version!==1||!Array.isArray(bundle.records)||bundle.records.length>MAX_RECORDS||
      !Array.isArray(bundle.diagnostics??[])||(bundle.diagnostics??[]).length>100||! /^[a-f0-9]{8}$/.test(bundle.collectionCode??''))throw Error('Unsupported receiver capture bundle.');
  validateSessionId(bundle.sessionId);validateProfile(bundle.profile);
  for(const record of bundle.records) {
    if(record.info?.collectionCode!==bundle.collectionCode||Object.entries(bundle.profile).some(([key,value])=>record.info?.[key]!==value))
      throw Error('Capture disagrees with its declared collection code/profile.');
  }
  return bundle;
}
function capturePixelsPerCell(record) {
  const value=record.pixelsPerCell??4;
  if(![2,4].includes(value))throw Error('Unsupported capture pixels per cell.');
  return value;
}
function requireEstablishedRaster(sample) {
  if(capturePixelsPerCell(sample)!==4)throw Error('This training/synthetic operation requires 4x4-pixel cells. Multires 2x2 captures can be joined now; multires training/synthetic integration is separate.');
}
function flags(args) {
  const result={};
  for(let i=0;i<args.length;i++) {if(!args[i].startsWith('--')||!args[i+1]||args[i+1].startsWith('--'))throw Error('Options require values.');const key=args[i].slice(2);if(key in result)throw Error(`Duplicate option --${key}`);result[key]=args[++i];}
  return result;
}
function only(options,names){for(const key of Object.keys(options))if(!names.includes(key))throw Error(`Unknown option --${key}`);}
function fileList(value) {
  const paths=Array.isArray(value)?value:typeof value==='string'?value.split(','):[];
  if(!paths.length||paths.some(path=>typeof path!=='string'||!path))throw Error('Expected nonempty input file paths.');
  return paths;
}
function safePath(root,name) {
  if(typeof name!=='string'||!name||isAbsolute(name)||name.includes('\\')||name.split('/').some(part=>!part||part==='.'||part==='..'))throw Error('Unsafe dataset path.');
  const target=resolve(root,name);if(!target.startsWith(resolve(root)+sep))throw Error('Dataset path escapes root.');return target;
}
async function safeReadPath(root,name) {
  const candidate=safePath(root,name),actual=await realpath(candidate),base=await realpath(root);
  if(!actual.startsWith(base+sep))throw Error('Dataset symlink escapes root.');
  return actual;
}
async function jsonFile(path,limit=MAX_MANIFEST_BYTES){if((await stat(path)).size>limit)throw Error(`JSON file exceeds bound: ${path}`);return JSON.parse(await readFile(path,'utf8'));}
async function freshDirectory(path){await mkdir(path,{recursive:true});if((await readdir(path)).length)throw Error('Output dataset directory must be empty; use a new output directory.');}
async function putSample(out,index,metadata,{input,labels,mask,baseline,expected,samples,original}) {
  const prefix=`sample-${String(index).padStart(6,'0')}`;
  const identity={sessionId:metadata.sessionId,payloadId:metadata.payloadId,info:metadata.info,timestamp:metadata.timestamp};
  const result={id:`sample-${hash(Buffer.from(JSON.stringify(identity))).slice(0,24)}`,...metadata,hashes:{}};
  const arrays={input:floatBytes(input),labels,mask,baseline:floatBytes(baseline),expected};
  if(samples)arrays.samples=floatBytes(samples);
  if(original)arrays.original=original;
  for(const [kind,bytes] of Object.entries(arrays)) {
    const ext=['input','baseline','samples'].includes(kind)?'f32':['labels','mask'].includes(kind)?'u8':kind==='original'?'png':'bin';
    const name=`${prefix}.${kind}.${ext}`;await writeFile(join(out,name),bytes);result[kind]=name;result.hashes[kind]=hash(bytes);
  }
  return result;
}
function maskFor(info){const mask=new Uint8Array(info.size**2);for(const cell of gridLayout(info.size,info.colors,info.gridBlock).cells)mask[cell]=1;return mask;}
function cleanCrop(frame) {
  const pixels=frame.size===64?1024:2048,scale=pixels/256;
  const learned=extractLearnedInput(renderFrame(frame,pixels),{transform:[scale,0,0,0,scale,0,0,0,1]});
  if(!learned.ok)throw Error(`Clean canonical crop failed: ${learned.error}`);
  return learned.input;
}

/** Calibrated-crop baseline for synthetic inputs; no clean truth used. */
export function denseClassicalScores(input,info) {
  const g=info.size,w=4*g,pixels=w*w;
  if(!(input instanceof Float32Array)||input.length!==3*pixels)throw Error('Wrong calibrated crop length.');
  const samples=new Float32Array(g*g*3+96),offsets=[[1,1],[2,1],[1,2],[2,2],[0,1],[3,2],[1,0],[2,3]];
  for(let y=0;y<g;y++)for(let x=0;x<g;x++)for(let c=0;c<3;c++) {
    let value=0;for(const [dx,dy] of offsets)value+=(input[c*pixels+(4*y+dy)*w+4*x+dx]+1)*127.5;
    samples[3*(y*g+x)+c]=value/offsets.length;
  }
  let at=g*g*3;for(let side=0;side<2;side++)for(const rgb of PALETTE){samples.set(rgb,at);at+=3;}
  return new Float32Array(classifySamples(info,samples,{local:true}).scores);
}

export async function generate(options) {
  only(options,['out','count','size','colors','phases','seed','split']);
  if(!options.out)throw Error('generate requires --out DIRECTORY.');
  const out=resolve(options.out),size=int(options.size,64,64,128),colors=int(options.colors,16,4,16),phases=int(options.phases,4,1,16),count=generationCount(options.count),seed=int(options.seed,7,0,0xffffffff),split=splitName(options.split);
  if(![64,128].includes(size)||![4,16].includes(colors)||![1,2,4,8,16].includes(phases))throw Error('Unsupported generation profile.');
  await freshDirectory(out);
  const sessionId=`synthetic-${seed}-${size}-${colors}`,dataset={format:DATASET_FORMAT,version:1,created:new Date().toISOString(),
    description:'Clean canonical OBM5 renderer crops. Corruption/training is a separate Python step.',samples:[]};
  for(let i=0;i<count;i++) {
    // Established OBM5 synthetic experiment; it is independent of Neural-QR collection.
    const profile={size,colors,phases,gridBlock:8,parity:32,holdMs:4000,permute:phases>1};
    let state=(seed+i)>>>0;const bytes=new Uint8Array(capacity(size,colors,profile)+8);
    for(let at=0;at<bytes.length;at++){state=(Math.imul(state,1664525)+1013904223)>>>0;bytes[at]=state>>>24;}
    const payload=bytes.slice(8),frames=phaseFrames(encodeSequence(payload,{...profile,id:bytes.slice(0,8),contentType:'raw'}));
    for(const frame of frames) {
      const input=cleanCrop(frame),info=parseHeader(frame.header);
      dataset.samples.push(await putSample(out,dataset.samples.length,{sessionId,payloadId:hash(payload),split,source:'synthetic',size,colors,info,
        burstId:opticalCaptureKey(info),timestamp:info.phase*500,provenance:{generator:'OBM5 renderFrame + validated dense crop',seed:(seed+i)>>>0,corruption:'none',baseline:'calibrated-crop approximation'}},
        {input,labels:frame.symbols,mask:maskFor(info),baseline:denseClassicalScores(input,info),expected:frame.chunk}));
    }
  }
  await writeFile(join(out,'manifest.json'),JSON.stringify(dataset,null,2)+'\n');
  return {dataset:out,samples:dataset.samples.length,split,size,colors};
}

export async function joinCaptures(options) {
  only(options,['out','senders','captures','split','exclude']);
  if(!options.out||!options.senders||!options.captures)throw Error('join requires --out DIRECTORY --senders FILE[,FILE] --captures FILE[,FILE].');
  const out=resolve(options.out),split=splitName(options.split),senders=new Map(),senderIds=new Map();
  for(const path of fileList(options.senders)) {
    const manifest=await jsonFile(path,1024*1024),frames=await senderFrames(manifest);
    recordingSplit(manifest,split,path);
    for(const frame of frames) {
      const key=captureKey(frame),signature=canonicalJSON({payloadId:manifest.payloadId,sessionId:manifest.sessionId,info:manifest.frames[frame.phase].info});
      if(senders.has(key)&&senders.get(key).signature!==signature)throw Error('Conflicting sender manifests reuse a collection code/profile.');
      const previous=senderIds.get(frame.id);if(previous&&previous!==manifest.payloadId)throw Error('Conflicting payloads reuse a collection code.');
      senderIds.set(frame.id,manifest.payloadId);senders.set(key,{signature,frame,manifest});
    }
  }
  let exclusions=[];if(options.exclude){exclusions=await jsonFile(options.exclude,65536);if(!Array.isArray(exclusions)||exclusions.some(x=>typeof x!=='string'))throw Error('Exclusions must be capture IDs in a JSON array.');}
  const dataset={format:NEURAL_QR_DATASET_FORMAT,version:1,created:new Date().toISOString(),association:'Operator-entered public collection code; not optically authenticated.',samples:[],excluded:[],diagnostics:[]},ids=new Set();
  await freshDirectory(out);
  for(const path of fileList(options.captures)) {
    const bundle=validateCaptureBundle(await readRecordingJSON(path,{maxBytes:MAX_CAPTURE_JSON,maxArrayItems:MAX_RECORDS}));
    validateSessionId(bundle.sessionId);
    recordingSplit(bundle,split,path);
    for(let i=0;i<bundle.records.length;i++) {
      const captureId=`${bundle.sessionId}/${bundle.records[i].timestamp}`;
      if(ids.has(captureId))throw Error('Duplicate capture session/record across files.');ids.add(captureId);
      if(exclusions.includes(captureId)){dataset.excluded.push({captureId,reason:'Explicit offline exclusion'});continue;}
      const record=validateCapture(bundle.records[i]);
      const guard=record.stability;
      if(!guard||!Number.isInteger(guard.guardMs)||guard.guardMs<500||guard.guardMs>5000||!Number.isFinite(guard.durationMs)||guard.durationMs<guard.guardMs||!Number.isInteger(guard.observations)||guard.observations<2)throw Error('Capture has no stable marker/phase guard evidence.');
      const sender=senders.get(record.key);
      if(!sender){dataset.excluded.push({captureId,reason:'No matching sender collection code/profile/phase; remains unlabeled'});continue;}
      const info=sender.manifest.frames[sender.frame.phase].info;
      // Exact declared profile and phase match. Public code is not optical proof of membership.
      if(canonicalJSON(info)!==canonicalJSON(record.info))throw Error('Capture metadata disagrees with sender acquisition profile.');
      const pixelsPerCell=capturePixelsPerCell(record),side=pixelsPerCell*info.size;
      const input=floatsFromBytes(base64ToBytes(record.inputBase64),3*side**2),samples=floatsFromBytes(base64ToBytes(record.samplesBase64),info.size**2*3+96);
      const baseline=new Float32Array(classifySamples(info,samples,{local:true}).scores);
      const original=record.original?base64ToBytes(record.original.base64,1024*1024):undefined;
      dataset.samples.push(await putSample(out,dataset.samples.length,{sessionId:bundle.sessionId,senderSessionId:sender.manifest.sessionId,payloadId:sender.manifest.payloadId,split,source:'real',size:info.size,colors:info.colors,pixelsPerCell,inputShape:[3,side,side],info,
        burstId:`${bundle.sessionId}/${record.key}`,timestamp:record.timestamp,
        devicePair:{sender:sender.manifest.environment?.deviceId??null,receiver:bundle.environment?.deviceId??null},
        provenance:{captureId,...(bundle.collectionSessionId?{collectionSessionId:bundle.collectionSessionId}:{}),stats:record.stats,camera:record.camera??bundle.camera,
          senderEnvironment:sender.manifest.environment,receiverEnvironment:bundle.environment,
          senderDatasetSplit:sender.manifest.datasetSplit,receiverDatasetSplit:bundle.datasetSplit,
          transform:record.transform,colorTransform:record.colorTransform,stability:record.stability,notes:bundle.notes??''}},
        {input,labels:sender.frame.symbols,mask:maskFor(info),baseline,expected:sender.frame.chunk,samples,original}));
    }
    for(const diagnostic of (bundle.diagnostics??[]).slice(0,100))dataset.diagnostics.push({sessionId:bundle.sessionId,...diagnostic});
  }
  if(!dataset.samples.length)throw Error('No labeled captures matched; check collection codes, profiles, manifests and exclusions.');
  await writeFile(join(out,'manifest.json'),JSON.stringify(dataset,null,2)+'\n');
  return {dataset:out,samples:dataset.samples.length,excluded:dataset.excluded,diagnostics:dataset.diagnostics.length,split};
}

/** Join server uploads by declared split/grid without altering any recordings.
 * Read all identities before creating outputs so leakage and missing counterparts
 * fail early. Directory names organize uploads; declared public codes/profiles pair them.
 */
export async function joinUploads(options) {
  only(options,['recordings-root','out']);
  if(!options['recordings-root']||!options.out)throw Error('join-uploads requires --recordings-root DIRECTORY --out NEW_DIRECTORY.');
  const root=resolve(options['recordings-root']),out=resolve(options.out);
  if(out===root||out.startsWith(root+sep)||root.startsWith(out+sep))throw Error('Joined output and source recordings must be separate directories, not ancestors of one another.');
  const rootInfo=await lstat(root);
  if(rootInfo.isSymbolicLink()||!rootInfo.isDirectory()||await realpath(root)!==root)throw Error('Recordings root must be a real directory without symlink components.');
  const paths=[];
  async function scan(path) {
    for(const entry of (await readdir(path,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))) {
      const child=join(path,entry.name);
      if(entry.isSymbolicLink())throw Error(`Recording symlinks are not supported: ${child}`);
      if(entry.isDirectory())await scan(child);
      else if(entry.isFile()&&entry.name.endsWith('.json'))paths.push(child);
    }
  }
  for(const split of ['train','validation','test']) {
    const folder=join(root,split);let info;
    try{info=await lstat(folder);}catch(error){if(error.code==='ENOENT')continue;throw error;}
    if(info.isSymbolicLink()||!info.isDirectory())throw Error(`Invalid upload split directory: ${folder}`);
    await scan(folder);
  }
  const sessionOwners=new Map(),payloadOwners=new Map(),associations=new Map(),groups=new Map(),captures=new Set(),entries=[];
  const own=(owners,id,split,label)=>{
    if(owners.has(id)&&owners.get(id)!==split)throw Error(`${label} crosses dataset splits: ${id}`);
    owners.set(id,split);
  };
  for(const path of paths) {
    const parts=relative(root,path).split(sep),[split,grid,kind]=parts;
    if(parts.length!==4||!['sender','receiver'].includes(kind)||!(/^(?:g(?:8|16|32|64|128|256)|unknown)$/.test(grid)))throw Error(`Unexpected upload location: ${path}`);
    const bundle=kind==='sender'?await jsonFile(path,1024*1024):await readRecordingJSON(path,{maxBytes:MAX_CAPTURE_JSON,maxArrayItems:MAX_RECORDS});
    validateSessionId(bundle.sessionId);recordingSplit(bundle,split,path);
    own(sessionOwners,bundle.sessionId,split,'Physical session');
    const entry={path,split,grid,kind,records:[],sessionId:bundle.sessionId};
    if(kind==='sender') {
      const frames=await senderFrames(bundle);
      if(grid!==`g${bundle.profile.size}`)throw Error(`Sender grid disagrees with upload folder: ${path}`);
      own(payloadOwners,bundle.payloadId,split,'Payload');
      for(const frame of frames) {
        const key=captureKey(frame),signature=canonicalJSON([bundle.sessionId,bundle.payloadId,bundle.frames[frame.phase].info]);
        const previous=associations.get(key);
        if(previous&&previous.signature!==signature)throw Error(`Conflicting sender collection code/profile: ${path}`);
        associations.set(key,{entry,signature,info:bundle.frames[frame.phase].info});
      }
    }else {
      validateCaptureBundle(bundle);
      if(!bundle.records.length&&grid!=='unknown')throw Error(`Diagnostic-only capture requires unknown grid folder: ${path}`);
      for(const raw of bundle.records) {
        const record=validateCapture(raw),guard=record.stability;
        if(!guard||!Number.isInteger(guard.guardMs)||guard.guardMs<500||guard.guardMs>5000||!Number.isFinite(guard.durationMs)||guard.durationMs<guard.guardMs||!Number.isInteger(guard.observations)||guard.observations<2)throw Error(`Capture has no stable marker/phase guard evidence: ${path}`);
        if(grid!==`g${record.info.size}`)throw Error(`Capture grid disagrees with upload folder: ${path}`);
        const identity=`${bundle.sessionId}/${record.timestamp}`;
        if(captures.has(identity))throw Error(`Duplicate capture across uploads: ${identity}`);
        captures.add(identity);entry.records.push({key:record.key,info:record.info});
      }
    }
    entries.push(entry);
  }
  let diagnosticFiles=0;
  for(const entry of entries)if(entry.kind==='receiver') {
    if(!entry.records.length){diagnosticFiles++;continue;}
    const key=`${entry.split}/${entry.grid}`,group=groups.get(key)??{split:entry.split,grid:entry.grid,senders:new Set(),captures:[],records:0};
    for(const record of entry.records) {
      const sender=associations.get(record.key);
      if(!sender)throw Error(`No matching uploaded sender for ${entry.path}; upload its sender manifest first.`);
      if(sender.entry.split!==entry.split)throw Error(`Sender and receiver cross dataset splits: ${entry.path}`);
      if(canonicalJSON(sender.info)!==canonicalJSON(record.info))throw Error(`Capture metadata disagrees with sender acquisition profile: ${entry.path}`);
      group.senders.add(sender.entry.path);group.records++;
    }
    group.captures.push(entry.path);groups.set(key,group);
  }
  if(!groups.size)throw Error('No matched uploaded captures to join. Upload sender manifests and receiver captures first.');
  await freshDirectory(out);
  const datasets=[];
  for(const [key,group]of [...groups].sort(([a],[b])=>a.localeCompare(b))) {
    const result=await joinCaptures({out:join(out,group.split,group.grid),split:group.split,senders:[...group.senders],captures:group.captures});
    if(result.samples!==group.records||result.excluded.length)throw Error(`Unexpected join exclusions for ${key}; uploaded recordings changed during processing.`);
    datasets.push({split:group.split,size:Number(group.grid.slice(1)),dataset:result.dataset,samples:result.samples});
  }
  return {recordings:root,datasetRoot:out,datasets,diagnosticFiles,
    scope:'Joined evidence only. No recordings deleted; multires training/synthetic integration is separate from the established G64 recipe.'};
}

/** Re-render joined sender truth offline, once per physical session/frame/phase.
 * No captured pixels enter the clean image; keep session/payload ownership so
 * these targets and their future corruptions cannot cross held-out splits.
 */
export async function targets(options) {
  only(options,['dataset','out']);
  if(!options.dataset||!options.out)throw Error('targets requires --dataset DIRECTORY[,DIRECTORY] --out DIRECTORY.');
  const grouped=new Map(),ids=new Set(),owners={sessionId:new Map(),payloadId:new Map()},sources=[];
  let captures=0;
  for(const location of fileList(options.dataset)) {
    const root=resolve(location),manifestPath=await safeReadPath(root,'manifest.json'),manifest=await jsonFile(manifestPath);
    if(manifest.format===NEURAL_QR_DATASET_FORMAT)throw Error('Neural-QR multires synthetic/training integration is separate; this operation requires the established OBM5 4x4-pixel dataset.');
    if(manifest.format!==DATASET_FORMAT||manifest.version!==1||!Array.isArray(manifest.samples)||!manifest.samples.length)throw Error('Unsupported joined dataset manifest.');
    sources.push({manifestSha256:hash(await readFile(manifestPath))});
    for(const sample of manifest.samples) {
      if(!sample||sample.source!=='real')throw Error('Target reconstruction requires joined real captures.');
      requireEstablishedRaster(sample);
      validateSessionId(sample.sessionId);validateSessionId(sample.senderSessionId);
      const split=splitName(sample.split),info=sample.info;
      if(sample.split===undefined||typeof sample.id!=='string'||!sample.id||ids.has(sample.id))throw Error('Missing/duplicate joined sample identity or split.');
      ids.add(sample.id);
      if(!/^[a-f0-9]{64}$/.test(sample.payloadId)||![64,128].includes(sample.size)||![4,16].includes(sample.colors)||
        !info||info.size!==sample.size||info.colors!==sample.colors||!/^[a-f0-9]{16}$/.test(info.id))throw Error('Invalid joined target profile or identity.');
      for(const [kind,values] of Object.entries({sessionId:[sample.sessionId,sample.senderSessionId],payloadId:[sample.payloadId]}))for(const value of values) {
        const previous=owners[kind].get(value);if(previous&&previous!==split)throw Error(`${kind} crosses dataset splits.`);
        owners[kind].set(value,split);
      }
      // The capture app emits one static logical frame and its palette phases.
      // Refuse partial multipart truth rather than inventing the missing bytes.
      if(info.count!==1||info.index!==0)throw Error('Target reconstruction requires the capture app\'s complete single-frame payload.');
      const expected=await readSampleFile(root,sample,'expected',65536),labels=await readSampleFile(root,sample,'labels',sample.size**2),mask=await readSampleFile(root,sample,'mask',sample.size**2);
      if(hash(expected)!==sample.payloadId)throw Error('Joined expected bytes disagree with whole-payload identity.');
      const frames=phaseFrames(encodeSequence(expected,{...info,id:Uint8Array.from(info.id.match(/../g),s=>parseInt(s,16)),contentType:info.content?.type}));
      const frame=frames[info.phase];
      if(!frame||canonicalJSON(parseHeader(frame.header))!==canonicalJSON(info))throw Error('Regenerated target header disagrees with joined metadata.');
      if(!sameBytes(frame.symbols,labels)||!sameBytes(maskFor(info),mask))throw Error('Regenerated target labels or mask disagree with joined truth.');
      const key=canonicalJSON([sample.sessionId,sample.payloadId,opticalCaptureKey(info)]),previous=grouped.get(key);
      if(previous&&(previous.sample.senderSessionId!==sample.senderSessionId||previous.sample.split!==split))throw Error('Conflicting identities for one target matrix.');
      if(!previous)grouped.set(key,{sample,frame});
      captures++;
    }
  }
  const out=resolve(options.out);await freshDirectory(out);
  const dataset={format:DATASET_FORMAT,version:1,created:new Date().toISOString(),
    description:'Canonical clean targets reconstructed from joined sender truth; no camera input used.',sources,samples:[]};
  for(const [key,{sample,frame}] of [...grouped].sort(([a],[b])=>a.localeCompare(b))) {
    const input=cleanCrop(frame),info=parseHeader(frame.header);
    dataset.samples.push(await putSample(out,dataset.samples.length,{
      id:`target-${hash(Buffer.from(key)).slice(0,24)}`,sessionId:sample.sessionId,senderSessionId:sample.senderSessionId,
      payloadId:sample.payloadId,split:sample.split,source:'synthetic',size:sample.size,colors:sample.colors,info,
      burstId:sample.burstId,timestamp:info.phase*500,
      provenance:{generator:'OBM5 renderFrame + validated joined sender truth',corruption:'none',
        baseline:'calibrated-crop approximation',targetKey:key}},
      {input,labels:frame.symbols,mask:maskFor(info),baseline:denseClassicalScores(input,info),expected:frame.chunk}));
  }
  await writeFile(join(out,'manifest.json'),JSON.stringify(dataset,null,2)+'\n');
  return {dataset:out,captures,samples:dataset.samples.length,deduplicated:captures-dataset.samples.length,
    splits:[...new Set(dataset.samples.map(sample=>sample.split))]};
}

function recoveryResult(info,scores,expected) {
  const result=decodeScores(info,scores,{}, {recovery:'erasures'});
  const exact=result.ok&&sameBytes(result.chunk,expected);
  return {accepted:result.ok,exact,wrongAccepted:result.ok&&!exact,rawCRC:result.stats.rawCRC,hardCRC:result.stats.hardCRC,
    recovery:result.stats.recovery,correctedBytes:result.stats.correctedBytes,erasedBytes:result.stats.erasedBytes};
}
async function readSampleFile(root,sample,kind,maxBytes) {
  const path=await safeReadPath(root,sample[kind]);if((await stat(path)).size>maxBytes)throw Error(`Oversized ${kind} file.`);
  const bytes=new Uint8Array(await readFile(path));
  if(!sample.hashes||hash(bytes)!==sample.hashes[kind])throw Error(`Dataset hash mismatch: ${kind}.`);
  return bytes;
}
/** Refresh derived synthetic baselines after Python materializes corruption. */
export async function baselines(options) {
  only(options,['dataset']);if(!options.dataset)throw Error('baselines requires --dataset DIRECTORY.');
  const root=resolve(options.dataset),manifestPath=await safeReadPath(root,'manifest.json'),manifest=await jsonFile(manifestPath);
  if(manifest.format===NEURAL_QR_DATASET_FORMAT)throw Error('Neural-QR multires baseline/training integration is separate; this operation requires the established OBM5 4x4-pixel dataset.');
  if(manifest.format!==DATASET_FORMAT||manifest.version!==1||!Array.isArray(manifest.samples)||!manifest.samples.length||manifest.samples.some(s=>s.source!=='synthetic'))throw Error('Baseline refresh accepts only a synthetic dataset.');
  if(new Set(manifest.samples.map(s=>s.id)).size!==manifest.samples.length)throw Error('Duplicate dataset sample ID.');
  for(const sample of manifest.samples) {
    requireEstablishedRaster(sample);
    const g=sample.size,info=sample.info;
    if(![64,128].includes(g)||![4,16].includes(sample.colors)||info.size!==g||info.colors!==sample.colors)throw Error('Unsupported dataset shape.');
    const input=floatsFromBytes(await readSampleFile(root,sample,'input',3*16*g*g*4),3*16*g*g);
    const baseline=floatBytes(denseClassicalScores(input,info)),target=await safeReadPath(root,sample.baseline);
    await writeFile(target,baseline);sample.hashes.baseline=hash(baseline);
  }
  await writeFile(manifestPath,JSON.stringify(manifest,null,2)+'\n');
  return {dataset:root,samples:manifest.samples.length,baseline:'Recomputed from current calibrated synthetic crop; no clean input or labels used.'};
}
export async function evaluate(options) {
  only(options,['dataset','predictions','split']);if(!options.dataset||!options.predictions)throw Error('evaluate requires --dataset DIRECTORY --predictions FILE.');
  const root=resolve(options.dataset),manifest=await jsonFile(join(root,'manifest.json')),predictionPath=resolve(options.predictions),predictions=await jsonFile(predictionPath);
  if(manifest.format===NEURAL_QR_DATASET_FORMAT)throw Error('Neural-QR multires recovery evaluation is not yet integrated; OBM5 packet CRC evaluation cannot be applied to unframed training matrices.');
  if(manifest.format!==DATASET_FORMAT||manifest.version!==1||!Array.isArray(manifest.samples)||!manifest.samples.length||!Array.isArray(predictions.samples)||!predictions.samples.length||!validFrameCount(predictions.framesPerInput)||!['physical','logical'].includes(predictions.outputLabelSpace))throw Error('Unsupported dataset/prediction manifest.');
  if(new Set(manifest.samples.map(s=>s.id)).size!==manifest.samples.length)throw Error('Duplicate dataset sample ID.');
  if(options.split)splitName(options.split);
  const eligible=manifest.samples.filter(s=>!options.split||s.split===options.split),byId=new Map(eligible.map(s=>[s.id,s]));
  for(const sample of eligible)requireEstablishedRaster(sample);
  const frames=predictions.framesPerInput,phaseGrouped=predictions.outputLabelSpace==='logical',used=new Set(),anchors=new Set(),selected=[];
  if(phaseGrouped&&(frames<4||frames>8))throw Error('Phase-grouped predictions require four to eight captures.');
  // Public capture identities define membership; no labels or successful decode
  // choose a member. Distinct, nonoverlapping groups match the Python evaluator.
  for(const prediction of predictions.samples) {
    if(typeof prediction.id!=='string'||anchors.has(prediction.id))throw Error('Duplicate/invalid prediction ID.');
    anchors.add(prediction.id);
    if(!Array.isArray(prediction.memberIds)||prediction.memberIds.length!==frames||prediction.memberIds[0]!==prediction.id||new Set(prediction.memberIds).size!==frames)throw Error('Prediction requires distinct member IDs with its anchor first.');
    const members=prediction.memberIds.map(id=>{
      if(typeof id!=='string'||!byId.has(id))throw Error('Prediction member is absent from the selected dataset split.');
      if(used.has(id))throw Error('Evaluation capture reused across candidates.');
      used.add(id);return byId.get(id);
    });
    const sample=members[0],identity=row=>canonicalJSON([row.source,row.sessionId,row.senderSessionId??null,row.payloadId,row.split,row.size,row.colors,row.info,row.burstId??null]);
    if(phaseGrouped) {
      const phaseIdentity=row=>{const {phase,...info}=row.info;return canonicalJSON([row.source,row.sessionId,row.senderSessionId??null,row.payloadId,row.split,row.size,row.colors,info,row.source==='synthetic'?row.synthetic?.phaseGroupId:null]);};
      const counts=[0,0,0,0];
      for(const member of members) {
        const info=member.info;
        if(member.size!==64||member.colors!==16||info.size!==64||info.colors!==16||info.phases!==4||info.permute!==true||!Number.isInteger(info.phase)||info.phase<0||info.phase>3||!['real','synthetic'].includes(member.source)||!member.burstId)throw Error('Phase-grouped members require four permuted G64/16 phases.');
        if(member.source==='synthetic'&&(typeof member.synthetic?.phaseGroupId!=='string'||!member.synthetic.phaseGroupId))throw Error('Synthetic phase groups require a shared corruption identity.');
        if(phaseIdentity(member)!==phaseIdentity(sample))throw Error('Phase-grouped members must share source, session, payload, logical header and corruption identity.');
        counts[info.phase]++;
      }
      if(counts.some(count=>count<1||count>2))throw Error('Phase-grouped inputs require one or two captures of every phase.');
    } else if(frames>1&&(!sample.burstId||members.some(row=>identity(row)!==identity(sample))))throw Error('Burst members must share source, session, payload, header/phase and burst identity.');
    const checkedFields=phaseGrouped?['mask','expected']:['labels','mask','expected'];
    if(frames>1&&members.some(row=>checkedFields.some(field=>row.hashes?.[field]!==sample.hashes?.[field])))throw Error('Burst target hashes disagree.');
    selected.push({sample,members,prediction});
  }
  const results=[];
  for(const {sample,members,prediction} of selected) {
    const info=sample.info,g=sample.size;
    if(![64,128].includes(g)||![4,16].includes(sample.colors)||info.size!==g||info.colors!==sample.colors)throw Error('Unsupported dataset shape.');
    const logitsPath=await safeReadPath(dirname(predictionPath),prediction.logits),logitLength=16*g*g*4;
    if((await stat(logitsPath)).size!==logitLength)throw Error('Wrong logits file length.');
    const logits=floatsFromBytes(new Uint8Array(await readFile(logitsPath)),16*g*g);
    const count=gridLayout(g,info.colors,info.gridBlock).cells.length*info.colors,baseline=new Float32Array(count);
    for(const member of members) {
      const input=floatsFromBytes(await readSampleFile(root,member,'input',3*16*g*g*4),3*16*g*g);
      const scores=member.source==='synthetic'?denseClassicalScores(input,member.info):floatsFromBytes(await readSampleFile(root,member,'baseline',count*4),count);
      for(let cell=0;cell<count;cell++)baseline[cell]+=scores[cell]/frames;
    }
    const learnedScores=logitsToLogicalScores(logits,info,{labelSpace:predictions.outputLabelSpace}),expected=await readSampleFile(root,sample,'expected',65536);
    // Only compare expected bytes after independent CRC/RS acceptance.
    results.push({id:sample.id,memberIds:prediction.memberIds,source:sample.source,size:g,colors:info.colors,classical:recoveryResult(info,baseline,expected),learned:recoveryResult(info,learnedScores,expected)});
  }
  const summary={samples:results.length,inputCaptures:used.size,framesPerInput:frames,outputLabelSpace:predictions.outputLabelSpace};
  for(const method of ['classical','learned']) {summary[method]={};for(const field of ['rawCRC','hardCRC','accepted','exact','wrongAccepted'])summary[method][field]=results.filter(r=>r[method][field]).length;}
  return {format:'neural-corrector-recovery-report',version:1,scope:`Each candidate uses the same ${frames} distinct exposures for both methods; classical logical scores are averaged, with nonoverlapping groups and no throughput claim.`,summary,samples:results};
}

async function main() {
  const [command,...args]=process.argv.slice(2),options=flags(args);
  const action={generate,join:joinCaptures,'join-uploads':joinUploads,targets,baselines,evaluate}[command];
  if(!action)throw Error('Usage: corrector-dataset.mjs generate|join|join-uploads|targets|baselines|evaluate [--option value].');
  console.log(JSON.stringify(await action(options),null,2));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(error=>{console.error(`Corrector dataset failed: ${error.message}`);process.exitCode=1;});

