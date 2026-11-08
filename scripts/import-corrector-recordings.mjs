#!/usr/bin/env node
/** Explicit offline recovery of the three collected corpora into one 4ppc contract.
 * No reader in this file is used by the live receiver. Original recordings are read-only.
 */
import {readFile,readdir,lstat,mkdir,stat,realpath} from 'node:fs/promises';
import {resolve,join,relative,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {visitRecordingJSON} from './lib/recording-json.mjs';
import {progress} from './lib/progress.mjs';
import {writeDatasetJSON} from './lib/dataset-json.mjs';
import {putSample,denseClassicalScores} from './corrector-dataset.mjs';
import {captureProfile,captureKey,senderFrames,validateSessionId,base64ToBytes,bytesToBase64,floatsFromBytes,canonicalJSON} from '../apps/color-optical-bench/dist/js/dataset/contract.js';
import {validateCapture} from '../apps/color-optical-bench/dist/js/dataset/capture.js';
import {encodeSequence,phaseFrames,parseHeader,frameKey,crc32,sameBytes} from '../apps/color-optical-bench/dist/js/format.js';
import {gridLayout,paletteMap} from '../apps/color-optical-bench/dist/js/grid.js';
const hash=value=>createHash('sha256').update(value).digest('hex');
const SIZES=[16,32,64,128],SPLITS=['train','validation','test'];
const oldKey=info=>`${frameKey(info)}/${info.phase}`;
const equal=(a,b)=>canonicalJSON(a)===canonicalJSON(b);
const splitName=s=>s==='val'?'validation':s;
function requireSplit(value){const split=splitName(value);if(!SPLITS.includes(split))throw Error(`Unsupported split: ${value}`);return split;}
function own(map,id,split,label){if(map.has(id)&&map.get(id)!==split)throw Error(`${label} crosses dataset splits: ${id}`);map.set(id,split);}
function bundleSplit(bundle,entry){if(bundle.datasetSplit!==undefined&&requireSplit(bundle.datasetSplit)!==entry.split)throw Error(`Recording split disagrees with folder: ${entry.path}`);}
function phaseGuard(record){const s=record.stability;if(!s||!Number.isInteger(s.guardMs)||s.guardMs<500||s.guardMs>5000||!Number.isFinite(s.durationMs)||s.durationMs<s.guardMs||!Number.isInteger(s.observations)||s.observations<2)throw Error('Capture has no stable phase/header guard evidence.');}
function canonicalFrames(logical,size,holdMs){
  const profile={...captureProfile(size),holdMs},collectionCode=crc32(logical).toString(16).padStart(8,'0'),payloadId=hash(logical),layout=gridLayout(size,16,8);
  if(logical.length!==size*size||logical.some(v=>v>=16)||Array.from(layout.pilots).some(i=>logical[i]!==layout.reference[i]))throw Error('Incompatible logical symbols or reference grid.');
  return Array.from({length:4},(_,phase)=>{const info={...profile,id:collectionCode,collectionCode,phase,bits:4},symbols=logical.slice();for(const cell of layout.cells)symbols[cell]=paletteMap(symbols[cell],info);return {info,symbols,logical,payloadId};});
}
async function sourceFrames(manifest,origin){
  validateSessionId(manifest.sessionId);
  if(origin!=='header-g64'){
    const version=origin==='native'?2:1,ppc=origin==='native'?4:2;
    if(manifest.format!=='neural-qr-sender'||manifest.version!==version||manifest.profile?.pixelsPerCell!==ppc)throw Error(`Expected ${origin} sender contract.`);
    if(!SIZES.includes(manifest.profile.size))throw Error('Excluded matrix size reached importer.');
    const converted={...manifest,version:2,profile:{...manifest.profile,pixelsPerCell:4},frames:manifest.frames?.map(row=>({...row,info:{...row.info,pixelsPerCell:4}}))};
    const frames=await senderFrames(converted);
    return frames.map((frame,i)=>({key:manifest.frames[i].key,sourceInfo:manifest.frames[i].info,frame:{info:frame.info,symbols:frame.symbols,logical:frame.logicalSymbols,payloadId:manifest.payloadId}}));
  }
  const p=manifest.profile;
  if(manifest.format!=='optical-decoder-sender'||manifest.version!==1||p?.size!==64||p.colors!==16||p.phases!==4||p.gridBlock!==8||!Number.isInteger(p.holdMs)||p.holdMs<500||p.holdMs>10000||! /^[a-f0-9]{16}$/.test(manifest.sequenceId??''))throw Error('Old G64 importer requires four phases, 16 colors, grid 8 and an OBM5 sender.');
  const payload=base64ToBytes(manifest.payloadBase64,1024*1024);if(hash(payload)!==manifest.payloadId)throw Error('Old sender payload hash mismatch.');
  const id=Uint8Array.from(manifest.sequenceId.match(/../g),x=>parseInt(x,16));
  const frames=phaseFrames(encodeSequence(payload,{...p,id,permute:true,contentType:'raw',maxFrames:8}));
  if(manifest.frames?.length!==frames.length)throw Error('Old sender phase count mismatch.');
  const canonical=new Map();
  return frames.map((frame,i)=>{
    const sourceInfo=parseHeader(frame.header),row=manifest.frames[i];
    if(row.key!==oldKey(sourceInfo)||row.headerBase64!==bytesToBase64(frame.header)||!equal(row.info,sourceInfo))throw Error('Old sender header or phase metadata mismatch.');
    if(!canonical.has(sourceInfo.index)){
      const logical=frame.symbols.slice();for(const cell of gridLayout(64,16,8).cells)logical[cell]=paletteMap(logical[cell],sourceInfo);
      canonical.set(sourceInfo.index,canonicalFrames(logical,64,p.holdMs));
    }
    const result=canonical.get(sourceInfo.index)[sourceInfo.phase];
    if(!sameBytes(frame.symbols,result.symbols))throw Error('Old phase mapping differs from canonical targets.');
    return {key:row.key,sourceInfo,frame:result};
  });
}
function captureEvidence(record,entry){
  if(entry.origin==='native')validateCapture(record);
  else {
    const info=record?.info,ppc=entry.origin==='resampled'?2:4;
    if(record?.ok!==true||!info||!SIZES.includes(info.size)||info.colors!==16||info.phases!==4||info.gridBlock!==8||!info.permute||!Number.isInteger(info.phase)||info.phase<0||info.phase>3||!Number.isFinite(record.timestamp)||record.timestamp<0||record.key!==(entry.origin==='resampled'?captureKey(info):oldKey(info)))throw Error('Invalid offline capture record/profile.');
    if(entry.origin==='resampled'&&(record.pixelsPerCell!==2||info.pixelsPerCell!==2||record.acquisition?.format!=='neural-qr'||record.acquisition.association!=='manual-public-code'||record.acquisition.phase!=='repeated-color-cues'))throw Error('Invalid version-1 Neural-QR acquisition evidence.');
    if(entry.origin==='header-g64'&&info.size!==64)throw Error('Only G64 header-based captures are imported.');
    const input=floatsFromBytes(base64ToBytes(record.inputBase64,3*(ppc*info.size)**2*4),3*(ppc*info.size)**2);
    if(input.some(v=>v < -1.5||v > 1.5))throw Error('Input tensor outside recording normalization bounds.');
    floatsFromBytes(base64ToBytes(record.samplesBase64,(info.size**2*3+96)*4),info.size**2*3+96);
  }
  phaseGuard(record);
  if(entry.size&&entry.size!==record.info.size)throw Error(`Capture size disagrees with directory: ${entry.path}`);
  return record;
}
async function scanRoot(root,origin,skipped){
  const entries=[];root=resolve(root);
  async function walk(folder){let rows;try{rows=await readdir(folder,{withFileTypes:true});}catch(e){if(e.code==='ENOENT'&&folder===root)return;throw e;}
    for(const row of rows.sort((a,b)=>a.name.localeCompare(b.name))){const path=join(folder,row.name),parts=relative(root,path).split(sep);
      if(row.isSymbolicLink())throw Error(`Recording symlinks are not accepted: ${path}`);
      if(row.isDirectory()){
        if(origin!=='header-g64'&&parts.length===2&&row.name==='unknown'){skipped.push({path,reason:'Diagnostic-only captures have no target geometry'});continue;}
        if(origin==='resampled'&&/^g(?:8|256)$/.test(row.name)){skipped.push({path,reason:'Excluded target size; directory not opened'});continue;}
        await walk(path);continue;
      }
      if(parts.length===1&&row.name==='collection_session.json'){skipped.push({path,reason:'Active collection coordination metadata'});continue;}
      if(!row.isFile()||!row.name.endsWith('.json')){if(row.name.endsWith('.tmp'))skipped.push({path,reason:'Unfinished upload'});continue;}
      let split,kind,size,receiverDevice;
      if(origin==='header-g64'){
        if(parts.length!==3||!['pc','phone'].includes(parts[0])||!['train','val','validation','test'].includes(parts[1]))throw Error(`Unexpected old recording location: ${path}`);
        split=requireSplit(parts[1]);size=64;receiverDevice=parts[0];kind=/_decoder-sender-/.test(row.name)?'sender':/_decoder-captures-/.test(row.name)?'receiver':null;
      }else{
        if(parts.length!==4||!['sender','receiver'].includes(parts[2])||!/^g(?:16|32|64|128)$/.test(parts[1]))throw Error(`Unexpected recording location: ${path}`);
        split=requireSplit(parts[0]);kind=parts[2];size=Number(parts[1].slice(1));
      }
      if(!kind)throw Error(`Cannot identify recording kind: ${path}`);
      entries.push({path,relativePath:relative(root,path),root,origin,split,size,kind,receiverDevice});
    }
  }
  const rootStat=await lstat(root).catch(e=>{if(e.code==='ENOENT')return null;throw e;});if(rootStat?.isSymbolicLink()||rootStat&&!rootStat.isDirectory())throw Error(`Invalid recording root: ${root}`);
  if(rootStat&&await realpath(root)!==root)throw Error(`Recording root has a symlink ancestor: ${root}`);
  await walk(root);return entries;
}
/** Plan reads files once, retains only small metadata, and never modifies recordings. */
export async function planRecordingImports({nativeRoot='local-data/recordings',resampledRoot='local-data/recordings-1',headerRoot='local-data/recordings-0'}={}){
  const roots={native:resolve(nativeRoot),resampled:resolve(resampledRoot),'header-g64':resolve(headerRoot)},skipped=[],entries=[];
  for(const [origin,root] of Object.entries(roots)){
    console.error(`Scanning ${origin} recordings: ${root}`);
    entries.push(...await scanRoot(root,origin,skipped));
  }
  const senders=entries.filter(x=>x.kind==='sender'),receivers=entries.filter(x=>x.kind==='receiver');
  const planned=progress('Checking recording files',senders.length+receivers.length);let files=0,captures=0;planned(0);
  const sessionOwners=new Map(),payloadOwners=new Map(),associations=new Map(),captureIds=new Set(),groups=new Map();
  for(const entry of senders){
    if((await stat(entry.path)).size>1024*1024)throw Error(`Sender manifest too large: ${entry.path}`);
    const bytes=await readFile(entry.path),manifest=JSON.parse(bytes);entry.sha256=hash(bytes);entry.manifest=manifest;entry.sessionId=manifest.sessionId;
    bundleSplit(manifest,entry);own(sessionOwners,manifest.sessionId,entry.split,'Physical session');own(payloadOwners,manifest.payloadId,entry.split,'Source payload');
    if(manifest.profile?.size!==entry.size)throw Error(`Sender size disagrees with directory: ${entry.path}`);
    for(const value of await sourceFrames(manifest,entry.origin)){
      own(payloadOwners,value.frame.payloadId,entry.split,'Logical target');
      const key=`${entry.origin}/${value.key}`;
      if(associations.has(key))throw Error(`Duplicate/conflicting sender identity: ${entry.path}`);
      associations.set(key,{...value,entry});
    }
    planned(++files,`${captures} captures checked`);
  }
  for(const entry of receivers){
    entry.records=[];
    const result=await visitRecordingJSON(entry.path,{onRecord:async raw=>{
      const record=captureEvidence(raw,entry),sender=associations.get(`${entry.origin}/${record.key}`);
      if(!sender)throw Error(`Missing matching sender for ${entry.path}: ${record.key}`);
      if(sender.entry.split!==entry.split)throw Error(`Sender and receiver cross dataset splits: ${entry.path}`);
      if(!equal(sender.sourceInfo,record.info))throw Error(`Capture metadata disagrees with sender: ${entry.path}`);
      entry.records.push({key:record.key,info:record.info,timestamp:record.timestamp,payloadId:sender.frame.payloadId,senderPath:sender.entry.path});
      captures++;planned(files,`${captures} captures checked`);
    }});
    planned(++files,`${captures} captures checked`);
    const bundle=result.metadata;entry.sha256=result.sha256;entry.metadata=bundle;entry.sessionId=bundle.sessionId;
    validateSessionId(bundle.sessionId);bundleSplit(bundle,entry);own(sessionOwners,bundle.sessionId,entry.split,'Physical session');
    const expectedFormat=entry.origin==='header-g64'?'optical-decoder-captures':'neural-qr-captures',expectedVersion=entry.origin==='native'?2:1;
    if(bundle.format!==expectedFormat||bundle.version!==expectedVersion)throw Error(`Unexpected capture contract: ${entry.path}`);
    if(entry.origin!=='header-g64'&&(bundle.profile?.size!==entry.size||entry.records.some(r=>r.info.collectionCode!==bundle.collectionCode||Object.entries(bundle.profile).some(([k,v])=>r.info[k]!==v))))throw Error(`Capture bundle profile mismatch: ${entry.path}`);
    if(!entry.records.length){skipped.push({path:entry.path,reason:'No retained captures'});continue;}
    const key=`${entry.split}/${entry.size}/${entry.origin}`,group=groups.get(key)??{split:entry.split,size:entry.size,origin:entry.origin,records:0,targets:new Map()};
    for(const r of entry.records){
      const id=`${entry.sessionId}/${r.timestamp}`;if(captureIds.has(id))throw Error(`Duplicate capture across recordings: ${id}`);captureIds.add(id);
      const target=`${entry.sessionId}/${r.payloadId}`,counts=group.targets.get(target)??[0,0,0,0];counts[r.info.phase]++;group.targets.set(target,counts);group.records++;
    }
    groups.set(key,group);
  }
  if(!groups.size)throw Error('No matched eligible recording samples found.');
  return {format:'neural-corrector-import-plan',version:1,roots,entries,skipped,groups:[...groups.values()].sort((a,b)=>a.split.localeCompare(b.split)||a.size-b.size||a.origin.localeCompare(b.origin)).map(g=>{
    const counts=[...g.targets.entries()].sort(([a],[b])=>a.localeCompare(b)).map(([,v])=>v);
    return {...g,targets:counts.length,targetCaptureCounts:counts.map(v=>v.reduce((a,b)=>a+b,0)),minCapturesPerPhase:Math.min(...counts.flat()),completeTargets:counts.filter(v=>v.every(n=>n>0)).length};
  })};
}
/** Half-pixel, separable Catmull-Rom bicubic interpolation. No 8-bit quantization.
 * Preserve calibrated overshoot; clamp interpolation overshoot only to the recording range.
 */
export function upsampleBicubic2x(input,side){
  if(!(input instanceof Float32Array)||input.length!==3*side*side)throw Error('Invalid bicubic input shape.');
  const cubic=x=>{x=Math.abs(x);return x<1?1.5*x*x*x-2.5*x*x+1:x<2?-.5*x*x*x+2.5*x*x-4*x+2:0;};
  const outSide=side*2,tmp=new Float32Array(3*side*outSide),out=new Float32Array(3*outSide*outSide);
  const taps=Array.from({length:outSide},(_,i)=>{const p=(i+.5)/2-.5,b=Math.floor(p);return [-1,0,1,2].map(d=>[Math.max(0,Math.min(side-1,b+d)),cubic(p-b-d)]);});
  for(let c=0;c<3;c++)for(let y=0;y<side;y++)for(let x=0;x<outSide;x++){let v=0;for(const [ix,w] of taps[x])v+=input[c*side*side+y*side+ix]*w;tmp[c*side*outSide+y*outSide+x]=v;}
  for(let c=0;c<3;c++)for(let y=0;y<outSide;y++)for(let x=0;x<outSide;x++){let v=0;for(const [iy,w] of taps[y])v+=tmp[c*side*outSide+iy*outSide+x]*w;out[c*outSide*outSide+y*outSide+x]=Math.max(-1.5,Math.min(1.5,v));}
  return out;
}
export async function importRecordings({plan,out,split='train'}={}){
  if(plan?.format!=='neural-corrector-import-plan'||plan.version!==1)throw Error('Expected validated recording import plan.');
  split=requireSplit(split);out=resolve(out);await mkdir(out,{recursive:true});if((await readdir(out)).length)throw Error('Import output must be empty.');
  const dataset={format:'neural-qr-dataset',version:2,created:new Date().toISOString(),description:'Canonical 4ppc targets with native, resampled and header-G64 real evidence; offline import only.',samples:[]},senders=new Map();
  for(const entry of plan.entries.filter(x=>x.kind==='sender'&&x.split===split)){
    const bytes=await readFile(entry.path);if(hash(bytes)!==entry.sha256)throw Error(`Sender changed after planning: ${entry.path}`);
    for(const value of await sourceFrames(JSON.parse(bytes),entry.origin))senders.set(`${entry.origin}/${value.key}`,{...value,entry});
  }
  const receivers=plan.entries.filter(x=>x.kind==='receiver'&&x.split===split&&x.records.length);
  const imported=progress(`Importing ${split} captures`,receivers.reduce((sum,entry)=>sum+entry.records.length,0));imported(0);
  for(const entry of receivers){
    let index=0;const bundle=entry.metadata;
    const result=await visitRecordingJSON(entry.path,{onRecord:async raw=>{
      const record=captureEvidence(raw,entry),planned=entry.records[index++],sender=senders.get(`${entry.origin}/${record.key}`);
      if(!planned||record.timestamp!==planned.timestamp||!equal(record.info,planned.info)||!sender)throw Error(`Capture changed after planning: ${entry.path}`);
      const {frame}=sender,info=frame.info,g=info.size,sourcePpc=entry.origin==='resampled'?2:4;
      let input=floatsFromBytes(base64ToBytes(record.inputBase64),3*(sourcePpc*g)**2);if(sourcePpc===2)input=upsampleBicubic2x(input,2*g);
      const mask=new Uint8Array(g*g);for(const cell of gridLayout(g,16,8).cells)mask[cell]=1;
      const directoryDirection=entry.origin==='header-g64'?(entry.receiverDevice==='pc'?'phone-to-pc':'pc-to-phone'):null;
      const senderEnvironment=sender.entry.manifest.environment??(directoryDirection?{deviceClass:entry.receiverDevice==='pc'?'mobile':'desktop',inferredFrom:'explicit recording directory'}:undefined);
      const receiverEnvironment=bundle.environment??(directoryDirection?{deviceClass:entry.receiverDevice==='pc'?'desktop':'mobile',inferredFrom:'explicit recording directory'}:undefined);
      const metadata={sessionId:entry.sessionId,senderSessionId:sender.entry.sessionId,payloadId:frame.payloadId,split,source:'real',origin:entry.origin,size:g,colors:16,pixelsPerCell:4,inputShape:[3,4*g,4*g],info,
        burstId:`${entry.sessionId}/${captureKey(info)}`,timestamp:record.timestamp,devicePair:{sender:senderEnvironment?.deviceId??null,receiver:receiverEnvironment?.deviceId??null},
        provenance:{origin:entry.origin,captureId:`${entry.sessionId}/${record.timestamp}`,sourceRecording:entry.relativePath,sourceRecordingSha256:entry.sha256,sourceFormat:bundle.format,sourceVersion:bundle.version,sourcePixelsPerCell:sourcePpc,sourcePayloadId:sender.entry.manifest.payloadId,sourceFrameKey:record.key,sourceInfo:record.info,
          interpolation:sourcePpc===2?'Catmull-Rom bicubic half-pixel 2x; clamp [-1.5,1.5]':'none',senderEnvironment,receiverEnvironment,directoryDirection,
          senderRecording:sender.entry.relativePath,senderRecordingSha256:sender.entry.sha256,collectionSessionId:bundle.collectionSessionId,stability:record.stability,transform:record.transform,colorTransform:record.colorTransform,camera:record.camera??bundle.camera,stats:record.stats}};
      dataset.samples.push(await putSample(out,dataset.samples.length,metadata,{input,labels:frame.symbols,mask,baseline:denseClassicalScores(input,info,4),expected:frame.logical}));
      imported(dataset.samples.length,`${entry.origin} G${g}`);
    }});
    if(result.sha256!==entry.sha256||index!==entry.records.length)throw Error(`Capture recording changed after planning: ${entry.path}`);
  }
  if(!dataset.samples.length)throw Error(`No eligible ${split} samples.`);
  await writeDatasetJSON(join(out,'manifest.json'),dataset);
  return {dataset:out,split,samples:dataset.samples.length,origins:Object.fromEntries(['native','resampled','header-g64'].map(origin=>[origin,dataset.samples.filter(x=>x.origin===origin).length]))};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  try{const options={};for(let i=2;i<process.argv.length;i+=2){const k=process.argv[i];if(!/^--[a-z-]+$/.test(k)||!process.argv[i+1]||process.argv[i+1].startsWith('--')||k in options)throw Error('Options require unique values.');options[k]=process.argv[i+1];}
    for(const key of Object.keys(options))if(!['--native','--resampled','--header-g64','--out','--split','--plan'].includes(key))throw Error(`Unknown option ${key}`);
    const plan=await planRecordingImports({nativeRoot:options['--native'],resampledRoot:options['--resampled'],headerRoot:options['--header-g64']});
    if(options['--plan']==='true')console.log(JSON.stringify({roots:plan.roots,groups:plan.groups,skipped:plan.skipped},null,2));
    else {if(!options['--out'])throw Error('Use --out DIRECTORY or --plan true.');console.log(JSON.stringify(await importRecordings({plan,out:options['--out'],split:options['--split']}),null,2));}
  }catch(error){console.error(`Import failed: ${error.message}`);process.exitCode=1;}
}

