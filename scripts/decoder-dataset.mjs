#!/usr/bin/env node
/** Canonical OBM5 data generation, offline label joining, and exact recovery. */
import {readFile,writeFile,mkdir,stat,realpath,readdir} from 'node:fs/promises';
import {resolve,dirname,join,sep,isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {createSender,senderFrames,captureKey,base64ToBytes,floatBytes,floatsFromBytes,DATASET_FORMAT,validateSessionId,canonicalJSON} from '../apps/color-optical-bench/dist/js/dataset/contract.js';
import {validateCapture} from '../apps/color-optical-bench/dist/js/dataset/capture.js';
import {renderFrame,parseHeader,PALETTE,sameBytes} from '../apps/color-optical-bench/dist/js/format.js';
import {gridLayout} from '../apps/color-optical-bench/dist/js/grid.js';
import {extractLearnedInput,logitsToLogicalScores} from '../apps/color-optical-bench/dist/js/learned-input.js';
import {classifySamples,decodeScores} from '../apps/color-optical-bench/dist/js/vision.js';

const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const MAX_JSON=36*1024*1024;
function int(value,def,min,max){const n=value===undefined?def:Number(value);if(!Number.isInteger(n)||n<min||n>max)throw Error('Invalid numeric argument.');return n;}
function splitName(value='train'){if(!['train','validation','test'].includes(value))throw Error('Use split train, validation, or test.');return value;}
function flags(args) {
  const result={};
  for(let i=0;i<args.length;i++) {if(!args[i].startsWith('--')||!args[i+1]||args[i+1].startsWith('--'))throw Error('Options require values.');const key=args[i].slice(2);if(key in result)throw Error(`Duplicate option --${key}`);result[key]=args[++i];}
  return result;
}
function only(options,names){for(const key of Object.keys(options))if(!names.includes(key))throw Error(`Unknown option --${key}`);}
function safePath(root,name) {
  if(typeof name!=='string'||!name||isAbsolute(name)||name.includes('\\')||name.split('/').some(part=>!part||part==='.'||part==='..'))throw Error('Unsafe dataset path.');
  const target=resolve(root,name);if(!target.startsWith(resolve(root)+sep))throw Error('Dataset path escapes root.');return target;
}
async function safeReadPath(root,name) {
  const candidate=safePath(root,name),actual=await realpath(candidate),base=await realpath(root);
  if(!actual.startsWith(base+sep))throw Error('Dataset symlink escapes root.');
  return actual;
}
async function jsonFile(path,limit=MAX_JSON){if((await stat(path)).size>limit)throw Error(`JSON file exceeds bound: ${path}`);return JSON.parse(await readFile(path,'utf8'));}
async function freshDirectory(path){await mkdir(path,{recursive:true});if((await readdir(path)).length)throw Error('Output dataset directory must be empty; use a new output directory.');}
async function putSample(out,index,metadata,{input,labels,mask,baseline,expected,samples,original}) {
  const prefix=`sample-${String(index).padStart(6,'0')}`;
  const identity={sessionId:metadata.sessionId,payloadId:metadata.payloadId,key:captureKey(metadata.info),timestamp:metadata.timestamp};
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
  const out=resolve(options.out),size=int(options.size,64,64,128),colors=int(options.colors,16,4,16),phases=int(options.phases,4,1,16),count=int(options.count,8,1,256),seed=int(options.seed,7,0,0xffffffff),split=splitName(options.split);
  if(![64,128].includes(size)||![4,16].includes(colors)||![1,2,4,8,16].includes(phases))throw Error('Unsupported generation profile.');
  await freshDirectory(out);
  const sessionId=`synthetic-${seed}-${size}-${colors}`,dataset={format:DATASET_FORMAT,version:1,created:new Date().toISOString(),
    description:'Clean canonical OBM5 renderer crops. Corruption/training is a separate Python step.',samples:[]};
  for(let i=0;i<count;i++) {
    const generated=await createSender({size,colors,phases,seed:(seed+i)>>>0,sessionId});
    for(const frame of generated.frames) {
      const pixels=size===64?1024:2048,image=renderFrame(frame,pixels),scale=pixels/256;
      const learned=extractLearnedInput(image,{transform:[scale,0,0,0,scale,0,0,0,1]});
      if(!learned.ok)throw Error(`Clean canonical crop failed: ${learned.error}`);
      const info=parseHeader(frame.header);
      dataset.samples.push(await putSample(out,dataset.samples.length,{sessionId,payloadId:generated.manifest.payloadId,split,source:'synthetic',size,colors,info,
        burstId:captureKey(info),timestamp:info.phase*500,provenance:{generator:'OBM5 renderFrame + validated dense crop',seed:(seed+i)>>>0,corruption:'none',baseline:'calibrated-crop approximation'}},
        {input:learned.input,labels:frame.symbols,mask:maskFor(info),baseline:denseClassicalScores(learned.input,info),expected:frame.chunk}));
    }
  }
  await writeFile(join(out,'manifest.json'),JSON.stringify(dataset,null,2)+'\n');
  return {dataset:out,samples:dataset.samples.length,split,size,colors};
}

export async function joinCaptures(options) {
  only(options,['out','senders','captures','split','exclude']);
  if(!options.out||!options.senders||!options.captures)throw Error('join requires --out DIRECTORY --senders FILE[,FILE] --captures FILE[,FILE].');
  const out=resolve(options.out),split=splitName(options.split),senders=new Map(),senderIds=new Map();
  for(const path of options.senders.split(',')) {
    const manifest=await jsonFile(path,1024*1024),frames=await senderFrames(manifest);
    for(const frame of frames) {
      const key=captureKey(frame),signature=JSON.stringify({payloadId:manifest.payloadId,sessionId:manifest.sessionId,header:Buffer.from(frame.header).toString('hex')});
      if(senders.has(key)&&senders.get(key).signature!==signature)throw Error('Conflicting sender manifests for an optical header.');
      const previous=senderIds.get(frame.id);if(previous&&previous!==manifest.payloadId)throw Error('Conflicting payloads reuse a sequence ID.');
      senderIds.set(frame.id,manifest.payloadId);senders.set(key,{signature,frame,manifest});
    }
  }
  let exclusions=[];if(options.exclude){exclusions=await jsonFile(options.exclude,65536);if(!Array.isArray(exclusions)||exclusions.some(x=>typeof x!=='string'))throw Error('Exclusions must be capture IDs in a JSON array.');}
  const dataset={format:DATASET_FORMAT,version:1,created:new Date().toISOString(),samples:[],excluded:[],diagnostics:[]},ids=new Set();
  await freshDirectory(out);
  for(const path of options.captures.split(',')) {
    const bundle=await jsonFile(path);if(bundle.format!=='optical-decoder-captures'||bundle.version!==1||!Array.isArray(bundle.records)||bundle.records.length>20)throw Error('Unsupported receiver capture bundle.');
    validateSessionId(bundle.sessionId);
    for(let i=0;i<bundle.records.length;i++) {
      const captureId=`${bundle.sessionId}/${bundle.records[i].timestamp}`;
      if(ids.has(captureId))throw Error('Duplicate capture session/record across files.');ids.add(captureId);
      if(exclusions.includes(captureId)){dataset.excluded.push({captureId,reason:'Explicit offline exclusion'});continue;}
      const record=validateCapture(bundle.records[i]);
      const guard=record.stability;
      if(!guard||!Number.isInteger(guard.guardMs)||guard.guardMs<500||guard.guardMs>5000||!Number.isFinite(guard.durationMs)||guard.durationMs<guard.guardMs||!Number.isInteger(guard.observations)||guard.observations<2)throw Error('Capture has no stable-header guard evidence.');
      const sender=senders.get(record.key);
      if(!sender){dataset.excluded.push({captureId,reason:'No matching sender/header; remains unlabeled'});continue;}
      const info=parseHeader(sender.frame.header);
      // Exact canonical metadata match, not just a sequence ID or payload checksum.
      if(canonicalJSON(info)!==canonicalJSON(record.info))throw Error('Capture metadata disagrees with sender header.');
      const input=floatsFromBytes(base64ToBytes(record.inputBase64),3*16*info.size**2),samples=floatsFromBytes(base64ToBytes(record.samplesBase64),info.size**2*3+96);
      const baseline=new Float32Array(classifySamples(info,samples,{local:true}).scores);
      const original=record.original?base64ToBytes(record.original.base64,1024*1024):undefined;
      dataset.samples.push(await putSample(out,dataset.samples.length,{sessionId:bundle.sessionId,senderSessionId:sender.manifest.sessionId,payloadId:sender.manifest.payloadId,split,source:'real',size:info.size,colors:info.colors,info,
        burstId:`${bundle.sessionId}/${record.key}`,timestamp:record.timestamp,
        provenance:{captureId,stats:record.stats,camera:record.camera,transform:record.transform,colorTransform:record.colorTransform,stability:record.stability,notes:bundle.notes??''}},
        {input,labels:sender.frame.symbols,mask:maskFor(info),baseline,expected:sender.frame.chunk,samples,original}));
    }
    for(const diagnostic of (bundle.diagnostics??[]).slice(0,100))dataset.diagnostics.push({sessionId:bundle.sessionId,...diagnostic});
  }
  if(!dataset.samples.length)throw Error('No labeled captures matched; check headers/manifests and exclusions.');
  await writeFile(join(out,'manifest.json'),JSON.stringify(dataset,null,2)+'\n');
  return {dataset:out,samples:dataset.samples.length,excluded:dataset.excluded,diagnostics:dataset.diagnostics.length,split};
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
  if(manifest.format!==DATASET_FORMAT||manifest.version!==1||!Array.isArray(manifest.samples)||manifest.samples.length>100000||manifest.samples.some(s=>s.source!=='synthetic'))throw Error('Baseline refresh accepts only a synthetic dataset.');
  if(new Set(manifest.samples.map(s=>s.id)).size!==manifest.samples.length)throw Error('Duplicate dataset sample ID.');
  for(const sample of manifest.samples) {
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
  if(manifest.format!==DATASET_FORMAT||manifest.version!==1||!Array.isArray(manifest.samples)||manifest.samples.length>100000||!Array.isArray(predictions.samples)||predictions.samples.length>100000)throw Error('Unsupported dataset/prediction manifest.');
  if(new Set(manifest.samples.map(s=>s.id)).size!==manifest.samples.length)throw Error('Duplicate dataset sample ID.');
  if(options.split)splitName(options.split);
  const eligible=manifest.samples.filter(s=>!options.split||s.split===options.split),eligibleIds=new Set(eligible.map(s=>s.id)),byId=new Map();
  for(const prediction of predictions.samples){if(typeof prediction.id!=='string'||byId.has(prediction.id))throw Error('Duplicate/invalid prediction ID.');byId.set(prediction.id,prediction);}
  for(const id of byId.keys())if(!eligibleIds.has(id))throw Error(`Prediction ID is absent from the selected dataset split: ${id}`);
  const selected=eligible.filter(sample=>byId.has(sample.id));
  if(!selected.length)throw Error('No predictions selected for recovery evaluation.');
  const results=[];
  for(const sample of selected) {
    const prediction=byId.get(sample.id);if(!prediction)throw Error(`Missing prediction for ${sample.id}.`);
    const info=sample.info,g=sample.size;
    if(![64,128].includes(g)||![4,16].includes(sample.colors)||info.size!==g||info.colors!==sample.colors)throw Error('Unsupported dataset shape.');
    const input=floatsFromBytes(await readSampleFile(root,sample,'input',3*16*g*g*4),3*16*g*g),expected=await readSampleFile(root,sample,'expected',65536);
    const logitsPath=await safeReadPath(dirname(predictionPath),prediction.logits),logitLength=16*g*g*4;
    if((await stat(logitsPath)).size!==logitLength)throw Error('Wrong logits file length.');
    const logits=floatsFromBytes(new Uint8Array(await readFile(logitsPath)),16*g*g);
    const count=gridLayout(g,info.colors,info.gridBlock).cells.length*info.colors;
    const baseline=sample.source==='synthetic'?denseClassicalScores(input,info):floatsFromBytes(await readSampleFile(root,sample,'baseline',count*4),count);
    const learnedScores=logitsToLogicalScores(logits,info);
    // Only compare expected bytes after independent CRC/RS acceptance.
    results.push({id:sample.id,source:sample.source,size:g,colors:info.colors,classical:recoveryResult(info,baseline,expected),learned:recoveryResult(info,learnedScores,expected)});
  }
  const summary={samples:results.length};
  for(const method of ['classical','learned']) {summary[method]={};for(const field of ['rawCRC','hardCRC','accepted','exact','wrongAccepted'])summary[method][field]=results.filter(r=>r[method][field]).length;}
  return {format:'optical-decoder-recovery-report',version:1,scope:'Single captured exposure per candidate; synthetic baseline reconstructed from the same calibrated crop; no temporal pooling.',summary,samples:results};
}

async function main() {
  const [command,...args]=process.argv.slice(2),options=flags(args);
  const action={generate,join:joinCaptures,baselines,evaluate}[command];
  if(!action)throw Error('Usage: decoder-dataset.mjs generate|join|baselines|evaluate [--option value].');
  console.log(JSON.stringify(await action(options),null,2));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(error=>{console.error(`Decoder dataset failed: ${error.message}`);process.exitCode=1;});
