#!/usr/bin/env node
/** Rebuild the current paired real/synthetic corpora from sorted recordings. */
import {readFile,readdir,lstat,realpath,rm,open} from 'node:fs/promises';
import {resolve,join,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import {senderFrames,captureKey,canonicalJSON,validateSessionId,MAX_CAPTURE_BYTES,MAX_RECORDS} from '../apps/color-optical-bench/dist/js/dataset/contract.js';
import {validateCapture} from '../apps/color-optical-bench/dist/js/dataset/capture.js';
import {gridLayout} from '../apps/color-optical-bench/dist/js/grid.js';
import {frameKey} from '../apps/color-optical-bench/dist/js/format.js';
import {joinCaptures,targets} from './corrector-dataset.mjs';

const REPO=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const PYTHON=join(REPO,'experiments','neural-corrector');
export const MANAGED_OUTPUTS=Object.freeze(['real-train','real-validation','target-clean-train',
  'target-clean-validation','target-synthetic-train','target-synthetic-validation','session-groups.json']);
const SPLITS=[['train','train'],['val','validation']];
const DEFAULTS=Object.freeze({dataRoot:'local-data',python:'python3',frames:12,burstsPerTarget:32,
  validationBurstsPerTarget:4,seed:41,severityMin:.45,severity:.9,maxOutputMib:2**20,threads:1,dryRun:false});
const RECIPE=['--glare-probability','.7','--glare-strength','.65','--white-balance-stops','.8',
  '--exposure-stops','1.75','--shade-opacity','.55','--rolling-shutter-stops','.8',
  '--capture-probability','.75','--capture-min-scale','.25','--capture-max-scale','1',
  '--capture-blur-sigma','.35','--rgb-offset-probability','.6','--rgb-offset-pixels','.5'];

function config(options) {
  for(const key of Object.keys(options))if(!(key in DEFAULTS))throw Error(`Unknown rebuild option: ${key}`);
  const value={...DEFAULTS,...options};
  for(const [key,min,max] of [['frames',2,16],['burstsPerTarget',1,2**20],['validationBurstsPerTarget',1,2**20],
    ['seed',0,0xfffffffe],['maxOutputMib',0,Number.MAX_SAFE_INTEGER],['threads',1,64]]) {
    if(!Number.isSafeInteger(value[key])||value[key]<min||value[key]>max)throw Error(`${key} must be an integer in [${min}, ${max}].`);
  }
  if(!Number.isFinite(value.severity)||!Number.isFinite(value.severityMin)||value.severityMin<0||value.severity>1||value.severityMin>value.severity)throw Error('Require 0 <= severity-min <= severity <= 1.');
  if(typeof value.python!=='string'||!value.python||typeof value.dataRoot!=='string'||!value.dataRoot||typeof value.dryRun!=='boolean')throw Error('Invalid rebuild paths/options.');
  return {...value,dataRoot:resolve(value.dataRoot)};
}
async function directory(path) {
  const info=await lstat(path);
  if(info.isSymbolicLink()||!info.isDirectory()||await realpath(path)!==resolve(path))throw Error(`Use a real directory without symlink components: ${path}`);
}
async function scan(path) {
  await directory(path);const files=[];
  for(const item of (await readdir(path,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))) {
    const child=join(path,item.name);
    if(item.isSymbolicLink())throw Error(`Recording symlinks are not supported: ${child}`);
    if(item.isDirectory())files.push(...await scan(child));
    else if(item.isFile()&&item.name.toLowerCase().endsWith('.json'))files.push(child);
  }
  return files;
}
async function outputState(root) {
  const found=[];
  for(const name of MANAGED_OUTPUTS) {
    const path=join(root,name);let info;
    try{info=await lstat(path);}catch(error){if(error.code==='ENOENT')continue;throw error;}
    if(info.isSymbolicLink()||(name.endsWith('.json')?!info.isFile():!info.isDirectory()))throw Error(`Refusing unexpected or symlink output: ${path}`);
    found.push(name);
  }
  return found;
}
function owner(map,key,split,label) {
  const previous=map.get(key);
  if(previous&&previous!==split)throw Error(`${label} crosses train/validation: ${key}`);
  map.set(key,split);
}
function guard(record,path) {
  const g=record.stability;
  if(!g||!Number.isInteger(g.guardMs)||g.guardMs<500||g.guardMs>5000||!Number.isFinite(g.durationMs)||g.durationMs<g.guardMs||!Number.isInteger(g.observations)||g.observations<2)throw Error(`Capture has no valid stable-header guard: ${path}`);
}
// Read one bounded bundle at a time. Retain only header/session metadata, never
// the large tensor/base64 arrays from all of a user's capture files together.
async function inspect(path,device,split) {
  const stat=await lstat(path);
  if(stat.isSymbolicLink()||!stat.isFile()||stat.size>MAX_CAPTURE_BYTES+4*1024*1024)throw Error(`Invalid or oversized recording: ${path}`);
  const bundle=JSON.parse(await readFile(path,'utf8'));
  validateSessionId(bundle.sessionId);
  const entry={path,device,split,sessionId:bundle.sessionId,stat:{size:stat.size,mtimeMs:stat.mtimeMs,ino:stat.ino}};
  if(bundle.format==='optical-decoder-sender') {
    if(stat.size>1024*1024)throw Error(`Oversized sender manifest: ${path}`);
    const frames=await senderFrames(bundle);
    return {...entry,kind:'sender',payloadId:bundle.payloadId,sequenceId:bundle.sequenceId,
      frames:frames.map(frame=>({key:captureKey(frame),info:bundle.frames.find(row=>row.key===captureKey(frame)).info})),matched:false};
  }
  if(bundle.format!=='optical-decoder-captures'||bundle.version!==1||!Array.isArray(bundle.records)||bundle.records.length>MAX_RECORDS)throw Error(`Unsupported current recording: ${path}`);
  if(bundle.diagnostics!==undefined&&(!Array.isArray(bundle.diagnostics)||bundle.diagnostics.length>100))throw Error(`Invalid capture diagnostics: ${path}`);
  const records=[];
  for(const record of bundle.records) {
    validateCapture(record);guard(record,path);
    records.push({key:record.key,info:record.info,timestamp:record.timestamp});
  }
  return {...entry,kind:'capture',records,diagnostics:(bundle.diagnostics??[]).length};
}

/** Validate all current inputs and split ownership without writing any files. */
export async function planRebuild(options={}) {
  const cfg=config(options),root=cfg.dataRoot;await directory(root);
  const files=[];
  for(const device of ['phone','pc']) {
    const deviceRoot=join(root,device);await directory(deviceRoot);
    // A stray export outside a split is usually a missed move, not an exclusion.
    for(const item of await readdir(deviceRoot,{withFileTypes:true})) {
      if(item.isSymbolicLink())throw Error(`Recording symlinks are not supported: ${join(deviceRoot,item.name)}`);
      if(item.isFile()&&item.name.toLowerCase().endsWith('.json'))throw Error(`Move recording into train/ or val/ first: ${join(deviceRoot,item.name)}`);
    }
    for(const [folder,split] of SPLITS)for(const path of await scan(join(deviceRoot,folder)))files.push({path,device,split});
  }
  const entries=[];
  for(const file of files)entries.push(await inspect(file.path,file.device,file.split));
  const sessions=new Map(),payloads=new Map(),sequences=new Map(),headers=new Map(),groups=new Map(),captures=new Set();
  for(const entry of entries) {
    owner(sessions,entry.sessionId,entry.split,'Physical session');
    if(entry.kind!=='sender')continue;
    owner(payloads,entry.payloadId,entry.split,'Payload');
    const prior=sequences.get(entry.sequenceId);
    if(prior)throw Error(`Duplicate/conflicting sender sequence ${entry.sequenceId}: ${prior.path} and ${entry.path}`);
    sequences.set(entry.sequenceId,entry);
    for(const frame of entry.frames) {
      if(frame.info.count!==1||frame.info.index!==0)throw Error(`Capture rebuild requires one logical matrix and its palette phases: ${entry.path}`);
      if(headers.has(frame.key))throw Error(`Duplicate/conflicting sender header: ${entry.path}`);
      headers.set(frame.key,{entry,info:frame.info});
    }
  }
  const summaries={train:{senders:0,captureFiles:0,records:0,bursts:new Map(),phaseGroups:new Map(),diagnostics:0},
    validation:{senders:0,captureFiles:0,records:0,bursts:new Map(),phaseGroups:new Map(),diagnostics:0}};
  for(const entry of entries) {
    const summary=summaries[entry.split];
    if(entry.kind==='sender'){summary.senders++;continue;}
    summary.captureFiles++;summary.diagnostics+=entry.diagnostics;
    const direction=entry.device==='phone'?'pc-to-phone':'phone-to-pc',previous=groups.get(entry.sessionId);
    if(previous&&previous!==direction)throw Error(`Receiver session is assigned to both directions: ${entry.sessionId}`);
    groups.set(entry.sessionId,direction);
    for(const record of entry.records) {
      const identity=`${entry.sessionId}/${record.timestamp}`;
      if(captures.has(identity))throw Error(`Duplicate receiver capture ${identity}: ${entry.path}`);
      captures.add(identity);
      const sender=headers.get(record.key);
      if(!sender)throw Error(`Unmatched capture header in ${entry.path}: ${record.key}`);
      if(sender.entry.split!==entry.split)throw Error(`Sender/receiver pair crosses train/validation: ${entry.path}`);
      if(sender.entry.device===entry.device)throw Error(`Sender and receiver are in the same device folder: ${entry.path}`);
      if(canonicalJSON(sender.info)!==canonicalJSON(record.info))throw Error(`Capture metadata disagrees with canonical sender header: ${entry.path}`);
      sender.entry.matched=true;summary.records++;
      summary.bursts.set(`${entry.sessionId}/${record.key}`,record.info);
      const info=record.info;
      if(info.size!==64||info.colors!==16||info.phases!==4||!info.permute)throw Error(`Phase-grouped rebuild requires G64, 16 colors and four permuted phases: ${entry.path}`);
      const key=`${entry.sessionId}/${frameKey(info)}`,counts=summary.phaseGroups.get(key)??[0,0,0,0];
      counts[info.phase]++;summary.phaseGroups.set(key,counts);
    }
  }
  for(const entry of entries)if(entry.kind==='sender'&&!entry.matched)throw Error(`Sender has no matching captures: ${entry.path}`);
  for(const [split,summary] of Object.entries(summaries)) {
    if(!summary.records)throw Error(`No matched real captures in ${split}. Both train and val folders need captures.`);
    for(const counts of summary.phaseGroups.values())if(counts.some(count=>count<2))throw Error(`${split} requires all four phases with at least two captures each per receiver session/frame.`);
    summary.phaseGroups=summary.phaseGroups.size;
    const repetitions=split==='train'?cfg.burstsPerTarget:cfg.validationBurstsPerTarget;
    // Same binary estimate as synthetic.generate; manifests need extra disk.
    let binaryBytes=0;
    for(const info of summary.bursts.values())binaryBytes+=3*16*info.size**2*4+2*info.size**2+
      gridLayout(info.size,info.colors,info.gridBlock).cells.length*info.colors*4+info.length;
    summary.syntheticCaptures=summary.bursts.size*repetitions*cfg.frames;
    summary.estimatedSyntheticMiB=binaryBytes*repetitions*cfg.frames/1024**2;
    if(cfg.maxOutputMib&&summary.estimatedSyntheticMiB>cfg.maxOutputMib)throw Error(`${split} synthetic estimate exceeds --max-output-mib; increase it before rebuilding.`);
    summary.bursts=summary.bursts.size;
  }
  return {config:cfg,entries,summary:summaries,existingOutputs:await outputState(root),
    outputs:MANAGED_OUTPUTS.map(name=>join(root,name))};
}

function run(command,args,{input}={}) {
  return new Promise((accept,reject)=>{
    const child=spawn(command,args,{cwd:REPO,stdio:[input===undefined?'ignore':'pipe','inherit','inherit']});
    child.on('error',reject);
    child.on('close',code=>code===0?accept():reject(Error(`${command} exited with ${code}.`)));
    if(input!==undefined){child.stdin.on('error',reject);child.stdin.end(input);}
  });
}
function syntheticArgs(plan,split) {
  const c=plan.config;
  return [join(PYTHON,'synthetic.py'),'--clean',join(c.dataRoot,`target-clean-${split}`),
    '--out',join(c.dataRoot,`target-synthetic-${split}`),'--phase-grouped','--frames',String(c.frames),
    '--bursts-per-target',String(split==='train'?c.burstsPerTarget:c.validationBurstsPerTarget),
    '--seed',String(c.seed+(split==='validation'?1:0)),'--severity-min',String(c.severityMin),
    '--severity',String(c.severity),'--max-output-mib',String(c.maxOutputMib),
    '--threads',String(c.threads),'--node',process.execPath,...RECIPE];
}

export async function rebuild(options={}, {log=console.log}={}) {
  const plan=await planRebuild(options),c=plan.config;
  log(JSON.stringify({dataRoot:c.dataRoot,summary:plan.summary,replace:plan.existingOutputs,
    outputs:plan.outputs,synthetics:{phaseGrouped:true,framesPerPhase:c.frames,burstsPerTarget:c.burstsPerTarget,
      validationBurstsPerTarget:c.validationBurstsPerTarget,trainSeed:c.seed,validationSeed:c.seed+1}},null,2));
  // Dependency errors occur before deletion, including unavailable Python envs.
  await run(c.python,['-c','import sys; sys.path.insert(0, sys.argv[1]); import synthetic, session_groups',PYTHON]);
  if(c.dryRun){log('Dry run complete; no files changed.');return {...plan,dryRun:true};}
  const lockPath=join(c.dataRoot,'.decoder-rebuild.lock');
  let lock;
  try{lock=await open(lockPath,'wx');}catch(error){if(error.code==='EEXIST')throw Error(`A rebuild lock exists: ${lockPath}. Do not run concurrent rebuilds; remove a stale lock only after its process has stopped.`);throw error;}
  let replacing=false;
  try {
    await lock.writeFile(`${process.pid}\n`);
    await outputState(c.dataRoot);
    for(const entry of plan.entries) {
      const now=await lstat(entry.path);
      if(now.isSymbolicLink()||now.size!==entry.stat.size||now.mtimeMs!==entry.stat.mtimeMs||now.ino!==entry.stat.ino)throw Error(`Recording changed during preflight: ${entry.path}`);
    }
    replacing=true;
    for(const name of MANAGED_OUTPUTS)await rm(join(c.dataRoot,name),{recursive:!name.endsWith('.json'),force:true});
    for(const split of ['train','validation']) {
      const rows=plan.entries.filter(entry=>entry.split===split);
      log(`Joining ${split} recordings…`);
      const result=await joinCaptures({out:join(c.dataRoot,`real-${split}`),split,
        senders:rows.filter(row=>row.kind==='sender').map(row=>row.path),
        captures:rows.filter(row=>row.kind==='capture').map(row=>row.path)});
      if(result.excluded.length||result.samples!==plan.summary[split].records)throw Error(`Unexpected ${split} join count; recordings changed or captures were excluded.`);
      log(`Reconstructing ${split} clean targets…`);
      await targets({dataset:[join(c.dataRoot,`real-${split}`)],out:join(c.dataRoot,`target-clean-${split}`)});
    }
    const groupInput={datasets:['train','validation'].map(split=>join(c.dataRoot,`real-${split}`)),
      out:join(c.dataRoot,'session-groups.json'),
      pc_to_phone:plan.entries.filter(e=>e.kind==='capture'&&e.device==='phone').map(e=>e.path),
      phone_to_pc:plan.entries.filter(e=>e.kind==='capture'&&e.device==='pc').map(e=>e.path)};
    await run(c.python,['-c',
      'import json,sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from session_groups import create_session_groups; p=json.load(sys.stdin); create_session_groups(p["datasets"],Path(p["out"]),pc_to_phone=p["pc_to_phone"],phone_to_pc=p["phone_to_pc"])',PYTHON],
      {input:JSON.stringify(groupInput)});
    for(const split of ['train','validation']) {
      log(`Generating ${split} paired synthetics…`);
      await run(c.python,syntheticArgs(plan,split));
    }
    log('Rebuild complete. Raw recordings and checkpoints were not changed.');
    return {dataRoot:c.dataRoot,summary:plan.summary,outputs:plan.outputs,dryRun:false};
  }catch(error) {
    if(replacing)error.message=`Rebuild incomplete: ${error.message} Raw recordings are unchanged; correct the error and rerun to replace these derived outputs.`;
    throw error;
  }finally{await lock.close();await rm(lockPath,{force:true});}
}

const HELP=`Usage: node scripts/rebuild-corrector-data.mjs [options]

Read local-data/{phone,pc}/{train,val}/ recursively. Match complete optical
headers; filenames and filesystem dates are not used for pairing. Validate
inputs first, then replace ONLY the fixed real-*, target-clean-* and
target-synthetic-* train/validation directories plus session-groups.json.
Require G64/16-color recordings with all four palette phases and at least two
captures per phase. Generate coherent phase-grouped synthetic trajectories.
Stop training that uses these paths before rebuilding. Raw recordings and
checkpoints are never deleted. Older differently named datasets are untouched.

  --data-root PATH                    default: local-data
  --python EXECUTABLE                 default: python3 (active venv)
  --frames N                          synthetic captures per phase burst, 2–16 (12)
  --bursts-per-target N                training corruption bursts (32)
  --validation-bursts-per-target N     validation corruption bursts (4)
  --seed N                            train seed (41); validation uses N+1
  --severity-min N                    minimum corruption severity (.45)
  --severity N                        maximum corruption severity (.9)
  --max-output-mib N                  per-split synthetic budget (1048576); 0 disables
  --threads N                         Python CPU threads (1)
  --dry-run                           validate and print plan; change nothing
  --help                              show this help
`;
function parse(args) {
  const result={},names={'data-root':'dataRoot',python:'python',frames:'frames','bursts-per-target':'burstsPerTarget',
    'validation-bursts-per-target':'validationBurstsPerTarget',seed:'seed','severity-min':'severityMin',
    severity:'severity','max-output-mib':'maxOutputMib',threads:'threads'};
  for(let i=0;i<args.length;i++) {
    const key=args[i];
    if(key==='--dry-run'){if(result.dryRun)throw Error('Duplicate --dry-run.');result.dryRun=true;continue;}
    const name=key.startsWith('--')?names[key.slice(2)]:undefined;
    if(!name||name in result||args[i+1]===undefined||args[i+1].startsWith('--'))throw Error(`Unknown, duplicate or incomplete option: ${key}`);
    result[name]=['dataRoot','python'].includes(name)?args[++i]:Number(args[++i]);
  }
  return result;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  if(process.argv.slice(2).includes('--help'))console.log(HELP);
  else Promise.resolve().then(()=>rebuild(parse(process.argv.slice(2)))).catch(error=>{console.error(`Corrector rebuild failed: ${error.message}`);process.exitCode=1;});
}
