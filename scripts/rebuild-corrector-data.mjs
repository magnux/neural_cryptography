#!/usr/bin/env node
/** Rebuild the 4-pixel W-tube corpus, without modifying source recordings. */
import {lstat,realpath,readFile,readdir,mkdir,mkdtemp,writeFile,rename,rm,statfs} from 'node:fs/promises';
import {resolve,dirname,join,sep,basename} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {planRecordingImports,importRecordings} from './import-corrector-recordings.mjs';
import {targets} from './corrector-dataset.mjs';

export const BUILD_FORMAT='neural-corrector-w-tube-build';
export const MANAGED_OUTPUTS=Object.freeze(['real','clean','synthetic','build.json']);
const ROOT=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const DEFAULTS={recordingsRoot:'recordings',resampledRoot:'recordings-1',headerRoot:'recordings-0',
  out:'local-data/w-tube',python:'python3',frames:8,syntheticRatio:4,severity:1,severityMin:.5,seed:41,maxOutputMib:2**20};
function settings(options={}) {
  for(const key of Object.keys(options))if(!(key in DEFAULTS))throw Error(`Unknown rebuild option: ${key}`);
  const value={...DEFAULTS,...options};
  for(const [key,min,max]of [['frames',2,16],['syntheticRatio',1,64],['seed',0,0xffffffff],['maxOutputMib',0,2**30]])
    if(!Number.isSafeInteger(value[key])||value[key]<min||value[key]>max)throw Error(`Invalid ${key}.`);
  if(!Number.isFinite(value.severity)||value.severity<0||value.severity>1||!Number.isFinite(value.severityMin)||value.severityMin<0||value.severityMin>value.severity)throw Error('Severity must satisfy 0 <= minimum <= maximum <= 1.');
  if(typeof value.python!=='string'||!value.python)throw Error('Python must be a nonempty executable path.');
  for(const name of ['recordingsRoot','resampledRoot','headerRoot','out']) {
    if(typeof value[name]!=='string'||!value[name])throw Error(`${name} must be a nonempty path.`);
    value[name]=resolve(value[name]);
  }
  return value;
}
async function exists(path){try{return await lstat(path);}catch(error){if(error.code==='ENOENT')return null;throw error;}}
async function validateDestination(out) {
  let parent=dirname(out);while(!await exists(parent))parent=dirname(parent);
  if(await realpath(parent)!==parent)throw Error('Dataset output must not have symlink parents.');
  const info=await exists(out);if(!info)return null;
  if(info.isSymbolicLink()||!info.isDirectory()||await realpath(out)!==out)throw Error('Dataset output must be a real directory.');
  const entries=await readdir(out,{withFileTypes:true});
  if(!entries.length)return `empty:${info.dev}:${info.ino}:${info.mtimeMs}`;
  if(entries.some(entry=>entry.isSymbolicLink()||!MANAGED_OUTPUTS.includes(entry.name)))throw Error('Refusing to replace a directory containing files outside the managed W-tube dataset.');
  let marker,bytes;try{bytes=await readFile(join(out,'build.json'));marker=JSON.parse(bytes);}catch{throw Error('Existing output is not a managed W-tube dataset (missing build.json).');}
  if(marker.format!==BUILD_FORMAT||marker.version!==1)throw Error('Unsupported build marker; choose a new output directory.');
  return `${info.dev}:${info.ino}:${info.mtimeMs}:${createHash('sha256').update(bytes).digest('hex')}`;
}
function assertSeparate(a,b) {
  if(a===b||a.startsWith(b+sep)||b.startsWith(a+sep))throw Error('Dataset output and recording roots must be separate, non-nested directories.');
}
async function prepare(options) {
  const config=settings(options),roots=[config.recordingsRoot,config.resampledRoot,config.headerRoot];
  for(const source of roots)assertSeparate(config.out,source);
  const destinationStamp=await validateDestination(config.out);
  const imports=await planRecordingImports({nativeRoot:config.recordingsRoot,resampledRoot:config.resampledRoot,headerRoot:config.headerRoot});
  const quantum=4*config.frames;
  const groups=imports.groups.map(group=>{
    if(group.minCapturesPerPhase<1)throw Error(`${group.split}/G${group.size}/${group.origin} has a target missing a phase; phase-grouped inputs require at least one capture of each phase.`);
    if(!Array.isArray(group.targetCaptureCounts)||group.targetCaptureCounts.length!==group.targets)throw Error('Import plan lacks per-target capture counts.');
    const captures=group.targetCaptureCounts.reduce((sum,count)=>sum+quantum*Math.ceil(config.syntheticRatio*count/quantum),0),clean=4*group.targets;
    // Conservative binary estimate: RGB192G² + baseline<=64G² + labels/mask/expected3G²,
    // plus 8 KiB/sample for metadata. Old output coexists until replacement succeeds.
    const estimatedBytes=(group.records+clean+captures)*(259*group.size**2+8192);
    return {split:group.split,size:group.size,origin:group.origin,targets:group.targets,realCaptures:group.records,
      minCapturesPerPhase:group.minCapturesPerPhase,someTargetsLackK8:group.minCapturesPerPhase<2,cleanCaptures:clean,syntheticCaptures:captures,achievedRatio:captures/group.records,estimatedBytes};
  });
  if(!groups.length)throw Error('No eligible complete recordings found.');
  const estimatedBytes=groups.reduce((sum,group)=>sum+group.estimatedBytes,0);
  if(config.maxOutputMib&&estimatedBytes>config.maxOutputMib*1024**2)throw Error(`Estimated output ${(estimatedBytes/1024**3).toFixed(2)} GiB exceeds --max-output-mib ${config.maxOutputMib}.`);
  const summary={format:BUILD_FORMAT,version:1,config,replacesExisting:destinationStamp!==null,groups,estimatedBytes,
    pixelsPerCell:4,syntheticCaptureQuantum:quantum,requestedSyntheticRatio:config.syntheticRatio,
    skipped:imports.skipped??[],trainingPipelineReady:false,
    note:'New corpus only: 4x4-cell G16/G32/G64/G128. Existing 2x2 multires training cannot consume this corpus; W-tube training is the next milestone. Space estimate is for the new corpus; keep room for both it and any output being replaced.'};
  return {config,imports,destinationStamp,summary};
}
export async function planRebuild(options={}) {return (await prepare(options)).summary;}
function run(executable,args,{quiet=false}={}) {
  return new Promise((accept,reject)=>{
    const child=spawn(executable,args,{cwd:ROOT,stdio:quiet?['ignore','ignore','pipe']:['ignore','inherit','inherit']});let stderr='';
    child.stderr?.on('data',chunk=>{stderr=(stderr+chunk).slice(-8192);});
    child.on('error',reject);child.on('close',code=>code===0?accept():reject(Error(`${basename(executable)} failed (${code})${stderr?`: ${stderr.trim()}`:''}`)));
  });
}
export async function rebuild(options={}) {
  const {config,imports,destinationStamp,summary}=await prepare(options);
  // Dependencies are checked before creating staging directories or touching output.
  await run(config.python,['-c','import numpy, torch'],{quiet:true});
  let existingParent=dirname(config.out);while(!await exists(existingParent))existingParent=dirname(existingParent);
  const disk=await statfs(existingParent,{bigint:true}),free=disk.bavail*disk.bsize;
  if(BigInt(summary.estimatedBytes)>free)throw Error(`Estimated new corpus ${(summary.estimatedBytes/1024**3).toFixed(2)} GiB exceeds free disk space ${(Number(free)/1024**3).toFixed(2)} GiB; the old corpus must coexist until replacement succeeds.`);
  await mkdir(dirname(config.out),{recursive:true});
  const stage=await mkdtemp(join(dirname(config.out),`.${basename(config.out)}-building-`));
  let backup=null,committed=false;
  try {
    const datasets=[];
    const splits=[...new Set(summary.groups.map(group=>group.split))].sort((a,b)=>['train','validation','test'].indexOf(a)-['train','validation','test'].indexOf(b));
    for(const [index,split] of splits.entries()) {
      const real=join(stage,'real',split),clean=join(stage,'clean',split),synthetic=join(stage,'synthetic',split);
      console.error(`Building ${split}: importing captures, rendering clean targets, then generating paired synthetic bursts.`);
      const imported=await importRecordings({plan:imports,out:real,split,python:config.python});
      const rendered=await targets({dataset:real,out:clean});
      await run(config.python,[join(ROOT,'experiments/neural-corrector/synthetic.py'),
        '--clean',clean,'--out',synthetic,'--paired-real',real,'--synthetic-ratio',String(config.syntheticRatio),
        '--recipe','w-tube','--frames',String(config.frames),'--severity',String(config.severity),
        '--severity-min',String(config.severityMin),'--seed',String((config.seed+index)>>>0),
        '--max-output-mib',String(config.maxOutputMib),'--node',process.execPath]);
      const manifest=JSON.parse(await readFile(join(synthetic,'manifest.json'),'utf8'));
      datasets.push({split,real:`real/${split}`,clean:`clean/${split}`,synthetic:`synthetic/${split}`,
        importedCaptures:imported.samples??summary.groups.filter(group=>group.split===split).reduce((sum,group)=>sum+group.realCaptures,0),
        cleanCaptures:rendered.samples,syntheticCaptures:manifest.samples.length});
    }
    const result={...summary,created:new Date().toISOString(),datasets};
    await writeFile(join(stage,'build.json'),JSON.stringify(result,null,2)+'\n');
    // Do not replace an output whose identity/ownership changed during this build.
    if(await validateDestination(config.out)!==destinationStamp)throw Error('Dataset output changed during rebuilding; refusing replacement.');
    if(destinationStamp!==null) {
      backup=await mkdtemp(join(dirname(config.out),`.${basename(config.out)}-previous-`));
      await rm(backup,{recursive:true});await rename(config.out,backup);
    }
    try {await rename(stage,config.out);committed=true;}
    catch(error) {if(backup){await rename(backup,config.out);backup=null;}throw error;}
    if(backup)await rm(backup,{recursive:true,force:true});
    return {...result,out:config.out};
  } finally {
    if(!committed)await rm(stage,{recursive:true,force:true});
  }
}

const HELP=`Build 4-pixel W-tube datasets from current and explicitly selected earlier recordings.
  node scripts/rebuild-corrector-data.mjs [--plan]
    [--recordings-root recordings] [--resampled-root recordings-1]
    [--header-root recordings-0] [--out local-data/w-tube] [--python python3]
    [--synthetic-ratio 4] [--frames 8] [--severity 1] [--severity-min .5]
    [--seed 41] [--max-output-mib 1048576]

Native v2 captures keep their 4x4 raster; v1 G16..G128 captures are bicubic-upsampled;
header-based G64 crops retain their original 4x4 raster. G8/G256 and .tmp are skipped.
Each split has real/, clean/ and synthetic/ manifests containing the supported sizes.
Synthetics target 4 captures per real capture, rounded up per target to full coherent
bursts (4 phases x --frames captures). This does not change 50/50 training sampling.
--plan validates source identity/splits and prints counts/disk estimates without writes.
Builds stage a complete replacement; only this script's managed output is replaced.
Raw recordings and checkpoints are never modified. W-tube training follows separately.`;
function args(argv) {
  const names={'recordings-root':'recordingsRoot','resampled-root':'resampledRoot','header-root':'headerRoot',out:'out',python:'python','synthetic-ratio':'syntheticRatio',frames:'frames',severity:'severity','severity-min':'severityMin',seed:'seed','max-output-mib':'maxOutputMib'},options={};let plan=false;
  for(let i=0;i<argv.length;i++) {
    if(argv[i]==='--plan'){plan=true;continue;}
    const key=names[argv[i].replace(/^--/,'')];if(!argv[i].startsWith('--')||!key||i+1===argv.length||argv[i+1].startsWith('--')||key in options)throw Error('Unknown, duplicate or missing rebuild option. Use --help.');
    options[key]=['recordingsRoot','resampledRoot','headerRoot','out','python'].includes(key)?argv[++i]:Number(argv[++i]);
  }
  return {options,plan};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  if(process.argv.slice(2).includes('--help'))console.log(HELP);
  else {const main=async()=>{const parsed=args(process.argv.slice(2));const result=await(parsed.plan?planRebuild:rebuild)(parsed.options);console.log(JSON.stringify(result,null,2));};main().catch(error=>{console.error(`Corrector rebuild failed: ${error.message}`);process.exitCode=1;});}
}
