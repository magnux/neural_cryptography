#!/usr/bin/env node
/** Rebuild current Neural-QR datasets without modifying recordings/checkpoints. */
import {mkdir,mkdtemp,lstat,realpath,readFile,writeFile,readdir,rename,rm,open} from 'node:fs/promises';
import {resolve,dirname,join,sep,basename} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import {planUploadJoin,joinUploads,targets} from './corrector-dataset.mjs';

export const BUILD_FORMAT='neural-qr-multires-build';
export const MANAGED_OUTPUTS=Object.freeze(['real','clean','synthetic','build.json']);
const ROOT=fileURLToPath(new URL('../',import.meta.url));
const DEFAULTS={recordingsRoot:'local-data/recordings',out:'local-data/multires',python:'python3',
  frames:8,burstsPerTarget:8,severity:.8,severityMin:.35,seed:41,maxOutputMib:2**20};
function settings(options={}) {
  for(const key of Object.keys(options))if(!(key in DEFAULTS))throw Error(`Unknown rebuild option: ${key}`);
  const value={...DEFAULTS,...options};
  for(const [key,min,max]of [['frames',2,16],['burstsPerTarget',1,2**20],['seed',0,0xffffffff],['maxOutputMib',0,2**30]])
    if(!Number.isSafeInteger(value[key])||value[key]<min||value[key]>max)throw Error(`Invalid ${key}.`);
  if(!Number.isFinite(value.severity)||value.severity<0||value.severity>1||!Number.isFinite(value.severityMin)||value.severityMin<0||value.severityMin>value.severity)throw Error('Severity must satisfy 0 <= minimum <= maximum <= 1.');
  if(typeof value.python!=='string'||!value.python)throw Error('Python must be a nonempty executable path.');
  value.recordingsRoot=resolve(value.recordingsRoot);value.out=resolve(value.out);return value;
}
async function exists(path){try{return await lstat(path);}catch(error){if(error.code==='ENOENT')return null;throw error;}}
async function validateDestination(out) {
  // Resolve every existing parent to prevent a symlink from redirecting cleanup.
  let parent=dirname(out);while(!await exists(parent))parent=dirname(parent);
  if(await realpath(parent)!==parent)throw Error('Dataset output must not have symlink parents.');
  const info=await exists(out);if(!info)return false;
  if(info.isSymbolicLink()||!info.isDirectory()||await realpath(out)!==out)throw Error('Dataset output must be a real directory.');
  const entries=await readdir(out,{withFileTypes:true});if(!entries.length)return true;
  if(entries.some(entry=>entry.isSymbolicLink()||!MANAGED_OUTPUTS.includes(entry.name)))throw Error('Refusing to replace a directory containing files outside the managed multires dataset.');
  let marker;try{marker=JSON.parse(await readFile(join(out,'build.json'),'utf8'));}catch{throw Error('Existing output is not a managed multires dataset (missing build.json).');}
  if(marker.format!==BUILD_FORMAT||marker.version!==1)throw Error('Unsupported multires build marker; choose a new output directory.');
  return true;
}
export async function planRebuild(options={}) {
  const config=settings(options);
  if(config.out===config.recordingsRoot||config.out.startsWith(config.recordingsRoot+sep)||config.recordingsRoot.startsWith(config.out+sep))throw Error('Dataset output and recordings must be separate, non-nested directories.');
  const replacesExisting=await validateDestination(config.out);
  const plan=await planUploadJoin({'recordings-root':config.recordingsRoot,out:join(config.out,'real')});
  const groups=plan.groups.map(group=>{
    if(group.minCapturesPerPhase<2)throw Error(`${group.key} has a target with fewer than two captures per phase; multires K4..8 training requires two or more captures of each of the four phases.`);
    const captures=4*config.frames*config.burstsPerTarget*group.targets,clean=4*group.targets;
    // Conservative serialized arrays: input48G² + baseline<=64G² + labels/mask/expected3G², plus JSON metadata.
    const estimatedBytes=(group.records+clean+captures)*(115*group.size**2+8192);
    return {split:group.split,size:group.size,targets:group.targets,realCaptures:group.records,syntheticCaptures:captures,estimatedBytes};
  });
  const estimatedBytes=groups.reduce((sum,group)=>sum+group.estimatedBytes,0);
  if(config.maxOutputMib&&estimatedBytes>config.maxOutputMib*1024**2)throw Error(`Estimated output ${(estimatedBytes/1024**3).toFixed(2)} GiB exceeds --max-output-mib ${config.maxOutputMib}.`);
  return {format:BUILD_FORMAT,version:1,config,replacesExisting,groups,estimatedBytes,
    syntheticCapturesPerTarget:4*config.frames*config.burstsPerTarget,
    note:'Full replacement is staged before swapping the managed output. Allow space for the existing output plus this estimate. Raw recordings and checkpoints are never removed.'};
}
function run(command,args) {
  return new Promise((done,reject)=>{
    const child=spawn(command,args,{cwd:ROOT,stdio:'inherit'});
    child.once('error',reject);child.once('exit',(code,signal)=>code===0?done():reject(Error(`${command} failed (${signal??code}).`)));
  });
}
export async function rebuild(options={}) {
  const plan=await planRebuild(options),{config}=plan,parent=dirname(config.out);
  await mkdir(parent,{recursive:true});
  const lockPath=join(parent,`.${basename(config.out)}.rebuild.lock`),lock=await open(lockPath,'wx').catch(error=>{if(error.code==='EEXIST')throw Error(`Rebuild lock exists: ${lockPath}`);throw error;});
  let stage,backup,movedOld=false;
  try {
    await lock.writeFile(JSON.stringify({pid:process.pid,started:new Date().toISOString(),out:config.out})+'\n');
    // Check dependencies before copying captures or generating large arrays.
    await run(config.python,['-c','import numpy, torch; print("NumPy/PyTorch ready for multires generation")']);
    stage=await mkdtemp(join(parent,`.${basename(config.out)}.build-`));
    const joined=await joinUploads({'recordings-root':config.recordingsRoot,out:join(stage,'real')});
    const datasets=[];
    for(const [index,group]of joined.datasets.entries()) {
      const suffix=join(group.split,`g${group.size}`),clean=join(stage,'clean',suffix),synthetic=join(stage,'synthetic',suffix);
      const target=await targets({dataset:group.dataset,out:clean});
      if(target.synthesizedMissingPhases)throw Error(`Missing recorded phases in ${suffix}; each real target needs all four phases for phase-grouped training.`);
      await run(config.python,[join(ROOT,'experiments/neural-corrector/synthetic.py'),'--clean',clean,'--out',synthetic,
        '--phase-grouped','--bursts-per-target',String(config.burstsPerTarget),'--frames',String(config.frames),
        '--severity',String(config.severity),'--severity-min',String(config.severityMin),'--capture-min-scale','.5',
        '--seed',String((config.seed+index)>>>0),'--max-output-mib',String(config.maxOutputMib),'--node',process.execPath]);
      datasets.push({split:group.split,size:group.size,realCaptures:group.samples,targets:target.targets,
        syntheticCaptures:target.targets*4*config.frames*config.burstsPerTarget});
    }
    const report={...plan,created:new Date().toISOString(),datasets};
    await writeFile(join(stage,'build.json'),JSON.stringify(report,null,2)+'\n');
    // Revalidate immediately before replacing; unknown user files are never deleted.
    const existsNow=await validateDestination(config.out);
    backup=stage+'.previous';
    if(existsNow){await rename(config.out,backup);movedOld=true;}
    try{await rename(stage,config.out);}catch(error){if(movedOld){await rename(backup,config.out);movedOld=false;}throw error;}
    if(movedOld){await rm(backup,{recursive:true});movedOld=false;}
    return report;
  } finally {
    // On ordinary filesystems rename removes this path; also clear any residual
    // staging entries on overlay/synchronized workspaces. This path is ours only.
    if(stage)await rm(stage,{recursive:true,force:true});
    // If restoring a failed atomic swap itself failed, preserve the previous data.
    await lock.close();await rm(lockPath,{force:true});
  }
}

const HELP=`Rebuild size-separated Neural-QR real/clean/synthetic datasets from server uploads.
  node scripts/rebuild-corrector-data.mjs [--recordings-root local-data/recordings]
    [--out local-data/multires] [--python python3] [--plan]
    [--bursts-per-target 8] [--frames 8] [--severity .8] [--severity-min .35]
    [--seed 41] [--max-output-mib 1048576]

Default: 8 coherent bursts x 8 captures/phase x 4 phases = 256 synthetics/target.
--plan validates recordings and prints counts/estimated disk usage without writing.
A successful rebuild replaces only a previously marked output directory; generation
failures preserve the prior output. Raw recordings and checkpoints remain untouched.`;
function args(argv) {
  const names={'recordings-root':'recordingsRoot',out:'out',python:'python','bursts-per-target':'burstsPerTarget',frames:'frames',severity:'severity','severity-min':'severityMin',seed:'seed','max-output-mib':'maxOutputMib'},options={};let plan=false;
  for(let i=0;i<argv.length;i++) {
    if(argv[i]==='--plan'){plan=true;continue;}
    const key=names[argv[i].replace(/^--/,'')];if(!argv[i].startsWith('--')||!key||i+1===argv.length||argv[i+1].startsWith('--')||key in options)throw Error('Unknown, duplicate or missing rebuild option. Use --help.');
    options[key]=['recordingsRoot','out','python'].includes(key)?argv[++i]:Number(argv[++i]);
  }
  return {options,plan};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  if(process.argv.slice(2).includes('--help'))console.log(HELP);
  else {const main=async()=>{const parsed=args(process.argv.slice(2));const result=await(parsed.plan?planRebuild:rebuild)(parsed.options);console.log(JSON.stringify(result,null,2));};main().catch(error=>{console.error(`Corrector rebuild failed: ${error.message}`);process.exitCode=1;});}
}
