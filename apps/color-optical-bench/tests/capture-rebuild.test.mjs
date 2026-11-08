import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,writeFile,readFile,rm,readdir,access,symlink} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {createSender,bytesToBase64,floatBytes,floatsFromBytes} from '../dist/js/dataset/contract.js';
import {CaptureBuffer} from '../dist/js/dataset/capture.js';
import {PALETTE} from '../dist/js/format.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';
import {joinUploads,targets,baselines,denseClassicalScores} from '../../../scripts/corrector-dataset.mjs';
import {BUILD_FORMAT,planRebuild,rebuild} from '../../../scripts/rebuild-corrector-data.mjs';
const base=fileURLToPath(new URL('../../../validation/',import.meta.url));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
async function temp(fn){await mkdir(base,{recursive:true});const root=await mkdtemp(join(base,'multires-pipeline-'));try{await fn(root);}finally{await rm(root,{recursive:true,force:true});}}
async function upload(root,size=16,split='train',seed=71) {
  const sender=await createSender({size,seed,sessionId:`display-${size}-${split}`});sender.manifest.datasetSplit=split;
  sender.manifest.environment={deviceId:'phone',deviceClass:'mobile'};
  const buffer=new CaptureBuffer();let timestamp=1000;
  for(const frame of sender.frames) {
    const samples=new Float32Array(3*size*size+96),input=new Float32Array(3*(4*size)**2);
    for(let cell=0;cell<size*size;cell++)samples.set(PALETTE[frame.symbols[cell]],cell*3);
    for(let i=0;i<32;i++)samples.set(PALETTE[i%16],3*size*size+3*i);
    const record={ok:true,info:frame.info,key:sender.manifest.frames[frame.phase].key,pixelsPerCell:4,timestamp,
      acquisition:{format:'neural-qr',version:2,geometry:'repeated-binary-size-cues',association:'manual-public-code',phase:'repeated-color-cues'},
      transform:[1,0,0,0,1,0,0,0,1],inputBase64:bytesToBase64(floatBytes(input)),samplesBase64:bytesToBase64(floatBytes(samples))};
    buffer.add(record);buffer.add({...record,timestamp:timestamp+600});buffer.add({...record,timestamp:timestamp+1200});timestamp+=2000;
  }
  const receiver={...buffer.export({sessionId:`receiver-${size}-${split}`}),datasetSplit:split,environment:{deviceId:'pc',deviceClass:'desktop'}};
  const directory=join(root,split,`g${size}`);await mkdir(join(directory,'sender'),{recursive:true});await mkdir(join(directory,'receiver'),{recursive:true});
  await writeFile(join(directory,'sender','sender.json'),JSON.stringify(sender.manifest));await writeFile(join(directory,'receiver','captures.json'),JSON.stringify(receiver));
  return {sender,receiver};
}

test('multires clean targets retain ownership and crop-only baselines never use target truth',async()=>temp(async root=>{
  const recordings=join(root,'recordings');await upload(recordings,16);await upload(recordings,128,'validation',72);
  const joined=await joinUploads({'recordings-root':recordings,out:join(root,'real')});
  for(const item of joined.datasets) {
    const out=join(root,`clean-${item.size}`);await targets({dataset:item.dataset,out});
    const path=join(out,'manifest.json'),manifest=JSON.parse(await readFile(path,'utf8'));
    assert.equal(manifest.samples.length,4);
    for(const sample of manifest.samples){
      assert.equal(sample.provenance.senderEnvironment.deviceClass,'mobile');assert.deepEqual(sample.devicePair,{sender:'phone',receiver:'pc'});
      const input=floatsFromBytes(new Uint8Array(await readFile(join(out,sample.input))),3*(4*item.size)**2);
      const labels=new Uint8Array(await readFile(join(out,sample.labels))),expected=new Uint8Array(await readFile(join(out,sample.expected))),mask=new Uint8Array(await readFile(join(out,sample.mask)));
      const layout=gridLayout(item.size,16,8),scores=denseClassicalScores(input,sample.info,4);
      for(const [index,cell]of layout.cells.entries()){
        assert.equal(labels[cell],paletteMap(expected[cell],sample.info));assert.equal(mask[cell],1);
        assert.equal(scores.subarray(index*16,index*16+16).indexOf(Math.max(...scores.subarray(index*16,index*16+16))),expected[cell]);
      }
      const side=4*item.size;assert.equal(input[side*side+4*side+4],Math.fround(PALETTE[labels[item.size+1]][1]/127.5-1));
    }
    const sample=manifest.samples[0],cleanBaseline=sample.hashes.baseline,blank=floatBytes(new Float32Array(3*(4*item.size)**2));
    await writeFile(join(out,sample.input),blank);sample.hashes.input=hash(blank);
    // Baseline refresh must not read the labels or expected bytes at all.
    await rm(join(out,sample.labels));await rm(join(out,sample.expected));
    await writeFile(path,JSON.stringify(manifest));await baselines({dataset:out});
    const refreshed=JSON.parse(await readFile(path,'utf8'));assert.notEqual(refreshed.samples[0].hashes.baseline,cleanBaseline);
  }
}));

function rebuildOptions(root, overrides={}) {return {recordingsRoot:join(root,'recordings'),resampledRoot:join(root,'absent-v1'),headerRoot:join(root,'absent-header'),out:join(root,'w-tube'),...overrides};}

test('rebuild plan is read-only and failures preserve managed outputs and raw recordings',async()=>temp(async root=>{
  const recordings=join(root,'recordings'),out=join(root,'w-tube');await upload(recordings);
  const before=await readFile(join(recordings,'train/g16/receiver/captures.json'));
  const plan=await planRebuild(rebuildOptions(root));
  assert.equal(plan.syntheticCaptureQuantum,32);assert.equal(plan.groups[0].syntheticCaptures,32);assert.equal(plan.groups[0].realCaptures,8);
  await assert.rejects(access(out));
  await mkdir(join(out,'real'),{recursive:true});await writeFile(join(out,'real','keep'),'previous dataset');
  await writeFile(join(out,'build.json'),JSON.stringify({format:BUILD_FORMAT,version:1}));
  await assert.rejects(rebuild(rebuildOptions(root,{python:join(root,'missing-python')})),/ENOENT/);
  assert.equal(await readFile(join(out,'real','keep'),'utf8'),'previous dataset');assert.deepEqual(await readFile(join(recordings,'train/g16/receiver/captures.json')),before);
  assert.deepEqual((await readdir(root)).sort(),['recordings','w-tube']);
  await writeFile(join(out,'private-notes.txt'),'keep');
  await assert.rejects(planRebuild(rebuildOptions(root)),/outside the managed/);
  await assert.rejects(planRebuild(rebuildOptions(root,{out:join(recordings,'derived')})),/separate/);
  await assert.rejects(planRebuild(rebuildOptions(root,{out:join(root,'new'),maxOutputMib:1})),/exceeds/);
}));

test('rebuild accepts small complete pools but rejects a missing phase before creating outputs',async()=>temp(async root=>{
  const recordings=join(root,'recordings'),out=join(root,'w-tube');await upload(recordings);
  const file=join(recordings,'train/g16/receiver/captures.json'),bundle=JSON.parse(await readFile(file,'utf8'));
  bundle.records.splice(bundle.records.findIndex(record=>record.info.phase===0),1);
  await writeFile(file,JSON.stringify(bundle));
  const plan=await planRebuild(rebuildOptions(root));assert.equal(plan.groups[0].someTargetsLackK8,true);
  bundle.records=bundle.records.filter(record=>record.info.phase!==0);await writeFile(file,JSON.stringify(bundle));
  await assert.rejects(planRebuild(rebuildOptions(root)),/missing a phase|every phase|all four phases/);
  await assert.rejects(access(out));
}));

const python=process.env.PYTHON??'python3';
const hasPython=spawnSync(python,['-c','import numpy, torch'],{stdio:'ignore'}).status===0;
test('full rebuild produces paired corpora and replaces only its managed output',{skip:!hasPython},async()=>temp(async root=>{
  const recordings=join(root,'recordings'),out=join(root,'w-tube');await upload(recordings);
  const source=join(recordings,'train/g16/receiver/captures.json'),before=await readFile(source);
  const options=rebuildOptions(root,{python,frames:2,syntheticRatio:1,severity:0,severityMin:0});
  const result=await rebuild(options);
  assert.equal(result.datasets.length,1);assert.equal(result.datasets[0].syntheticCaptures,8);
  for(const [kind,count]of [['real',8],['clean',4],['synthetic',8]]) {
    const data=JSON.parse(await readFile(join(out,kind,'train/manifest.json'),'utf8'));
    assert.equal(data.format,'neural-qr-dataset');assert.equal(data.version,2);assert.equal(data.samples.length,count);
    assert.ok(data.samples.every(sample=>sample.pixelsPerCell===4));
  }
  await writeFile(join(out,'real','old-managed-file'),'old');
  await rebuild(options);
  await assert.rejects(access(join(out,'real','old-managed-file')));
  assert.deepEqual(await readFile(source),before);
  assert.deepEqual((await readdir(root)).sort(),['recordings','w-tube']);
}));

test('a failed build removes staging and preserves the previous managed corpus',async()=>temp(async root=>{
  const recordings=join(root,'recordings'),out=join(root,'w-tube');await upload(recordings);
  await mkdir(join(out,'real'),{recursive:true});await writeFile(join(out,'real/keep'),'previous');
  await writeFile(join(out,'build.json'),JSON.stringify({format:BUILD_FORMAT,version:1}));
  const fake=join(root,'fake-python');
  // The dependency probe succeeds, but synthetic generation exits nonzero.
  await writeFile(fake,'#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nexit 23\n',{mode:0o700});
  await assert.rejects(rebuild(rebuildOptions(root,{python:fake,frames:2,syntheticRatio:1})),/failed \(23\)/);
  assert.equal(await readFile(join(out,'real/keep'),'utf8'),'previous');
  assert.deepEqual((await readdir(root)).sort(),['fake-python','recordings','w-tube']);
}));

test('symlink output and unsupported output markers are never replaced',async()=>temp(async root=>{
  await upload(join(root,'recordings'));await mkdir(join(root,'other'));
  await symlink(join(root,'other'),join(root,'w-tube'));
  await assert.rejects(planRebuild(rebuildOptions(root)),/real directory/);
  await rm(join(root,'w-tube'));await mkdir(join(root,'w-tube'));
  await writeFile(join(root,'w-tube/build.json'),JSON.stringify({format:'other-experiment',version:1}));
  await assert.rejects(planRebuild(rebuildOptions(root)),/Unsupported build marker/);
}));



test('default recording roots live under local-data and plan progress stays on stderr',async()=>temp(async root=>{
  await upload(join(root,'local-data/recordings'));
  const script=fileURLToPath(new URL('../../../scripts/rebuild-corrector-data.mjs',import.meta.url));
  const result=spawnSync(process.execPath,[script,'--plan'],{cwd:root,encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
  const plan=JSON.parse(result.stdout);
  for(const [key,folder]of [['recordingsRoot','recordings'],['resampledRoot','recordings-1'],['headerRoot','recordings-0']])
    assert.equal(plan.config[key],join(root,'local-data',folder));
  assert.equal(plan.config.out,join(root,'local-data/w-tube'));
  assert.match(result.stderr,/Scanning native recordings:/);
  assert.match(result.stderr,/Checking recording files: 2\/2/);
  assert.equal(plan.groups[0].realCaptures,8);
  assert.equal(plan.trainingPipelineReady,true);
  await assert.rejects(access(join(root,'local-data/w-tube')));
}));

test('a mismatched synthetic summary preserves the old corpus',async()=>temp(async root=>{
  const out=join(root,'w-tube');await upload(join(root,'recordings'));
  await mkdir(join(out,'real'),{recursive:true});await writeFile(join(out,'real/keep'),'previous');
  await writeFile(join(out,'build.json'),JSON.stringify({format:BUILD_FORMAT,version:1}));
  const fake=join(root,'fake-python');
  await writeFile(fake,`#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nprintf '{"samples":999}\\n'\n`,{mode:0o700});
  await assert.rejects(rebuild(rebuildOptions(root,{python:fake,frames:2,syntheticRatio:1})),/Synthetic capture count differs from plan/);
  assert.equal(await readFile(join(out,'real/keep'),'utf8'),'previous');
  assert.deepEqual((await readdir(root)).sort(),['fake-python','recordings','w-tube']);
}));
