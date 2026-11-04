import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,readFile,writeFile,rm,access,symlink} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createSender} from '../dist/js/dataset/contract.js';
import {captureFrame,CaptureBuffer} from '../dist/js/dataset/capture.js';
import {renderFrame} from '../dist/js/format.js';
import {joinUploads,joinCaptures,targets} from '../../../scripts/corrector-dataset.mjs';
import {planRebuild} from '../../../scripts/rebuild-corrector-data.mjs';

const base=fileURLToPath(new URL('../../../validation/',import.meta.url));
async function temp(fn){await mkdir(base,{recursive:true});const root=await mkdtemp(join(base,'upload-join-'));try{await fn(root);}finally{await rm(root,{recursive:true,force:true});}}
const copied=x=>JSON.parse(JSON.stringify(x));
async function pair(size,split,seed=19) {
  const sender=await createSender({size,phases:4,colors:16,seed,sessionId:`display-${size}-${split}`});
  sender.manifest.datasetSplit=split;
  sender.manifest.environment={deviceId:'screen-device',deviceLabel:'Laptop',deviceClass:'desktop',browser:{userAgent:'test browser'}};
  const b=new CaptureBuffer();let time=1000;
  for(const frame of sender.frames){
    const record=captureFrame(renderFrame(frame,1024),{pixelsPerCell:2,timestamp:time,camera:{label:'Front camera',deviceId:'origin-scoped-camera',facingMode:'user'}});
    assert.equal(record.ok,true);b.add(record);b.add({...record,timestamp:time+600});time+=2000;
  }
  const receiver={...b.export({sessionId:`receiver-${size}-${split}`}),datasetSplit:split,
    environment:{deviceId:'camera-device',deviceLabel:'Phone',deviceClass:'mobile',browser:{userAgent:'test mobile browser'}},camera:{label:'Front camera'}};
  return {sender:sender.manifest,receiver};
}
async function upload(root,{sender,receiver},split=sender.datasetSplit) {
  const folder=join(root,split,`g${sender.profile.size}`),a=join(folder,'sender','sender.json'),b=join(folder,'receiver','capture.json');
  await mkdir(join(folder,'sender'),{recursive:true});await mkdir(join(folder,'receiver'),{recursive:true});
  await writeFile(a,JSON.stringify(sender));await writeFile(b,JSON.stringify(receiver));return {a,b};
}

test('uploaded recordings join by size and split with exact 2x2 crops and both endpoint metadata',async()=>temp(async root=>{
  const recordings=join(root,'recordings'),out=join(root,'joined'),small=await pair(8,'train'),medium=await pair(16,'validation');
  const first=await upload(recordings,small);await upload(recordings,medium);
  const before=await readFile(first.b),result=await joinUploads({'recordings-root':recordings,out});
  assert.equal(result.datasets.length,2);
  for(const {size,split}of result.datasets){
    const directory=join(out,split,`g${size}`),manifest=JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));
    assert.equal(manifest.samples.length,4);assert.equal(new Set(manifest.samples.map(s=>s.burstId)).size,4);
    for(const sample of manifest.samples){
      assert.equal(sample.pixelsPerCell,2);assert.deepEqual(sample.inputShape,[3,size*2,size*2]);
      assert.equal((await readFile(join(directory,sample.input))).length,3*(size*2)**2*4);
      assert.deepEqual(sample.devicePair,{sender:'screen-device',receiver:'camera-device'});
      assert.equal(sample.provenance.senderEnvironment.deviceLabel,'Laptop');
      assert.equal(sample.provenance.receiverEnvironment.deviceClass,'mobile');
      assert.equal(sample.provenance.camera.label,'Front camera');
      assert.equal(sample.provenance.receiverDatasetSplit,split);
    }
  }
  assert.deepEqual(await readFile(first.b),before,'raw uploads remain untouched');
  await assert.rejects(targets({dataset:join(out,'train','g8'),out:join(root,'targets')}),/Multires 2x2 captures/);
  await assert.rejects(access(join(root,'targets')));
  await assert.rejects(joinUploads({'recordings-root':recordings,out:join(recordings,'joined')}),/separate directories/);
}));

test('upload joining rejects cross-split sessions and missing counterparts before output creation',async()=>temp(async root=>{
  const recordings=join(root,'recordings'),out=join(root,'joined'),a=await pair(8,'train'),b=await pair(16,'validation');
  b.receiver.sessionId=a.receiver.sessionId;
  await upload(recordings,a);const second=await upload(recordings,b);
  await assert.rejects(joinUploads({'recordings-root':recordings,out}),/Physical session crosses dataset splits/);
  await assert.rejects(access(out));
  b.receiver.sessionId='receiver-validation';await writeFile(second.b,JSON.stringify(b.receiver));await rm(second.a);
  await assert.rejects(joinUploads({'recordings-root':recordings,out}),/No matching uploaded sender/);
  await assert.rejects(access(out));
}));

test('upload joining checks split declarations, grid folders, and symlink inputs',async()=>temp(async root=>{
  const recordings=join(root,'recordings'),out=join(root,'joined'),p=await pair(8,'train'),paths=await upload(recordings,p);
  const wrong=copied(p.receiver);wrong.datasetSplit='test';await writeFile(paths.b,JSON.stringify(wrong));
  await assert.rejects(joinUploads({'recordings-root':recordings,out}),/split disagrees/);
  await writeFile(paths.b,JSON.stringify(p.receiver));
  await symlink(paths.b,join(recordings,'train','g8','receiver','link.json'));
  await assert.rejects(joinUploads({'recordings-root':recordings,out}),/symlinks/);
  await rm(join(recordings,'train','g8','receiver','link.json'));
  await assert.rejects(joinCaptures({senders:paths.a,captures:paths.b,out,split:'validation'}),/split disagrees/);
}));

test('established synthetic rebuild rejects 2x2 captures before replacing existing datasets',async()=>temp(async root=>{
  for(const device of ['phone','pc'])for(const split of ['train','val'])await mkdir(join(root,device,split),{recursive:true});
  const p=await pair(64,'train');await writeFile(join(root,'pc/train/sender.json'),JSON.stringify(p.sender));
  await writeFile(join(root,'phone/train/capture.json'),JSON.stringify(p.receiver));
  await mkdir(join(root,'real-train'));await writeFile(join(root,'real-train/keep'),'untouched');
  await assert.rejects(planRebuild({dataRoot:root}),/4x4-pixel cells/);
  assert.equal(await readFile(join(root,'real-train/keep'),'utf8'),'untouched');
}));
