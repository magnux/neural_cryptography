import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,rm,readFile,writeFile,symlink,rename,access,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {createSender} from '../dist/js/dataset/contract.js';
import {CaptureBuffer,captureFrame} from '../dist/js/dataset/capture.js';
import {renderFrame} from '../dist/js/format.js';
import {planRebuild,rebuild,MANAGED_OUTPUTS} from '../../../scripts/rebuild-decoder-data.mjs';

const repo=fileURLToPath(new URL('../../../',import.meta.url)),base=join(repo,'validation');
const python=process.env.OPTICAL_DECODER_TEST_PYTHON??'python3';
const probe=spawnSync(python,['-c','import numpy, torch'],{encoding:'utf8'}),hasPython=probe.status===0;
if(process.env.OPTICAL_DECODER_TEST_PYTHON&&!hasPython)throw Error(`OPTICAL_DECODER_TEST_PYTHON is unusable: ${probe.error?.message??probe.stderr}`);
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const fixtures=[];
for(const [seed,session,receiver] of [[701,'sender-train','receiver-train'],[702,'sender-val','receiver-val']]) {
  const sender=await createSender({seed,sessionId:session,phases:4,colors:16,size:64});
  const buffer=new CaptureBuffer();
  for(const [phase,frame] of sender.frames.entries()) {
    const timestamp=1000+phase*3000,record=captureFrame(renderFrame(frame,1024),{timestamp});
    buffer.add(record);buffer.add({...record,timestamp:timestamp+600});buffer.add({...record,timestamp:timestamp+1200});
  }
  fixtures.push({sender:sender.manifest,capture:buffer.export({sessionId:receiver})});
}
const copy=x=>JSON.parse(JSON.stringify(x));
async function fixture(fn) {
  await mkdir(base,{recursive:true});const root=await mkdtemp(join(base,'rebuild,fixture-'));
  try {
    for(const device of ['phone','pc'])for(const split of ['train','val'])await mkdir(join(root,device,split),{recursive:true});
    const paths={trainSender:join(root,'pc/train/arbitrary,sender.json'),trainCapture:join(root,'phone/train/arbitrary,capture.json'),
      valSender:join(root,'phone/val/z_sender.json'),valCapture:join(root,'pc/val/z_capture.json')};
    await writeFile(paths.trainSender,JSON.stringify(fixtures[0].sender));await writeFile(paths.trainCapture,JSON.stringify(fixtures[0].capture));
    await writeFile(paths.valSender,JSON.stringify(fixtures[1].sender));await writeFile(paths.valCapture,JSON.stringify(fixtures[1].capture));
    await mkdir(join(root,'real-train'));await writeFile(join(root,'real-train/previous.txt'),'previous dataset');
    await fn(root,paths);
  } finally {await rm(root,{recursive:true,force:true});}
}
async function previous(root){assert.equal(await readFile(join(root,'real-train/previous.txt'),'utf8'),'previous dataset');}

test('rebuild discovers both directions by folder and full headers, independent of filenames and commas',async()=>fixture(async(root,paths)=>{
  await mkdir(join(root,'phone/train/nested'));
  await rename(paths.trainCapture,join(root,'phone/train/nested/misleading-prefix.json'));
  const plan=await planRebuild({dataRoot:root});
  assert.equal(plan.entries.length,4);assert.equal(plan.summary.train.records,8);assert.equal(plan.summary.validation.records,8);
  assert.equal(plan.summary.train.bursts,4);assert.equal(plan.summary.validation.bursts,4);
  assert.equal(plan.summary.train.syntheticCaptures,1536);assert.equal(plan.summary.validation.syntheticCaptures,192);
  assert.deepEqual(plan.existingOutputs,['real-train']);await previous(root);
}));

test('incomplete phase coverage fails before replacing datasets',async()=>fixture(async(root,paths)=>{
  const missing=copy(fixtures[0].capture);missing.records=missing.records.filter(row=>row.info.phase!==3);
  await writeFile(paths.trainCapture,JSON.stringify(missing));
  await assert.rejects(planRebuild({dataRoot:root}),/all four phases/);await previous(root);
}));

test('rebuild rejects unmatched headers and wrong-device pairs before replacing datasets',async()=>fixture(async(root,paths)=>{
  const capture=copy(fixtures[0].capture);capture.records[0].info.id='0000000000000000';
  // Keep the record's own key consistent so this reaches sender matching.
  capture.records[0].key=capture.records[0].key.replace(fixtures[0].sender.sequenceId,'0000000000000000');
  await writeFile(paths.trainCapture,JSON.stringify(capture));
  await assert.rejects(planRebuild({dataRoot:root}),/Unmatched capture/);await previous(root);
  await writeFile(paths.trainCapture,JSON.stringify(fixtures[0].capture));
  await rename(paths.trainCapture,join(root,'pc/train/wrong-side.json'));
  await assert.rejects(planRebuild({dataRoot:root}),/same device/);await previous(root);
}));

test('rebuild rejects duplicate captures, malformed tensor data, and unassigned files',async()=>fixture(async(root,paths)=>{
  const duplicate=join(root,'phone/train/duplicate.json');
  await writeFile(duplicate,await readFile(paths.trainCapture));
  await assert.rejects(planRebuild({dataRoot:root}),/Duplicate receiver capture/);await rm(duplicate);
  const broken=copy(fixtures[0].capture);broken.records[0].inputBase64='AAAA';
  await writeFile(paths.trainCapture,JSON.stringify(broken));
  await assert.rejects(planRebuild({dataRoot:root}),/length/);
  await writeFile(paths.trainCapture,JSON.stringify(fixtures[0].capture));
  await writeFile(join(root,'phone/unassigned.json'),await readFile(paths.trainCapture));
  await assert.rejects(planRebuild({dataRoot:root}),/Move recording/);await previous(root);
}));

test('rebuild rejects physical-session, payload and paired-header leakage across splits',async()=>fixture(async(root,paths)=>{
  const sameSession=copy(fixtures[1].sender);sameSession.sessionId=fixtures[0].sender.sessionId;
  await writeFile(paths.valSender,JSON.stringify(sameSession));
  await assert.rejects(planRebuild({dataRoot:root}),/Physical session crosses/);
  const samePayload=copy(fixtures[0].sender);samePayload.sessionId='separate-validation-session';
  await writeFile(paths.valSender,JSON.stringify(samePayload));
  await assert.rejects(planRebuild({dataRoot:root}),/Payload crosses/);
  await writeFile(paths.valSender,JSON.stringify(fixtures[1].sender));
  const crossed=copy(fixtures[0].capture);crossed.sessionId='separate-validation-receiver';
  await writeFile(paths.valCapture,JSON.stringify(crossed));
  await assert.rejects(planRebuild({dataRoot:root}),/pair crosses/);await previous(root);
}));

test('rebuild rejects conflicting receiver directions and unmatched senders',async()=>fixture(async(root,paths)=>{
  const sameSession=copy(fixtures[1].capture);sameSession.sessionId=fixtures[0].capture.sessionId;
  // Put the new receiver file in train first so direction conflict is checked
  // without the earlier session-split ownership failure.
  await writeFile(join(root,'pc/train/conflict.json'),JSON.stringify(sameSession));
  await assert.rejects(planRebuild({dataRoot:root}),/both directions/);
  await rm(join(root,'pc/train/conflict.json'));
  const unused=await createSender({seed:703,sessionId:'sender-unused',phases:1,colors:4});
  await writeFile(join(root,'pc/train/unused.json'),JSON.stringify(unused.manifest));
  await assert.rejects(planRebuild({dataRoot:root}),/no matching captures/);await previous(root);
}));

test('rebuild refuses symlink inputs and outputs without touching targets',async()=>fixture(async(root,paths)=>{
  await symlink(paths.trainSender,join(root,'pc/train/link.json'));
  await assert.rejects(planRebuild({dataRoot:root}),/symlink/);await rm(join(root,'pc/train/link.json'));
  await symlink(join(root,'phone'),join(root,'target-synthetic-train'));
  await assert.rejects(planRebuild({dataRoot:root}),/symlink output/);
  assert.equal(JSON.parse(await readFile(paths.trainCapture,'utf8')).sessionId,'receiver-train');await previous(root);
}));

test('invalid config, output budget and missing Python fail before deletion',async()=>fixture(async(root)=>{
  await assert.rejects(planRebuild({dataRoot:root,frames:17}),/frames/);
  await assert.rejects(planRebuild({dataRoot:root,severityMin:.95}),/severity/);
  await assert.rejects(planRebuild({dataRoot:root,maxOutputMib:1}),/estimate exceeds/);
  await assert.rejects(rebuild({dataRoot:root,python:join(root,'missing-python')},{log:()=>{}}),/ENOENT/);
  await previous(root);
}));

test('dry run validates dependencies but does not replace data or raw exports',{skip:!hasPython},async()=>fixture(async(root,paths)=>{
  const before=await readFile(paths.trainCapture),result=await rebuild({dataRoot:root,python,dryRun:true},{log:()=>{}});
  assert.equal(result.dryRun,true);await previous(root);
  assert.deepEqual(await readFile(paths.trainCapture),before);
  await assert.rejects(access(join(root,'.decoder-rebuild.lock')));
}));

test('small full rebuild regenerates joined bursts, exact targets, groups and separate paired synthetics',{skip:!hasPython},async()=>fixture(async(root,paths)=>{
  const rawBefore=new Map(await Promise.all(Object.values(paths).map(async path=>[path,sha(await readFile(path))])));
  await mkdir(join(root,'checkpoints'));await writeFile(join(root,'checkpoints/keep.pt'),'checkpoint');
  await mkdir(join(root,'old-custom-dataset'));await writeFile(join(root,'old-custom-dataset/keep'),'custom');
  const result=await rebuild({dataRoot:root,python,frames:2,burstsPerTarget:1,validationBurstsPerTarget:1},{log:()=>{}});
  assert.equal(result.dryRun,false);assert.equal(result.outputs.length,MANAGED_OUTPUTS.length);
  assert.ok(!(await readdir(join(root,'real-train'))).includes('previous.txt'),'previous dataset files must be removed');
  await assert.rejects(readFile(join(root,'real-train/previous.txt')));
  const groups=JSON.parse(await readFile(join(root,'session-groups.json'),'utf8'));
  assert.deepEqual(groups,{format:'optical-decoder-session-groups',version:1,
    sessions:{'receiver-train':'pc-to-phone','receiver-val':'phone-to-pc'}});
  for(const split of ['train','validation']) {
    const read=async name=>JSON.parse(await readFile(join(root,name,'manifest.json'),'utf8'));
    const real=await read(`real-${split}`),clean=await read(`target-clean-${split}`),synthetic=await read(`target-synthetic-${split}`);
    assert.equal(real.samples.length,8);assert.equal(clean.samples.length,4);assert.equal(synthetic.samples.length,8);
    assert.equal(synthetic.generator.phaseGrouped,true);
    assert.equal(new Set(synthetic.samples.map(s=>s.synthetic.phaseGroupId)).size,1);
    assert.equal(new Set(real.samples.map(s=>s.burstId)).size,4);
    assert.equal(new Set(synthetic.samples.map(s=>s.burstId)).size,4);
    for(const sample of [...real.samples,...clean.samples,...synthetic.samples])assert.equal(sample.split,split);
    for(const sample of synthetic.samples) {
      assert.equal(sample.sessionId,real.samples[0].sessionId);
      assert.equal(sample.payloadId,real.samples[0].payloadId);
      const samePhase=real.samples.find(row=>row.info.phase===sample.info.phase);
      for(const field of ['labels','mask','expected'])assert.equal(sample.hashes[field],samePhase.hashes[field]);
    }
  }
  for(const [path,hash] of rawBefore)assert.equal(sha(await readFile(path)),hash);
  assert.equal(await readFile(join(root,'checkpoints/keep.pt'),'utf8'),'checkpoint');
  assert.equal(await readFile(join(root,'old-custom-dataset/keep'),'utf8'),'custom');
  await assert.rejects(access(join(root,'.decoder-rebuild.lock')));
}));
