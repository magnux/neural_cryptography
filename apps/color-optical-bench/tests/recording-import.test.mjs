import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,copyFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {planRecordingImports,importRecordings,upsampleBicubic2x} from '../../../scripts/import-corrector-recordings.mjs';
import {visitRecordingJSON,readRecordingJSON} from '../../../scripts/lib/recording-json.mjs';
import {targets} from '../../../scripts/corrector-dataset.mjs';
import {createSender,bytesToBase64,floatBytes,captureKey} from '../dist/js/dataset/contract.js';
import {capacity,encodeSequence,phaseFrames,parseHeader,frameKey,PALETTE} from '../dist/js/format.js';
const hash=x=>createHash('sha256').update(x).digest('hex');
const crop=(frame,ppc)=>{const g=frame.size??frame.info.size,w=g*ppc,result=new Float32Array(3*w*w);for(let c=0;c<3;c++)for(let y=0;y<w;y++)for(let x=0;x<w;x++)result[c*w*w+y*w+x]=PALETTE[frame.symbols[Math.floor(y/ppc)*g+Math.floor(x/ppc)]][c]/127.5-1;return result;};
async function put(path,value){await mkdir(join(path,'..'),{recursive:true});await writeFile(path,JSON.stringify(value));}
function evidence(frame,origin,timestamp){const info=frame.info??parseHeader(frame.header),ppc=origin==='resampled'?2:4;return {ok:true,info,key:origin==='header-g64'?`${frameKey(info)}/${info.phase}`:captureKey(info),timestamp,pixelsPerCell:ppc,
  inputBase64:bytesToBase64(floatBytes(crop(frame,ppc))),samplesBase64:bytesToBase64(floatBytes(new Float32Array(info.size**2*3+96))),transform:[1,0,0,0,1,0,0,0,1],stability:{durationMs:500,guardMs:500,observations:2},
  ...(origin==='header-g64'?{}:{acquisition:{format:'neural-qr',association:'manual-public-code',phase:'repeated-color-cues',...(origin==='native'?{version:2,geometry:'repeated-binary-size-cues'}:{})}})};}
async function fixtures(root,{multipart=false}={}){
  const nativeRoot=join(root,'recordings'),resampledRoot=join(root,'recordings-1'),headerRoot=join(root,'recordings-0');
  for(const [origin,dir,size,seed] of [['native',nativeRoot,16,11],['resampled',resampledRoot,32,12]]){
    const s=await createSender({size,seed,sessionId:`sender-${origin}`});
    if(origin==='resampled'){s.manifest.version=1;s.manifest.profile.pixelsPerCell=2;for(const row of s.manifest.frames)row.info.pixelsPerCell=2;for(const frame of s.frames){frame.pixelsPerCell=2;frame.info.pixelsPerCell=2;}}
    await put(join(dir,'train',`g${size}`,'sender','one.json'),s.manifest);
    const bundle={format:'neural-qr-captures',version:origin==='native'?2:1,sessionId:`receiver-${origin}`,profile:s.manifest.profile,collectionCode:s.manifest.collectionCode,records:s.frames.map((f,i)=>evidence(f,origin,i+1000)),diagnostics:[]};
    await put(join(dir,'train',`g${size}`,'receiver','one.json'),bundle);
  }
  const profile={size:64,colors:16,phases:4,gridBlock:8,parity:32,holdMs:4000};
  const payload=Uint8Array.from({length:capacity(64,16,profile)*(multipart?2:1)},(_,i)=>(i*37+5)&255),id=new Uint8Array(8).fill(3),frames=phaseFrames(encodeSequence(payload,{...profile,id,permute:true,maxFrames:2}));
  const sender={format:'optical-decoder-sender',version:1,sessionId:'sender-header',profile,payloadId:hash(payload),payloadBase64:bytesToBase64(payload),sequenceId:Buffer.from(id).toString('hex'),frames:frames.map(f=>({key:`${frameKey(f)}/${f.phase}`,info:parseHeader(f.header),headerBase64:bytesToBase64(f.header)}))};
  await put(join(headerRoot,'phone','train','old_decoder-sender-one.json'),sender);
  await put(join(headerRoot,'pc','train','old_decoder-captures-one.json'),{format:'optical-decoder-captures',version:1,sessionId:'receiver-header',records:frames.map((f,i)=>evidence(f,'header-g64',i+1000))});
  await mkdir(join(resampledRoot,'train','g256','receiver'),{recursive:true});await writeFile(join(resampledRoot,'train','g256','receiver','invalid.json'),'not JSON');
  await writeFile(join(nativeRoot,'train','g16','sender','unfinished.json.tmp'),'not JSON');
  await put(join(nativeRoot,'collection_session.json'),{session:'still recording'});
  await put(join(nativeRoot,'train','unknown','receiver','diagnostic.json'),{records:[]});
  return {nativeRoot,resampledRoot,headerRoot,oldPayload:payload,oldFrames:frames};
}

test('streamed record visitation awaits each callback and returns metadata without capture retention',async()=>{
  const root=await mkdtemp(join(tmpdir(),'recording-stream-'));try{
    const path=join(root,'sample.json'),source={records:[{i:1},{i:2}],diagnostics:[{error:'none'}],sessionId:'after-arrays'};await put(path,source);
    const seen=[],result=await visitRecordingJSON(path,{highWaterMark:7,onRecord:async r=>{await Promise.resolve();seen.push(r.i);}});
    assert.deepEqual(seen,[1,2]);assert.deepEqual(result.metadata.records,[]);assert.equal(result.metadata.sessionId,'after-arrays');assert.deepEqual(result.counts,{records:2,diagnostics:1});assert.equal(result.sha256,hash(await readFile(path)));assert.deepEqual(await readRecordingJSON(path),source);
  }finally{await rm(root,{recursive:true,force:true});}
});
test('three recording generations join exact canonical targets without modifying originals',async()=>{
  const root=await mkdtemp(join(tmpdir(),'recording-import-'));try{
    const f=await fixtures(root),plan=await planRecordingImports(f);assert.equal(plan.groups.length,3);assert.equal(plan.skipped.length,4);assert.deepEqual(plan.groups.map(g=>g.targetCaptureCounts),[[4],[4],[4]]);
    const out=join(root,'joined');await importRecordings({plan,out});const manifest=JSON.parse(await readFile(join(out,'manifest.json')));assert.equal(manifest.version,2);assert.equal(manifest.samples.length,12);
    for(const sample of manifest.samples){assert.equal(sample.pixelsPerCell,4);const expected=await readFile(join(out,sample.expected));assert.equal(hash(expected),sample.payloadId);assert.equal(expected.length,sample.size**2);assert.equal((await readFile(join(out,sample.input))).length,3*(4*sample.size)**2*4);}
    const old=manifest.samples.find(s=>s.origin==='header-g64');assert.equal(old.provenance.sourcePayloadId,hash(f.oldPayload));assert.notEqual(old.payloadId,old.provenance.sourcePayloadId);assert.equal(old.provenance.receiverEnvironment.deviceClass,'desktop');assert.deepEqual(new Uint8Array(await readFile(join(out,old.labels))),f.oldFrames[0].symbols);
    const clean=join(root,'clean');await targets({dataset:out,out:clean});const cleanManifest=JSON.parse(await readFile(join(clean,'manifest.json')));assert.deepEqual(new Set(cleanManifest.samples.map(s=>s.origin)),new Set(['native','resampled','header-g64']));assert.equal(cleanManifest.samples.length,12);
    for(const entry of plan.entries){assert.equal(hash(await readFile(entry.path)),entry.sha256);}
  }finally{await rm(root,{recursive:true,force:true});}
});
test('import preflight rejects physical session leakage and unmatched truth',async()=>{
  const root=await mkdtemp(join(tmpdir(),'recording-leak-'));try{
    const f=await fixtures(root);const source=join(f.nativeRoot,'train','g16','sender','one.json'),dest=join(f.nativeRoot,'validation','g16','sender','one.json');await mkdir(join(dest,'..'),{recursive:true});await copyFile(source,dest);await assert.rejects(planRecordingImports(f),/crosses dataset splits/);
    await rm(join(f.nativeRoot,'validation'),{recursive:true});await rm(source);await assert.rejects(planRecordingImports(f),/Missing matching sender/);
  }finally{await rm(root,{recursive:true,force:true});}
});
test('bicubic preserves float evidence and clamps only recording-range overshoot',()=>{
  const x=new Float32Array(3*4*4).fill(1.25),y=upsampleBicubic2x(x,4);assert.equal(y.length,3*8*8);assert.ok(y.every(v=>v===1.25));x[1]=-1.5;x[2]=1.5;const z=upsampleBicubic2x(x,4);assert.ok(z.every(v=>v>=-1.5&&v<=1.5));assert.ok(z.some(v=>Math.abs(v)>1));
});

test('multipart OBM messages become separate logical matrices with phase-consistent targets',async()=>{
  const root=await mkdtemp(join(tmpdir(),'recording-multipart-'));try{
    const f=await fixtures(root,{multipart:true}),plan=await planRecordingImports(f);
    const group=plan.groups.find(g=>g.origin==='header-g64');assert.equal(group.records,8);assert.equal(group.targets,2);assert.deepEqual(group.targetCaptureCounts,[4,4]);
    const out=join(root,'joined');await importRecordings({plan,out});const rows=JSON.parse(await readFile(join(out,'manifest.json'))).samples.filter(s=>s.origin==='header-g64');
    assert.equal(new Set(rows.map(s=>s.payloadId)).size,2);assert.equal(new Set(rows.map(s=>s.provenance.sourcePayloadId)).size,1);
    for(const index of [0,1]){const phases=rows.filter(s=>s.provenance.sourceInfo.index===index);assert.deepEqual(phases.map(s=>s.info.phase),[0,1,2,3]);assert.equal(new Set(phases.map(s=>s.payloadId)).size,1);
      for(const row of phases)assert.deepEqual(new Uint8Array(await readFile(join(out,row.labels))),f.oldFrames[index*4+row.info.phase].symbols);
    }
  }finally{await rm(root,{recursive:true,force:true});}
});

test('canonical imports from all three origins load and generate paired Python phase bursts', {skip:!process.env.PYTHON}, async()=>{
  const root=await mkdtemp(join(tmpdir(),'recording-python-'));try{
    const f=await fixtures(root,{multipart:true}),plan=await planRecordingImports(f),real=join(root,'real'),clean=join(root,'clean'),synthetic=join(root,'synthetic');
    await importRecordings({plan,out:real});await targets({dataset:real,out:clean});
    const repo=fileURLToPath(new URL('../../../',import.meta.url));
    const check=`import sys
from pathlib import Path
sys.path.insert(0, str(Path('experiments/neural-corrector').resolve()))
from data import Dataset
d=Dataset([Path(sys.argv[1])])
assert len(d.samples)==16
assert {s.metadata['origin'] for s in d.samples}=={'native','resampled','header-g64'}
for s in d.samples:
    x,y,m=s.tensors()
    assert x.shape==(3,4*s.shape[0],4*s.shape[0])
`;
    let run=spawnSync(process.env.PYTHON,['-c',check,real],{cwd:repo,encoding:'utf8'});assert.equal(run.status,0,run.stderr);
    run=spawnSync(process.env.PYTHON,['experiments/neural-corrector/synthetic.py','--clean',clean,'--out',synthetic,'--paired-real',real,'--synthetic-ratio','1','--frames','2','--severity','.1','--severity-min','.1','--seed','123','--node',process.execPath],{cwd:repo,encoding:'utf8',maxBuffer:4*1024*1024});assert.equal(run.status,0,run.stderr);
    const rows=JSON.parse(await readFile(join(synthetic,'manifest.json'))).samples;assert.equal(rows.length,32);
    assert.deepEqual(Object.fromEntries(['native','resampled','header-g64'].map(origin=>[origin,rows.filter(r=>r.origin===origin).length])),{native:8,resampled:8,'header-g64':16});
    for(const payloadId of new Set(rows.map(r=>r.payloadId))){const target=rows.filter(r=>r.payloadId===payloadId);assert.deepEqual([0,1,2,3].map(phase=>target.filter(r=>r.info.phase===phase).length),[2,2,2,2]);}
    run=spawnSync(process.env.PYTHON,['-c',check.replace('==16','==32'),synthetic],{cwd:repo,encoding:'utf8'});assert.equal(run.status,0,run.stderr);
  }finally{await rm(root,{recursive:true,force:true});}
});
