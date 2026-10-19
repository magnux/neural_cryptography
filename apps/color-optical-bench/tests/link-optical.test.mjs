import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {Worker} from 'node:worker_threads';
import {capacity,renderFrame} from '../dist/js/format.js';
import {TemporalDecoder} from '../dist/js/temporal.js';
import {CONTROL_PROFILE,DATA_PROFILE,normalizeProfile,getChunkBytes,getControlCapacity,packetToFrames} from '../dist/js/link/optical.js';

if(!globalThis.crypto)globalThis.crypto=webcrypto;
const packet=(seed=0,length=88)=>Uint8Array.from({length},(_,i)=>(i*53+seed)&255);

test('link packet capacities include wire overhead and fit the largest control',()=>{
  assert.equal(getControlCapacity(),150);
  assert.equal(getChunkBytes(),5428);
  assert.equal(getChunkBytes(CONTROL_PROFILE),106);
  assert.equal(getControlCapacity({...CONTROL_PROFILE,colors:16}),300);
  assert.equal(capacity(DATA_PROFILE.size,DATA_PROFILE.colors,DATA_PROFILE),5472);
  assert.equal(normalizeProfile({phases:1}).permute,false);
  assert.throws(()=>getControlCapacity({...CONTROL_PROFILE,size:16,colors:16}),/control packet/);
  assert.throws(()=>getChunkBytes({...CONTROL_PROFILE,size:8}),/small|cannot fit/);
  for(const profile of [{phases:3},{holdMs:100},{phases:1,permute:true},{colors:32},null]){
    assert.throws(()=>normalizeProfile(profile));
  }
});

test('one link packet becomes one raw OBM5 frame with independent palette observations',async()=>{
  const bytes=packet(1,getChunkBytes()+44),frames=await packetToFrames(bytes);
  assert.equal(frames.length,8);
  for(let i=0;i<frames.length;i++){
    const f=frames[i];assert.equal(f.phase,i);assert.equal(f.count,1);assert.equal(f.index,0);
    assert.equal(f.content.type,'raw');assert.equal(f.total,bytes.length);assert.deepEqual(f.chunk,bytes);
    assert.equal(f.id,frames[0].id);
  }
  await assert.rejects(packetToFrames(packet(0,5473)),/holds 5472/);
  await assert.rejects(packetToFrames(new Uint8Array()),/must contain bytes/);
});

test('optical sequence IDs follow packet bytes and remain stable through retransmission',async()=>{
  const original=packet(),mutable=original.slice(),pending=packetToFrames(mutable,CONTROL_PROFILE);
  mutable[0]^=1;
  const first=(await pending)[0],repeat=(await packetToFrames(original,CONTROL_PROFILE))[0],changed=(await packetToFrames(mutable,CONTROL_PROFILE))[0];
  assert.equal(first.id,repeat.id);assert.deepEqual(first.header,repeat.header);
  assert.deepEqual(first.chunk,original);assert.notEqual(first.id,changed.id);
  assert.notEqual(first.id,'0000000000000000');
});

test('streaming receiver decodes consecutive independent packets; ordinary receiver stays locked',async()=>{
  const a=(await packetToFrames(packet(1),CONTROL_PROFILE))[0],b=(await packetToFrames(packet(2),CONTROL_PROFILE))[0];
  const options={filter:'none',minPhases:1,preview:false};
  const locked=new TemporalDecoder();assert.equal(locked.decode(renderFrame(a,512),options).ok,true);
  const refused=locked.decode(renderFrame(b,512),options);
  assert.equal(refused.ok,false);assert.match(refused.error,/Different sequence/);
  const stream=new TemporalDecoder();
  for(const frame of [a,a,b,b,a]){
    const result=stream.decode(renderFrame(frame,512),{...options,streaming:true});
    assert.equal(result.ok,true,result.error);assert.deepEqual(result.chunk,frame.chunk);
    assert.equal(stream.groups.size,1);
  }
});

test('streaming retains repeats and phases of one packet but never mixes different packet evidence',async()=>{
  const profile={...CONTROL_PROFILE,colors:16,phases:2,permute:true};
  const a=await packetToFrames(packet(3),profile),b=await packetToFrames(packet(4),profile),decoder=new TemporalDecoder();
  const options={streaming:true,filter:'mean',window:2,minPhases:2,preview:false};
  let result;
  for(const frame of [a[0],a[0],a[1]])result=decoder.decode(renderFrame(frame,512),options);
  assert.equal(result.ok,false);assert.deepEqual(result.stats.temporal.counts,[2,1]);
  result=decoder.decode(renderFrame(b[1],512),options);
  assert.equal(result.ok,false);assert.deepEqual(result.stats.temporal.counts,[0,1]);
  for(const frame of [b[1],b[0],b[0]])result=decoder.decode(renderFrame(frame,512),options);
  assert.equal(result.ok,true,result.error);assert.deepEqual(result.chunk,b[0].chunk);
  assert.deepEqual(result.stats.temporal.counts,[2,2]);assert.equal(decoder.groups.size,1);
});

test('invalid headers do not reset streaming evidence and new valid packets restart settling',async()=>{
  const [a]=(await packetToFrames(packet(5),CONTROL_PROFILE)),[b]=(await packetToFrames(packet(6),CONTROL_PROFILE));
  let now=0;const decoder=new TemporalDecoder({clock:()=>now}),options={streaming:true,filter:'mean',window:2,minPhases:1,settleMs:100,preview:false};
  assert.equal(decoder.decode(renderFrame(a,512),options).stage,'settle');now=101;
  let result=decoder.decode(renderFrame(a,512),options);assert.deepEqual(result.stats.temporal.counts,[1]);
  const invalid={...b,header:b.header.slice()};invalid.header[60]^=1;
  result=decoder.decode(renderFrame(invalid,512),options);assert.equal(result.info,undefined);
  result=decoder.decode(renderFrame(a,512),options);assert.equal(result.ok,true,result.error);
  assert.equal(decoder.decode(renderFrame(b,512),options).stage,'settle');assert.equal(decoder.bytes,0);
  now=202;result=decoder.decode(renderFrame(b,512),options);assert.deepEqual(result.stats.temporal.counts,[1]);
});

test('actual decoder worker forwards streaming options for back-to-back link packets',async()=>{
  const worker=new Worker(new URL('./worker-harness.mjs',import.meta.url));let id=0;
  const run=frame=>new Promise((resolve,reject)=>{
    const success=data=>{worker.off('error',failure);resolve(data.result);};
    const failure=error=>{worker.off('message',success);reject(error);};
    worker.once('message',success);worker.once('error',failure);
    worker.postMessage({id:++id,image:renderFrame(frame,512),options:{streaming:true,filter:'none',minPhases:1,preview:false}});
  });
  try{
    for(const seed of [7,8,7]){
      const frame=(await packetToFrames(packet(seed),CONTROL_PROFILE))[0],result=await run(frame);
      assert.equal(result.ok,true,result.error);assert.deepEqual(result.chunk,frame.chunk);
    }
  }finally{await worker.terminate();}
});
