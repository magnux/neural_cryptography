import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {readRecordingJSON} from '../../../scripts/lib/recording-json.mjs';

const base=fileURLToPath(new URL('../../../validation/',import.meta.url));
async function withFile(content,fn){
  await mkdir(base,{recursive:true});const dir=await mkdtemp(join(base,'stream-recording-'));
  try{const path=join(dir,'recording.json');await writeFile(path,content);await fn(path);}
  finally{await rm(dir,{recursive:true,force:true});}
}

test('stream reader preserves current recording values across tiny UTF-8/token boundaries',async()=>{
  const expected={format:'neural-qr-captures',version:2,notes:'á 😀 \\" ] } , \n',
    profile:{size:16,colors:16,phases:4,enabled:true,optional:null},
    records:[{key:'a/0',inputBase64:'AQID'.repeat(400),nested:[false,null,3.25,-1e-7,{},[]]},
      {key:'a/1',camera:{label:'“front”'},escaped:'\\\\\\"x'}],
    diagnostics:[{error:'{quoted: "]"}',timestamp:0}],after:[1,{deep:['x']}],empty:[]};
  await withFile(JSON.stringify(expected),async path=>{
    for(const highWaterMark of [1,2,3,7,64,65536])
      assert.deepEqual(await readRecordingJSON(path,{highWaterMark}),expected);
  });
});

test('streamed arrays may exceed the token bound while each capture remains bounded',async()=>{
  const expected={records:Array.from({length:100},(_,i)=>({id:i,data:'A'.repeat(120)})),diagnostics:[]};
  await withFile(JSON.stringify(expected),async path=>{
    assert.deepEqual(await readRecordingJSON(path,{maxTokenBytes:200,highWaterMark:31}),expected);
    await assert.rejects(readRecordingJSON(path,{maxTokenBytes:64}),/per-value byte limit/);
    await assert.rejects(readRecordingJSON(path,{maxArrayItems:99}),/item limit/);
    await assert.rejects(readRecordingJSON(path,{maxBytes:100}),/byte limit/);
  });
});

test('reader rejects malformed, truncated and trailing JSON instead of accepting a partial recording',async()=>{
  const invalid=['', '[1]', 'null', '{', '{"records":[', '{"records":[{}', '{"records":[{},]}',
    '{"records":[,{}]}', '{"records":[],,"x":0}', '{"records":[],}', '{"x":}',
    '{"x":01}', '{"x":+1}', '{"x":NaN}', '{"x":Infinity}', '{"x":tru}',
    '{"x":true false}', '{"x":"bad\\q"}', '{"x":"bad\nline"}', '{"x":[1,]}',
    '{"x":{"a":1]}', '{"x":"unclosed}', '{} false', '{}{}', '\ufeff{}'];
  for(const source of invalid)await withFile(source,async path=>{
    await assert.rejects(readRecordingJSON(path,{highWaterMark:2}),undefined,source);
  });
});

test('top-level duplicate keys reject and special property names remain ordinary own fields',async()=>{
  await withFile('{"records":[],"records":[]}',async path=>{
    await assert.rejects(readRecordingJSON(path),/Duplicate/);
  });
  await withFile('{"__proto__":{"flag":true},"constructor":3,"records":[]}',async path=>{
    const result=await readRecordingJSON(path,{highWaterMark:5});
    assert.equal(Object.getPrototypeOf(result),Object.prototype);
    assert.equal(Object.hasOwn(result,'__proto__'),true);
    assert.deepEqual(result.__proto__,{flag:true});assert.equal(result.constructor,3);
  });
});

test('UTF-8, nesting and invalid parser bounds fail explicitly',async()=>{
  await withFile(Buffer.from([123,34,120,34,58,34,0xc3,34,125]),async path=>{
    await assert.rejects(readRecordingJSON(path,{highWaterMark:1}),/encoded data/);
  });
  await withFile('{"records":[{"nested":[[[1]]]}]}',async path=>{
    await assert.rejects(readRecordingJSON(path,{maxDepth:2}),/nesting/);
    for(const options of [{maxBytes:0},{maxTokenBytes:-1},{maxDepth:1.5},{maxArrayItems:NaN},{highWaterMark:2**21}])
      await assert.rejects(readRecordingJSON(path,options),/limit|chunks/);
  });
});
