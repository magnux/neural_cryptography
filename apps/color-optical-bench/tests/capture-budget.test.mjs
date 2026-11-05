import test from 'node:test';
import assert from 'node:assert/strict';
import {MAX_CAPTURE_BYTES,DEFAULT_CAPTURE_BYTES,CAPTURE_BUDGET_MIB,MAX_RECORDS,recommendedPhaseMaximum} from '../dist/js/dataset/contract.js';
import {CaptureBuffer} from '../dist/js/dataset/capture.js';
import {recordingBlob} from '../dist/js/dataset/recording-json.js';

const MiB=1024*1024;
test('budgets fit all four phases at requested large-grid pool sizes',()=>{
  assert.equal(MAX_CAPTURE_BYTES,800*MiB);assert.equal(DEFAULT_CAPTURE_BYTES,400*MiB);
  assert.deepEqual(CAPTURE_BUDGET_MIB,[200,400,800]);assert.equal(MAX_RECORDS,128);
  for(const budget of CAPTURE_BUDGET_MIB){
    assert.equal(recommendedPhaseMaximum(128,4,budget*MiB),32);
    assert.equal(recommendedPhaseMaximum(256,4,budget*MiB),budget/25);
    for(const size of [8,16,32,64,128,256]){
      const count=recommendedPhaseMaximum(size,4,budget*MiB);
      const input=3*(2*size)**2*4,samples=(3*size*size+96)*4;
      const perRecord=4*Math.ceil(input/3)+4*Math.ceil(samples/3)+8192;
      assert.ok(count*4*perRecord+3*MiB<=budget*MiB);
      assert.ok(count*4<=128);
    }
  }
  assert.throws(()=>recommendedPhaseMaximum(256,4,801*MiB),/budget/);
  assert.throws(()=>recommendedPhaseMaximum(256,4,NaN),/budget/);
});

test('the byte budget changes only on an empty batch; previous smaller limits remain valid',()=>{
  const buffer=new CaptureBuffer();assert.equal(buffer.summary().maxBytes,400*MiB);
  buffer.setBudget(800*MiB);assert.equal(buffer.summary().maxBytes,800*MiB);
  buffer.add({ok:false,timestamp:1000,stage:'phase',error:'not yet aligned'});
  assert.throws(()=>buffer.setBudget(200*MiB),/clear/);
  buffer.reset();buffer.setBudget(200*MiB);assert.equal(buffer.summary().maxBytes,200*MiB);
  assert.throws(()=>buffer.setBudget(801*MiB),/budget/);
  const old=new CaptureBuffer({maxBytes:160*MiB});
  const saved=old.export({sessionId:'existing-small-batch'});
  assert.equal(saved.format,'neural-qr-captures');assert.equal(saved.version,1);
  assert.equal(saved.limits.maxBytes,160*MiB);
});

test('recording Blob serializes each record independently while preserving exact JSON and metadata',async t=>{
  const recording={format:'neural-qr-captures',version:1,notes:'Unicode 🎥 and \\" \n \t',
    optional:undefined,profile:{size:256},records:[{inputBase64:'A'.repeat(65536),timestamp:1},{text:'}],\\"nested\\":['},null],
    diagnostics:[{error:'"comma, brace} and slash\\'}],limits:{maxBytes:800*MiB},collectionSessionId:'session'};
  const expected=JSON.stringify(recording),stringify=JSON.stringify;
  t.mock.method(JSON,'stringify',function(value,...rest){
    assert.notEqual(value,recording,'never serialize the whole recording');
    assert.notEqual(value,recording.records,'never serialize the whole records array');
    assert.notEqual(value,recording.diagnostics,'never serialize all diagnostics at once');
    return stringify(value,...rest);
  });
  const blob=recordingBlob(recording);
  assert.equal(blob.type,'application/json');assert.equal(blob.size,new TextEncoder().encode(expected).length);
  assert.equal(await blob.text(),expected);
});

test('recording Blob handles optional fields and arrays like JSON serialization and rejects circular data',async()=>{
  const value={records:[undefined,NaN,{optional:undefined}],diagnostics:[],ignored:undefined};
  assert.equal(await recordingBlob(value).text(),JSON.stringify(value));
  assert.equal(await recordingBlob({}).text(),'{}');
  for(const value of [null,undefined,[],1])assert.throws(()=>recordingBlob(value),/recording/);
  const circular={};circular.self=circular;
  assert.throws(()=>recordingBlob({records:[circular]}),/circular/i);
});
