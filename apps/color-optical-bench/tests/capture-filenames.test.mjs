import test from 'node:test';
import assert from 'node:assert/strict';
import {createSender} from '../dist/js/dataset/contract.js';
import {recordingPrefix,recordingFilename} from '../dist/js/dataset/filenames.js';

const {manifest}=await createSender({seed:19,sessionId:'sender-filename-test'});
const created='2026-11-03T12:34:56.789Z';manifest.created=created;
function capture(phase=0){return {format:'neural-qr-captures',version:2,created,sessionId:'receiver-filename-test',
  collectionCode:manifest.collectionCode,profile:manifest.profile,records:[{ok:true,info:manifest.frames[phase].info}],diagnostics:[]};}

test('sender and receiver names share the explicitly assigned public collection code across phases',()=>{
  const prefix=manifest.collectionCode;
  assert.equal(recordingFilename(manifest),`${prefix}_neural-qr-sender-sender-filename-test.json`);
  for(let phase=0;phase<4;phase++){
    assert.equal(recordingPrefix(capture(phase)),prefix);
    assert.equal(recordingFilename(capture(phase)),`${prefix}_neural-qr-captures-receiver-filename-test-2026-11-03T12-34-56-789Z.json`);
  }
  assert.ok(manifest.frames.every(frame=>!('headerBase64' in frame)));
});

test('names keep export identity; diagnostics retain the assigned code and mixed assignments reject',()=>{
  const first=capture();
  assert.notEqual(recordingFilename(first),recordingFilename({...first,sessionId:'receiver-other'}));
  assert.notEqual(recordingFilename(first),recordingFilename({...first,created:'2026-11-03T12:34:57.789Z'}));
  assert.equal(recordingPrefix({...first,records:[]}),manifest.collectionCode);
  assert.throws(()=>recordingPrefix({...first,records:[...first.records,{info:{collectionCode:'12345678'}}]}),/mixes/);
  assert.throws(()=>recordingFilename({...first,sessionId:'../bad'}),/session ID/);
  assert.throws(()=>recordingFilename({...first,created:'not-a-date'}),/timestamp/);
  assert.throws(()=>recordingPrefix({...manifest,format:'optical-decoder-sender'}),/current Neural-QR/);
  assert.throws(()=>recordingPrefix({...manifest,frames:[]}),/frame list/);
  for(const collectionCode of ['',1,'123456789','ABCDABCD',undefined])
    assert.throws(()=>recordingPrefix({...first,collectionCode}),/collection code/);
});
