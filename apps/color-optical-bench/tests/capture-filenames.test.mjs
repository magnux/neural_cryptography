import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,readFile,writeFile,rm,readdir,symlink,stat} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createSender} from '../dist/js/dataset/contract.js';
import {crc32} from '../dist/js/format.js';
import {recordingPrefix,recordingFilename} from '../dist/js/dataset/filenames.js';
import {renameRecordings} from '../../../scripts/rename-corrector-recordings.mjs';

const base=fileURLToPath(new URL('../../../validation/',import.meta.url));
const {manifest,frames}=await createSender({seed:19,sessionId:'sender-filename-test'});
const created='2026-11-03T12:34:56.789Z';
manifest.created=created;
function capture(phase=0) {
  return {format:'optical-decoder-captures',version:1,created,sessionId:'receiver-filename-test',
    records:[{ok:true,info:manifest.frames[phase].info}],diagnostics:[]};
}
async function temp(fn){await mkdir(base,{recursive:true});const root=await mkdtemp(join(base,'filenames-'));try{await fn(root);}finally{await rm(root,{recursive:true,force:true});}}

test('sender and receiver filenames share the payload CRC across palette phases',()=>{
  const prefix=crc32(frames[0].chunk).toString(16).padStart(8,'0');
  assert.equal(recordingPrefix(manifest),prefix);
  assert.equal(recordingFilename(manifest),`${prefix}_decoder-sender-${manifest.sequenceId}.json`);
  for(let phase=0;phase<frames.length;phase++) {
    assert.equal(recordingPrefix(capture(phase)),prefix);
    assert.match(recordingFilename(capture(phase)),new RegExp(`^${prefix}_decoder-captures-receiver-filename-test-`));
  }
  assert.notEqual(new DataView(frames[0].header.buffer).getUint32(60),new DataView(frames[1].header.buffer).getUint32(60));
  assert.equal(recordingPrefix({...manifest,frames:[{info:{chunkCRC:1}}]}),'00000001');
  assert.equal(recordingPrefix({...manifest,frames:[{info:{chunkCRC:0xffffffff}}]}),'ffffffff');
});

test('names keep export identity, name mixed captures from first record, and label diagnostic-only files',()=>{
  const first=capture(),other={...capture(),sessionId:'receiver-other'};
  assert.notEqual(recordingFilename(first),recordingFilename(other));
  assert.notEqual(recordingFilename(first),recordingFilename({...first,created:'2026-11-03T12:34:57.789Z'}));
  first.records.push({ok:true,info:{...first.records[0].info,chunkCRC:42}});
  assert.equal(recordingPrefix(first),recordingPrefix(manifest));
  assert.equal(recordingPrefix({...first,records:[]}), 'unpaired');
  assert.match(recordingFilename({...first,records:[]}),/^unpaired_decoder-captures-/);
  assert.throws(()=>recordingFilename({...first,sessionId:'../bad'}),/session ID/);
  assert.throws(()=>recordingFilename({...first,created:'not-a-date'}),/timestamp/);
  assert.throws(()=>recordingPrefix({...manifest,version:0}),/current/);
  assert.throws(()=>recordingPrefix({...manifest,frames:[]}),/frame list/);
  for(const chunkCRC of [-1,2**32,1.5,'19',undefined])
    assert.throws(()=>recordingPrefix({...manifest,frames:[{info:{chunkCRC}}]}),/CRC32/);
});

test('recursive rename replaces only manual prefixes and preserves exact suffixes and split folders',async()=>temp(async root=>{
  const phone=join(root,'phone','val'),pc=join(root,'pc','val');await mkdir(phone,{recursive:true});await mkdir(pc,{recursive:true});
  const senderSuffix=`decoder-sender-${manifest.sequenceId}.json`;
  // Deliberately differs from JSON.created; existing names must not be rebuilt.
  const receiverSuffix='decoder-captures-receiver-filename-test-2026-10-29T18-06-23-716Z.json';
  const senderPath=join(pc,`0_${senderSuffix}`),receiverPath=join(phone,`a_${receiverSuffix}`);
  const senderBytes=JSON.stringify(manifest,null,2)+'\n',receiverBytes=JSON.stringify(capture())+'\n';
  await writeFile(senderPath,senderBytes);await writeFile(receiverPath,receiverBytes);
  await writeFile(join(pc,'notes.json'),'{"notes":"unrelated"}');
  const before=await stat(receiverPath);
  const dry=await renameRecordings([root]);assert.equal(dry.mode,'dry-run');assert.equal(dry.renames.length,2);assert.equal(dry.skipped.length,1);
  assert.equal(await readFile(receiverPath,'utf8'),receiverBytes);
  const applied=await renameRecordings([root],{apply:true});assert.equal(applied.renames.length,2);
  const prefix=recordingPrefix(manifest);
  assert.equal(await readFile(join(pc,`${prefix}_${senderSuffix}`),'utf8'),senderBytes);
  const newReceiver=join(phone,`${prefix}_${receiverSuffix}`);assert.equal(await readFile(newReceiver,'utf8'),receiverBytes);
  assert.deepEqual((await readdir(phone)).sort(),[`${prefix}_${receiverSuffix}`]);
  assert.deepEqual((await readdir(pc)).sort(),[`${prefix}_${senderSuffix}`,'notes.json'].sort());
  assert.equal((await stat(newReceiver)).mtimeMs,before.mtimeMs);
  await assert.rejects(readFile(senderPath),{code:'ENOENT'});await assert.rejects(readFile(receiverPath),{code:'ENOENT'});
  const again=await renameRecordings([root],{apply:true});assert.equal(again.renames.length,0);assert.equal(again.unchanged.length,2);
}));

test('any destination conflict stops all renames before mutation',async()=>temp(async root=>{
  await writeFile(join(root,'a_sender.json'),JSON.stringify(manifest));
  await writeFile(join(root,'b_receiver.json'),JSON.stringify(capture()));
  const destination=join(root,`${recordingPrefix(capture())}_receiver.json`);
  await writeFile(destination,'{"existing":"content"}');
  const before=(await readdir(root)).sort();
  await assert.rejects(renameRecordings([root],{apply:true}),/destination already exists/);
  assert.deepEqual((await readdir(root)).sort(),before);
  assert.equal(await readFile(destination,'utf8'),'{"existing":"content"}');
}));

test('duplicate exports and symlink sources are refused without deleting originals',async()=>temp(async root=>{
  await writeFile(join(root,'a_same.json'),JSON.stringify(manifest));await writeFile(join(root,'b_same.json'),JSON.stringify(manifest));
  await assert.rejects(renameRecordings([root],{apply:true}),/same filename/);
  assert.deepEqual((await readdir(root)).sort(),['a_same.json','b_same.json']);
  await rm(join(root,'b_same.json'));await symlink(join(root,'a_same.json'),join(root,'linked.json'));
  await assert.rejects(renameRecordings([root],{apply:true}),/symlink/);
  assert.deepEqual((await readdir(root)).sort(),['a_same.json','linked.json']);
}));

test('rename adds a prefix when absent and keeps diagnostic-only unpaired names idempotent',async()=>temp(async root=>{
  const senderSuffix=`decoder-sender-${manifest.sequenceId}.json`,diagnosticSuffix='decoder-captures-receiver-diagnostics-2026-10-29T18-06-23-716Z.json';
  await writeFile(join(root,senderSuffix),JSON.stringify(manifest));
  await writeFile(join(root,`i_${diagnosticSuffix}`),JSON.stringify({...capture(),records:[]}));
  const result=await renameRecordings([root],{apply:true});assert.equal(result.renames.length,2);
  assert.deepEqual((await readdir(root)).sort(),[`${recordingPrefix(manifest)}_${senderSuffix}`,`unpaired_${diagnosticSuffix}`].sort());
  const again=await renameRecordings([root],{apply:true});assert.equal(again.renames.length,0);assert.equal(again.unchanged.length,2);
}));
