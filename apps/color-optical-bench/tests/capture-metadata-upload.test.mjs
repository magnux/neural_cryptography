import test from 'node:test';
import assert from 'node:assert/strict';
import {collectionIdentity,environmentMetadata,cameraMetadata} from '../dist/js/dataset/metadata.js';
import {recordingUploadConfig,uploadRecording} from '../dist/js/dataset/upload.js';

test('collection identity persists per origin and unavailable storage stays session-local',()=>{
  const values=new Map(),storage={getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value)};
  const uuid='aabbccdd-1234-4321-8888-aabbccddeeff';
  assert.equal(collectionIdentity({storage,randomUUID:()=>uuid}).deviceId,uuid);
  const again=collectionIdentity({storage,randomUUID:()=>{throw Error('must reuse stored ID');}});
  assert.equal(again.deviceId,uuid);assert.equal(again.persisted,true);
  const blocked=collectionIdentity({storage:{getItem(){throw Error('blocked');}},randomUUID:()=>uuid});
  assert.equal(blocked.deviceId,uuid);assert.equal(blocked.persisted,false);
});

test('metadata uses available browser hints and camera settings without requesting fingerprinting hints',()=>{
  const identity={deviceId:'origin-device',deviceLabel:'Test phone',persisted:true};
  const nav={userAgent:'Android Mobile',platform:'Linux',language:'en',userAgentData:{mobile:true,platform:'Android',brands:[{brand:'Example',version:'1'}],getHighEntropyValues(){throw Error('not allowed');}}};
  const environment=environmentMetadata(identity,{navigator:nav,screen:{width:400,height:800},innerWidth:390,innerHeight:760,devicePixelRatio:2});
  assert.equal(environment.deviceClass,'mobile');assert.equal(environment.deviceLabel,'Test phone');assert.equal(environment.display.viewportWidth,390);
  assert.equal(environment.browser.clientHints.platform,'Android');assert.match(environment.deviceIdScope,/origin/);
  const current=cameraMetadata({label:'Rear camera',getSettings:()=>({deviceId:'scoped-id',groupId:'scoped-group',width:1280,height:720,frameRate:30,facingMode:'environment'})},{requestedHeight:1080});
  assert.equal(current.label,'Rear camera');assert.equal(current.width,1280);assert.equal(current.requestedHeight,1080);
  const stopped=cameraMetadata(null,{previous:current});assert.equal(stopped.active,false);assert.equal(stopped.label,current.label);assert.equal(stopped.deviceId,current.deviceId);
  assert.equal(environmentMetadata(identity,{navigator:{userAgent:'unspecified'},screen:{}}).deviceClass,'unknown');
});

const config={enabled:true,maxUploadBytes:4096,uploadToken:'csrf-token'};
const recording={format:'neural-qr-captures',version:1,datasetSplit:'validation',created:'2026-11-04T12:00:00.000Z',records:[]};
class Request{
  constructor(result={ok:true,storedPath:'validation/g64/receiver/example.json',duplicate:false}){this.upload={};this.headers={};this.result=result;this.status=201;}
  open(method,url){this.method=method;this.url=url;}
  setRequestHeader(key,value){this.headers[key]=value;}
  send(body){this.body=body;queueMicrotask(()=>{this.upload.onprogress?.({loaded:body.size,total:body.size,lengthComputable:true});this.responseText=JSON.stringify(this.result);this.onload?.();});}
  abort(){this.onabort?.();}
}
test('server configuration keeps downloads viable when uploads are disabled or unavailable',async()=>{
  assert.equal((await recordingUploadConfig(async()=>({ok:false,status:404}))).enabled,false);
  assert.equal((await recordingUploadConfig(async()=>({ok:true,json:async()=>({enabled:false})}))).enabled,false);
  assert.deepEqual(await recordingUploadConfig(async()=>({ok:true,json:async()=>config})),config);
});
test('upload sends exact recording bytes, split and token; retries preserve the snapshot',async()=>{
  const first=new Request(),progress=[];
  await uploadRecording(recording,{split:'validation',config,xhrFactory:()=>first,onProgress:value=>progress.push(value)});
  assert.equal(first.method,'POST');assert.equal(first.url,'/api/recordings?split=validation');
  assert.equal(first.headers['X-Upload-Token'],'csrf-token');assert.equal(first.headers['Content-Type'],'application/json');
  assert.equal(await first.body.text(),JSON.stringify(recording));assert.equal(progress.at(-1).loaded,first.body.size);
  const retry=new Request({ok:true,storedPath:'validation/g64/receiver/example.json',duplicate:true});
  const result=await uploadRecording(recording,{split:'validation',config,xhrFactory:()=>retry});
  assert.equal(result.duplicate,true);assert.equal(await retry.body.text(),await first.body.text());
});
test('upload rejects wrong splits, oversize data, failed server writes and cancellation',async()=>{
  await assert.rejects(uploadRecording(recording,{split:'train',config}),/split/);
  await assert.rejects(uploadRecording(recording,{split:'validation',config:{...config,maxUploadBytes:1}}),/limit/);
  const rejected=new Request({error:'Filename already contains different bytes'});rejected.status=409;
  await assert.rejects(uploadRecording(recording,{split:'validation',config,xhrFactory:()=>rejected}),/different bytes/);
  const controller=new AbortController(),request=new Request();request.send=()=>{};
  const pending=uploadRecording(recording,{split:'validation',config,xhrFactory:()=>request,signal:controller.signal});
  controller.abort();await assert.rejects(pending,/cancelled/);
});
