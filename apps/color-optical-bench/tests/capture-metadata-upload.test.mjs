import test from 'node:test';
import assert from 'node:assert/strict';
import {collectionIdentity,environmentMetadata,cameraMetadata} from '../dist/js/dataset/metadata.js';
import {recordingUploadConfig,uploadRecording,fetchCollectionSession,stageCollectionSession,abandonCollectionSession,validateCollectionSession} from '../dist/js/dataset/upload.js';

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
const recording={format:'neural-qr-captures',version:2,datasetSplit:'validation',created:'2026-11-04T12:00:00.000Z',records:[]};
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


const profile={size:64,colors:16,phases:4,gridBlock:8,parity:0,holdMs:4000,permute:true,pixelsPerCell:4};
const descriptor={format:'neural-qr-collection-session',version:2,id:'public-session',collectionCode:'1234abcd',senderSessionId:'sender-physical',profile,datasetSplit:'validation',created:'2026-11-05T12:00:00.000Z'};
test('session discovery copies only public assignment metadata and never sender truth',async()=>{
  let request;
  const response={...descriptor,payloadBase64:'forbidden',labels:[1,2],seed:123,profile:{...profile,expectedSymbols:[1,2]}};
  const value=await fetchCollectionSession({config,fetcher:async(url,options)=>{request={url,options};return {ok:true,json:async()=>({enabled:true,session:response})};}});
  assert.deepEqual(value,descriptor);assert.equal(request.url,'/api/collection-session');assert.equal(request.options.cache,'no-store');
  assert.equal(await fetchCollectionSession({config,fetcher:async()=>({ok:true,json:async()=>({enabled:true,session:null})})}),null);
  assert.equal(await fetchCollectionSession({config:{enabled:false},fetcher:()=>{throw Error('must not fetch');}}),null);
  assert.throws(()=>validateCollectionSession({...descriptor,profile:{...profile,colors:4}}),/profile/);
  await assert.rejects(fetchCollectionSession({config,fetcher:async()=>({ok:false,json:async()=>({error:'unavailable'})})}),/unavailable/);
});
test('sender staging is explicit and bound to the manifest; completed retries do not reopen sessions',async()=>{
  const sender={format:'neural-qr-sender',version:2,sessionId:descriptor.senderSessionId,collectionCode:descriptor.collectionCode,profile,datasetSplit:'validation'};
  const request=new Request({ok:true,storedPath:'validation/g64/sender/sender.json.tmp',session:descriptor});
  const staged=await stageCollectionSession(sender,{split:'validation',config,xhrFactory:()=>request});
  assert.deepEqual(staged.session,descriptor);assert.equal(request.url,'/api/collection-session?split=validation');assert.equal(request.timeout,300000);
  const wrong=new Request({ok:true,storedPath:'wrong.json.tmp',session:{...descriptor,collectionCode:'ffffffff'}});
  await assert.rejects(stageCollectionSession(sender,{split:'validation',config,xhrFactory:()=>wrong}),/does not match/);
  const completed=new Request({ok:true,storedPath:'sender.json',session:null,sessionCompleted:true,duplicate:true});
  assert.equal((await stageCollectionSession(sender,{split:'validation',config,xhrFactory:()=>completed})).session,null);
  await assert.rejects(stageCollectionSession(recording,{split:'validation',config}),/sender manifest/);
});
test('abandon targets the observed session and receiver uploads have a bounded large-body timeout',async()=>{
  let request;
  await abandonCollectionSession(descriptor.id,{config,fetcher:async(url,options)=>{request={url,options};return {ok:true,json:async()=>({ok:true,abandoned:true})};}});
  assert.equal(request.url,'/api/collection-session/abandon');assert.deepEqual(JSON.parse(request.options.body),{id:descriptor.id});assert.equal(request.options.headers['X-Upload-Token'],config.uploadToken);
  const receiver=new Request();await uploadRecording({...recording,collectionSessionId:descriptor.id},{split:'validation',config,xhrFactory:()=>receiver});
  assert.equal(receiver.timeout,900000);assert.equal(JSON.parse(await receiver.body.text()).collectionSessionId,descriptor.id);
});
