import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {Worker as NodeWorker} from 'node:worker_threads';
import {contentDescriptor,parseContentDescriptor,decodeContent,imageType,samplePng,sampleWav,inspectWav} from '../dist/js/content.js';
import {encodeSequence,parseHeader,renderFrame} from '../dist/js/format.js';
import {decodeImage,simulateCapture} from '../dist/js/vision.js';
import {ReceiveSession} from '../dist/js/receive-session.js';
import {DecoderClient} from '../dist/js/decoder-client.js';

const id=Uint8Array.of(8,7,6,5,4,3,2,1);
const resultFor=f=>({ok:true,stage:'complete',info:parseHeader(f.header),chunk:f.chunk,stats:{}});
function framesFor(bytes,type,sequence=id){return encodeSequence(bytes,{size:32,colors:4,maxFrames:8,id:sequence,contentType:type});}
const encodeText=s=>new TextEncoder().encode(s);
const roundTrip=(bytes,type)=>decodeContent(bytes,parseContentDescriptor(contentDescriptor(bytes,type)));

test('content descriptor has an exact eight-byte, big-endian layout',()=>{
  assert.deepEqual(contentDescriptor(encodeText('A'),'text'),Uint8Array.of(79,67,49,1,0,0,0,1));
  assert.deepEqual(contentDescriptor(new Uint8Array(),'raw'),Uint8Array.of(79,67,49,0,0,0,0,0));
});
test('complete UTF-8 preserves newlines, emoji, BOM, and multi-frame text',()=>{
  const text='\uFEFF'+('Light · € 👋\n日本語\n'.repeat(55)),bytes=encodeText(text),received=roundTrip(bytes,'text');
  assert.ok(bytes.length>512);assert.equal(received.text,text);assert.deepEqual(received.bytes,bytes);
  assert.equal(roundTrip(new Uint8Array(),'text').text,'');
});
test('raw binary is never inferred to be nested content; a descriptor is mandatory',()=>{
  const bytes=Uint8Array.of(79,67,49,1,0,255,128,0),decoded=roundTrip(bytes,'raw');
  assert.equal(decoded.type,'raw');assert.equal(decoded.text,undefined);assert.deepEqual(decoded.bytes,bytes);
  assert.throws(()=>decodeContent(bytes),/descriptor/);
});
test('PNG, JPEG, and WAV preserve original bytes and download types',()=>{
  const jpeg=new Uint8Array(readFileSync(new URL('fixtures/tile.jpg',import.meta.url)));
  for(const [bytes,type,mime,extension] of [[samplePng(),'png','image/png','png'],[jpeg,'jpeg','image/jpeg','jpg'],[sampleWav(),'wav','audio/wav','wav']]){
    const decoded=roundTrip(bytes,type);assert.equal(decoded.mime,mime);assert.equal(decoded.extension,extension);assert.deepEqual(decoded.bytes,bytes);
    if(type!=='wav')assert.equal(imageType(bytes),type);
  }
});
test('invalid UTF-8, unknown descriptors, and inconsistent lengths fail explicitly',()=>{
  assert.throws(()=>contentDescriptor(Uint8Array.of(0xc3),'text'),/UTF-8/);
  const header=contentDescriptor(encodeText('a'),'text');
  const unknown=header.slice();unknown[3]=99;assert.throws(()=>parseContentDescriptor(unknown),/type/);
  const version=header.slice();version[2]=50;assert.throws(()=>parseContentDescriptor(version),/version/);
  assert.throws(()=>parseContentDescriptor(header.subarray(0,7)),/Truncated/);
  assert.throws(()=>decodeContent(encodeText('abc'),{type:'text',length:1}),/length/);
  assert.throws(()=>contentDescriptor(new Uint8Array(),'mp3'),/Unsupported/);
  assert.throws(()=>imageType(encodeText('not a picture')),/PNG or JPEG/);
});
test('WAV sample validates as mono 8 kHz PCM and rejects compressed/truncated audio',()=>{
  const wav=sampleWav(),info=inspectWav(wav);assert.equal(info.sampleRate,8000);assert.equal(info.channels,1);assert.equal(info.bits,8);assert.equal(info.duration,.12);
  const compressed=wav.slice();compressed[20]=3;assert.throws(()=>contentDescriptor(compressed,'wav'),/PCM WAV/);
  assert.throws(()=>inspectWav(wav.subarray(0,wav.length-1)),/length/);
  const badAlign=wav.slice();badAlign[32]=2;assert.throws(()=>inspectWav(badAlign),/layout/);
});
test('Unicode content survives multiple real image decodes and out-of-order assembly',()=>{
  const text='á € 👋\n'.repeat(85),bytes=encodeText(text),frames=framesFor(bytes,'text'),session=new ReceiveSession();
  assert.ok(frames.length>1);
  for(const frame of [...frames].reverse()){
    const capture=simulateCapture(renderFrame(frame)),result=decodeImage(capture,{preview:false});
    assert.equal(result.ok,true,result.error);session.accept(result);
  }
  assert.equal(session.content.text,text);assert.deepEqual(session.content.bytes,bytes);
});
test('image and audio files survive the complete optical and content pipeline',()=>{
  const jpeg=new Uint8Array(readFileSync(new URL('fixtures/tile.jpg',import.meta.url)));
  for(const [bytes,type] of [[samplePng(),'png'],[jpeg,'jpeg'],[sampleWav(),'wav']]){
    const frames=framesFor(bytes,type),session=new ReceiveSession();
    for(const frame of frames){const result=decodeImage(simulateCapture(renderFrame(frame)),{preview:false});assert.equal(result.ok,true,result.error);session.accept(result);}
    assert.equal(session.content.type,type);assert.deepEqual(session.content.bytes,bytes);
  }
});
test('live and laboratory transfers cannot overwrite or reset each other',()=>{
  const live=new ReceiveSession(),lab=new ReceiveSession();
  const liveText='Live message 👋'.repeat(35),liveFrames=framesFor(encodeText(liveText),'text');
  live.accept(resultFor(liveFrames[0]));assert.equal(live.content,null);
  const labFrame=framesFor(samplePng(),'png',new Uint8Array(8))[0];lab.accept(resultFor(labFrame));assert.equal(lab.content.type,'png');
  assert.equal(live.assembler.parts.size,1);assert.equal(live.content,null);lab.reset();assert.equal(live.assembler.parts.size,1);
  for(const f of liveFrames.slice(1))live.accept(resultFor(f));assert.equal(live.content.text,liveText);assert.equal(lab.received,null);
});
test('duplicate complete frames reuse the content object and do not restart a media preview',()=>{
  const frame=framesFor(samplePng(),'png')[0],session=new ReceiveSession();
  const first=session.accept(resultFor(frame)),second=session.accept(resultFor(frame));
  assert.equal(first.fresh,true);assert.equal(second.fresh,false);assert.strictEqual(first.content,second.content);
  session.reset();assert.equal(session.content,null);assert.equal(session.accept(resultFor(frame)).fresh,true);
});
test('transport integrity and content validation remain distinct',()=>{
  const frame=encodeSequence(Uint8Array.of(255),{size:32,colors:4,id})[0],session=new ReceiveSession();
  const result=resultFor(frame);result.info={...result.info,content:{type:'text',length:1}};
  const accepted=session.accept(result);assert.equal(accepted.valid,true);assert.ok(accepted.complete);
  assert.match(accepted.contentError,/UTF-8/);assert.equal(session.content,null);
});
test('two worker clients decode independently with actual transferable buffers',async()=>{
  const previous=globalThis.Worker,underlying=[];
  globalThis.Worker=class {
    constructor(){this.worker=new NodeWorker(new URL('./worker-harness.mjs',import.meta.url));underlying.push(this.worker);this.worker.on('message',data=>this.onmessage?.({data}));this.worker.on('error',error=>this.onerror?.({message:error.message}));}
    postMessage(...args){this.worker.postMessage(...args);}
    terminate(){return this.worker.terminate();}
  };
  try{
    const first=new DecoderClient(),second=new DecoderClient();
    const image1=renderFrame(framesFor(encodeText('live'),'text')[0]),image2=renderFrame(framesFor(samplePng(),'png')[0]);
    const [r1,r2]=await Promise.all([first.decode(image1),second.decode(image2)]);
    assert.equal(first.busy,false);assert.equal(second.busy,false);assert.equal(image1.data.byteLength,0);assert.equal(image2.data.byteLength,0);
    assert.equal(decodeContent(r1.chunk,r1.info.content).text,'live');assert.equal(decodeContent(r2.chunk,r2.info.content).type,'png');
    second.reset();
    const frame=framesFor(encodeText('worker simulation'),'text')[0];
    const simulated=await second.simulate(frame,768,{width:960,height:760,noise:0,moire:0},{preview:false});
    assert.equal(simulated.ok,true,simulated.error);assert.equal(decodeContent(simulated.chunk,simulated.info.content).text,'worker simulation');
    assert.deepEqual(simulated.inputDimensions,{width:960,height:760});assert.ok(simulated.inputPreview.data.length>0);
    assert.ok(frame.chunk.byteLength>0,'simulation must not detach the outgoing frame');
  }finally{globalThis.Worker=previous;await Promise.all(underlying.map(w=>w.terminate()));}
});
