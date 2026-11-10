import test from 'node:test';
import assert from 'node:assert/strict';
import {MAX_CAPTURE_PIXELS,cameraRequest,captureDimensions,openCaptureCamera,cameraStatus} from '../dist/js/dataset/camera.js';
import {cameraMetadata} from '../dist/js/dataset/metadata.js';
import {captureFrame} from '../dist/js/dataset/capture.js';
import {createSender,captureProfile,base64ToBytes} from '../dist/js/dataset/contract.js';
import {renderNeuralQR} from '../dist/js/neural-qr/layout.js';

test('1440p and 4K requests are ideals; selected camera identity remains exact',async()=>{
  const calls=[],track={getSettings:()=>({width:1920,height:1080,frameRate:30})};
  const media={getUserMedia:async options=>{calls.push(options);return {getVideoTracks:()=>[track]};}};
  for(const [resolution,width,height] of [['1440',2560,1440],['2160',3840,2160]]){
    const result=await openCaptureCamera(media,{resolution,deviceId:'rear'});
    assert.deepEqual(calls.at(-1),{audio:false,video:{width:{ideal:width},height:{ideal:height},resizeMode:{ideal:'none'},deviceId:{exact:'rear'}}});
    const info=cameraMetadata(track,result.request);
    assert.equal(info.width,1920);assert.equal(info.height,1080);assert.equal(info.requestedWidth,width);assert.equal(info.requestedResolution,resolution);
    assert.match(cameraStatus(info),/actual camera 1920×1080/);
  }
});

test('native maximum uses capability bounds and reports negotiated actual settings',async()=>{
  let applied;
  const track={getCapabilities:()=>({width:{max:8000},height:{max:6000}}),applyConstraints:async c=>{applied=c;},
    getSettings:()=>({width:4096,height:3072,frameRate:15}),label:'Rear camera'};
  const {request}=await openCaptureCamera({getUserMedia:async()=>({getVideoTracks:()=>[track]})},{resolution:'max'});
  assert.ok(applied.width.ideal*applied.height.ideal<=MAX_CAPTURE_PIXELS);
  assert.ok(applied.width.ideal>3840);
  const info=cameraMetadata(track,request);
  assert.equal(info.width,4096);assert.equal(info.requestedResolution,'max');
  assert.match(cameraStatus(info),/maximum available; actual camera 4096×3072, 15 fps/);
});

test('maximum request without capability API still returns the negotiated stream',async()=>{
  const stream={getVideoTracks:()=>[{}]},media={getUserMedia:async()=>stream};
  assert.equal((await openCaptureCamera(media,{resolution:'max'})).stream,stream);
});

test('resolution fallback keeps the selected camera and records its reason',async()=>{
  const calls=[],stream={getVideoTracks:()=>[{}]},media={getUserMedia:async options=>{
    calls.push(options);if(calls.length===1)throw Object.assign(Error('Dimensions unsupported'),{name:'OverconstrainedError',constraint:'width'});return stream;
  }};
  const result=await openCaptureCamera(media,{resolution:'2160',deviceId:'same-camera'});
  assert.equal(result.stream,stream);assert.deepEqual(calls[1],{audio:false,video:{deviceId:{exact:'same-camera'}}});
  assert.match(result.request.resolutionFallback,/constraints rejected/);
});

test('permission errors and unavailable camera selection never trigger fallback',async()=>{
  for(const error of [Object.assign(Error('denied'),{name:'NotAllowedError'}),Object.assign(Error('device gone'),{name:'OverconstrainedError',constraint:'deviceId'})]){
    let calls=0;
    await assert.rejects(openCaptureCamera({getUserMedia:async()=>{calls++;throw error;}},{resolution:'2160',deviceId:'missing'}),e=>e===error);
    assert.equal(calls,1);
  }
});

test('failed optional maximum adjustment retains the working stream',async()=>{
  const track={getCapabilities:()=>({width:{max:3840},height:{max:2160}}),applyConstraints:async()=>{throw Error('unsupported');}};
  const stream={getVideoTracks:()=>[track]},result=await openCaptureCamera({getUserMedia:async()=>stream},{resolution:'max'});
  assert.equal(result.stream,stream);assert.match(result.request.resolutionFallback,/keeping/);
});

test('4K and portrait source pixels survive acquisition; larger frames have an explicit bound',()=>{
  assert.deepEqual(captureDimensions(3840,2160),{width:3840,height:2160,scale:1});
  assert.deepEqual(captureDimensions(2160,3840),{width:2160,height:3840,scale:1});
  assert.deepEqual(captureDimensions(4096,3072),{width:4096,height:3072,scale:1});
  const capped=captureDimensions(8000,6000);assert.ok(capped.width*capped.height<=MAX_CAPTURE_PIXELS);assert.ok(capped.scale<1);
  assert.match(cameraStatus({width:8000,height:6000,requestedResolution:'max'},capped),/20-megapixel processing bound/);
  assert.throws(()=>captureDimensions(0,0),/no valid/);assert.throws(()=>cameraRequest('garbage'),/supported/);
});

test('requested and negotiated camera metadata survives stopping',()=>{
  const previous=cameraMetadata({getSettings:()=>({width:1280,height:720,frameRate:30})},
    {requestedResolution:'2160',requestedWidth:3840,requestedHeight:2160,resolutionFallback:'Unavailable'});
  assert.deepEqual(cameraMetadata(null,{previous}),{...previous,active:false});
});

test('higher-resolution source improves measured G128 density without changing recording tensors',async()=>{
  const sender=await createSender({...captureProfile(128),seed:77,sessionId:'high-resolution'});
  const options={collectionCode:sender.manifest.collectionCode,profile:sender.manifest.profile};
  const low=captureFrame(renderNeuralQR(sender.frames[0],1024),{...options,probe:true});
  const high=captureFrame(renderNeuralQR(sender.frames[0],2048),options);
  assert.equal(low.ok,true);assert.equal(high.ok,true);
  assert.ok(high.stats.pixelsPerCell>low.stats.pixelsPerCell*1.9);
  assert.equal(high.info.size,128);assert.equal(high.info.pixelsPerCell,4);
  assert.equal(base64ToBytes(high.inputBase64).byteLength,3*512*512*4);
});
