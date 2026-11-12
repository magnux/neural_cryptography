import test from 'node:test';
import assert from 'node:assert/strict';
import {CameraControls,cameraControlCapabilities,readCameraControls,acquisitionDiagnostics} from '../dist/js/dataset/camera-controls.js';

const capabilities=()=>({exposureMode:['continuous','manual'],exposureCompensation:{min:-3,max:3,step:.25},
  exposureTime:{min:10,max:1000,step:10},focusMode:['continuous','single-shot','manual'],focusDistance:{min:0,max:100,step:5}});
function camera({capabilities:caps=capabilities(),settings={},apply}={}){
  const actual={exposureMode:'continuous',exposureCompensation:0,exposureTime:100,focusMode:'continuous',focusDistance:50,...settings},calls=[];
  const track={readyState:'live',getCapabilities:()=>caps,getSettings:()=>({...actual}),async applyConstraints(value){
    calls.push(value);if(apply)await apply(value,actual);else Object.assign(actual,value.advanced[0]);
  }};
  return {track,actual,calls};
}

test('capabilities are optional, finite and changeable; constructing controls never applies them',()=>{
  assert.deepEqual(cameraControlCapabilities({}),{});
  assert.deepEqual(cameraControlCapabilities({applyConstraints(){},getCapabilities(){throw Error('unavailable');}}),{});
  const fixture=camera({capabilities:{exposureMode:['none','manual','bogus'],focusMode:['manual','manual'],
    exposureCompensation:{min:-2,max:2,step:0},exposureTime:{min:0,max:Infinity},focusDistance:{min:1,max:1}}});
  assert.deepEqual(cameraControlCapabilities(fixture.track),{});
  const normal=camera(),controller=new CameraControls(normal.track);
  assert.equal(normal.calls.length,0);assert.deepEqual(controller.snapshot().requested,{});
  assert.equal(controller.snapshot().actual.exposureMode,'continuous');
  assert.equal(readCameraControls({getSettings:()=>({deviceId:'not-for-report',exposureMode:'bogus',focusDistance:NaN})}).deviceId,undefined);
});

test('requested values clamp and quantize to capabilities; actual driver values remain distinct',async()=>{
  const fixture=camera({apply:async(request,actual)=>{actual.exposureCompensation=Math.min(1,request.advanced[0].exposureCompensation);}});
  const controller=new CameraControls(fixture.track);
  await controller.apply({exposureCompensation:99});
  assert.deepEqual(fixture.calls[0],{advanced:[{exposureCompensation:3}]});
  assert.equal(controller.snapshot().requested.exposureCompensation,3);
  assert.equal(controller.snapshot().actual.exposureCompensation,1);
  await controller.apply({exposureCompensation:-.38});
  assert.equal(fixture.calls[1].advanced[0].exposureCompensation,-.5);
  assert.equal(controller.snapshot().pending,0);
  for(const value of [NaN,Infinity,'1'])await assert.rejects(controller.apply({exposureCompensation:value}),/finite/);
  await assert.rejects(controller.apply({zoom:2}),/does not expose/);
  await assert.rejects(controller.apply({exposureMode:'automatic'}),/available/);
});

test('manual values wait for a confirmed manual mode and unsupported or ignored modes do not unlock them',async()=>{
  const fixture=camera(),controller=new CameraControls(fixture.track);
  await assert.rejects(controller.apply({exposureTime:60}),/Manual exposure/);
  assert.equal(fixture.calls.length,0);
  await controller.apply({exposureMode:'manual'});await controller.apply({exposureTime:63});
  assert.equal(fixture.calls[1].advanced[0].exposureTime,60);
  await controller.apply({focusMode:'manual',focusDistance:37});
  assert.equal(controller.snapshot().actual.focusDistance,35);
  const ignored=camera({apply:async()=>{}}),other=new CameraControls(ignored.track);
  await other.apply({focusMode:'manual'});
  await assert.rejects(other.apply({focusDistance:25}),/Manual focus/);
  assert.equal(ignored.calls.length,1);
});

test('camera requests are serialized and a rejected setting does not poison later requests',async()=>{
  let release,active=0,maxActive=0;
  const fixture=camera({apply:async(request,actual)=>{
    active++;maxActive=Math.max(active,maxActive);
    if(request.advanced[0].exposureCompensation===-1)await new Promise(resolve=>{release=resolve;});
    active--;Object.assign(actual,request.advanced[0]);
  }}),controller=new CameraControls(fixture.track);
  const first=controller.apply({exposureCompensation:-1}),second=controller.apply({exposureCompensation:-2});
  await new Promise(resolve=>setImmediate(resolve));assert.equal(fixture.calls.length,1);
  release();await Promise.all([first,second]);assert.equal(maxActive,1);assert.equal(fixture.calls.length,2);
  fixture.track.applyConstraints=async value=>{fixture.calls.push(value);if('exposureTime' in value.advanced[0])throw new DOMException('Combination rejected','OverconstrainedError');Object.assign(fixture.actual,value.advanced[0]);};
  await controller.apply({exposureMode:'manual'});
  await assert.rejects(controller.apply({exposureTime:50}),/Combination rejected/);
  assert.match(controller.snapshot().lastError,/Combination rejected/);
  await controller.apply({focusMode:'manual'});assert.equal(controller.snapshot().actual.focusMode,'manual');
});

test('not-supported driver errors remove only the affected exposed control',async()=>{
  const fixture=camera({apply:async()=>{const e=new DOMException('Not available','NotSupportedError');e.constraint='exposureCompensation';throw e;}});
  const controller=new CameraControls(fixture.track);
  await assert.rejects(controller.apply({exposureCompensation:-1}),/Not available/);
  assert.equal(controller.snapshot().capabilities.exposureCompensation,undefined);
  assert.deepEqual(controller.snapshot().capabilities.focusMode,['continuous','single-shot','manual']);
  await assert.rejects(controller.apply({exposureCompensation:-2}),/does not expose/);
  assert.equal(fixture.calls.length,1);
});

test('closing invalidates in-flight results, queued operations and the controller for a new track',async()=>{
  let release;
  const fixture=camera({apply:()=>new Promise(resolve=>{release=resolve;})}),controller=new CameraControls(fixture.track);
  const first=controller.apply({exposureCompensation:-1}),second=controller.apply({focusMode:'manual'});
  const failures=Promise.all([assert.rejects(first,{name:'AbortError'}),assert.rejects(second,{name:'AbortError'})]);
  await new Promise(resolve=>setImmediate(resolve));controller.close();release();await failures;
  assert.equal(fixture.calls.length,1);assert.equal(controller.snapshot().pending,0);
  await assert.rejects(controller.apply({exposureCompensation:0}),{name:'AbortError'});
  const next=camera(),replacement=new CameraControls(next.track);await replacement.apply({exposureCompensation:-.25});
  assert.equal(next.calls.length,1);assert.deepEqual(replacement.snapshot().requested,{exposureCompensation:-.25});
});

test('queued updates are bounded and ended camera tracks reject new requests',async()=>{
  let release;const fixture=camera({apply:()=>new Promise(resolve=>{release=resolve;})}),controller=new CameraControls(fixture.track);
  const pending=Array.from({length:8},()=>controller.apply({exposureCompensation:-1}));
  const rejected=pending.map(p=>assert.rejects(p,{name:'AbortError'}));
  await assert.rejects(controller.apply({exposureCompensation:0}),/Wait/);
  await new Promise(resolve=>setImmediate(resolve));controller.close();release();await Promise.all(rejected);
  const ended=camera();ended.track.readyState='ended';await assert.rejects(new CameraControls(ended.track).apply({exposureCompensation:0}),{name:'AbortError'});
});

test('last acquisition diagnostics omit camera pixels and bound every diagnostic string or array',()=>{
  const value=acquisitionDiagnostics({stage:'orientation',error:'a'.repeat(400),input:new Float32Array(1000),stats:{
    markerCount:4,geometrySource:'local',orientationMargin:.12,finderContrasts:Array(100).fill(55),finderBlackLevels:[16,20,22,90],finderWhiteLevels:[240,240,240,220],
    markerClippedFraction:.1,detectionMode:'adaptive',detectionMaxSide:1600,detectionWidth:1600,detectionHeight:900,detectionThreshold:132,quadCount:2,
    orientationError:.05,sizeContrasts:[200,100],pixels:new Uint8Array(1000),colorTransform:Array(1000).fill(2),ms:NaN}});
  assert.equal(value.error.length,300);assert.equal(value.stats.finderContrasts.length,8);
  assert.equal(value.stats.markerCount,4);assert.equal(value.stats.ms,undefined);
  assert.equal(value.stats.markerClippedFraction,.1);assert.equal(value.stats.orientationError,.05);
  assert.deepEqual(value.stats.finderBlackLevels,[16,20,22,90]);assert.deepEqual(value.stats.sizeContrasts,[200,100]);
  assert.equal(value.stats.detectionMode,'adaptive');assert.equal(value.stats.quadCount,2);
  assert.equal(value.input,undefined);assert.equal(value.stats.pixels,undefined);assert.equal(value.stats.colorTransform,undefined);
});
