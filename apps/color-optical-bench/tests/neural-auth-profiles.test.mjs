import test from 'node:test';
import assert from 'node:assert/strict';
import {deviceProfile,transportProfile,transferMetrics} from '../dist/js/neural-auth/profiles.js';

test('both physical screen profiles allow G32 through G128 in either direction',()=>{
  assert.deepEqual(transportProfile('pc',128),{device:'pc',outgoingSize:128,incomingMaxSize:128,controlSize:32});
  assert.deepEqual(transportProfile('phone',64),{device:'phone',outgoingSize:64,incomingMaxSize:128,controlSize:32});
  for(const device of ['pc','phone']){
    assert.deepEqual(deviceProfile(device).outgoingSizes,[32,64,128]);
    for(const size of [32,64,128]){
      assert.deepEqual(transportProfile(device,size),{device,outgoingSize:size,incomingMaxSize:128,controlSize:32});
    }
    for(const size of [16,256,64.5,NaN,Infinity])assert.throws(()=>transportProfile(device,size));
  }
  for(const device of ['verifier','prover','desktop',null])assert.throws(()=>deviceProfile(device));
});

test('profile defaults are immutable and throughput is measured in delivered bytes',()=>{
  assert.equal(deviceProfile('pc').defaultOutgoingSize,128);
  assert.equal(deviceProfile('phone').defaultOutgoingSize,64);
  for(const device of ['pc','phone']){
    assert.throws(()=>deviceProfile(device).outgoingSizes.push(256));
    assert.throws(()=>{deviceProfile(device).incomingMaxSize=256;});
  }
  assert.deepEqual(transferMetrics(1200,2400),{bytes:1200,elapsedMs:2400,goodputBytesPerSecond:500});
  assert.equal(transferMetrics(0,0).goodputBytesPerSecond,0);
  assert.throws(()=>transferMetrics(-1,100));assert.throws(()=>transferMetrics(100,NaN));
});
