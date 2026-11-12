import test from 'node:test';
import assert from 'node:assert/strict';
import {deviceProfile,transportProfile,transferMetrics} from '../dist/js/neural-auth/profiles.js';

test('physical screen profiles cap each direction independently of proof roles',()=>{
  assert.deepEqual(transportProfile('pc',128),{device:'pc',outgoingSize:128,incomingMaxSize:64,controlSize:32});
  assert.deepEqual(transportProfile('phone',64),{device:'phone',outgoingSize:64,incomingMaxSize:128,controlSize:32});
  for(const size of [32,64])assert.equal(transportProfile('phone',size).outgoingSize,size);
  for(const size of [32,64,128])assert.equal(transportProfile('pc',size).outgoingSize,size);
  for(const size of [16,128,256,64.5,NaN,Infinity])assert.throws(()=>transportProfile('phone',size));
  for(const device of ['verifier','prover','desktop',null])assert.throws(()=>deviceProfile(device));
});

test('profile defaults are immutable and throughput is measured in delivered bytes',()=>{
  assert.equal(deviceProfile('pc').defaultOutgoingSize,128);
  assert.equal(deviceProfile('phone').defaultOutgoingSize,64);
  assert.throws(()=>deviceProfile('phone').outgoingSizes.push(128));
  assert.deepEqual(transferMetrics(1200,2400),{bytes:1200,elapsedMs:2400,goodputBytesPerSecond:500});
  assert.equal(transferMetrics(0,0).goodputBytesPerSecond,0);
  assert.throws(()=>transferMetrics(-1,100));assert.throws(()=>transferMetrics(100,NaN));
});
