import test from 'node:test';
import assert from 'node:assert/strict';
import {CameraPreparation} from '../dist/camera-preparation.js';

const profile={size:64,backend:'auto'},good={modelStatus:'trained',ready:true,offlineReady:true,parity:{passed:true}};
function fixture(){
  const correctors=[],cryptos=[],states=[],lost=[];
  const make=(list,opts,method,close)=>{
    let resolve,reject;const result=new Promise((a,b)=>{resolve=a;reject=b;});
    const obj={resolve,calls:[],closed:false,invalidate:opts.onInvalidate,[close](){this.closed=true;reject(Error('closed'));opts.onInvalidate?.(Error('closed'));},
      [method](...args){this.calls.push(args);return result;}};
    list.push(obj);return obj;
  };
  const manager=new CameraPreparation({makeCrypto:opts=>make(cryptos,opts,'request','destroy'),makeCorrector:opts=>make(correctors,opts,'call','cancel'),
      onState:(...s)=>states.push(s),onLost:e=>lost.push(e)});
  return {manager,correctors,cryptos,states,lost};
}

test('Start while loading joins the same two workers and ready waits for the crypto worker as well as the model',async()=>{
  const {manager,correctors,cryptos,states}=fixture(),waiting=manager.prepare(profile);
  assert.equal(manager.prepare(profile),waiting);
  assert.deepEqual(cryptos[0].calls,[['selfTest',{}]]);
  correctors[0].resolve(good);await Promise.resolve();
  assert.equal(states.at(-1)[0],'loading');
  cryptos[0].resolve({passed:true});const session=await waiting;
  assert.equal(session.corrector,correctors[0]);assert.equal(session.cryptoClient,cryptos[0]);
  assert.equal(manager.prepare(profile),waiting);assert.equal(correctors.length,1);assert.equal(states.at(-1)[0],'ready');
});

test('changing the receive grid cancels both old workers; tab exit cancels the replacement and ignores late invalidation',async()=>{
  const {manager,correctors,cryptos,states,lost}=fixture(),old=manager.prepare(profile);
  const next=manager.prepare({...profile,size:128});
  assert.ok(correctors[0].closed&&cryptos[0].closed);await assert.rejects(old);
  correctors[0].invalidate(Error('late worker'));assert.equal(lost.length,0);
  correctors[1].resolve(good);cryptos[1].resolve({passed:true});await next;assert.equal(states.at(-1)[0],'ready');
  manager.clear();assert.ok(correctors[1].closed&&cryptos[1].closed);assert.equal(states.at(-1)[0],'idle');
});

test('failed model/parity never advertises flight mode; retry starts fresh and a dead idle worker invalidates ready',async()=>{
  const {manager,correctors,cryptos,states,lost}=fixture(),bad=manager.prepare(profile);
  cryptos[0].resolve({passed:true});correctors[0].resolve({...good,offlineReady:false});await assert.rejects(bad,/readiness/);
  assert.equal(states.at(-1)[0],'error');assert.ok(correctors[0].closed&&cryptos[0].closed);
  const retry=manager.prepare(profile);correctors[1].resolve(good);cryptos[1].resolve({passed:true});await retry;
  correctors[1].invalidate(Error('worker gone'));assert.equal(states.at(-1)[0],'error');assert.equal(lost.length,1);
});
