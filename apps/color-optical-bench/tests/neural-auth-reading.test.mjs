import test from 'node:test';
import assert from 'node:assert/strict';
import {ReadingCue} from '../dist/js/neural-auth/reading.js';

const firstSample={ok:false,stage:'collect',info:{size:64,phase:0},updated:true,duplicateExposure:false,collected:1};
function harness(){
  const changes=[];let started=0;
  const cue=new ReadingCue({onChange:value=>changes.push(value),onReadingStarted:()=>started++});
  return {cue,changes,get started(){return started;}};
}

test('first valid sample marks provisional reading once per displayed packet',()=>{
  const h=harness();h.cue.displayed('message/data/0');
  assert.equal(h.cue.observe({ok:false,stage:'detect',markers:[{}]}),false);
  assert.equal(h.started,0,'markers alone are not a collected sample');
  assert.equal(h.cue.observe(firstSample),true);
  assert.equal(h.started,1);assert.deepEqual(h.changes,[true]);
  h.cue.observe({...firstSample,info:{size:64,phase:1},collected:2});
  assert.equal(h.started,1);assert.deepEqual(h.changes,[true]);

  h.cue.observe({ok:false,stage:'phase'});assert.equal(h.cue.active,false);
  h.cue.observe(firstSample);assert.equal(h.cue.active,true);
  assert.equal(h.started,1,'reacquiring the same candidate cannot repeatedly dismiss notices');
  h.cue.settle();h.cue.displayed('message/data/0');h.cue.observe(firstSample);
  assert.equal(h.started,1,'settling does not erase the displayed-packet history');

  h.cue.displayed('message/data/1');assert.equal(h.cue.active,false);
  h.cue.observe(firstSample);assert.equal(h.started,2);
});

test('duplicate exposures and FEC/CRC retries preserve the cue; recovery or unsupported geometry clears it',()=>{
  const h=harness();h.cue.displayed('message/ack/0');
  h.cue.observe({...firstSample,updated:false,duplicateExposure:true});
  assert.equal(h.cue.active,false);assert.equal(h.started,0);
  h.cue.observe(firstSample);
  for(const result of [
    {...firstSample,updated:false,duplicateExposure:true},
    {ok:false,stage:'fec',info:{size:64}},
    {ok:false,stage:'crc',info:{size:64}},
  ])assert.equal(h.cue.observe(result),true);
  assert.deepEqual(h.changes,[true]);assert.equal(h.started,1);
  assert.equal(h.cue.observe({ok:true,info:{size:64}}),false);
  h.cue.observe(firstSample);
  assert.equal(h.cue.observe({...firstSample,stage:'profile'}),false);
  assert.equal(h.started,1);
});

test('new runs rearm reading even when the displayed key is unchanged',()=>{
  const h=harness();h.cue.displayed('message/start');h.cue.observe(firstSample);
  h.cue.reset();assert.equal(h.cue.active,false);
  h.cue.displayed('message/start');h.cue.observe(firstSample);
  assert.equal(h.started,2);assert.deepEqual(h.changes,[true,false,true]);
});
