import test from 'node:test';
import assert from 'node:assert/strict';
import {ReadingCue,READING_DECAY_MS} from '../dist/js/neural-auth/reading.js';
import {TurnNotices} from '../dist/js/neural-auth/notices.js';

const firstSample={ok:false,stage:'collect',info:{size:64,phase:0},updated:true,duplicateExposure:false,collected:1};
function harness(){
  const changes=[],timers=new Map();let now=0,next=0;
  const cue=new ReadingCue({onChange:value=>changes.push(value),
    setTimer:(callback,ms)=>{assert.equal(ms,3000);timers.set(++next,{callback,due:now+ms});return next;},
    clearTimer:id=>timers.delete(id)});
  return {cue,changes,timers,advance(ms){
    now+=ms;
    for(const [id,timer] of [...timers])if(timer.due<=now){timers.delete(id);timer.callback();}
  }};
}

test('first sample starts reading; misses fade for three seconds without restarting the decay',()=>{
  const h=harness();h.cue.displayed('message/data/0');
  assert.equal(h.cue.observe({ok:false,stage:'detect',markers:[{}]}),'idle');
  assert.equal(h.timers.size,0);
  assert.equal(h.cue.observe(firstSample),'reading');
  h.cue.observe({...firstSample,info:{size:64,phase:1},collected:2});
  assert.deepEqual(h.changes,['reading']);
  assert.equal(h.cue.observe({ok:false,stage:'phase'}),'decaying');
  const timer=[...h.timers.values()][0];
  h.advance(1000);h.cue.observe({ok:false,stage:'detect'});
  assert.equal([...h.timers.values()][0],timer,'continued misses must not prolong the fade');
  h.advance(1999);assert.equal(h.cue.state,'decaying');
  h.advance(1);assert.equal(h.cue.state,'idle');assert.equal(h.timers.size,0);
  assert.equal(READING_DECAY_MS,3000);
  assert.deepEqual(h.changes,['reading','decaying','idle']);
});

test('reacquisition cancels the fade and invalidates its stale callback',()=>{
  const h=harness();h.cue.observe(firstSample);h.cue.observe({stage:'detect'});
  const stale=[...h.timers.values()][0].callback;
  h.advance(1500);assert.equal(h.cue.observe(firstSample),'reading');
  assert.equal(h.timers.size,0);stale();assert.equal(h.cue.state,'reading');
  h.cue.observe({stage:'detect'});h.advance(2999);assert.equal(h.cue.state,'decaying');
  h.advance(1);assert.equal(h.cue.state,'idle');
  assert.deepEqual(h.changes,['reading','decaying','reading','decaying','idle']);
});

test('duplicate exposures and FEC/CRC retries cannot start or restart the reading cue',()=>{
  const h=harness();h.cue.displayed('message/ack/0');
  assert.equal(h.cue.observe({...firstSample,updated:false,duplicateExposure:true}),'idle');
  h.cue.observe(firstSample);
  for(const result of [
    {...firstSample,updated:false,duplicateExposure:true},
    {ok:false,stage:'fec',info:{size:64}},
    {ok:false,stage:'crc',info:{size:64}},
  ])assert.equal(h.cue.observe(result),'reading');
  h.cue.observe({stage:'detect'});const timer=[...h.timers.values()][0];
  h.cue.observe({...firstSample,updated:false,duplicateExposure:true});
  assert.equal(h.cue.state,'decaying');assert.equal([...h.timers.values()][0],timer);
});

test('packet recovery, a new display, explicit settling and reset remove the overlay immediately',()=>{
  for(const finish of [
    cue=>cue.observe({ok:true,info:{size:64}}),
    cue=>cue.observe({...firstSample,stage:'profile'}),
    cue=>cue.displayed('message/data/1'),
    cue=>cue.settle(),
    cue=>cue.reset(),
  ]){
    const h=harness();h.cue.displayed('message/data/0');h.cue.observe(firstSample);h.cue.observe({stage:'detect'});
    const stale=[...h.timers.values()][0].callback;
    finish(h.cue);assert.equal(h.cue.state,'idle');assert.equal(h.timers.size,0);
    h.cue.observe(firstSample);stale();assert.equal(h.cue.state,'reading');
    h.cue.displayed(h.cue.key);assert.equal(h.cue.state,'reading','unchanged display does not clear activity');
  }
});

test('default decay timers use the browser global receiver',t=>{
  let callback,cleared=0;
  t.mock.method(globalThis,'setTimeout',function(fn,ms){assert.equal(this,globalThis);assert.equal(ms,3000);callback=fn;return 7;});
  t.mock.method(globalThis,'clearTimeout',function(id){assert.equal(this,globalThis);assert.equal(id,7);cleared++;});
  const cue=new ReadingCue();cue.observe(firstSample);cue.observe({stage:'detect'});
  cue.reset();assert.equal(cleared,1);
  cue.observe(firstSample);callback();assert.equal(cue.state,'reading');
});

test('reading and decay leave an actual handoff notice visible for its full independent lifetime',()=>{
  const pending=new Map(),noticesChanged=[];let now=0,next=0;
  const timers={setTimer:(callback,ms)=>{pending.set(++next,{callback,due:now+ms});return next;},
    clearTimer:id=>pending.delete(id)};
  const advance=ms=>{
    now+=ms;
    for(const [id,timer] of [...pending])if(timer.due<=now){pending.delete(id);timer.callback();}
  };
  const notices=new TurnNotices({...timers,onChange:value=>noticesChanged.push(value)});
  const cue=new ReadingCue(timers);cue.displayed('message/data/0');
  notices.update({state:'awaiting-hello'},{});
  const notice=noticesChanged.at(-1);assert.equal(notice.title,'Ready to receive');
  advance(500);cue.observe(firstSample);
  advance(700);cue.observe({stage:'detect'});
  advance(1000);cue.observe(firstSample);
  advance(300);cue.observe({stage:'detect'});
  advance(499);
  assert.equal(now,2999);assert.equal(noticesChanged.at(-1),notice);
  assert.equal(noticesChanged.length,1,'reading, misses and reacquisition never dismiss the notice');
  advance(1);
  assert.equal(noticesChanged.at(-1),null,'the notice expires at its own original three-second deadline');
  assert.equal(cue.state,'decaying','the reading cue has its own later deadline');
  advance(2499);assert.equal(cue.state,'decaying');
  advance(1);assert.equal(cue.state,'idle');
  assert.equal(pending.size,0);assert.deepEqual(noticesChanged,[notice,null]);
});
