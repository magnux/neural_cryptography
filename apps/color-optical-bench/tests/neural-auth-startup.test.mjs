import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {setImmediate as settle, setTimeout as delay} from 'node:timers/promises';
import {AuthSession} from '../dist/js/auth-core/session.js';
import {CryptoOperations} from '../dist/js/crypto-lab/operations.js';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {TurnNotices, NOTICE_DURATION_MS} from '../dist/js/neural-auth/notices.js';

globalThis.crypto ??= webcrypto;

// Node's timers accept arbitrary receivers; Window timers do not. Use the
// default TurnNotices timers so extracting a native method is caught here.
function browserTimers(t){
  const pending=new Map(),receivers=[];let next=0,cleared=0;
  t.mock.method(globalThis,'setTimeout',function(callback,ms){
    receivers.push(this);
    if(this!==globalThis)throw new TypeError('Illegal invocation');
    assert.equal(ms,NOTICE_DURATION_MS);
    pending.set(++next,callback);return next;
  });
  t.mock.method(globalThis,'clearTimeout',function(id){
    receivers.push(this);
    if(this!==globalThis)throw new TypeError('Illegal invocation');
    cleared++;pending.delete(id);
  });
  return {pending,receivers,get cleared(){return cleared;}};
}

for(const role of ['verifier','prover']){
  test(`${role} startup survives browser timer receiver checks and retains its key`,async t=>{
    const timers=browserTimers(t),changes=[],operations=new CryptoOperations();
    let current={},cancellations=0;
    const notices=new TurnNotices({onChange:value=>changes.push(value)});
    const channel=new TurnChannel({initiator:role==='verifier',
      onChange:()=>notices.update(current,channel.snapshot())});
    const session=new AuthSession({role,deviceId:'11111111-1111-4111-8111-111111111111',repetitions:1,
      cryptoClient:{request:(type,payload,options={})=>operations.run(type,payload,options.onProgress),
        cancel:()=>{cancellations++;operations.clear();}},
      send:bytes=>channel.enqueue(bytes),
      onChange:snapshot=>{current=snapshot;notices.update(snapshot,channel.snapshot());}});
    t.after(()=>{session.close();channel.close();notices.reset();});

    await session.start();
    // HELLO framing includes asynchronous WebCrypto digests. Wait only until
    // first transmission (or a terminal failure), not for optical delivery.
    if(role==='verifier'){
      for(let i=0;i<200&&channel.snapshot().mode!=='send'&&session.snapshot().state!=='failed';i++)await delay(5);
      assert.equal(session.snapshot().state,'awaiting-identity');
      assert.equal(channel.snapshot().mode,'send');
      assert.ok(channel.current(),'the first HELLO packet is ready to display');
      assert.equal(changes.at(-1).title,'Sending this round');
      assert.ok(timers.cleared>0,'sending replaces the queued-message notice');
    }else{
      assert.equal(session.snapshot().state,'awaiting-hello');
      assert.equal(channel.snapshot().mode,'receive');
      assert.equal(changes.at(-1).title,'Ready to receive');
    }
    assert.equal(session.snapshot().error,null);
    assert.equal(cancellations,0,'notification rendering must not discard the ephemeral key');
    assert.ok(session.snapshot().localKeyId);
    assert.equal(timers.pending.size,1);
    const staleCallback=[...timers.pending.values()][0];
    notices.reset();
    assert.equal(timers.pending.size,0,'reset clears the native timer with its global receiver');
    assert.equal(changes.at(-1),null);
    const count=changes.length;staleCallback();
    assert.equal(changes.length,count,'a reset invalidates the old notice callback');
    assert.ok(timers.receivers.length>0);
    assert.ok(timers.receivers.every(receiver=>receiver===globalThis));
  });
}

test('startup failure is available before cancellation invalidates the worker',async()=>{
  const operations=new CryptoOperations(),failure='Initial transmission failed';let invalidation=null;
  const session=new AuthSession({role:'verifier',deviceId:'11111111-1111-4111-8111-111111111111',repetitions:1,
    cryptoClient:{request:(type,payload,options={})=>operations.run(type,payload,options.onProgress),
      cancel:()=>{
        operations.clear();
        invalidation={workerError:'Operation cancelled; the ephemeral key was discarded',auth:session.snapshot()};
      }},
    send:async()=>{throw new Error(failure);}});
  await session.start();await settle();
  assert.equal(invalidation?.auth.state,'failed');
  assert.equal(invalidation.auth.error,failure,'cleanup can preserve the primary error instead of displaying cancellation');
  assert.notEqual(invalidation.auth.error,invalidation.workerError);
});
