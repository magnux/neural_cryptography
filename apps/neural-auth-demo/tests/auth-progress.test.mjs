import test from 'node:test';
import assert from 'node:assert/strict';
import {exchangeStep} from '../dist/auth-progress.js';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {decodePacket} from '../dist/js/link/protocol.js';

test('both screens track the same logical step through chunks, held final ACKs, REQUESTs and slow proof preparation', async t=>{
  const verifier=new TurnChannel({initiator:true,automaticTurns:true});
  const prover=new TurnChannel({automaticTurns:true});
  t.after(()=>{verifier.close();prover.close();});
  const labels=['Hello','Identity','Challenge','Proof','Result'];
  const steps=()=>[exchangeStep('verifier',verifier.snapshot()),exchangeStep('prover',prover.snapshot())];
  const both=n=>steps().forEach(step=>assert.deepEqual(step,{number:n,total:5,label:labels[n-1]}));
  both(1);
  for(let n=1;n<=5;n++){
    const [sender,receiver]=n%2?[verifier,prover]:[prover,verifier];
    const pending=sender.enqueue(new Uint8Array(n===4?12_345:20),{expectReply:n<5});
    while(sender.snapshot().preparing) await new Promise(r=>setTimeout(r,0));
    both(n);
    let packets=0;
    while(sender.snapshot().outgoing){
      const wire=sender.current(),delivery=await receiver.receive(wire),ack=receiver.current();
      assert.equal(delivery.accepted,true);assert.equal(decodePacket(ack).kind,'ack');
      // The sender can miss the ACK and repeat its DATA. Queueing the next
      // response or displaying progress must not count another auth message.
      if(delivery.delivery){assert.ok((await receiver.receive(wire)).accepted);both(n);}
      assert.equal((await sender.receive(ack)).accepted,true);packets++;
      if(sender.snapshot().outgoing)both(n);
    }
    if(n===4)assert.ok(packets>2); // many proof chunks, one auth step
    else assert.equal(packets,1,'one small envelope consumes exactly one acknowledged packet');
    await pending;
    if(n<5){
      const request=sender.current();assert.equal(decodePacket(request).kind,'poll');
      assert.deepEqual(steps().map(s=>s.number),n%2?[n+1,n]:[n,n+1]);
      assert.equal((await receiver.receive(request)).accepted,true);
      both(n+1);
      assert.equal((await receiver.receive(request)).reason,'duplicate-request');
      both(n+1); // No reply is even queued yet (e.g. the proof is still computing).
    } else both(5);
  }
});
