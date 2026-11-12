import test from 'node:test';
import assert from 'node:assert/strict';
import {TurnNotices,NOTICE_DURATION_MS} from '../dist/js/neural-auth/notices.js';

function harness(){
  let next=0;const timers=new Map(),changes=[];
  const notices=new TurnNotices({onChange:value=>changes.push(value),
    setTimer:(callback,delay)=>{assert.equal(delay,3000);timers.set(++next,callback);return next;},
    clearTimer:id=>timers.delete(id)});
  return {notices,timers,changes,last:()=>changes.at(-1)};
}
test('round delivery notifies once; duplicate ACKs and progress cannot extend three seconds',()=>{
  const h=harness(),link={mode:'send',outgoing:{messageId:'identity'},receivedMessages:0};
  h.notices.update({state:'awaiting-identity'},link);
  assert.match(h.last().title,/Sending/);
  h.notices.update({state:'awaiting-identity'},{mode:'receive',lastTransfer:{state:'delivered',messageId:'identity'}});
  assert.equal(h.last().title,'ACK received · round delivered');
  const count=h.changes.length,timer=[...h.timers.values()][0];
  for(let i=0;i<10;i++)h.notices.update({state:'awaiting-identity',progress:i},{mode:'receive',lastTransfer:{state:'delivered',messageId:'identity'}});
  assert.equal(h.changes.length,count);assert.equal(h.timers.size,1);
  timer();assert.equal(h.last(),null);assert.equal(NOTICE_DURATION_MS,3000);
});
test('received rounds and queued replies preserve the final ACK instruction',()=>{
  const h=harness();h.notices.update({}, {mode:'receive',receivedMessages:0});
  h.notices.update({}, {mode:'ack',receivedMessages:1});
  assert.match(h.last().title,/Round received/);assert.match(h.last().message,/other camera/);
  h.notices.update({}, {mode:'ack',receivedMessages:1,canSendNext:true});
  assert.match(h.last().title,/next turn/);assert.match(h.last().message,/until the other device advances/);
  const count=h.changes.length;
  h.notices.update({}, {mode:'ack',receivedMessages:1,canSendNext:true});
  assert.equal(h.changes.length,count);
});
test('pairing notice waits for local identity delivery; processing progress is quiet',()=>{
  const h=harness(),auth={state:'pairing',canConfirm:true};
  h.notices.update(auth,{mode:'ack',receivedMessages:1});assert.doesNotMatch(h.last().title,/Compare/);
  h.notices.update(auth,{mode:'ack',receivedMessages:1,canSendNext:true});
  h.notices.update(auth,{mode:'send',receivedMessages:1,outgoing:{messageId:'identity'}});
  h.notices.update(auth,{mode:'receive',receivedMessages:1,lastTransfer:{state:'delivered',messageId:'identity'}});
  assert.equal(h.last().title,'Compare the pairing codes');
  h.notices.update({state:'proving'}, {mode:'ack',receivedMessages:2});
  h.notices.update({state:'verifying'}, {mode:'ack',receivedMessages:2});
  assert.equal(h.last().title,'Checking the proof');const count=h.changes.length;
  h.notices.update({state:'verifying',progress:99}, {mode:'ack',receivedMessages:2});
  assert.equal(h.changes.length,count);
});
test('a local verdict is not delivery, and peer-reported success keeps the ACK visible',()=>{
  const h=harness();h.notices.update({state:'verified',result:true},{mode:'ack'});
  assert.notEqual(h.last().title,'Authentication complete');
  h.notices.update({state:'verified',result:true,resultDelivered:true},{mode:'receive'});
  assert.equal(h.last().title,'Authentication complete');
  h.notices.reset();h.notices.update({state:'peer-accepted',result:true,peerReported:true},{mode:'ack'});
  assert.equal(h.last().title,'Verifier reports acceptance');assert.match(h.last().message,/final ACK/);
});
test('new transitions replace stale notices; reset prevents old timers hiding a new attempt',()=>{
  const h=harness();h.notices.update({state:'proving'},{});const old=[...h.timers.values()][0];
  h.notices.reset();h.notices.update({state:'awaiting-hello'},{});const current=h.last();
  old();assert.equal(h.last(),current);assert.equal(h.timers.size,1);
  h.notices.reset();assert.equal(h.timers.size,0);assert.equal(h.last(),null);
});
test('dismissing a notice retains packet history and completed-round pairing eligibility',()=>{
  const h=harness(),link={mode:'receive',lastTransfer:{state:'delivered',messageId:'identity'}};
  h.notices.update({state:'awaiting-identity'},{mode:'send',outgoing:{messageId:'identity'}});
  h.notices.update({state:'awaiting-identity'},link);const old=[...h.timers.values()][0];
  h.notices.dismiss();assert.equal(h.last(),null);assert.equal(h.timers.size,0);
  const changes=h.changes.length;
  h.notices.update({state:'awaiting-identity'},link);
  assert.equal(h.changes.length,changes,'the old delivery must not reannounce after dismissal');
  h.notices.update({state:'pairing',canConfirm:true},link);
  assert.equal(h.last().title,'Compare the pairing codes');
  old();assert.equal(h.last().title,'Compare the pairing codes');assert.equal(h.timers.size,1);
});

test('automatic handoffs announce the packet actually displayed and keep repeated updates quiet',()=>{
  const h=harness(),base={automaticTurns:true,receivedMessages:1};
  h.notices.update({}, {...base,mode:'ack',canSendNext:true});
  assert.equal(h.last().title,'Reply queued');assert.match(h.last().message,/automatically/);
  const outgoing={messageId:'reply',index:-1,chunkCount:2};
  h.notices.update({}, {...base,mode:'ack',outgoing,repeatingAck:true});
  assert.equal(h.last().title,'Finishing previous ACK');assert.doesNotMatch(h.last().message,/new message-start matrix is displayed/);
  const count=h.changes.length,timer=[...h.timers.values()][0];
  for(let i=0;i<3;i++)h.notices.update({}, {...base,mode:'ack',outgoing,repeatingAck:true});
  assert.equal(h.changes.length,count);assert.equal([...h.timers.values()][0],timer);
  h.notices.update({}, {...base,mode:'send',outgoing,repeatingAck:false});
  assert.equal(h.last().title,'Sending this round');
  h.notices.update({}, {...base,mode:'ack',outgoing,repeatingAck:true});
  assert.equal(h.last().title,'Repeating final ACK');
  h.notices.update({}, {...base,mode:'send',outgoing:{...outgoing,index:0},repeatingAck:false});
  assert.equal(h.last().title,'ACK received · sending chunk 1/2');
  for(const value of h.changes.filter(Boolean))assert.doesNotMatch(`${value.title} ${value.message}`,/tap|click|Send next turn/i);
});

test('automatic pairing displays its POC policy without manual confirmation prompts',()=>{
  const h=harness(),base={automaticTurns:true};
  h.notices.update({state:'awaiting-identity'},{...base,mode:'send',outgoing:{messageId:'identity',index:0}});
  const received={...base,mode:'receive',lastTransfer:{state:'delivered',messageId:'identity'}};
  h.notices.update({state:'awaiting-identity'},received);
  h.notices.update({state:'pairing',canConfirm:true},received);
  assert.equal(h.last().title,'Pairing continues automatically');assert.match(h.last().message,/without a manual comparison/);
  const count=h.changes.length;
  h.notices.update({state:'pairing',canConfirm:true},received);assert.equal(h.changes.length,count);
  h.notices.update({state:'awaiting-challenge',confirmed:true},received);
  assert.equal(h.changes.length,count);
  assert.doesNotMatch(h.last().title,/Compare/);
  assert.doesNotMatch(h.last().message,/tap|click|confirm only|confirm the codes/i);
});
