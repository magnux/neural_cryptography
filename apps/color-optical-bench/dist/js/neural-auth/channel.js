/** UI-independent turn scheduling with recoverable automatic handoffs. */
import {createTurnSender,TurnReceiver} from './transport.js';
import {decodePacket,randomId} from '../link/protocol.js';
import {chunkBytesForSize} from '../neural-qr/packet.js';

export class TurnChannel {
  constructor({initiator=false,outgoingSize=64,automaticTurns=false,handoffHoldMs=3000,
    now=()=>performance.now(),onChange=()=>{},onMessage=()=>{},onError=()=>{}}={}) {
    chunkBytesForSize(outgoingSize);this.outgoingSize=outgoingSize;
    if(typeof automaticTurns!=='boolean'||!Number.isSafeInteger(handoffHoldMs)||handoffHoldMs<100||handoffHoldMs>120000)
      throw Error('Invalid automatic handoff settings.');
    this.automaticTurns=automaticTurns;this.handoffHoldMs=handoffHoldMs;this.handoffSwitchAt=null;
    this.now=now;this.onChange=onChange;this.onMessage=onMessage;this.onError=onError;
    this.sessionId=initiator?randomId(16):null;
    this.receiver=new TurnReceiver({sessionId:this.sessionId,maxMessageBytes:524288});
    this.outgoing=null;this.pending=null;this.stickyAck=null;this.lastTransfer=null;
    this.handoffAck=null;this.repeatingAck=false;
    this.started=false;this.closed=false;this.epoch=0;this.preparing=false;this.receivedMessages=0;
    this.initialAuto=initiator;this.lastNow=0;
  }
  clock(){this.lastNow=Math.max(this.lastNow,this.now());return this.lastNow;}
  notify(){this.onChange(this.snapshot());}
  snapshot(){return {mode:this.repeatingAck?'ack':this.outgoing?'send':this.stickyAck?'ack':'receive',
    queued:!!this.pending,preparing:this.preparing,canSendNext:!!this.pending&&!this.outgoing&&!this.preparing,
    repeatingAck:this.repeatingAck,canRepeatAck:!!this.handoffAck&&this.outgoing?.sender.snapshot().state==='awaiting-start',
    automaticTurns:this.automaticTurns,handoffAuto:this.automaticTurns&&!!this.handoffAck,
    handoffHoldMs:this.handoffHoldMs,handoffSwitchAt:this.handoffSwitchAt,
    receivedMessages:this.receivedMessages,sessionId:this.sessionId,outgoingSize:this.outgoingSize,
    outgoing:this.outgoing?.sender.snapshot()??null,incoming:this.receiver.snapshot(),lastTransfer:this.lastTransfer,closed:this.closed};}
  setOutgoingSize(size){
    chunkBytesForSize(size);
    if(this.closed)throw Error('Optical channel is closed.');
    if(this.outgoing||this.preparing)throw Error('Change outgoing geometry between messages, not during a transfer.');
    this.outgoingSize=size;this.notify();
  }
  enqueue(bytes){
    if(this.closed)return Promise.reject(Error('Optical channel is closed.'));
    if(this.pending||this.outgoing||this.preparing)return Promise.reject(Error('Another authentication message is pending.'));
    if(!(bytes instanceof Uint8Array)||!bytes.length||bytes.length>524288)return Promise.reject(Error('Invalid authentication message size.'));
    const promise=new Promise((resolve,reject)=>{this.pending={bytes:bytes.slice(),resolve,reject};});
    this.notify();
    if(this.automaticTurns||this.initialAuto&&!this.started){
      this.initialAuto=false;void this.nextTurn().catch(error=>this.fail(error));
    }
    return promise;
  }
  async nextTurn(){
    if(this.closed||!this.pending||this.outgoing||this.preparing)return false;
    if(!this.sessionId)throw Error('Receive the verifier identity before sending.');
    const epoch=this.epoch,pending=this.pending;this.preparing=true;this.notify();
    try{
      const sender=await createTurnSender(pending.bytes,{sessionId:this.sessionId,now:this.clock(),chunkBytes:chunkBytesForSize(this.outgoingSize)});
      if(this.closed||epoch!==this.epoch)return false;
      // Keep the final receipt until the peer acknowledges the new START.
      // Automatic handoffs display that ACK first, then alternate it with the
      // exact same START. A missed final ACK can therefore recover without a
      // manual button, a new message ID, or an assumed delivery.
      this.handoffAck=this.stickyAck&&decodePacket(this.stickyAck).status==='complete'?this.stickyAck.slice():null;
      this.repeatingAck=this.automaticTurns&&!!this.handoffAck;
      this.handoffSwitchAt=this.repeatingAck?this.clock()+this.handoffHoldMs:null;
      this.outgoing={...pending,sender};this.pending=null;this.stickyAck=null;this.started=true;
      // Only current() marks START as displayed; an ACK for an unseen START
      // must not end the retained-ACK slot or advance the sender.
      return true;
    }catch(error){if(epoch===this.epoch)this.fail(error);return false;}
    finally{if(epoch===this.epoch){this.preparing=false;this.notify();}}
  }
  repeatPreviousAck(){
    if(this.closed||!this.snapshot().canRepeatAck)return false;
    if(!this.repeatingAck){
      this.repeatingAck=true;if(this.automaticTurns)this.handoffSwitchAt=this.clock()+this.handoffHoldMs;
      this.notify();
    }return true;
  }
  resumeMessage(){
    if(this.closed||!this.repeatingAck)return false;
    this.repeatingAck=false;if(this.automaticTurns)this.handoffSwitchAt=this.clock()+this.handoffHoldMs;
    this.notify();return true;
  }
  current(){
    if(this.closed)return null;
    if(this.repeatingAck)return this.handoffAck.slice();
    return this.outgoing?.sender.next(this.clock())??this.stickyAck;
  }
  async receive(bytes){
    if(this.closed)return {accepted:false,reason:'closed'};
    const epoch=this.epoch;let packet;
    try{packet=decodePacket(bytes);}catch{return {accepted:false,reason:'malformed'};}
    if(this.outgoing){
      if(packet.kind!=='ack')return {accepted:false,reason:'sending'};
      const result=this.outgoing.sender.receive(bytes,this.clock());
      if(result.accepted){this.handoffAck=null;this.repeatingAck=false;this.handoffSwitchAt=null;}
      if(result.delivered){
        const current=this.outgoing;this.lastTransfer=current.sender.snapshot();this.outgoing=null;
        current.resolve();
      }
      if(result.accepted)this.notify();return result;
    }
    if(packet.kind==='ack')return {accepted:false,reason:'not-sending'};
    const result=await this.receiver.receive(bytes,this.clock());
    if(this.closed||epoch!==this.epoch)return {accepted:false,reason:'closed'};
    if(result.accepted){
      if(!this.sessionId){this.sessionId=packet.sessionId;this.receiver.sessionId=this.sessionId;}
      if(result.ack)this.stickyAck=result.ack;
      if(result.delivery){
        this.receivedMessages++;this.lastTransfer=this.receiver.snapshot();
        // Proof computation must not block camera handling and ACK display.
        Promise.resolve().then(()=>{if(!this.closed&&epoch===this.epoch)return this.onMessage(result.delivery.bytes);})
          .catch(error=>{if(!this.closed&&epoch===this.epoch)this.fail(error);});
      }
      this.notify();
    }
    return result;
  }
  tick(){
    if(this.closed)return;
    const now=this.clock(),outgoing=this.outgoing?.sender.tick(now),incoming=this.receiver.tick(now);
    if(['timed-out','failed'].includes(outgoing?.state)||['timed-out','failed'].includes(incoming.state)){
      this.fail(Error(outgoing?.error||incoming.error));return;
    }
    if(this.automaticTurns&&this.handoffAck&&outgoing?.state==='awaiting-start'&&now>=this.handoffSwitchAt){
      this.repeatingAck=!this.repeatingAck;this.handoffSwitchAt=now+this.handoffHoldMs;
      this.notify();
    }
  }
  fail(error){this.close(error);this.onError(error);}
  close(error=new Error('Optical channel stopped.')){
    if(this.closed)return;this.closed=true;this.epoch++;
    this.outgoing?.sender.cancel();this.receiver.cancel();
    this.outgoing?.reject(error);this.pending?.reject(error);
    this.outgoing=null;this.pending=null;this.stickyAck=null;this.handoffAck=null;this.repeatingAck=false;this.handoffSwitchAt=null;this.preparing=false;this.notify();
  }
}
