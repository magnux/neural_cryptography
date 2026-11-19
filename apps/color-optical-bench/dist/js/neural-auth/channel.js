/** UI-independent, request-driven optical turn scheduling. */
import {createTurnSender,TurnReceiver,TURN_LIMITS} from './transport.js';
import {decodePacket,encodePacket,randomId} from '../link/protocol.js';
import {chunkBytesForSize} from '../neural-qr/packet.js';

const sameTurn=(a,b)=>a.sessionId===b.sessionId&&a.messageId===b.messageId&&a.messageType===b.messageType;

export class TurnChannel {
  constructor({initiator=false,outgoingSize=64,automaticTurns=false,
    now=()=>performance.now(),onChange=()=>{},onMessage=()=>{},onError=()=>{}}={}) {
    chunkBytesForSize(outgoingSize);this.outgoingSize=outgoingSize;
    if(typeof automaticTurns!=='boolean')throw Error('Invalid automatic handoff settings.');
    this.automaticTurns=automaticTurns;
    this.now=now;this.onChange=onChange;this.onMessage=onMessage;this.onError=onError;
    this.sessionId=initiator?randomId(16):null;
    this.receiver=new TurnReceiver({sessionId:this.sessionId,maxMessageBytes:524288});
    this.outgoing=null;this.pending=null;this.stickyAck=null;this.lastTransfer=null;
    this.receivedTurn=null;this.replyRequest=null;this.handoffDeadline=null;
    this.started=false;this.closed=false;this.epoch=0;this.preparing=false;this.receivedMessages=0;
    this.initialAuto=initiator;this.lastNow=0;this.sessionDeadline=this.clock()+TURN_LIMITS.maxTimeoutMs;
  }
  clock(){
    const now=this.now();if(!Number.isFinite(now)||now<0)throw Error('Invalid transport clock.');
    this.lastNow=Math.max(this.lastNow,now);return this.lastNow;
  }
  notify(){this.onChange(this.snapshot());}
  mayStart(){return this.initialAuto&&!this.started&&!this.receivedMessages||!!this.receivedTurn?.requested&&!this.receivedTurn.used;}
  snapshot(){
    const waitingForRequest=!!this.receivedTurn&&!this.receivedTurn.requested&&!this.receivedTurn.used;
    return {mode:this.outgoing?'send':this.replyRequest?'request':this.stickyAck?'ack':'receive',
      expected:this.outgoing?'ack':waitingForRequest||this.receivedTurn?.requested&&!this.receivedTurn.used?'request':'data',
      queued:!!this.pending,preparing:this.preparing,canSendNext:!!this.pending&&!this.outgoing&&!this.preparing&&this.mayStart(),
      automaticTurns:this.automaticTurns,waitingForRequest,requestedReply:!!this.replyRequest,
      receivedMessages:this.receivedMessages,sessionId:this.sessionId,outgoingSize:this.outgoingSize,
      outgoing:this.outgoing?.sender.snapshot()??null,incoming:this.receiver.snapshot(),lastTransfer:this.lastTransfer,closed:this.closed};
  }
  setOutgoingSize(size){
    chunkBytesForSize(size);
    if(this.closed)throw Error('Optical channel is closed.');
    if(this.outgoing||this.preparing)throw Error('Change outgoing geometry between messages, not during a transfer.');
    this.outgoingSize=size;this.notify();
  }
  enqueue(bytes,{expectReply=true}={}){
    if(this.closed)return Promise.reject(Error('Optical channel is closed.'));
    if(this.pending||this.outgoing||this.preparing)return Promise.reject(Error('Another authentication message is pending.'));
    if(!(bytes instanceof Uint8Array)||!bytes.length||bytes.length>524288)return Promise.reject(Error('Invalid authentication message size.'));
    if(typeof expectReply!=='boolean')return Promise.reject(Error('Invalid reply expectation.'));
    const promise=new Promise((resolve,reject)=>{this.pending={bytes:bytes.slice(),expectReply,resolve,reject};});
    this.notify();
    if(this.automaticTurns||this.initialAuto&&!this.started)void this.nextTurn().catch(error=>this.fail(error));
    return promise;
  }
  async nextTurn(){
    this.tick();
    if(this.closed||!this.pending||this.outgoing||this.preparing||this.replyRequest||!this.mayStart())return false;
    if(!this.sessionId)throw Error('Receive the verifier identity before sending.');
    const epoch=this.epoch,pending=this.pending;this.preparing=true;this.notify();
    try{
      const sender=await createTurnSender(pending.bytes,{sessionId:this.sessionId,now:this.clock(),chunkBytes:chunkBytesForSize(this.outgoingSize)});
      if(this.closed||epoch!==this.epoch)return false;
      this.outgoing={...pending,sender};this.pending=null;this.stickyAck=null;this.started=true;this.initialAuto=false;
      if(this.receivedTurn)this.receivedTurn.used=true;
      this.handoffDeadline=null;
      // current() marks this exact START as displayed. It then remains current
      // until chunk zero is acknowledged, including the final ACK for a singleton.
      return true;
    }catch(error){if(epoch===this.epoch)this.fail(error);return false;}
    finally{if(epoch===this.epoch){this.preparing=false;this.notify();}}
  }
  current(){
    if(this.closed)return null;
    return this.outgoing?.sender.next(this.clock())??this.replyRequest?.slice()??this.stickyAck?.slice()??null;
  }
  expectedAck(){
    if(this.closed||!this.outgoing)return null;
    const now=this.clock();
    return now<this.sessionDeadline?this.outgoing.sender.expectedAck(now):null;
  }
  receiveRequest(packet){
    const previous=this.receivedTurn;
    if(packet.sessionId!==this.sessionId)return {accepted:false,reason:'foreign-session'};
    if(!previous||!sameTurn(packet,previous)||packet.base!==previous.base)return {accepted:false,reason:'stale-request'};
    if(previous.requested)return {accepted:true,reason:'duplicate-request'};
    previous.requested=true;this.stickyAck=null;
    // Remember the grant even when proof computation has not queued its reply.
    // Duplicate requests never extend the deadline or restart sender preparation.
    if(this.automaticTurns)void this.nextTurn().catch(error=>this.fail(error));
    this.notify();return {accepted:true,reason:'requested'};
  }
  async receive(bytes){
    this.tick();
    if(this.closed)return {accepted:false,reason:'closed'};
    const epoch=this.epoch;let packet;
    try{packet=decodePacket(bytes);}catch{return {accepted:false,reason:'malformed'};}
    if(packet.kind==='poll')return this.receiveRequest(packet);
    if(this.outgoing){
      if(packet.kind!=='ack')return {accepted:false,reason:'sending'};
      const result=this.outgoing.sender.receive(bytes,this.clock());
      if(result.delivered){
        const current=this.outgoing;this.lastTransfer=current.sender.snapshot();this.outgoing=null;
        if(current.expectReply){
          const {sessionId,messageId,messageType,chunkCount}=this.lastTransfer;
          this.replyRequest=encodePacket({kind:'poll',sessionId,messageId,messageType,base:chunkCount-1});
          this.handoffDeadline=this.clock()+TURN_LIMITS.timeoutMs;
        }
        current.resolve();
      }
      if(result.accepted)this.notify();return result;
    }
    if(packet.kind==='ack')return {accepted:false,reason:'not-sending'};
    if(packet.kind==='start'&&this.receivedTurn&&!this.receivedTurn.used&&!sameTurn(packet,this.receivedTurn))
      return {accepted:false,reason:'waiting-for-request'};
    const result=await this.receiver.receive(bytes,this.clock());
    if(this.closed||epoch!==this.epoch)return {accepted:false,reason:'closed'};
    if(result.accepted){
      if(!this.sessionId){this.sessionId=packet.sessionId;this.receiver.sessionId=this.sessionId;}
      if(result.reason!=='already-delivered'){
        // Only a fresh START satisfies our request. Old START/DATA receipts must
        // never replace a current REQUEST or a newer receipt with an old ACK.
        if(packet.kind==='start'&&result.reason!=='duplicate-start'){
          this.replyRequest=null;this.handoffDeadline=null;
        }
        // TurnReceiver keeps the newest ACK even when it returns an ACK for an
        // older duplicate DATA packet, so a delayed camera frame cannot regress it.
        if(result.ack)this.stickyAck=this.receiver.ack();
      }
      if(result.delivery){
        this.receivedMessages++;this.lastTransfer=this.receiver.snapshot();
        this.receivedTurn={sessionId:packet.sessionId,messageId:packet.messageId,messageType:packet.messageType,
          base:this.lastTransfer.chunkCount-1,requested:false,used:false};
        this.handoffDeadline=this.clock()+TURN_LIMITS.timeoutMs;
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
    if(now>=this.sessionDeadline||this.handoffDeadline!==null&&now>=this.handoffDeadline)
      this.fail(Error(now>=this.sessionDeadline?'Session deadline reached.':'Turn handoff deadline reached.'));
  }
  fail(error){this.close(error);this.onError(error);}
  close(error=new Error('Optical channel stopped.')){
    if(this.closed)return;this.closed=true;this.epoch++;
    this.outgoing?.sender.cancel();this.receiver.cancel();
    this.outgoing?.reject(error);this.pending?.reject(error);
    this.outgoing=null;this.pending=null;this.stickyAck=null;this.replyRequest=null;this.receivedTurn=null;this.handoffDeadline=null;this.preparing=false;this.notify();
  }
}
