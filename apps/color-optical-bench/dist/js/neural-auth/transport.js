/** Stop-and-wait OLM1 transport for the Neural-ZKBoo camera demo.
 * A displayed packet stays current until its matching optical ACK arrives.
 * CRC/SHA-256 provide channel integrity, not peer authentication.
 */
import {encodePacket, decodePacket, sha256, randomId, hexToBytes, DATA_OVERHEAD, firstChunkCapacity, inlineChunkCount} from '../link/protocol.js';
import {chunkBytesForSize} from '../neural-qr/packet.js';

export const TURN_LIMITS = Object.freeze({chunkBytes: chunkBytesForSize(128), defaultChunkBytes: 1200, maxMessageBytes: 2 * 1024 * 1024,
  maxMessages: 32, timeoutMs: 10 * 60 * 1000, maxTimeoutMs: 60 * 60 * 1000});
const terminal = new Set(['delivered', 'failed', 'timed-out', 'cancelled']);
const same = (a,b) => a.sessionId===b.sessionId && a.messageId===b.messageId && a.messageType===b.messageType;
const fields = p => ({sessionId:p.sessionId,messageId:p.messageId,messageType:p.messageType});
const equal = (a,b) => a.length===b.length && a.every((value,i)=>value===b[i]);
const metadataEqual = (a,b) => same(a,b) && ['totalBytes','chunkBytes','chunkCount','windowSize','digest'].every(key=>a[key]===b[key]);
function integer(value,min,max,label) {if(!Number.isSafeInteger(value)||value<min||value>max)throw Error(`Invalid ${label}.`);return value;}
function clock(now) {if(!Number.isFinite(now)||now<0)throw Error('Invalid transport clock.');return now;}
function packet(bytes) {
  if(!(bytes instanceof Uint8Array)||bytes.length>TURN_LIMITS.chunkBytes+DATA_OVERHEAD)throw Error('Invalid turn packet length.');
  return decodePacket(bytes);
}
function ack(meta,index,status) {
  return encodePacket({...fields(meta),kind:'ack',base:index,bitmap:1,status,digest:meta.digest});
}
// START carries chunk zero. Its digest and length fields leave less room than DATA.
function offset(meta,index) {return index===0?0:firstChunkCapacity(meta.chunkBytes)+(index-1)*meta.chunkBytes;}
function expectedBytes(meta,index) {
  return Math.min(index===0?firstChunkCapacity(meta.chunkBytes):meta.chunkBytes,meta.totalBytes-offset(meta,index));
}

export async function createTurnSender(payload,options={}) {
  if(!(payload instanceof Uint8Array)||payload.length>TURN_LIMITS.maxMessageBytes)throw Error('A turn message must be Uint8Array of at most 2 MiB.');
  const bytes=payload.slice(), chunkBytes=integer(options.chunkBytes??TURN_LIMITS.defaultChunkBytes,1,TURN_LIMITS.chunkBytes,'chunk size');
  const meta={sessionId:options.sessionId??randomId(16),messageId:options.messageId??randomId(8),messageType:'raw',
    totalBytes:bytes.length,chunkBytes,chunkCount:inlineChunkCount(bytes.length,chunkBytes),windowSize:1,digest:await sha256(bytes)};
  hexToBytes(meta.sessionId,16);hexToBytes(meta.messageId,8);
  return new TurnSender(bytes,meta,options);
}

class TurnSender {
  constructor(bytes,meta,options) {
    this.bytes=bytes;this.meta=meta;this.timeoutMs=integer(options.timeoutMs??TURN_LIMITS.timeoutMs,1,TURN_LIMITS.maxTimeoutMs,'timeout');
    this.clock=clock(options.now??0);this.deadline=this.clock+this.timeoutMs;
    this.state='awaiting-ack';this.index=0;this.shown=false;this.ackedChunks=0;this.ackedBytes=0;this.error=null;
    this.current=encodePacket({...meta,kind:'start',data:bytes.subarray(0,expectedBytes(meta,0))});
  }
  tick(now) {
    this.clock=Math.max(this.clock,clock(now));
    if(!terminal.has(this.state)&&this.clock>=this.deadline)this.fail('timed-out','Transfer deadline reached.');
    return this.snapshot();
  }
  fail(state,error) {this.state=state;this.error=error;this.bytes=null;this.current=null;this.shown=false;}
  next(now) {this.tick(now);if(!this.current)return null;this.shown=true;return this.current.slice();}
  receive(bytes,now) {
    this.tick(now);
    if(terminal.has(this.state))return {accepted:false,reason:'terminal'};
    let p;try{p=packet(bytes);}catch{return {accepted:false,reason:'malformed'};}
    if(!same(p,this.meta))return {accepted:false,reason:'foreign'};
    if(p.kind!=='ack'||p.digest!==this.meta.digest)return {accepted:false,reason:'unexpected'};
    if(!this.shown)return {accepted:false,reason:'unsent-ack'};
    const final=this.index===this.meta.chunkCount-1;
    if(p.base!==this.index||p.bitmap!==1||p.status!==(final?'complete':'progress'))return {accepted:false,reason:'stale-ack'};
    this.ackedChunks++;this.ackedBytes+=expectedBytes(this.meta,this.index);
    if(final) {this.fail('delivered',null);return {accepted:true,delivered:true};}
    this.index++;
    const start=offset(this.meta,this.index);
    this.current=encodePacket({...fields(this.meta),kind:'data',index:this.index,
      data:this.bytes.subarray(start,start+expectedBytes(this.meta,this.index))});
    this.shown=false;
    return {accepted:true,delivered:false};
  }
  cancel() {if(!terminal.has(this.state))this.fail('cancelled','Cancelled locally.');}
  snapshot() {
    return {...this.meta,state:this.state,index:this.index,ackedChunks:this.ackedChunks,ackedBytes:this.ackedBytes,
      progress:this.state==='delivered'?1:this.meta.totalBytes?this.ackedBytes/this.meta.totalBytes:0,error:this.error,
      bufferedBytes:this.bytes?.byteLength??0,deadline:this.deadline};
  }
}

/** One in-flight message and a bounded receipt cache. Call reset for a fresh
 * authentication run. The channel retains final ACKs until a matching REQUEST;
 * duplicate packets reconstruct their receipt without regressing the current ACK.
 */
export class TurnReceiver {
  constructor(options={}) {
    this.timeoutMs=integer(options.timeoutMs??TURN_LIMITS.timeoutMs,1,TURN_LIMITS.maxTimeoutMs,'timeout');
    this.maxMessageBytes=integer(options.maxMessageBytes??TURN_LIMITS.maxMessageBytes,0,TURN_LIMITS.maxMessageBytes,'message limit');
    this.expectedSessionId=options.sessionId??null;if(this.expectedSessionId!==null)hexToBytes(this.expectedSessionId,16);
    this.reset();
  }
  reset() {this.epoch=(this.epoch??0)+1;this.sessionId=this.expectedSessionId;this.clock=0;this.active=null;this.receipts=[];this.currentAck=null;this.last={state:'idle'};this.verifying=false;}
  ack() {return this.currentAck?.slice()??null;}
  tick(now) {
    this.clock=Math.max(this.clock,clock(now));
    if(this.active&&this.clock>=this.active.deadline)this.finish('timed-out','Transfer deadline reached.');
    return this.snapshot();
  }
  finish(state,error=null) {
    const a=this.active;if(!a)return;
    this.last={...a.meta,state,receivedChunks:a.next,receivedBytes:a.receivedBytes,error};
    // Receipts recognize conflicting START replays without retaining payload bytes.
    this.receipts.push({meta:a.meta,state,firstDigest:a.firstDigest??null,
      ack:state==='delivered'?this.currentAck:null});
    this.active=null;this.verifying=false;if(state!=='delivered')this.currentAck=null;
  }
  async receive(bytes,now) {
    this.tick(now);
    let p;try{p=packet(bytes);}catch{return {accepted:false,reason:'malformed',ack:null};}
    if(this.sessionId!==null&&p.sessionId!==this.sessionId)return {accepted:false,reason:'foreign-session',ack:null};
    if(p.kind!=='start'&&p.kind!=='data')return {accepted:false,reason:'unexpected',ack:null};
    if(p.kind==='start'&&!(p.data instanceof Uint8Array))return {accepted:false,reason:'unsupported-message',ack:null};
    if(p.kind==='data'&&p.index===0)return {accepted:false,reason:'invalid-chunk',ack:null};
    if(this.active&&!same(this.active.meta,p))return {accepted:false,reason:'foreign',ack:null};
    if(this.verifying)return {accepted:false,reason:'verifying',ack:null};
    const previous=this.receipts.find(row=>same(row.meta,p));
    if(previous) {
      if(previous.state!=='delivered')return {accepted:false,reason:'terminal',ack:null};
      if(p.kind==='start') {
        if(!metadataEqual(previous.meta,p))return {accepted:false,reason:'conflicting-start',ack:null};
        const epoch=this.epoch;
        let digest;try{digest=await sha256(p.data);}catch{return {accepted:false,reason:'digest-error',ack:null};}
        if(epoch!==this.epoch)return {accepted:false,reason:'superseded',ack:null};
        if(digest!==previous.firstDigest)return {accepted:false,reason:'conflicting-start',ack:null};
      }
      if(p.kind==='data'&&(p.index>=previous.meta.chunkCount||p.data.length!==expectedBytes(previous.meta,p.index)))return {accepted:false,reason:'invalid-chunk',ack:null};
      // Answer from the receipt without replacing the newest displayed ACK.
      return {accepted:true,reason:'already-delivered',ack:previous.ack.slice()};
    }
    if(p.kind==='start') {
      if(p.windowSize!==1||p.chunkBytes>TURN_LIMITS.chunkBytes||p.totalBytes>this.maxMessageBytes||p.messageType!=='raw')return {accepted:false,reason:'unsupported-message',ack:null};
      if(this.active) {
        if(!metadataEqual(this.active.meta,p)||!equal(this.active.bytes.subarray(0,expectedBytes(p,0)),p.data))return {accepted:false,reason:'conflicting-start',ack:null};
        // A late START must not reset progress or replace a more recent ACK.
        return {accepted:true,reason:'duplicate-start',ack:this.ack()};
      }
      if(this.receipts.length>=TURN_LIMITS.maxMessages)return {accepted:false,reason:'message-count-limit',ack:null};
      if(p.data.length!==expectedBytes(p,0)||p.chunkCount!==inlineChunkCount(p.totalBytes,p.chunkBytes))return {accepted:false,reason:'invalid-chunk',ack:null};
      this.sessionId??=p.sessionId;
      const meta={...fields(p),chunkBytes:p.chunkBytes,totalBytes:p.totalBytes,chunkCount:p.chunkCount,windowSize:p.windowSize,digest:p.digest};
      this.active={meta,bytes:new Uint8Array(p.totalBytes),next:0,receivedBytes:0,deadline:this.clock+this.timeoutMs};
    }
    const a=this.active;
    if(!a)return {accepted:false,reason:'no-active-message',ack:null};
    const index=p.kind==='start'?0:p.index;
    if(index>=a.meta.chunkCount||p.data.length!==expectedBytes(a.meta,index))return {accepted:false,reason:'invalid-chunk',ack:null};
    const start=offset(a.meta,index);
    if(index<a.next) {
      if(!equal(a.bytes.subarray(start,start+p.data.length),p.data))return {accepted:false,reason:'conflicting-chunk',ack:null};
      // Reply for exactly the duplicate chunk, never advance sender on a guess.
      return {accepted:true,reason:'duplicate',ack:ack(a.meta,index,'progress')};
    }
    if(index!==a.next)return {accepted:false,reason:'out-of-order',ack:null};
    a.bytes.set(p.data,start);a.next++;a.receivedBytes+=p.data.length;
    if(a.next<a.meta.chunkCount) {this.currentAck=ack(a.meta,index,'progress');return {accepted:true,ack:this.ack()};}
    const epoch=this.epoch;this.verifying=true;
    let digest,firstDigest;
    try {
      // For singleton messages the full digest already identifies chunk zero.
      [digest,firstDigest]=await Promise.all([sha256(a.bytes),a.next===1?null:sha256(a.bytes.subarray(0,expectedBytes(a.meta,0)))]);
    }catch {
      if(this.active===a&&this.epoch===epoch)this.finish('failed','SHA-256 unavailable.');
      return {accepted:false,reason:'digest-error',ack:null};
    }
    if(this.active!==a||this.epoch!==epoch)return {accepted:false,reason:'superseded',ack:null};
    if(digest!==a.meta.digest) {this.finish('failed','Full-message SHA-256 mismatch.');return {accepted:false,reason:'digest-mismatch',ack:null};}
    a.firstDigest=firstDigest??digest;this.currentAck=ack(a.meta,index,'complete');
    const delivery={...fields(a.meta),digest,bytes:a.bytes};
    this.finish('delivered');
    return {accepted:true,ack:this.ack(),delivery};
  }
  cancel() {this.finish('cancelled','Cancelled locally.');this.currentAck=null;}
  snapshot() {
    const a=this.active,s=a?{...a.meta,state:this.verifying?'verifying':'receiving',receivedChunks:a.next,receivedBytes:a.receivedBytes,error:null}:this.last;
    return {...s,totalBytes:s.totalBytes??0,chunkCount:s.chunkCount??0,receivedChunks:s.receivedChunks??0,receivedBytes:s.receivedBytes??0,
      progress:s.state==='delivered'?1:s.totalBytes?(s.receivedBytes??0)/s.totalBytes:0,
      bufferedBytes:a?.bytes.byteLength??0,completedMessages:this.receipts.filter(row=>row.state==='delivered').length,receiptCount:this.receipts.length};
  }
}
