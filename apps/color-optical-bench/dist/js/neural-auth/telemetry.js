/** Local optical timings only. No camera frames, payloads or private keys. */
import {decodePacket} from '../link/protocol.js';

const duplicateReasons=new Set(['duplicate','duplicate-start','already-delivered','duplicate-request']);
const rounded=value=>value===null?null:Math.round(value*10)/10;
const messageKey=p=>`${p.sessionId}/${p.messageId}/${p.messageType}`;
// In this transport START is chunk zero, including a complete singleton.
const packetKind=p=>p.kind==='ack'?(p.base===0?'start':'data'):p.kind;
const packetIndex=p=>p.kind==='start'?0:p.kind==='ack'?p.base:p.index;
const rowKey=(direction,p)=>`${direction}/${messageKey(p)}/${packetKind(p)}/${packetIndex(p)??''}`;
const parse=bytes=>{try{return decodePacket(bytes);}catch{return null;}};
const elapsed=(end,start)=>end!==null&&start!==null&&end>=start?end-start:null;

// The input to this tracker is always the canonical logical ACK. Grid identifies
// its actual optical representation: a 32-byte digest on G16, the OLM packet on G32.
function validGrid(kind,grid){
  if(kind==='ack')return grid===16||grid===32;
  if(kind==='poll')return grid===32;
  return (kind==='data'||kind==='start')&&[32,64,128].includes(grid);
}
const ackBytes=(grid,bytes)=>grid===16?32:bytes.length;

function summarize(values){
  const sorted=values.filter(value=>value!==null).sort((a,b)=>a-b),count=sorted.length;
  if(!count)return {count:0,minMs:null,medianMs:null,p95Ms:null,meanMs:null,maxMs:null};
  const mid=Math.floor(count/2);
  return {count,minMs:rounded(sorted[0]),medianMs:rounded(count%2?sorted[mid]:(sorted[mid-1]+sorted[mid])/2),
    p95Ms:rounded(sorted[Math.ceil(.95*count)-1]),meanMs:rounded(sorted.reduce((a,b)=>a+b,0)/count),maxMs:rounded(sorted.at(-1))};
}

export class PacketTelemetry {
  constructor({now=()=>performance.now(),originMs=now(),maxRows=4096}={}){
    if(typeof now!=='function'||!Number.isFinite(originMs)||!Number.isSafeInteger(maxRows)||maxRows<1||maxRows>8192)
      throw Error('Invalid optical telemetry settings.');
    this.now=now;this.originMs=originMs;this.maxRows=maxRows;
    this.rows=new Map();this.requests=new Map();this.messages=new Map();this.droppedEvents=0;this.minimumRoundTripMs=null;
  }
  stamp(atMs){
    const absolute=atMs??this.now();
    return Number.isFinite(absolute)&&absolute>=this.originMs?absolute-this.originMs:null;
  }
  meta(p){
    if(p.kind!=='start'||this.messages.has(messageKey(p)))return;
    if(this.messages.size<this.maxRows)this.messages.set(messageKey(p),{digest:p.digest,chunkCount:p.chunkCount});
  }
  row(direction,p){
    const key=rowKey(direction,p),existing=this.rows.get(key);
    if(existing)return existing;
    if(this.rows.size+this.requests.size>=this.maxRows){this.droppedEvents++;return null;}
    const kind=packetKind(p),index=packetIndex(p),meta=this.messages.get(messageKey(p));
    const row={direction,sessionId:p.sessionId,messageId:p.messageId,messageType:p.messageType,kind,index,
      grid:null,packetBytes:null,payloadBytes:null,chunkCount:meta?.chunkCount??null,
      displayedMs:null,lastDisplayedMs:null,displayCount:0,receivedMs:null,
      ack:{status:null,grid:null,packetBytes:null,displayedMs:null,lastDisplayedMs:null,displayCount:0,receivedMs:null},
      digest:p.digest??meta?.digest??null};
    this.rows.set(key,row);return row;
  }
  request(direction,p,bytes,grid){
    const key=`${direction}/${messageKey(p)}/${p.base}`,existing=this.requests.get(key);
    if(existing)return existing;
    if(this.rows.size+this.requests.size>=this.maxRows){this.droppedEvents++;return null;}
    const row={direction,kind:'request',sessionId:p.sessionId,previousMessageId:p.messageId,previousFinalIndex:p.base,
      grid,packetBytes:bytes.length,displayedMs:null,lastDisplayedMs:null,displayCount:0,receivedMs:null};
    this.requests.set(key,row);return row;
  }
  describe(row,p,bytes,grid){
    const meta=this.messages.get(messageKey(p));
    row.grid=grid;row.packetBytes=bytes.length;row.payloadBytes=p.data?.length??0;
    row.chunkCount=p.kind==='start'?p.chunkCount:meta?.chunkCount??row.chunkCount;
    row.digest=p.digest??meta?.digest??row.digest;
  }
  matchesAck(row,p){
    if(row.digest!==null&&row.digest!==p.digest)return false;
    const status=row.chunkCount===null?null:row.index===row.chunkCount-1?'complete':'progress';
    return p.base===row.index&&p.bitmap===1&&['progress','complete'].includes(p.status)&&(!status||p.status===status);
  }
  /** Call once when a packet becomes visible, after painting its first phase.
   * Redisplaying a retained ACK/START/REQUEST increments displayCount without resetting
   * the first timestamp. Phase refreshes are not separate packet displays.
   */
  displayed(bytes,{grid,atMs}={}){
    const p=parse(bytes),at=this.stamp(atMs);
    if(!p||!validGrid(p.kind,grid)||at===null)return false;
    if(p.kind==='poll'){
      const row=this.request('sent',p,bytes,grid);if(!row)return false;
      row.displayedMs??=at;row.lastDisplayedMs=at;row.displayCount++;return true;
    }
    if(p.kind==='ack'){
      const row=this.row('received',p);if(!row||!this.matchesAck(row,p))return false;
      const ack=row.ack;ack.status=p.status;ack.grid=grid;ack.packetBytes=ackBytes(grid,bytes);
      ack.displayedMs??=at;ack.lastDisplayedMs=at;ack.displayCount++;return true;
    }
    this.meta(p);const row=this.row('sent',p);if(!row)return false;
    this.describe(row,p,bytes,grid);row.displayedMs??=at;row.lastDisplayedMs=at;row.displayCount++;return true;
  }
  /** Acceptance is mandatory. atMs is captured when the recovered packet enters
   * transport, before receive() can synchronously paint the reply. Correlation
   * supports ACK display occurring before this callback returns. Times use only
   * this device's monotonic clock; they never compare phone and PC clocks.
   */
  accepted(bytes,result,{grid,atMs}={}){
    if(!result?.accepted||duplicateReasons.has(result.reason))return false;
    const p=parse(bytes),at=this.stamp(atMs);
    if(!p||!validGrid(p.kind,grid)||at===null)return false;
    if(p.kind==='poll'){
      const row=this.request('received',p,bytes,grid);if(!row||row.receivedMs!==null)return false;
      row.receivedMs=at;return true;
    }
    if(p.kind==='ack'){
      const row=this.rows.get(rowKey('sent',p));
      if(!row||row.displayedMs===null||row.ack.receivedMs!==null||at<row.displayedMs||!this.matchesAck(row,p))return false;
      Object.assign(row.ack,{status:p.status,grid,packetBytes:ackBytes(grid,bytes),receivedMs:at});
      const rtt=at-row.displayedMs;
      this.minimumRoundTripMs=this.minimumRoundTripMs===null?rtt:Math.min(this.minimumRoundTripMs,rtt);
      return true;
    }
    this.meta(p);const row=this.row('received',p);if(!row||row.receivedMs!==null)return false;
    this.describe(row,p,bytes,grid);row.receivedMs=at;return true;
  }
  snapshot(){
    const records=[...this.rows.values()],requests=[...this.requests.values()],rtt=row=>elapsed(row.ack.receivedMs,row.displayedMs),
      ackDelay=row=>elapsed(row.ack.displayedMs,row.receivedMs);
    const sent=records.filter(row=>row.direction==='sent'),received=records.filter(row=>row.direction==='received');
    const groups=[];
    for(const grid of [32,64,128])for(const kind of ['start','data']){
      const rows=sent.filter(row=>row.grid===grid&&row.kind===kind);
      if(rows.length)groups.push({grid,kind,...summarize(rows.map(rtt))});
    }
    return {format:'neural-zkboo-packet-timings',version:1,maxRows:this.maxRows,droppedEvents:this.droppedEvents,
      truncated:this.droppedEvents>0,
      timingScope:'Local monotonic milliseconds since attempt start. RTT: first visible START/DATA to recovered matching ACK subsequently accepted by transport; includes optical capture, inference, retries and alignment. REQUEST controls are recorded separately and excluded from RTT.',
      roundTrips:{overall:summarize(sent.map(rtt)),byGridAndKind:groups},
      receiveToAckDisplay:summarize(received.map(ackDelay)),
      requestCounts:{sent:requests.filter(row=>row.direction==='sent').length,received:requests.filter(row=>row.direction==='received').length},
      requests:requests.map(row=>({...row,displayedMs:rounded(row.displayedMs),lastDisplayedMs:rounded(row.lastDisplayedMs),receivedMs:rounded(row.receivedMs)})),
      packets:records.map(row=>{
        const common={direction:row.direction,sessionId:row.sessionId,messageId:row.messageId,kind:row.kind,index:row.index,
          grid:row.grid,packetBytes:row.packetBytes,payloadBytes:row.payloadBytes,chunkCount:row.chunkCount};
        return row.direction==='sent'?{...common,displayedMs:rounded(row.displayedMs),lastDisplayedMs:rounded(row.lastDisplayedMs),displayCount:row.displayCount,
          ack:{status:row.ack.status,grid:row.ack.grid,packetBytes:row.ack.packetBytes,receivedMs:rounded(row.ack.receivedMs)},roundTripMs:rounded(rtt(row))}
          :{...common,receivedMs:rounded(row.receivedMs),ack:{status:row.ack.status,grid:row.ack.grid,packetBytes:row.ack.packetBytes,
            displayedMs:rounded(row.ack.displayedMs),lastDisplayedMs:rounded(row.ack.lastDisplayedMs),displayCount:row.ack.displayCount},ackDisplayDelayMs:rounded(ackDelay(row))};
      })};
  }
}
