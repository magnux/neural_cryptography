import {capacity,chunkToSymbols,sequenceKey} from './format.js';
import {gridLayout} from './grid.js';
export const PROBE_PATTERNS=Object.freeze({random:1,patches:2,checker:3});
export function probeChunk(info){
  const cap=capacity(info.size,info.colors,info),bits=Math.log2(info.colors),bytes=new Uint8Array(cap);
  let state=(info.probeSeed^Math.imul((info.index??0)+1,0x9e3779b9))>>>0;
  for(let i=0;i<Math.ceil(cap*8/bits);i++){
    state=(Math.imul(state,1664525)+1013904223)>>>0;
    const symbol=info.probePattern===2?(Math.floor(i/8)+Math.floor(i/info.size/8))%info.colors:info.probePattern===3?(i+Math.floor(i/info.size))%info.colors:(state>>>24)&(info.colors-1);
    for(let bit=0;bit<bits;bit++){const pos=i*bits+bit;if(pos<cap*8)bytes[pos>>>3]|=((symbol>>(bits-1-bit))&1)<<(7-(pos&7));}
  }
  return bytes;
}
export function probePayload(info){
  const cap=capacity(info.size,info.colors,info),bytes=new Uint8Array(cap*info.maxFrames);
  for(let index=0;index<info.maxFrames;index++)bytes.set(probeChunk({...info,index}),index*cap);return bytes;
}
/** Known data is generated only after classification. Never call from a decoder. */
export function measureErrors(info,symbols){
  if(!info.probePattern||!symbols)return null;
  const expected=chunkToSymbols(probeChunk(info),{...info,phase:0}),layout=gridLayout(info.size,info.colors,info.gridBlock);
  const confusion=Array(info.colors**2).fill(0),side=Math.min(32,info.size),errors=Array(side**2).fill(0),totals=Array(side**2).fill(0);let cellErrors=0,bitErrors=0;
  for(const cell of layout.cells){
    const truth=expected[cell],observed=symbols[cell],different=truth!==observed;cellErrors+=Number(different);
    let xor=truth^observed;while(xor){bitErrors+=xor&1;xor>>>=1;}confusion[truth*info.colors+observed]++;
    const x=Math.floor((cell%info.size)*side/info.size),y=Math.floor(Math.floor(cell/info.size)*side/info.size),bin=y*side+x;
    totals[bin]++;errors[bin]+=Number(different);
  }
  const cells=layout.cells.length,bits=cells*info.bits;
  return {cellErrors,cells,bitErrors,bits,symbolErrorRate:cellErrors/cells,bitErrorRate:bitErrors/bits,confusion,heatmap:{side,errors,totals}};
}
export function comparisonResult(info,result){
  return {ready:true,ok:result.ok,stage:result.stage,error:result.error,stats:result.stats,
    errors:measureErrors(info,result.symbols)};
}

/** Bounded report state, shared by live and lab UI; never stores payload bytes. */
export class ExperimentReport {
  constructor(){this.reset();}
  reset(){this.startedAt=null;this.captures=0;this.headers=0;this.rows={};this.history=[];this.profile=null;this.sequence=null;this.rejectedSequences=0;this.temporal=null;}
  accept(result,now=performance.now()){
    if(this.startedAt===null)this.startedAt=now;
    this.captures++;if(result.info){this.headers++;const i=result.info,key=sequenceKey(i);if(this.sequence&&this.sequence!==key){this.rejectedSequences++;return;}this.sequence=key;this.profile={size:i.size,colors:i.colors,id:i.id,gridBlock:i.gridBlock,parity:i.parity,phases:i.phases,permute:i.permute,holdMs:i.holdMs,probePattern:i.probePattern,probeSeed:i.probeSeed};}
    if(result.stats?.temporal)this.temporal=temporalSummary(result.stats.temporal);
    for(const method of METHODS){
      const r=result.colorComparison?.[method];if(!r)continue;
      if(!r.ready)continue;
      const row=this.rows[method]??={attempts:0,passes:0,rawPasses:0,hardPasses:0,erasurePasses:0,correctedBytes:0,erasedBytes:0,erasureTrials:0,firstVerifiedMs:null,firstVerifiedCapture:null,firstVerifiedCounts:null,frames:[],bytes:0,cells:0,cellErrors:0,bits:0,bitErrors:0,confusion:[],last:null};
      row.attempts++;row.correctedBytes+=r.stats?.correctedBytes??0;row.erasedBytes+=r.stats?.erasedBytes??0;row.erasureTrials+=r.stats?.erasureTrials??0;
      if(r.ok){
        row.passes++;if(r.stats?.rawCRC)row.rawPasses++;if(r.stats?.hardCRC)row.hardPasses++;if(r.stats?.recovery==='erasures')row.erasurePasses++;
        if(row.firstVerifiedMs===null){row.firstVerifiedMs=now-this.startedAt;row.firstVerifiedCapture=this.captures;row.firstVerifiedCounts=countsSummary(result.stats?.temporal?.counts);}
        if(result.info&&!row.frames.includes(result.info.index)){row.frames.push(result.info.index);row.bytes+=result.info.length;}
      }
      if(r.errors){
        for(const key of ['cells','cellErrors','bits','bitErrors'])row[key]+=r.errors[key];
        r.errors.confusion.slice(0,256).forEach((n,i)=>row.confusion[i]=(row.confusion[i]??0)+n);
      }
      row.last=resultSummary(r);
    }
    if(result.colorComparison){
      this.history.push({elapsedMs:now-this.startedAt,phase:result.info?.phase,index:result.info?.index,
        phaseSamples:countsSummary(result.stats?.temporal?.counts),temporal:temporalSummary(result.stats?.temporal),pixelsPerCell:result.stats?.pixelsPerCell,
        ...Object.fromEntries(METHODS.map(method=>[method,compact(result.colorComparison[method])]))});
      if(this.history.length>120)this.history.shift();
    }
  }
  snapshot(now=performance.now()){
    const elapsedMs=this.startedAt===null?0:now-this.startedAt;
    return {profile:this.profile,captures:this.captures,headers:this.headers,rejectedSequences:this.rejectedSequences,elapsedMs,temporal:this.temporal,
      methods:Object.fromEntries(Object.entries(this.rows).map(([key,row])=>[key,{...row,
        symbolErrorRate:row.cells?row.cellErrors/row.cells:null,bitErrorRate:row.bits?row.bitErrors/row.bits:null,
        uniqueVerifiedBytesPerSecond:elapsedMs>0?row.bytes/(elapsedMs/1000):0}])),history:this.history};
  }
}
const METHODS=['global','local','fusedGlobal','fusedLocal','robustGlobal','robustLocal','accepted'];
const countsSummary=counts=>Array.isArray(counts)?counts.slice(0,16):null;
function scalarFields(source,keys){const result={};for(const key of keys){const value=source?.[key];if(typeof value==='number'||typeof value==='boolean'||typeof value==='string'||value===null)result[key]=value;}return result;}
function temporalSummary(temporal){
  if(!temporal)return null;
  const summary=scalarFields(temporal,['filter','samples','target','requested','poolCapacity','requestedPool','phases','ready','eligiblePhases','minPhases','retainedPhaseLimit','memoryLimited','singleCRC','memoryBytes','evicted','ageLimitMs']);
  summary.counts=countsSummary(temporal.counts);
  if(Array.isArray(temporal.eligiblePhases))summary.eligiblePhases=temporal.eligiblePhases.slice(0,16);
  if(temporal.robust)summary.robust=Object.fromEntries(['global','local'].map(method=>[method,
    (temporal.robust[method]??[]).slice(0,16).map(s=>s?scalarFields(s,['phase','cells','candidates','kept','trimmedCells','ambiguousCells','meanSelectionDistance','meanDiscardedDistance']):null)]));
  return summary;
}
function resultSummary(result){
  const stats=scalarFields(result.stats,['rawCRC','hardCRC','correctedBytes','erasedBytes','erasureTrials','recovery','rsBlocks','confidence','lowConfidence','colorRMSE','acceptedDecoder','acceptedMethod']);
  if(result.stats?.references)stats.references=scalarFields(result.stats.references,['pilotCells','gridBlock','missingPrototypes','minimumSeparation']);
  let errors=null;
  if(result.errors){
    errors={...scalarFields(result.errors,['cellErrors','cells','bitErrors','bits','symbolErrorRate','bitErrorRate']),confusion:result.errors.confusion.slice(0,256)};
    const map=result.errors.heatmap;if(map)errors.heatmap={side:map.side,errors:map.errors.slice(0,1024),totals:map.totals.slice(0,1024)};
  }
  return {ready:result.ready,ok:result.ok,stage:result.stage,error:result.error,stats,errors};
}
function compact(result){return result?.ready?{ok:result.ok,rawCRC:result.stats?.rawCRC,hardCRC:result.stats?.hardCRC,correctedBytes:result.stats?.correctedBytes,erasedBytes:result.stats?.erasedBytes,erasureTrials:result.stats?.erasureTrials,recovery:result.stats?.recovery,stage:result.stage,cellErrors:result.errors?.cellErrors??null,cells:result.errors?.cells??null,confidence:result.stats?.confidence??null}:null;}
