import {sequenceKey,frameKey,sameBytes} from './format.js';
import {extractCapture,classifySamples,decodeScores} from './vision.js';
import {comparisonResult} from './diagnostics.js';
import {filterSamples,robustSamples} from './sampling.js';
export {filterSamples} from './sampling.js';

/** Calibrate and unpermute each exposure before fusion. Pools are FIFO: even
 * strong old consensus must leave. Selection is independent for every cell;
 * each eligible phase has equal weight, irrespective of its capture count. */
export class TemporalDecoder {
  constructor({maxBytes=64*1024*1024,maxAgeMs=null,clock=()=>performance.now()}={}){
    if(!Number.isFinite(maxBytes)||maxBytes<1||(maxAgeMs!==null&&(!Number.isFinite(maxAgeMs)||maxAgeMs<0)))throw Error('Invalid evidence budget.');
    this.maxBytes=maxBytes;this.maxAgeMs=maxAgeMs;this.clock=clock;this.reset();
  }
  reset(){this.groups=new Map();this.bytes=0;this.sequence=null;this.settings=null;this.evicted=0;this.transitionKey=null;this.transitionAt=null;}
  remove(key){const group=this.groups.get(key);if(group){this.bytes-=group.bytes;this.groups.delete(key);this.evicted++;}}
  accumulate(info,entry,window,{overhead=0,minPhases=4}={}){
    const sequence=sequenceKey(info),key=frameKey(info),now=this.clock();
    if(this.sequence&&this.sequence!==sequence)throw Error('Different sequence detected. Reset the receiver to accept it.');
    this.sequence=sequence;
    const entryBytes=entry.global.byteLength+(entry.local===entry.global?0:entry.local.byteLength);
    const required=Math.min(info.phases,minPhases),requestedPool=window+overhead;
    // Prefer retaining all phases. If even one capture per phase cannot fit,
    // retain a rolling subset, but never fewer than the requested first attempt.
    let available=Math.floor(this.maxBytes/(entryBytes*info.phases));
    if(available<1)available=Math.floor(this.maxBytes/(entryBytes*required));
    const poolCapacity=Math.min(requestedPool,available),target=Math.min(window,poolCapacity);
    if(target<1)throw Error('This profile and first-attempt threshold exceed the temporal memory budget. Use fewer colors or a lower phase threshold.');
    const retainedPhaseLimit=Math.min(info.phases,Math.floor(this.maxBytes/(entryBytes*poolCapacity)));
    const ageLimitMs=this.maxAgeMs??Math.max(60000,2*(info.holdMs??4000)*info.phases*info.count);
    for(const [id,group] of this.groups)if(now-group.at>ageLimitMs)this.remove(id);
    let group=this.groups.get(key);
    if(group&&(group.poolCapacity!==poolCapacity||group.target!==target)){this.remove(key);group=null;}
    if(!group)group={phases:Array.from({length:info.phases},()=>[]),bytes:0,target,poolCapacity,requestedPool,retainedPhaseLimit,ageLimitMs,at:now};
    group.requestedPool=requestedPool;
    const drop=samples=>{const old=samples.shift();group.bytes-=old.bytes;this.bytes-=old.bytes;};
    for(const samples of group.phases)while(samples.length&&now-samples[0].at>ageLimitMs)drop(samples);
    const samples=group.phases[info.phase];
    if(!samples.length){
      while(group.phases.filter(p=>p.length).length>=retainedPhaseLimit){
        let oldest=null;
        for(const p of group.phases)if(p.length&&(!oldest||p.at(-1).at<oldest.at(-1).at))oldest=p;
        while(oldest.length)drop(oldest);this.evicted++;
      }
    }
    while(samples.length>=poolCapacity)drop(samples);
    samples.push({...entry,at:now,bytes:entryBytes});group.bytes+=entryBytes;this.bytes+=entryBytes;group.at=now;
    this.groups.delete(key);this.groups.set(key,group);
    while(this.bytes>this.maxBytes||this.groups.size>8){const oldest=this.groups.keys().next().value;if(oldest===key)break;this.remove(oldest);}
    return group;
  }
  /** Exposed separately so transport tests can supply measured RGB directly. */
  process(info,values,{filter='none',window=5,overhead=2,minPhases=4,recovery='hard'}={}){
    validate({filter,window,overhead,minPhases,recovery});
    const options={recovery},global=classifySamples(info,values,{local:false}),local=info.gridBlock?classifySamples(info,values,{local:true}):global;
    const singleGlobal=decodeScores(info,global.scores,global,options),singleLocal=local===global?singleGlobal:decodeScores(info,local.scores,local,options);
    const requested=filter==='none'?1:window;
    const group=this.accumulate(info,{global:global.scores,local:local.scores},requested,{overhead:filter==='robust'?overhead:0,minPhases});
    const counts=group.phases.map(p=>p.length),eligible=group.phases.map((p,i)=>p.length>=group.target?i:-1).filter(i=>i>=0);
    const required=Math.min(info.phases,minPhases),ready=eligible.length>=required;
    const colorComparison={global:comparisonResult(info,singleGlobal),local:comparisonResult(info,singleLocal)},robust={global:[],local:[]};
    let selected={ok:false,stage:'accumulate',error:`${eligible.length}/${required} phases ready; samples ${counts.join(' / ')} (minimum ${group.target} each). Keep the sequence playing.`,stats:{}};
    if(ready){
      const candidates={};
      for(const method of ['global','local']){
        const base=method==='global'?global:local;
        for(const estimator of (filter==='robust'?['mean','robust']:[filter==='median'?'median':'mean'])){
          const scores=new Float32Array(global.scores.length);
          for(const phaseIndex of eligible){
            const samples=group.phases[phaseIndex].map(e=>e[method]);let phase;
            if(estimator==='robust'){
              const estimate=robustSamples(samples,info.colors,group.target);phase=estimate.scores;
              robust[method].push({phase:phaseIndex,...estimate.stats});
            }else phase=filterSamples(samples,estimator);
            for(let i=0;i<scores.length;i++)scores[i]+=phase[i]/eligible.length;
          }
          const name=`${estimator==='robust'?'robust':'fused'}${method==='global'?'Global':'Local'}`;
          const decoded=decodeScores(info,scores,{...base,decoder:method},options);
          colorComparison[name]=comparisonResult(info,decoded);candidates[name]=decoded;
        }
      }
      const order=['robustLocal','robustGlobal','fusedLocal','fusedGlobal'],valid=order.filter(k=>candidates[k]?.ok);
      const chosen=valid[0]??order.find(k=>candidates[k]);selected=candidates[chosen];
      if(valid.some(k=>!sameBytes(candidates[k].chunk,selected.chunk)))selected={ok:false,stage:'payload',error:'Conflicting CRC-valid decoder candidates. Reset and retry.',stats:{}};
      selected.stats.acceptedDecoder=selected.ok?selected.decoder:null;selected.stats.acceptedMethod=selected.ok?chosen:null;
      colorComparison.accepted=comparisonResult(info,selected);
    }
    const output={...selected,colorComparison,observedPalette:global.observedPalette,colorTransform:global.colorTransform,
      stats:{...selected.stats,temporal:{filter,samples:counts[info.phase],target:group.target,requested,poolCapacity:group.poolCapacity,requestedPool:group.requestedPool,
        counts,phases:info.phases,eligiblePhases:eligible,minPhases:required,retainedPhaseLimit:group.retainedPhaseLimit,
        memoryLimited:group.poolCapacity<group.requestedPool||group.retainedPhaseLimit<info.phases,ageLimitMs:group.ageLimitMs,
        ready,singleCRC:singleLocal.ok,memoryBytes:this.bytes,evicted:this.evicted,robust}}};
    delete output.symbols;delete output.scores;return output;
  }
  decode(image,{filter='none',window=5,overhead=2,minPhases=4,recovery='hard',preview=true,spatial='cross',settleMs=0,singlePhase=false,streaming=false}={}){
    const start=performance.now();validate({filter,window,overhead,minPhases,recovery});
    if(!['cross','gaussian'].includes(spatial)||!Number.isFinite(settleMs)||settleMs<0||settleMs>3000||typeof streaming!=='boolean')throw Error('Invalid receiver settings.');
    const setting=`${filter}/${window}/${overhead}/${minPhases}/${recovery}/${spatial}/${settleMs}/${singlePhase}/${streaming}`;
    if(this.settings!==setting){this.reset();this.settings=setting;}
    const capture=extractCapture(image,{preview,spatial}),values=capture.samples;delete capture.samples;
    if(!values)return capture;
    const info=capture.info,sequence=sequenceKey(info);
    if(this.sequence&&this.sequence!==sequence){
      if(!streaming)return {...capture,error:'Different sequence detected. Reset the receiver to accept it.'};
      // extractCapture has already validated the monochrome header and its CRC.
      // Link packets are independent sequences: discard every prior phase and
      // restart the transition guard before using the new packet's evidence.
      this.reset();this.settings=setting;
    }
    this.sequence=sequence;const key=`${frameKey(info)}/${info.phase}`,now=this.clock();
    if(key!==this.transitionKey){this.transitionKey=key;this.transitionAt=now;}
    if(now-this.transitionAt<settleMs)return {...capture,stage:'settle',error:'Waiting for the display transition to settle.'};
    try{
      if(singlePhase){
        const global=classifySamples(info,values,{local:false}),local=info.gridBlock?classifySamples(info,values,{local:true}):global;
        const g=decodeScores(info,global.scores,global,{recovery}),l=decodeScores(info,local.scores,local,{recovery}),colorComparison={global:comparisonResult(info,g),local:comparisonResult(info,l)};
        delete l.scores;delete l.symbols;return {...capture,...l,colorComparison,stats:{...capture.stats,...l.stats,ms:performance.now()-start}};
      }
      const result=this.process(info,values,{filter,window,overhead,minPhases,recovery});return {...capture,...result,stats:{...capture.stats,...result.stats,ms:performance.now()-start}};}
    catch(error){return {...capture,ok:false,stage:'accumulate',error:error.message,stats:{...capture.stats,ms:performance.now()-start}};}
  }
}
function validate({filter,window,overhead,minPhases,recovery}){
  if(!['none','mean','median','robust'].includes(filter)||!Number.isInteger(window)||window<1||window>9||
    !Number.isInteger(overhead)||overhead<0||overhead>2||![1,2,4,8,16].includes(minPhases)||!['hard','erasures'].includes(recovery))throw Error('Invalid receiver settings.');
}
