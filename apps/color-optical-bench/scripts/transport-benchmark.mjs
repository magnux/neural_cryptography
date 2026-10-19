// Deterministic optical stress test. Lab time measures CPU work, not optical rate.
// node scripts/transport-benchmark.mjs --sizes=64,128 --phases=8,16 --captures=9
import {encodeSequence,phaseFrames,renderFrame} from '../dist/js/format.js';
import {probePayload,ExperimentReport} from '../dist/js/diagnostics.js';
import {TemporalDecoder} from '../dist/js/temporal.js';
import {simulateCapture} from '../dist/js/vision.js';
const args=Object.fromEntries(process.argv.slice(2).map(a=>a.replace(/^--/,'').split('=')));
const sizes=(args.sizes??'64,128').split(',').map(Number),phases=(args.phases??'8,16').split(',').map(Number),captures=Number(args.captures??9);
if(sizes.some(n=>!Number.isInteger(n)||n<8||n>512)||phases.some(n=>![1,2,4,8,16].includes(n))||!Number.isInteger(captures)||captures<5||captures>50)throw Error('Invalid benchmark arguments.');
for(const size of sizes)for(const phaseCount of phases){
  const options={size,colors:16,gridBlock:8,parity:32,phases:phaseCount,permute:phaseCount>1,maxFrames:1,probePattern:1,probeSeed:1234,id:Uint8Array.of(1,2,3,4,5,6,7,8)};
  const frames=phaseFrames(encodeSequence(probePayload(options),options)),decoder=new TemporalDecoder(),report=new ExperimentReport();let ordinal=0,result;
  for(const frame of frames){const source=renderFrame(frame,Math.min(4096,Math.max(1024,Math.ceil(size*6.4/256)*256)));for(let i=0;i<captures;i++,ordinal++){
    const image=simulateCapture(source,{width:960,height:760,preset:'perspective',noise:12,moire:44,moirePeriod:16,curvature:1,colorDependence:1,phase:ordinal*1.1,seed:1234+ordinal*104729});
    result=decoder.decode(image,{filter:'robust',window:5,overhead:2,minPhases:4,recovery:'erasures',preview:false});report.accept(result);
  }}
  const s=report.snapshot();console.log(JSON.stringify({size,phases:phaseCount,exposuresPerPhase:captures,headers:s.headers,captures:s.captures,lastOK:result.ok,elapsedProcessingMs:Math.round(s.elapsedMs),
    methods:Object.fromEntries(Object.entries(s.methods).map(([name,r])=>[name,{SER:r.symbolErrorRate,raw:r.rawPasses,hard:r.hardPasses,erasure:r.erasurePasses,verified:r.passes,attempts:r.attempts,firstCapture:r.firstVerifiedCapture,firstCounts:r.firstVerifiedCounts,correctedBytes:r.correctedBytes}]))}));
}
