import {writeFile,mkdir} from 'node:fs/promises';
import {runSimulation} from '../dist/js/link/simulation.js';

const cases=[
  ['clean',{dropRate:0,duplicateRate:0,corruptRate:0}],
  ['default-faults',{dropRate:.15,duplicateRate:.15,corruptRate:.05}],
  ['heavier-faults',{dropRate:.25,duplicateRate:.3,corruptRate:.15,seed:91}],
  ['blackhole',{bytes:1024,dropRate:1,duplicateRate:0,corruptRate:0,timeoutMs:3000,retryMs:1000,maxRetries:100}],
];
const reports=[];
for(const [name,options] of cases){
  const report=await runSimulation(options);reports.push({name,...report});
  console.log(`${name}: ${report.status}, exact=${report.exact}, ${report.sender.chunkCount} chunks, ${report.sender.retries} retries, ${Math.round(report.elapsedMs)} ms processing, ${(report.simulatedMs/60000).toFixed(1)} min virtual optical time`);
  if(name==='blackhole'?report.status!=='timed-out':report.status!=='delivered'||!report.exact)process.exitCode=1;
}
const output=new URL('../local-results/link.json',import.meta.url);
await mkdir(new URL('../local-results/',import.meta.url),{recursive:true});
await writeFile(output,JSON.stringify({schema:'neural-key-link-validation-v1',createdAt:new Date().toISOString(),runtime:process.version,
  scope:'Real packet state machines and optical byte/symbol codec. No physical camera or browser visual QA.',reports},null,2)+'\n');
console.log(`Saved ${output.pathname}`);
