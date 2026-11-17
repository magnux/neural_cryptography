#!/usr/bin/env node
/** Local execution of the same full handshake simulator; never a phone benchmark. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
import {runFullHandshake,runSimulationComparison} from '../dist/simulation-core.js';
import {createSimulationCorrector} from '../dist/simulation-runtime.js';

const args=process.argv.slice(2),options={statistics:true,compare:true};let output=null,protocolOnly=false;
for(let index=0;index<args.length;index++){
  const flag=args[index];
  if(flag==='--protocol-only')protocolOnly=true;
  else if(flag==='--no-statistics')options.statistics=false;
  else if(flag==='--optimized-only')options.compare=false;
  else if(['--out','--models','--runtime'].includes(flag)&&args[index+1]){
    const value=resolve(args[++index]);
    if(flag==='--out')output=value;
    if(flag==='--models')options.modelRoot=pathToFileURL(value+'/');
    if(flag==='--runtime')options.runtimeRoot=pathToFileURL(value+'/');
  }else throw Error('Usage: benchmark-simulation.mjs [--protocol-only] [--no-statistics] [--optimized-only] [--models DIRECTORY] [--runtime DIRECTORY] [--out REPORT.json]');
}
const onProgress=message=>process.stderr.write(message+'\n');
let corrector;
try{
  let report;
  if(protocolOnly){
    const run=await runFullHandshake({optical:false,optimized:false,statistics:options.statistics,onProgress});
    report={format:'neural-auth-demo-simulation',version:1,kind:'protocol-only',success:run.success,
      timestamp:new Date().toISOString(),environment:`Node ${process.version}`,assumedOutput:false,
      inference:{measured:false,reason:'Explicit --protocol-only: no optical acquisition or neural inference executed.'},
      modelAccuracyValidated:false,physicalCameraTest:false,baseline:run,
      warnings:['Protocol-only benchmark: actual fresh-key authentication, TurnChannel framing, proof generation and verification; no optical or neural inference timing.']};
  }else{
    corrector=await createSimulationCorrector({...options,onProgress,read:async url=>{
      const bytes=await readFile(url);return bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
    }});
    report=await runSimulationComparison({...options,corrector,onProgress});
    report.environment=`Node ${process.version}; ONNX Runtime Web/WASM, single thread. Not a browser or phone measurement.`;
  }
  const json=JSON.stringify(report,null,2)+'\n';
  if(output){await mkdir(dirname(output),{recursive:true});await writeFile(output,json);}
  process.stdout.write(json);
  if(!report.success)process.exitCode=1;
}catch(error){process.stderr.write(`Simulation failed: ${error.message}\n`);process.exitCode=1;}
finally{await corrector?.close();}
