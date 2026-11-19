#!/usr/bin/env node
/** Join two diagnostic sweep reports locally, after capture. Never used by a reader. */
import {readFile,stat} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';

const type='neural-qr-corrected-timing-sweep',keys=['K4','K8'];
const witnessMin=36,marginMin=12;
function parse(s){
  const m=typeof s==='string'&&s.length===356&&/^s1:([a-f0-9]{96}):([a-f0-9]{256})$/.exec(s);
  if(!m)throw Error('Invalid or unsupported sampled-cell format.');
  return {witness:m[1],score:m[2]};
}
const matches=(a,b)=>{let count=0;for(let i=0;i<a.length;i++)if(a[i]===b[i])count++;return count;};
function steps(report,role){
  if(report?.format!==type||report.version!==1||report.configuration?.role!==role||
    ![32,64,128].includes(report.configuration.size)||![16,32].includes(report.configuration.ackSize)||
    !Array.isArray(report.steps)||report.steps.length>64)throw Error('Expected a bounded '+role+' trained sweep report.');
  const rows=new Map();
  for(const row of report.steps){
    if(!row||!Number.isInteger(row.step)||row.step<1||row.step>64||rows.has(row.step)||
      !Number.isFinite(row.phaseMs)||!Number.isFinite(row.captureMs))throw Error('Invalid or duplicate sweep step.');
    rows.set(row.step,row);
  }
  return rows;
}

export function scoreSweepCells(sender,receiver){
  const source=steps(sender,'sender'),dest=steps(receiver,'receiver');
  for(const k of ['size','ackSize','startPhaseMs','startCaptureMs','decrementMs'])
    if(sender.configuration[k]!==receiver.configuration[k])throw Error('Incompatible sweep configuration: '+k);
  const validId=id=>typeof id==='string'&&/^[a-f0-9]{32}$/.test(id);
  if(!validId(sender.runId)||(receiver.runId!==null&&(!validId(receiver.runId)||sender.runId!==receiver.runId)))
    throw Error('Mismatched or invalid run IDs; use both reports of the same run.');
  const refs=new Map();
  for(const [step,row] of source)if(row.cellSample?.reference)refs.set(step,parse(row.cellSample.reference));
  if(!refs.size)throw Error('No painted sender reference sketches in this report.');
  const make=()=>({scored:0,validated:0,likely:0,noInference:0,missingSketch:0,missingReference:0,foreign:0,
    weakWitness:0,otherStep:0,ambiguous:0,wrongCells:0,comparedCells:0,sampledCER:null});
  const byCapture={K4:make(),K8:make()};
  for(const [step,row] of dest){
    const sent=source.get(step);
    if(sent&&(sent.phaseMs!==row.phaseMs||sent.captureMs!==row.captureMs))throw Error('Mismatched cadence at step '+step);
    for(const k of keys){
      const b=byCapture[k],value=row.cellSample?.[k];
      if(!value){if(row.reader?.inferences?.[k]===0)b.noInference++;else b.missingSketch++;continue;}
      const observed=parse(value),expected=refs.get(step),status=row.cellSample?.status?.[k];
      if(status==='foreign'){b.foreign++;continue;}
      if(!expected){b.missingReference++;continue;}
      if(!['unconfirmed','matched'].includes(status))throw Error('Unknown identity status for recorded prediction.');
      const own=matches(observed.witness,expected.witness);
      let rival=0;
      for(const [other,ref] of refs)if(other!==step)rival=Math.max(rival,matches(observed.witness,ref.witness));
      if(rival>=witnessMin&&rival>own){b.otherStep++;continue;}
      if(own<witnessMin){b.weakWitness++;continue;}
      if(own-rival<marginMin){b.ambiguous++;continue;}
      b.scored++;
      if(status==='matched'&&receiver.runId!==null)b.validated++;else b.likely++;
      b.wrongCells+=256-matches(observed.score,expected.score);b.comparedCells+=256;
    }
  }
  for(const b of Object.values(byCapture))b.sampledCER=b.comparedCells?b.wrongCells/b.comparedCells:null;
  return {grid:sender.configuration.size,localColorCorrection:receiver.configuration.localColorCorrection,
    runBinding:receiver.runId===null?'receiver did not validate a run ID; witness-only pairing':'matching run IDs',
    receiverSteps:dest.size,paintedReferences:refs.size,byCapture,
    scope:'First K4 and first K8 DATA inference per armed step, never all attempts. A sample must match ≥36/96 witness cells and beat every other painted step by ≥12. CER uses 256 separate wire cells per eligible sketch; unavailable, mixed or weakly identified reads may be underrepresented. likely counts are not packet validation.'};
}

async function load(path){
  const info=await stat(path);
  if(!info.isFile()||info.size>16*1024*1024)throw Error('Expected a sweep JSON report under 16 MiB.');
  return JSON.parse(await readFile(path,'utf8'));
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  try{
    const args=process.argv.slice(2);
    if(args.length!==2)throw Error('Usage: node scripts/score-sweep-cells.mjs SENDER.json RECEIVER.json');
    const [sender,receiver]=await Promise.all(args.map(load));
    console.log(JSON.stringify(scoreSweepCells(sender,receiver),null,2));
  }catch(e){console.error(e.message);process.exitCode=1;}
}
