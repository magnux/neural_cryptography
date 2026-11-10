#!/usr/bin/env node
/** JSON-lines bridge for the explicitly diagnostic recorded-error replay. */
import {once} from 'node:events';
import {DEFAULT_PARITY,MAX_WIRE_BYTES,packetLayout,replayCase} from './lib/corrector-rs-replay.mjs';

const MAX_LINE_CHARS=4*Math.ceil(MAX_WIRE_BYTES/3)*2+4096;
function base64(value,name){
  if(typeof value!=='string'||!value.length||value.length>4*Math.ceil(MAX_WIRE_BYTES/3)||
     !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value))
    throw Error(`${name} must be canonical bounded base64.`);
  const decoded=Buffer.from(value,'base64');
  if(decoded.toString('base64')!==value)throw Error(`${name} is not canonical base64.`);
  return new Uint8Array(decoded);
}

async function main(){
  const args=process.argv.slice(2);let parity=DEFAULT_PARITY;
  if(args.length){
    if(args.length!==2||args[0]!=='--parity'||!/^\d+$/.test(args[1]))
      throw Error('Usage: node scripts/corrector-rs-replay.mjs [--parity 32] < cases.jsonl');
    parity=Number(args[1]);
  }
  packetLayout(255,parity);
  let pending='',number=0;
  const processLine=async line=>{
    number++;
    if(line.length>MAX_LINE_CHARS)throw Error(`Line ${number} exceeds the replay input bound.`);
    if(!line.trim())return;
    try{
      const row=JSON.parse(line);
      if(!row||typeof row!=='object'||Array.isArray(row))throw Error('Expected a JSON object.');
      if(row.id!==undefined&&(typeof row.id!=='string'||row.id.length>1024))throw Error('id must be a string of at most 1024 characters.');
      if(row.parity!==undefined&&row.parity!==parity)throw Error('Per-case parity must match the CLI parity.');
      const result=replayCase(base64(row.expected,'expected'),base64(row.predicted,'predicted'),{parity});
      const output=JSON.stringify({...result,...(row.id===undefined?{}:{id:row.id})})+'\n';
      if(!process.stdout.write(output))await once(process.stdout,'drain');
    }catch(error){throw Error(`Line ${number}: ${error.message}`);}
  };
  process.stdin.setEncoding('utf8');
  for await(const chunk of process.stdin){
    pending+=chunk;let newline;
    while((newline=pending.indexOf('\n'))>=0){await processLine(pending.slice(0,newline));pending=pending.slice(newline+1);}
    if(pending.length>MAX_LINE_CHARS)throw Error('Replay input line exceeds the size bound.');
  }
  if(pending.length)await processLine(pending);
}

main().catch(error=>{process.stderr.write(`RS replay failed: ${error.message}\n`);process.exitCode=1;process.stdin.destroy();});
