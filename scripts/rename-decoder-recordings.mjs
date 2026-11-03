#!/usr/bin/env node
/** Rename current recording exports using their shared first-matrix CRC32.
 * Only the leading filename prefix changes; its remaining suffix, JSON bytes,
 * and split folder stay intact. Unrelated JSON is skipped; malformed recordings
 * stop preflight.
 */
import {lstat,readFile,readdir,link,unlink,realpath} from 'node:fs/promises';
import {resolve,dirname,basename,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {recordingPrefix} from '../apps/color-optical-bench/dist/js/dataset/filenames.js';

const MAX_RECORDING_BYTES=164*1024*1024;
const FORMATS=new Set(['optical-decoder-sender','optical-decoder-captures']);
const defaults=['local-data/phone','local-data/pc'];
async function exists(path) {
  try{return await lstat(path);}catch(error){if(error.code==='ENOENT')return null;throw error;}
}
function signature(info){return [info.dev,info.ino,info.size,info.mtimeMs,info.ctimeMs].join('/');}

export async function planRecordingRenames(roots=defaults) {
  if(!Array.isArray(roots)||!roots.length)throw Error('Provide at least one recording directory.');
  const files=new Set(),visited=new Set();
  async function visit(path) {
    path=resolve(path);if(visited.has(path))return;visited.add(path);
    const info=await lstat(path);
    if(info.isSymbolicLink())throw Error(`Refusing recording symlink: ${path}`);
    if(info.isDirectory()) {
      for(const name of (await readdir(path)).sort())await visit(join(path,name));
    }else if(info.isFile()&&path.toLowerCase().endsWith('.json'))files.add(path);
  }
  for(const root of roots) {
    if(await realpath(root)!==resolve(root))throw Error(`Refusing recording path through a symlink: ${root}`);
    await visit(root);
  }
  const renames=[],unchanged=[],skipped=[],destinations=new Map();
  for(const source of [...files].sort()) {
    const info=await lstat(source);
    if(info.size>MAX_RECORDING_BYTES)throw Error(`Recording JSON exceeds 164 MiB: ${source}`);
    let recording;
    try{recording=JSON.parse(await readFile(source,'utf8'));}catch(error){throw Error(`Cannot read JSON ${source}: ${error.message}`);}
    if(!FORMATS.has(recording?.format)){skipped.push(source);continue;}
    const suffix=basename(source).replace(/^[a-z0-9]+_/i,'');
    const destination=join(dirname(source),`${recordingPrefix(recording)}_${suffix}`);
    if(destinations.has(destination)&&destinations.get(destination)!==source)
      throw Error(`Two recordings would have the same filename: ${destination}`);
    destinations.set(destination,source);
    if(destination===source){unchanged.push(source);continue;}
    if(await exists(destination))throw Error(`Rename destination already exists: ${destination}`);
    renames.push({source,destination,signature:signature(info)});
  }
  return {renames,unchanged,skipped};
}

export async function renameRecordings(roots=defaults,{apply=false}={}) {
  const plan=await planRecordingRenames(roots);
  if(apply) {
    // Preflight every source again before any mutation. Hard links make each
    // destination creation exclusive, so a concurrent file is never replaced.
    for(const row of plan.renames) {
      const info=await lstat(row.source);
      if(!info.isFile()||info.isSymbolicLink()||signature(info)!==row.signature)
        throw Error(`Recording changed during preflight: ${row.source}`);
      if(await exists(row.destination))throw Error(`Rename destination already exists: ${row.destination}`);
    }
    const linked=[];
    try {
      for(const row of plan.renames){await link(row.source,row.destination);linked.push(row);}
    }catch(error) {
      for(const row of linked)await unlink(row.destination);
      throw error;
    }
    // These links are in the same parent directory and on the same filesystem.
    // Removing the old name preserves the contents and modification time.
    for(const row of linked)await unlink(row.source);
  }
  return {mode:apply?'apply':'dry-run',...plan};
}

async function main() {
  let apply=false,mode=null;const roots=[];
  for(const arg of process.argv.slice(2)) {
    if(arg==='--help'||arg==='-h') {
      console.log('Usage: node scripts/rename-decoder-recordings.mjs [--dry-run|--apply] [DIRECTORY ...]\nDefaults: dry-run; recursively scan local-data/phone and local-data/pc.\nReplaces only the leading index/checksum prefix; filename suffixes and folders stay intact.\nExisting destinations and symlinks are refused.');return;
    }
    if(arg==='--apply'||arg==='--dry-run') {
      if(mode)throw Error('Choose only one of --apply or --dry-run.');mode=arg;apply=arg==='--apply';
    }else if(arg.startsWith('--'))throw Error(`Unknown option: ${arg}`);
    else roots.push(arg);
  }
  const result=await renameRecordings(roots.length?roots:defaults,{apply});
  for(const {source,destination} of result.renames)console.log(`${result.mode}: ${source}\n  -> ${destination}`);
  console.log(`${result.renames.length} ${apply?'renamed':'to rename'}; ${result.unchanged.length} already named; ${result.skipped.length} unrelated JSON skipped.`);
  if(!apply&&result.renames.length)console.log('Run with --apply to perform these filename changes.');
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))
  main().catch(error=>{console.error(`Recording rename failed: ${error.message}`);process.exitCode=1;});
