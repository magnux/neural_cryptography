import test from 'node:test';
import assert from 'node:assert/strict';
import {cp,readFile,mkdir,mkdtemp,rm,stat} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join,relative,resolve,dirname,isAbsolute} from 'node:path';

const root=fileURLToPath(new URL('../../../',import.meta.url));
const dist=join(root,'apps/neural-auth-demo/dist');
const optional=path=>path.startsWith('models/')||path==='vendor/onnxruntime'||path.startsWith('vendor/onnxruntime/');

function moduleSpecifiers(code){
  return [
    ...code.matchAll(/^[ \t]*(?:import|export)\s+(?:[^;'"`]*?\s+from\s*)?['"]([^'"]+)['"]/gm),
    ...code.matchAll(/\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g),
    ...code.matchAll(/\bnew\s+URL\s*\(\s*['"]([^'"]+)['"]\s*,\s*import\.meta\.url\s*\)/g),
  ].map(match=>match[1]);
}

async function crawl(base){
  const html=await readFile(join(base,'index.html'),'utf8');
  const queue=[...html.matchAll(/<(?:script|link)\b[^>]*\b(?:src|href)=["']([^"']+)["']/g)].map(match=>match[1].replace(/^\.\//,''));
  const visited=new Set(['index.html']);
  while(queue.length){
    const file=queue.pop();if(visited.has(file))continue;
    assert.ok(!file.startsWith('../')&&!isAbsolute(file),`Asset escapes static root: ${file}`);
    assert.ok((await stat(join(base,file))).isFile(),`Missing standalone asset ${file}`);
    visited.add(file);
    if(!/\.m?js$/.test(file))continue;
    const source=await readFile(join(base,file),'utf8');
    for(const spec of moduleSpecifiers(source)){
      assert.match(spec,/^\.{1,2}\//,`Nonlocal dependency in ${file}: ${spec}`);
      const local=relative(base,resolve(base,dirname(file),spec));
      assert.ok(!local.startsWith('../')&&!isAbsolute(local),`Dependency escapes static root: ${file} -> ${spec}`);
      if(!optional(local))queue.push(local);
    }
  }
  return visited;
}

async function isolated(t){
  await mkdir(join(root,'validation'),{recursive:true});
  const base=await mkdtemp(join(root,'validation/demo-dist-'));
  t.after(()=>rm(base,{recursive:true,force:true}));
  await cp(dist,base,{recursive:true,filter:source=>!optional(relative(dist,source))});
  return base;
}

test('prepared dependency closure matches canonical sources and carries notices',async()=>{
  const manifest=JSON.parse(await readFile(join(dist,'dependency-manifest.json'),'utf8'));
  assert.ok(manifest.files.length>25);
  for(const entry of manifest.files)assert.deepEqual(await readFile(join(dist,entry.file)),await readFile(join(root,entry.source)),entry.file);
  for(const file of ['LICENSE','NOTICE','CITATION.cff','THIRD_PARTY.md'])assert.deepEqual(await readFile(join(dist,file)),await readFile(join(root,file)));
  assert.deepEqual(await readFile(join(dist,'js/crypto/vendor/noble-hashes/LICENSE')),await readFile(join(root,'shared/browser-crypto/crypto/vendor/noble-hashes/LICENSE')));
});

test('isolated dist resolves fetches for every page, imported module and worker without sibling routes',async t=>{
  const base=await isolated(t),files=await crawl(base);
  for(const file of ['app.js','simulation.js','simulation-worker.js','simulation-core.js','simulation-runtime.js',
    'js/crypto-lab/worker.js','js/corrector-lab/worker.js','js/auth-core/session.js','js/neural-auth/receive-pipeline.js'])assert.ok(files.has(file),file);
  const requested=[];
  // Exercise Request/Response routing without a network dependency or browser.
  const origin='https://standalone.example/';
  const staticFetch=async url=>{
    const request=new Request(url);
    const name=new URL(request.url).pathname.slice(1)||'index.html';requested.push(name);
    const path=resolve(base,name);
    if(relative(base,path).startsWith('../'))return new Response('Forbidden',{status:403});
    try{return new Response(await readFile(path),{headers:{'Content-Type':path.endsWith('.js')?'text/javascript':'text/plain'}});}
    catch{return new Response('Not found',{status:404});}
  };
  for(const file of files){
    const response=await staticFetch(new URL(file,origin));assert.equal(response.status,200,file);
    assert.deepEqual(new Uint8Array(await response.arrayBuffer()),new Uint8Array(await readFile(join(base,file))),file);
  }
  assert.ok(requested.every(path=>!path.includes('color-optical-bench')&&!path.includes('qr-neural-key')));
  assert.equal((await staticFetch(new URL('models/neural-corrector/manifest.json',origin))).status,404);
  assert.equal((await staticFetch(new URL('models/simulation-corrector/manifest.json',origin))).status,404);
});

test('an omitted transitive worker asset is detected in a standalone copy',async t=>{
  const base=await isolated(t);
  await rm(join(base,'js/corrector-lab/worker.js'));
  await assert.rejects(crawl(base),error=>error.code==='ENOENT'&&error.path.endsWith('js/corrector-lab/worker.js'));
});
