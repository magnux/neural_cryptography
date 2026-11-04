import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,mkdtemp,rm} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {join,dirname} from 'node:path';

const root=fileURLToPath(new URL('../../../',import.meta.url));
const projectNotices=['LICENSE','NOTICE','CITATION.cff','THIRD_PARTY.md'];
const runtimeFiles=['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs','ort-wasm-simd-threaded.asyncify.wasm','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm'];
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
async function put(base,path,contents){await mkdir(dirname(join(base,path)),{recursive:true});await writeFile(join(base,path),contents);}
async function fixture(t,script){
  await mkdir(join(root,'validation'),{recursive:true});
  const base=await mkdtemp(join(root,'validation','licenses-'));
  t.after(()=>rm(base,{recursive:true,force:true}));
  await put(base,'package.json','{"type":"module"}');
  await put(base,`scripts/${script}`,await readFile(join(root,'scripts',script)));
  for(const name of projectNotices)await put(base,name,await readFile(join(root,name)));
  return base;
}

test('prepared standalone apps carry exact project attribution and unchanged vendor notices',async()=>{
  for(const app of ['color-optical-bench','qr-neural-key'])for(const file of projectNotices){
    assert.deepEqual(await readFile(join(root,'apps',app,'dist',file)),await readFile(join(root,file)),`${app}/${file}`);
  }
  for(const app of ['color-optical-bench','qr-neural-key']){
    const source=join(root,'shared/browser-crypto/crypto/vendor/noble-hashes/LICENSE');
    assert.deepEqual(await readFile(join(root,'apps',app,'dist/js/crypto/vendor/noble-hashes/LICENSE')),await readFile(source));
  }
  const qr=join(root,'apps/qr-neural-key/dist/vendor');
  for(const file of ['jsQR-LICENSE.txt','qrcodegen-LICENSE.txt'])assert.deepEqual(
    await readFile(join(root,'apps/color-optical-bench/dist/js/qr-core/vendor',file)),await readFile(join(qr,file)));
});

test('missing project notices fail preparation before existing app files are touched',async t=>{
  const base=await fixture(t,'prepare-apps.mjs');
  await rm(join(base,'NOTICE'));
  await put(base,'apps/color-optical-bench/dist/NOTICE','existing attribution');
  const result=spawnSync(process.execPath,[join(base,'scripts/prepare-apps.mjs'),'--app','color-optical-bench'],{encoding:'utf8'});
  assert.notEqual(result.status,0);assert.match(result.stderr,/Missing canonical file: .*NOTICE/);
  assert.equal(await readFile(join(base,'apps/color-optical-bench/dist/NOTICE'),'utf8'),'existing attribution');
});

test('model preparation carries project and runtime notices separately without altering the manifest',async t=>{
  const base=await fixture(t,'prepare-corrector.mjs');
  await put(base,'apps/color-optical-bench/dist/js/corrector-lab/runtime.js',await readFile(join(root,'apps/color-optical-bench/dist/js/corrector-lab/runtime.js')));
  const runtime='experiments/neural-corrector/browser-runtime';
  await put(base,`${runtime}/node_modules/onnxruntime-web/package.json`,'{"version":"1.23.2"}');
  for(const name of runtimeFiles)await put(base,`${runtime}/node_modules/onnxruntime-web/dist/${name}`,`test runtime ${name}`);
  for(const name of ['LICENSE.onnxruntime','ThirdPartyNotices.onnxruntime.txt'])await put(base,`${runtime}/${name}`,`third-party ${name}`);
  const artifact=async(file,bytes)=>{await put(base,`artifacts/${file}`,bytes);return {file,bytes:bytes.length,sha256:hash(bytes)};};
  // Tiny opaque model and correctly sized tensor files exercise packaging only.
  // No inference runtime or model is executed by preparation.
  const manifest={format:'neural-corrector',version:1,status:'untrained',modelProfile:'U-tube-2x-attn',
    framesPerInput:8,minFramesPerInput:4,frameCounts:[4,5,6,7,8],inputLayout:'phase-paired-rgb-presence-planes',
    outputLabelSpace:'logical',phaseMasks:[0,15,7,8],parameterCount:432848,pixelsPerCell:4,channels:32,classes:16,
    normalization:{divisor:127.5,offset:-1,clip:[-1.5,1.5]},networkInputClip:[-1,1],tolerance:{atol:1e-4,rtol:1e-3},
    inputName:'pixels',outputName:'logits',seed:7,opset:17,versions:{torch:'fixture'},cases:[{size:64,framesAvailable:4,
      inputShape:[1,32,256,256],outputShape:[1,16,64,64],model:await artifact('model-64.onnx',Buffer.from('packaging fixture')),
      input:await artifact('input-64.f32',Buffer.alloc(32*256*256*4)),expected:await artifact('expected-64.f32',Buffer.alloc(16*64*64*4))}]};
  await put(base,'artifacts/manifest.json',JSON.stringify(manifest));
  const run=()=>spawnSync(process.execPath,[join(base,'scripts/prepare-corrector.mjs'),'--artifacts',join(base,'artifacts')],{encoding:'utf8'});
  const result=run();assert.equal(result.status,0,result.stderr);
  const served=join(base,'apps/color-optical-bench/dist');
  for(const file of ['LICENSE','NOTICE'])assert.deepEqual(await readFile(join(served,'models/neural-corrector',file)),await readFile(join(base,file)));
  for(const file of ['LICENSE.onnxruntime','ThirdPartyNotices.onnxruntime.txt'])assert.equal(
    await readFile(join(served,'vendor/onnxruntime',file),'utf8'),`third-party ${file}`);
  assert.deepEqual(JSON.parse(await readFile(join(served,'models/neural-corrector/manifest.json'),'utf8')),manifest);
  await rm(join(base,'NOTICE'));
  const rejected=run();assert.notEqual(rejected.status,0);assert.match(rejected.stderr,/Missing project NOTICE/);
  assert.deepEqual(await readFile(join(served,'models/neural-corrector/NOTICE')),await readFile(join(root,'NOTICE')));
});
