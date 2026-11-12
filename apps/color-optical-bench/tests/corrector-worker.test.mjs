import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {packetFrames} from '../dist/js/neural-auth/optical.js';
import {encodePacket} from '../dist/js/link/protocol.js';
import {renderNeuralQR} from '../dist/js/neural-qr/layout.js';

const workerURL=new URL('../dist/js/corrector-lab/worker.js',import.meta.url);
const original=await readFile(workerURL,'utf8');
let serial=0;
function fixture(sizes=[32,64,128]) {
  const files=new Map();
  const artifact=(file,buffer)=>{const bytes=new Uint8Array(buffer);files.set(file,bytes);return {file,bytes:bytes.byteLength,sha256:createHash('sha256').update(bytes).digest('hex')};};
  const manifest={format:'neural-corrector',version:1,status:'trained',checkpointSha256:'b'.repeat(64),training:{steps:1},
    modelProfile:'VW',parameterCount:696400,datasetFormat:'neural-qr-dataset',datasetVersion:2,pixelsPerCell:4,
    inputLayout:'phase-paired-rgb-presence-planes',outputLabelSpace:'logical',framesPerInput:8,minFramesPerInput:4,
    frameCounts:[4,5,6,7,8],phaseMasks:[0,15,7,8],channels:32,classes:16,inputName:'pixels',outputName:'logits',
    normalization:{divisor:127.5,offset:-1,clip:[-1.5,1.5]},networkInputClip:[-1,1],tolerance:{atol:1e-4,rtol:1e-3},
    seed:7,opset:17,versions:{torch:'test'},anchor:{representation:'dct-low-frequency',maxModes:16,normalization:'ortho',spatialScale:'grid-over-modes'},
    anchorDeployment:'baked-selected-spatial-grid',bottleneck:{stages:3,blocks:['conv','attention'],layersPerBlock:2,scales:[4,2,1,2,4],fusion:'normalized-residual'},
    cases:sizes.map(size=>{
      const plane=(4*size)**2,input=new Float32Array(32*plane),expected=new Float32Array(16*size*size);
      for(const slot of [0,2,4,6])input.fill(1,(24+slot)*plane,(25+slot)*plane);
      return {size,framesAvailable:4,inputShape:[1,32,4*size,4*size],outputShape:[1,16,size,size],
        model:artifact(`model-${size}.onnx`,Uint8Array.of(size).buffer),input:artifact(`input-${size}.f32`,input.buffer),expected:artifact(`expected-${size}.f32`,expected.buffer)};
    })};
  files.set('manifest.json',new TextEncoder().encode(JSON.stringify(manifest)));
  return {manifest,files};
}
async function harness({sizes,gpu=false,badGPU=false,badWASM=false,corrupt=false}={}) {
  const {manifest,files}=fixture(sizes),state={offline:false,fetches:0,created:[],released:[],runs:[]};
  const saved=new Map(['fetch','navigator','location','self','__correctorTestRuntime'].map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const set=(name,value)=>Object.defineProperty(globalThis,name,{configurable:true,writable:true,value});
  set('location',{origin:'https://example.test'});set('navigator',{gpu:gpu?{}:undefined});set('self',{postMessage(){}});
  set('fetch',async url=>{
    state.fetches++;if(state.offline)throw Error('network offline');
    const file=new URL(url).pathname.split('/').at(-1),bytes=files.get(file);
    if(!bytes)throw Error(`Unknown test asset: ${file}`);
    const output=corrupt&&file.endsWith('.onnx')?Uint8Array.of(0):bytes;
    return {ok:true,url:`https://example.test/${file}`,body:new ReadableStream({start(controller){controller.enqueue(output);controller.close();}})};
  });
  set('__correctorTestRuntime',{
    env:{wasm:{},versions:{test:'mock'}},Tensor:class {constructor(type,data,dims){Object.assign(this,{type,data,dims});}dispose(){}},
    InferenceSession:{async create(buffer,options){
      const size=new Uint8Array(buffer)[0],backend=options.executionProviders[0],id=`${size}/${backend}/${state.created.length}`;
      state.created.push(id);
      return {inputNames:['pixels'],outputNames:['logits'],async release(){state.released.push(id);},async run(){
        state.runs.push(id);if(backend==='webgpu'&&badGPU)throw Error('GPU execution failed');
        const data=new Float32Array(16*size*size);if(backend==='wasm'&&badWASM)data[0]=1;
        return {logits:{type:'float32',dims:[1,16,size,size],data,dispose(){}}};
      }};
    }}
  });
  const source=original.replace(/from '(\.[^']+)'/g,(_,path)=>`from '${new URL(path,workerURL).href}'`)
    .replaceAll('import.meta.url',JSON.stringify(workerURL.href))
    .replace('let ort, busy = false','let ort=globalThis.__correctorTestRuntime, busy = false')+
    `\nexport {initializeTransport,load,sessions,assets,scanTransport};\n// harness ${serial++}`;
  const worker=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
  return {worker,state,manifest,restore(){for(const [name,descriptor]of saved){if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];}}};
}

test('live initialization verifies every receiving geometry and cached fallback remains usable offline',async()=>{
  const h=await harness();try{
    const report=await h.worker.initializeTransport({size:128,backend:'wasm'},()=>{});
    assert.equal(report.offlineReady,true);assert.equal(report.parity.passed,true);assert.equal(report.effectiveBackend,'wasm');
    assert.deepEqual(report.availableSizes,[32,64,128]);assert.equal(report.parity.cases.length,3);
    assert.equal(h.worker.sessions.size,2);assert.equal(h.worker.assets.size,3);
    assert.equal(h.state.released.length,1);
    const fetches=h.state.fetches;h.state.offline=true;
    const fallback=await h.worker.load({size:64,backend:'wasm'},()=>{},{transport:true,manifest:h.manifest});
    assert.equal(fallback.entry.size,64);assert.equal(fallback.parity.passed,true);
    assert.equal(h.state.fetches,fetches);assert.equal(h.worker.sessions.size,2);assert.equal(h.worker.assets.size,3);
  }finally{h.restore();}
});

test('auto backend warms WASM offline fallback and rechecks parity after WebGPU execution fails',async()=>{
  const h=await harness({sizes:[32],gpu:true,badGPU:true});try{
    const report=await h.worker.initializeTransport({size:32,backend:'auto'},()=>{});
    assert.equal(report.wasmFallbackReady,true);assert.equal(report.effectiveBackend,'wasm');
    assert.equal(report.parity.passed,true);assert.match(report.fallbackReason,/WebGPU inference\/parity failed/);
    assert.ok(h.state.created[0].includes('/wasm/'));assert.ok(h.state.runs.some(id=>id.includes('/webgpu/')));
    assert.ok(h.state.released.some(id=>id.includes('/webgpu/')));
  }finally{h.restore();}
});

test('wrong inference or artifact bytes cannot initialize live authentication',async()=>{
  for(const options of [{badWASM:true},{corrupt:true}]) {
    const h=await harness({sizes:[32],...options});try{
      await assert.rejects(h.worker.initializeTransport({size:32,backend:'wasm'},()=>{}),/parity failed|SHA-256/);
      assert.equal(h.worker.sessions.size,0);
    }finally{h.restore();}
  }
});


test('live worker forwards the configured reading interval through acquisition',async()=>{
  const h=await harness({sizes:[32]});try{
    await h.worker.initializeTransport({size:32,backend:'wasm'},()=>{});
    const packet=encodePacket({kind:'data',messageType:'raw',sessionId:'12'.repeat(16),messageId:'34'.repeat(8),index:0,data:new Uint8Array(16)});
    const frames=packetFrames(packet,{size:32});
    const scan=(phase,time,captureIntervalMs=500)=>h.worker.scanTransport({size:32,backend:'wasm'},()=>{},
      {image:renderNeuralQR(frames[phase],512),runId:1,mediaTime:time,presentedFrames:phase+1,expected:'data',captureIntervalMs});
    const first=await scan(0,0);assert.equal(first.stats.geometryMaxAgeMs,1000);
    assert.equal(first.stats.geometrySource,'detected');
    const second=await scan(1,.5);assert.equal(second.stats.geometrySource,'previous');
    assert.equal(second.stats.captureIntervalMs,500);
    const third=await scan(2,1);assert.equal(third.stats.geometrySource,'detected');
    await assert.rejects(scan(3,1.5,0),/interval/i);
  }finally{h.restore();}
});
