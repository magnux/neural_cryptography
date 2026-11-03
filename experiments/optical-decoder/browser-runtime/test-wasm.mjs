#!/usr/bin/env node
/** Optional ONNX Runtime Web/WASM parity check in Node; not a browser benchmark. */
import {readFile, writeFile, mkdir, stat} from 'node:fs/promises';
import {resolve, dirname, sep} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {validateManifest, verifyArtifact, decodeFloat32, compareLogits, accumulateParity, summarizeTimings, inputPresence, clipNetworkInput}
  from '../../../apps/color-optical-bench/dist/js/decoder-lab/runtime.js';

function options(args) {
  let artifacts = new URL('../artifacts/smoke/', import.meta.url), out = null;
  for (let i=0; i<args.length; i++) {
    const flag = args[i];
    if (!['--artifacts', '--out'].includes(flag) || !args[i+1] || args[i+1].startsWith('--')) {
      throw new Error('Usage: node test-wasm.mjs [--artifacts DIRECTORY] [--out REPORT.json]');
    }
    const value = resolve(args[++i]);
    if (flag === '--artifacts') artifacts = pathToFileURL(value+sep);
    else out = value;
  }
  return {artifacts, out};
}

async function main() {
  const {artifacts, out} = options(process.argv.slice(2));
  const manifestPath = new URL('manifest.json', artifacts);
  if ((await stat(manifestPath)).size > 65536) throw new Error('Decoder manifest exceeds 64 KiB.');
  const manifest = validateManifest(JSON.parse(await readFile(manifestPath, 'utf8')));
  let ort;
  try { ort = await import('./node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs'); }
  catch { throw new Error('Install the pinned browser runtime first: npm ci --prefix experiments/optical-decoder/browser-runtime'); }
  const runtimeRoot = new URL('./node_modules/onnxruntime-web/dist/', import.meta.url);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = runtimeRoot.href;
  const report = {format: 'optical-decoder-node-wasm-report', version: 1, modelStatus: manifest.status,
    modelProfile: manifest.modelProfile, inputLayout:manifest.inputLayout, frameCounts:manifest.frameCounts,
    minFramesPerInput:manifest.minFramesPerInput,outputLabelSpace:manifest.outputLabelSpace,phaseMasks:manifest.phaseMasks, networkInputClip: manifest.networkInputClip,
    framesPerInput: manifest.framesPerInput, inputChannels: manifest.channels,
    environment: `Node.js ${process.version}`, runtimeVersions: {...ort.env.versions},
    execution: 'ONNX Runtime Web WebGPU bundle, WASM provider, one thread. Not a browser or phone measurement.',
    artifacts: fileURLToPath(artifacts), warmups: 2, measuredRuns: 10, cases: []};
  async function artifact(entry) {
    const url = new URL(entry.file, artifacts);
    if ((await stat(url)).size !== entry.bytes) throw new Error(`Wrong byte length for ${entry.file}.`);
    const bytes = await readFile(url), buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset+bytes.byteLength);
    return verifyArtifact(buffer, entry);
  }
  for (const entry of manifest.cases) {
    const model = await artifact(entry.model);
    const input = decodeFloat32(await artifact(entry.input));
    const expected = decodeFloat32(await artifact(entry.expected));
    const present=inputPresence(input,manifest,entry.size);
    if(present.length!==entry.framesAvailable)throw Error('Fixture presence mask disagrees with framesAvailable.');
    const started = performance.now();
    const session = await ort.InferenceSession.create(model, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    const initMs = performance.now()-started;
    const tensor = new ort.Tensor('float32', input, entry.inputShape), samples = [];
    let parity = null, exportToleranceParity = null, clippingParity = null, maskedSlotParity = null;
    const frameCountChecks=[];
    try {
      if (session.inputNames.length !== 1 || session.inputNames[0] !== manifest.inputName ||
          session.outputNames.length !== 1 || session.outputNames[0] !== manifest.outputName) throw new Error('Export tensor names do not match the manifest.');
      for (let i=0; i<report.warmups+report.measuredRuns; i++) {
        const start = performance.now(), result = await session.run({[manifest.inputName]: tensor});
        try {
          const logits = result[manifest.outputName];
          if (logits.type !== 'float32' || logits.dims.length !== 4 || logits.dims.some((v,k)=>v!==entry.outputShape[k])) throw new Error('Unexpected output type or shape.');
          const values = new Float32Array(logits.data), elapsed = performance.now()-start;
          if (i >= report.warmups) {
            samples.push(elapsed);
            parity = accumulateParity(parity, compareLogits(values, expected));
            exportToleranceParity = accumulateParity(exportToleranceParity, compareLogits(values, expected, manifest.tolerance));
          }
        } finally { for (const value of Object.values(result)) value.dispose?.(); }
      }
      // Fixture storage retains affine-calibration overshoot. Running the same
      // graph on an explicitly clipped copy must give the identical logits.
      const [low, high] = manifest.networkInputClip;
      const clippedInput = clipNetworkInput(input,manifest,entry.size);
      const rgbLength=3*manifest.framesPerInput*(4*entry.size)**2;
      const changedValues = input.subarray(0,rgbLength).reduce((n, value) => n+Number(value<low || value>high), 0);
      if (!changedValues) throw new Error('Numerical fixture must exercise the network input clip.');
      const clippedTensor = new ort.Tensor('float32', clippedInput, entry.inputShape);
      let original, clipped;
      try {
        original = await session.run({[manifest.inputName]: tensor});
        clipped = await session.run({[manifest.inputName]: clippedTensor});
        clippingParity = {...compareLogits(original[manifest.outputName].data,
          clipped[manifest.outputName].data, {atol:0,rtol:0}), changedValues};
      } finally {
        clippedTensor.dispose();
        for (const result of [original, clipped]) for (const value of Object.values(result || {})) value.dispose?.();
      }
      if(manifest.inputLayout==='phase-paired-rgb-presence-planes') {
        const poisoned=input.slice(),one=3*(4*entry.size)**2;
        const missing=Array.from({length:manifest.framesPerInput},(_,i)=>i).filter(i=>!present.includes(i));
        if(!missing.length)throw Error('The variable-frame parity fixture must exercise missing slots.');
        for(const slot of missing)for(let i=slot*one;i<(slot+1)*one;i++)poisoned[i]=20+(i%37)*.1;
        const changed=new ort.Tensor('float32',poisoned,entry.inputShape);
        let reference,actual;
        try {
          reference=await session.run({[manifest.inputName]:tensor});
          actual=await session.run({[manifest.inputName]:changed});
          maskedSlotParity={...compareLogits(actual[manifest.outputName].data,
            reference[manifest.outputName].data,{atol:0,rtol:0}),missingSlots:missing};
        } finally {
          changed.dispose();
          for(const result of [reference,actual])for(const value of Object.values(result||{}))value.dispose?.();
        }
      }
      if(manifest.inputLayout==='phase-paired-rgb-presence-planes') {
        const plane=(4*entry.size)**2,one=3*plane;
        const first=[0,1,2,3].map(phase=>present.find(slot=>Math.floor(slot/2)===phase));
        const order=[...first,...Array.from({length:manifest.framesPerInput},(_,i)=>i).filter(i=>!first.includes(i))];
        // Exercise every advertised observation count. These are runtime/mask
        // checks, not extra Python fixtures or optical recovery measurements.
        for(const framesAvailable of manifest.frameCounts) {
          const values=input.slice(),active=order.slice(0,framesAvailable),missing=order.slice(framesAvailable);
          for(let slot=0;slot<manifest.framesPerInput;slot++) {
            if(active.includes(slot)&&!present.includes(slot))values.set(input.subarray(first[Math.floor(slot/2)]*one,(first[Math.floor(slot/2)]+1)*one),slot*one);
            if(!active.includes(slot))values.fill(0,slot*one,(slot+1)*one);
            values.fill(Number(active.includes(slot)),(3*manifest.framesPerInput+slot)*plane,(3*manifest.framesPerInput+slot+1)*plane);
          }
          inputPresence(values,manifest,entry.size);
          const observation=new ort.Tensor('float32',values,entry.inputShape);
          let result,changed,changedTensor;
          try {
            result=await session.run({[manifest.inputName]:observation});
            const logits=result[manifest.outputName].data;
            if(logits.some(value=>!Number.isFinite(value)))throw Error('Observation-count check returned non-finite logits.');
            let ignored=null;
            if(missing.length) {
              const poisoned=values.slice();
              for(const slot of missing)poisoned.fill(-23.5,slot*one,(slot+1)*one);
              changedTensor=new ort.Tensor('float32',poisoned,entry.inputShape);
              changed=await session.run({[manifest.inputName]:changedTensor});
              ignored=compareLogits(changed[manifest.outputName].data,logits,{atol:0,rtol:0});
            }
            frameCountChecks.push({framesAvailable,logitsFinite:true,maskedSlotParity:ignored,
              passed:ignored===null||ignored.passed});
          } finally {
            observation.dispose();changedTensor?.dispose();
            for(const output of [result,changed])for(const value of Object.values(output||{}))value.dispose?.();
          }
        }
      }
    } finally { tensor.dispose(); await session.release(); }
    report.cases.push({size: entry.size, modelSha256: entry.model.sha256, initMs, parity, exportToleranceParity, clippingParity, maskedSlotParity, frameCountChecks, framesAvailable:present.length,
      timings: summarizeTimings(samples), timingScope: 'session.run plus CPU output copy; excludes parity comparison'});
  }
  report.passed = report.cases.every(entry=>entry.parity.passed && entry.exportToleranceParity.passed && entry.clippingParity.passed && (entry.maskedSlotParity===null||entry.maskedSlotParity.passed) && entry.frameCountChecks.every(check=>check.passed));
  const text = JSON.stringify(report, null, 2)+'\n';
  if (out) { await mkdir(dirname(out), {recursive: true}); await writeFile(out, text); }
  process.stdout.write(text);
  if (!report.passed) process.exitCode = 1;
}
main().catch(error=>{console.error(`Decoder Web/WASM check failed: ${error.message}`);process.exitCode=1;});
