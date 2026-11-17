#!/usr/bin/env node
/** Prepare optional, local-only corrector inference assets. Never downloads or deploys. */
import {readFile, writeFile, mkdir, cp, stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {join, resolve} from 'node:path';
import {validateManifest} from '../apps/color-optical-bench/dist/js/corrector-lab/runtime.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const experiment = join(root, 'experiments/neural-corrector');
const applications = ['color-optical-bench', 'neural-auth-demo'];
const version = '1.23.2';
const runtimeFiles = [
  'ort.webgpu.min.mjs',
  'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm',
  'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
];
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

async function prepare() {
  const args = process.argv.slice(2);
  const options = new Map();
  for (let index=0; index<args.length; index+=2) {
    const name=args[index], value=args[index+1];
    if (!['--artifacts','--simulation-artifacts','--app'].includes(name) || !value || options.has(name)) {
      throw Error('Usage: node scripts/prepare-corrector.mjs [--artifacts directory | --simulation-artifacts directory] [--app color-optical-bench|neural-auth-demo]');
    }
    options.set(name,value);
  }
  const simulation=options.has('--simulation-artifacts');
  if (simulation && options.has('--artifacts')) throw Error('Prepare live and simulation artifacts separately; their model directories must stay distinct.');
  const selected=options.get('--app');
  if (selected && !applications.includes(selected)) throw Error(`Unknown corrector app: ${selected}`);
  if (simulation && selected && selected!=='neural-auth-demo') throw Error('Simulation assets belong only to neural-auth-demo.');
  const names=selected?[selected]:simulation?['neural-auth-demo']:applications;
  // These notices cover the project contribution, not independently licensed
  // checkpoint content. Read all inputs before replacing any served assets.
  const projectNotices = new Map();
  for (const file of ['LICENSE', 'NOTICE']) {
    try { projectNotices.set(file, await readFile(join(root, file))); }
    catch { throw Error(`Missing project ${file}. Restore it from Git before preparing model assets.`); }
  }
  const artifactOption=options.get(simulation?'--simulation-artifacts':'--artifacts');
  const artifacts = artifactOption ? resolve(artifactOption) : join(experiment, 'artifacts/smoke');
  const pkgPath = join(experiment, 'browser-runtime/node_modules/onnxruntime-web');
  let pkg;
  try { pkg = JSON.parse(await readFile(join(pkgPath, 'package.json'), 'utf8')); }
  catch { throw Error('Install the pinned runtime first: npm ci --prefix experiments/neural-corrector/browser-runtime'); }
  if (pkg.version !== version) throw Error(`Expected onnxruntime-web ${version}; found ${pkg.version}. Run npm ci in browser-runtime.`);
  let manifest;
  try {
    const path = join(artifacts, 'manifest.json');
    if ((await stat(path)).size > 65536) throw Error('Manifest too large.');
    manifest = validateManifest(JSON.parse(await readFile(path, 'utf8')));
  } catch (error) {
    throw Error(`Export the model first: python experiments/neural-corrector/export.py. ${error.message}`);
  }
  if (simulation && (manifest.status!=='untrained' || manifest.modelProfile!=='VW' || ![32,64].every(size=>manifest.cases.some(item=>item.size===size)))) {
    throw Error('Simulation requires an untrained VW export with G32 and G64. It cannot replace live trained corrector assets.');
  }

  // Validate every source before replacing any served file.
  const modelAssets = new Map();
  for (const item of manifest.cases) for (const kind of ['model', 'input', 'expected']) {
    const entry = item[kind];
    const path = join(artifacts, entry.file);
    if ((await stat(path)).size !== entry.bytes) throw Error(`Size mismatch: ${entry.file}`);
    const bytes = await readFile(path);
    if (hash(bytes) !== entry.sha256) throw Error(`Hash mismatch: ${entry.file}`);
    modelAssets.set(entry.file, bytes);
  }
  const runtime = [];
  for (const file of runtimeFiles) {
    const path = join(pkgPath, 'dist', file);
    const bytes = await readFile(path);
    runtime.push({file, bytes:bytes.length, sha256:hash(bytes), path});
  }
  const licenses = ['LICENSE.onnxruntime', 'ThirdPartyNotices.onnxruntime.txt'];
  for (const file of licenses) await stat(join(experiment, 'browser-runtime', file));

  for (const name of names) {
    const app=join(root,'apps',name,'dist');
    const modelTarget = join(app, simulation?'models/simulation-corrector':'models/neural-corrector');
    const runtimeTarget = join(app, 'vendor/onnxruntime');
    await mkdir(modelTarget, {recursive:true});
    await mkdir(runtimeTarget, {recursive:true});
    for (const [file, bytes] of modelAssets) await writeFile(join(modelTarget, file), bytes);
    for (const [file, bytes] of projectNotices) await writeFile(join(modelTarget, file), bytes);
    for (const entry of runtime) await cp(entry.path, join(runtimeTarget, entry.file));
    for (const file of licenses) await cp(join(experiment, 'browser-runtime', file), join(runtimeTarget, file));
    await writeFile(join(runtimeTarget, 'manifest.json'), JSON.stringify({
      package:'onnxruntime-web', version,
      source:`https://www.npmjs.com/package/onnxruntime-web/v/${version}`,
      files:runtime.map(({path, ...entry}) => entry), licenses,
    }, null, 2)+'\n');
    // Write the manifest last, so readers never see a new manifest with old files.
    await writeFile(join(modelTarget, 'manifest.json'), JSON.stringify(manifest, null, 2)+'\n');
    console.log(`Prepared ${name}: ${simulation?'simulation-only':'corrector'} assets and local runtime.`);
  }
  console.log(`Prepared ${manifest.status} ${manifest.frameCounts.join("/")}-observation corrector fixtures (${manifest.framesPerInput} slots) for ${manifest.cases.map(c=>c.size).join('/')} and ONNX Runtime Web ${version}.`);
  console.log('Serve the selected app dist directory. Assets remain local and ignored by Git.');
}

prepare().catch(error => { console.error(`Corrector preparation failed: ${error.message}`); process.exitCode=1; });
