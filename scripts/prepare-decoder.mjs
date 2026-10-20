#!/usr/bin/env node
/** Prepare optional, local-only M1 inference assets. Never downloads or deploys. */
import {readFile, writeFile, mkdir, cp, stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {join, resolve} from 'node:path';
import {validateManifest} from '../apps/color-optical-bench/dist/js/decoder-lab/runtime.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const experiment = join(root, 'experiments/optical-decoder');
const app = join(root, 'apps/color-optical-bench/dist');
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
  if (args.length && (args.length !== 2 || args[0] !== '--artifacts')) {
    throw Error('Usage: node scripts/prepare-decoder.mjs [--artifacts directory]');
  }
  const artifacts = args.length ? resolve(args[1]) : join(experiment, 'artifacts/smoke');
  const pkgPath = join(experiment, 'browser-runtime/node_modules/onnxruntime-web');
  let pkg;
  try { pkg = JSON.parse(await readFile(join(pkgPath, 'package.json'), 'utf8')); }
  catch { throw Error('Install the pinned runtime first: npm ci --prefix experiments/optical-decoder/browser-runtime'); }
  if (pkg.version !== version) throw Error(`Expected onnxruntime-web ${version}; found ${pkg.version}. Run npm ci in browser-runtime.`);
  let manifest;
  try {
    const path = join(artifacts, 'manifest.json');
    if ((await stat(path)).size > 65536) throw Error('Manifest too large.');
    manifest = validateManifest(JSON.parse(await readFile(path, 'utf8')));
  } catch (error) {
    throw Error(`Export the M1 model first: python experiments/optical-decoder/export.py. ${error.message}`);
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

  const modelTarget = join(app, 'models/optical-decoder');
  const runtimeTarget = join(app, 'vendor/onnxruntime');
  await mkdir(modelTarget, {recursive:true});
  await mkdir(runtimeTarget, {recursive:true});
  for (const [file, bytes] of modelAssets) await writeFile(join(modelTarget, file), bytes);
  for (const entry of runtime) await cp(entry.path, join(runtimeTarget, entry.file));
  for (const file of licenses) await cp(join(experiment, 'browser-runtime', file), join(runtimeTarget, file));
  await writeFile(join(runtimeTarget, 'manifest.json'), JSON.stringify({
    package:'onnxruntime-web', version,
    source:`https://www.npmjs.com/package/onnxruntime-web/v/${version}`,
    files:runtime.map(({path, ...entry}) => entry), licenses,
  }, null, 2)+'\n');
  // Write the manifest last, so readers never see a new manifest with old files.
  await writeFile(join(modelTarget, 'manifest.json'), JSON.stringify(manifest, null, 2)+'\n');
  console.log(`Prepared untrained decoder fixtures for ${manifest.cases.map(c=>c.size).join('/')} and ONNX Runtime Web ${version}.`);
  console.log('Serve apps/color-optical-bench/dist and open decoder.html. Assets remain local and ignored by Git.');
}

prepare().catch(error => { console.error(`Decoder preparation failed: ${error.message}`); process.exitCode=1; });
