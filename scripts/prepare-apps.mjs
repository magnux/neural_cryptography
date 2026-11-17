#!/usr/bin/env node
/** Copy canonical crypto and authentication into independently servable apps. */
import { cp, mkdir, readFile, writeFile, rm, stat, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, dirname, resolve, relative, isAbsolute } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const shared = join(root, 'shared', 'browser-crypto');
const fixtures = join(root, 'shared', 'fixtures', 'zk');
const projectNotices = ['LICENSE', 'NOTICE', 'CITATION.cff', 'THIRD_PARTY.md'];
const workers = ['client.js', 'operations.js', 'reference-fixture.js', 'worker.js'];
const applications = {
  'color-optical-bench': 'crypto-lab',
  'qr-neural-key': 'crypto-worker',
  'neural-auth-demo': 'crypto-lab',
};

/** Follow the demo's static imports and worker URLs back to canonical sources.
 * The mirrored JS tree is generated in full; it never imports a sibling app.
 * Runtime/model directories are optional and prepared by prepare-corrector.mjs.
 */
async function demoDependencies() {
  const dist = join(root, 'apps/neural-auth-demo/dist');
  const bench = join(root, 'apps/color-optical-bench/dist');
  const pending = (await readdir(dist)).filter(file => /\.m?js$/.test(file));
  if (!pending.includes('app.js')) throw Error('The demo must contain dist/app.js.');
  const visited = new Set(), generated = new Map();
  while (pending.length) {
    const local = pending.pop();
    if (visited.has(local)) continue;
    visited.add(local);
    let source = join(dist, local);
    if (local.startsWith('js/')) {
      if (local.startsWith('js/crypto/')) source = join(shared, 'crypto', local.slice('js/crypto/'.length));
      else if (local.startsWith('js/auth-core/')) source = join(shared, 'auth', local.slice('js/auth-core/'.length));
      else if (local.startsWith('js/crypto-lab/') && workers.includes(local.slice('js/crypto-lab/'.length))) {
        source = join(shared, 'worker', local.slice('js/crypto-lab/'.length));
      } else source = join(bench, local);
      generated.set(local, source);
    }
    await requirePath(source);
    const code = await readFile(source, 'utf8');
    const specs = [
      ...code.matchAll(/^[ \t]*(?:import|export)\s+(?:[^;'"`]*?\s+from\s*)?['"]([^'"]+)['"]/gm),
      ...code.matchAll(/\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g),
      ...code.matchAll(/\bnew\s+URL\s*\(\s*['"]([^'"]+)['"]\s*,\s*import\.meta\.url\s*\)/g),
    ];
    for (const [, spec] of specs) {
      if (!spec.startsWith('./') && !spec.startsWith('../')) throw Error(`Nonlocal demo dependency in ${source}: ${spec}`);
      const dependency = relative(dist, resolve(dist, dirname(local), spec));
      if (dependency.startsWith('../') || isAbsolute(dependency)) throw Error(`Demo dependency escapes dist: ${local} → ${spec}`);
      // Model/runtime directory URLs are fetched only after explicit preparation.
      if (dependency.startsWith('models/') || dependency==='vendor/onnxruntime' || dependency.startsWith('vendor/onnxruntime/')) continue;
      if (/\.m?js$/.test(dependency)) pending.push(dependency);
      else throw Error(`Unrecognized demo module dependency: ${local} → ${spec}`);
    }
  }
  return generated;
}

async function requirePath(path, directory = false) {
  const info = await stat(path).catch(() => null);
  if (!info || (directory ? !info.isDirectory() : !info.isFile())) {
    throw new Error(`Missing canonical ${directory ? 'directory' : 'file'}: ${path}. Restore it from Git before preparing the apps.`);
  }
}

async function prepare() {
  const args = process.argv.slice(2);
  let names = Object.keys(applications);
  if (args.length) {
    if (args.length !== 2 || args[0] !== '--app' || !Object.hasOwn(applications, args[1])) {
      throw new Error(`Usage: node scripts/prepare-apps.mjs [--app ${Object.keys(applications).join('|')}]`);
    }
    names = [args[1]];
  }

  // Validate all canonical inputs before replacing any generated output.
  const notices = new Map();
  for (const file of projectNotices) {
    await requirePath(join(root, file));
    notices.set(file, await readFile(join(root, file)));
  }
  await requirePath(join(shared, 'crypto'), true);
  await requirePath(fixtures, true);
  for (const name of ['bytes.js', 'circuit.js', 'hashes.js', 'proof.js',
    'vendor/noble-hashes/sha3.js', 'vendor/noble-hashes/_u64.js',
    'vendor/noble-hashes/utils.js', 'vendor/noble-hashes/LICENSE']) {
    await requirePath(join(shared, 'crypto', name));
  }
  await requirePath(join(fixtures, 'manifest.json'));
  const manifest = JSON.parse(await readFile(join(fixtures, 'manifest.json'), 'utf8'));
  if (manifest.version !== 2 || !Array.isArray(manifest.cases) || !manifest.cases.length) {
    throw new Error('The canonical fixture manifest must contain current compact proof cases.');
  }
  for (const fixture of manifest.cases) {
    if (typeof fixture.proofFile !== 'string' || !/^[a-z0-9-]+\.bin$/.test(fixture.proofFile)) {
      throw new Error('The canonical fixture manifest contains an invalid proof filename.');
    }
    await requirePath(join(fixtures, fixture.proofFile));
  }
  for (const name of workers) await requirePath(join(shared, 'worker', name));
  for (const name of ['envelope.js', 'session.js']) await requirePath(join(shared, 'auth', name));
  const demoFiles = names.includes('neural-auth-demo') ? await demoDependencies() : null;

  for (const name of names) {
    const app = join(root, 'apps', name);
    await requirePath(join(app, 'dist', 'index.html'));
    if (name === 'neural-auth-demo') {
      await rm(join(app, 'dist/js'), {recursive:true, force:true});
      for (const [local, source] of demoFiles) {
        await mkdir(dirname(join(app, 'dist', local)), {recursive:true});
        await cp(source, join(app, 'dist', local));
      }
      await writeFile(join(app, 'dist/dependency-manifest.json'), JSON.stringify({
        generatedBy:'scripts/prepare-apps.mjs',
        files:[...demoFiles].sort(([a], [b]) => a.localeCompare(b)).map(([file, source]) => ({file, source:relative(root, source)})),
      }, null, 2)+'\n');
    }
    for (const [file, bytes] of notices) await writeFile(join(app, 'dist', file), bytes);
    const crypto = join(app, 'dist', 'js', 'crypto');
    const localFixtures = join(app, 'tests', 'fixtures', 'zk');
    const authentication = join(app, 'dist', 'js', 'auth-core');
    // These directories contain only ignored generated files. Removing them
    // also removes obsolete modules when the canonical implementation changes.
    await rm(crypto, { recursive: true, force: true });
    await rm(localFixtures, { recursive: true, force: true });
    await rm(authentication, { recursive: true, force: true });
    await cp(join(shared, 'crypto'), crypto, { recursive: true });
    await cp(fixtures, localFixtures, { recursive: true });
    await cp(join(shared, 'auth'), authentication, { recursive: true });
    const worker = join(app, 'dist', 'js', applications[name]);
    if (name === 'qr-neural-key') await rm(worker, { recursive: true, force: true });
    await mkdir(worker, { recursive: true });
    // The color app keeps its own lab.js and reports.js beside these modules.
    for (const file of workers) await cp(join(shared, 'worker', file), join(worker, file));
    if (name === 'color-optical-bench') {
      // This generated directory is no longer used by Neural-QR authentication.
      await rm(join(app, 'dist', 'js', 'qr-core'), {recursive:true, force:true});
    }
    console.log(`Prepared ${name}: shared crypto, authentication, worker and public test fixtures.`);
  }
}

prepare().catch(error => {
  console.error(`App preparation failed: ${error.message}`);
  process.exitCode = 1;
});

