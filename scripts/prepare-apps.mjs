#!/usr/bin/env node
/** Copy the one canonical crypto implementation into independently servable apps. */
import { cp, mkdir, readFile, rm, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const shared = join(root, 'shared', 'browser-crypto');
const fixtures = join(root, 'shared', 'fixtures', 'zk');
const workers = ['client.js', 'operations.js', 'reference-fixture.js', 'worker.js'];
const applications = {
  'color-optical-bench': 'crypto-lab',
  'qr-neural-key': 'crypto-worker',
};

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
      throw new Error('Usage: node scripts/prepare-apps.mjs [--app color-optical-bench|qr-neural-key]');
    }
    names = [args[1]];
  }

  // Validate all canonical inputs before replacing any generated output.
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

  for (const name of names) {
    const app = join(root, 'apps', name);
    await requirePath(join(app, 'dist', 'index.html'));
    const crypto = join(app, 'dist', 'js', 'crypto');
    const localFixtures = join(app, 'tests', 'fixtures', 'zk');
    // These directories contain only ignored generated files. Removing them
    // also removes obsolete modules when the canonical implementation changes.
    await rm(crypto, { recursive: true, force: true });
    await rm(localFixtures, { recursive: true, force: true });
    await cp(join(shared, 'crypto'), crypto, { recursive: true });
    await cp(fixtures, localFixtures, { recursive: true });
    const worker = join(app, 'dist', 'js', applications[name]);
    if (name === 'qr-neural-key') await rm(worker, { recursive: true, force: true });
    await mkdir(worker, { recursive: true });
    // The color app keeps its own lab.js and reports.js beside these modules.
    for (const file of workers) await cp(join(shared, 'worker', file), join(worker, file));
    console.log(`Prepared ${name}: shared crypto, worker, and public test fixtures.`);
  }
}

prepare().catch(error => {
  console.error(`App preparation failed: ${error.message}`);
  process.exitCode = 1;
});
