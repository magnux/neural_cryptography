import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const dist = new URL('../dist/', import.meta.url);
async function filesAt(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const url = new URL(entry.name + (entry.isDirectory() ? '/' : ''), directory);
    files.push(...(entry.isDirectory() ? await filesAt(url) : [url]));
  }
  return files;
}

test('distribution has all HTML assets and UI element references', async () => {
  for (const [page, controller] of [['index.html', 'js/app.js'], ['auth.html', 'js/auth/app.js']]) {
  const html = await readFile(new URL(page, dist), 'utf8');
  const app = await readFile(new URL(controller, dist), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
  assert.equal(new Set(ids).size, ids.length, 'duplicate HTML IDs');
  for (const match of app.matchAll(/(?:\$|\bn)\('([^']+)'\)/g)) {
    assert.ok(ids.includes(match[1]), `missing control ${match[1]}`);
  }
  for (const match of html.matchAll(/(?:src|href)="(\.[^"]+)"/g)) {
    await stat(new URL(match[1], dist));
  }
  }
});

test('every shipped module parses and its local import/worker paths exist', async () => {
  const files = (await filesAt(dist)).filter(f => f.pathname.endsWith('.js'));
  for (const url of files) {
    const parsed = spawnSync(process.execPath, ['--check', fileURLToPath(url)], { encoding: 'utf8' });
    assert.equal(parsed.status, 0, parsed.stderr);
    const source = await readFile(url, 'utf8');
    for (const match of source.matchAll(/(?:from\s+|import\s*|new URL\(\s*)['"](\.[^'"]+)['"]/g)) {
      await stat(new URL(match[1], url));
    }
  }
});

test('bundled QR dependencies match the recorded hashes and include licenses', async () => {
  const manifest = JSON.parse(await readFile(new URL('vendor/manifest.json', dist)));
  for (const dependency of manifest.dependencies) {
    const source = await readFile(new URL(`vendor/${dependency.file}`, dist));
    assert.equal(createHash('sha256').update(source).digest('hex'), dependency.sha256);
  }
  for (const name of ['qrcodegen-LICENSE.txt', 'jsQR-LICENSE.txt']) {
    assert.ok((await readFile(new URL(`vendor/${name}`, dist), 'utf8')).length > 500);
  }
});
