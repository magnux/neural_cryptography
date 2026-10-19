#!/usr/bin/env node
/** Cross-language gate. Requires Python + NumPy; npm test itself does not. */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { fromHex } from '../dist/js/crypto/bytes.js';
import { prove, verify } from '../dist/js/crypto/proof.js';
import { shake256 } from '../dist/js/crypto/hashes.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const repository = fileURLToPath(new URL('../../../', import.meta.url));
const fixtureDirectory = join(repository, 'shared', 'fixtures', 'zk');
const python = process.env.PYTHON || 'python3';
const temporary = await mkdtemp(join(tmpdir(), 'neural-key-parity-'));
function reference(...args) {
  return JSON.parse(execFileSync(python, [join(root, 'scripts', 'zk-fixtures.py'), ...args],
    { encoding: 'utf8', cwd: root, timeout: 120000, maxBuffer: 1024 * 1024 }));
}
const elapsed = start => Math.round((performance.now() - start) * 100) / 100;
// This deterministic hook is confined to the test process, never the app.
async function deterministicProof(fixture) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  const native = globalThis.crypto;
  const source = shake256(fromHex(fixture.entropySeedHex, fixture.entropySeedHex.length / 2), fixture.expected.entropyBytes);
  let position = 0;
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: {
    subtle: native.subtle,
    getRandomValues(array) {
      assert.ok(position + array.byteLength <= source.length);
      new Uint8Array(array.buffer, array.byteOffset, array.byteLength).set(source.subarray(position, position + array.byteLength));
      position += array.byteLength;
      return array;
    },
  } });
  try {
    const result = await prove(fromHex(fixture.secretHex, 32), fixture.public, fixture.challenge);
    assert.equal(position, source.length);
    return result;
  } finally {
    Object.defineProperty(globalThis, 'crypto', descriptor);
  }
}
try {
  const storedManifestBytes = await readFile(join(fixtureDirectory, 'manifest.json'));
  const manifest = JSON.parse(storedManifestBytes);
  assert.equal(manifest.version, 2);
  const generatedDirectory = join(temporary, 'python');
  const generatedBrowser = join(temporary, 'reference-fixture.js');
  const regenerated = reference('generate', '--out', generatedDirectory, '--browser-fixture', generatedBrowser);
  assert.equal(regenerated.cases, manifest.cases.length);
  assert.deepEqual(await readFile(join(generatedDirectory, 'manifest.json')), storedManifestBytes,
    'Python fixture manifest is reproducible');
  assert.deepEqual(await readFile(generatedBrowser), await readFile(join(repository, 'shared', 'browser-crypto', 'worker', 'reference-fixture.js')),
    'Public-only browser fixture is reproducible');
  const cases = [], timings = [];
  for (const fixture of manifest.cases) {
    const regeneratedProof = new Uint8Array(await readFile(join(generatedDirectory, fixture.proofFile)));
    assert.deepEqual(regeneratedProof, new Uint8Array(await readFile(join(fixtureDirectory, fixture.proofFile))));
    assert.deepEqual(await deterministicProof(fixture), regeneratedProof,
      'Python and JavaScript compact proofs are byte-for-byte identical');
    let started = performance.now();
    assert.equal(await verify(fixture.public, fixture.challenge, regeneratedProof), true);
    const pythonProofVerifyMs = elapsed(started);
    started = performance.now();
    // Fresh native randomness, independent of the deterministic parity tests.
    const proof = await prove(fromHex(fixture.secretHex, 32), fixture.public, fixture.challenge);
    const proveMs = elapsed(started);
    assert.notDeepEqual(proof, regeneratedProof, 'fresh proof must differ from fixture proof');
    started = performance.now();
    assert.equal(await verify(fixture.public, fixture.challenge, proof), true);
    const verifyMs = elapsed(started);
    const proofFile = `${fixture.id}-js.bin`;
    await writeFile(join(temporary, proofFile), proof);
    const valid = { id: fixture.id, public: fixture.public, challenge: fixture.challenge, proofFile };
    cases.push(valid);
    const stricter = { ...fixture.challenge, required_repetitions: fixture.expected.repetitions + 1 };
    const alteredChallenges = [['stricter-policy', stricter]];
    // Low repetition counts are intentionally weak. Avoid a flaky test that
    // assumes a fresh r=3 proof cannot accidentally satisfy a changed digest.
    if (fixture.expected.repetitions >= 32) {
      const nonce = (fixture.challenge.nonce.startsWith('00') ? '01' : '00') + fixture.challenge.nonce.slice(2);
      alteredChallenges.push(['nonce', { ...fixture.challenge, nonce }],
        ['action', { ...fixture.challenge, action: 'different-action' }],
        ['policy-context', { ...fixture.challenge, required_repetitions: fixture.challenge.required_repetitions - 1 }]);
    }
    for (const [name, challenge] of alteredChallenges) {
      assert.equal(await verify(fixture.public, challenge, proof), false);
      cases.push({ ...valid, id: `${fixture.id}-wrong-${name}`, challenge, expectValid: false });
    }
    for (const name of ['magic', 'trit', ...(fixture.expected.repetitions >= 32 ? ['trace'] : [])]) {
      const damaged = proof.slice();
      if (name === 'magic') damaged[0] ^= 1;
      if (name === 'trit') damaged[42 + fixture.public.circuit.side ** 2 / 4] |= 0xc0;
      if (name === 'trace') damaged[damaged.length - 1] ^= 1;
      assert.equal(await verify(fixture.public, fixture.challenge, damaged), false);
      const damagedFile = `${fixture.id}-damaged-${name}.bin`;
      await writeFile(join(temporary, damagedFile), damaged);
      cases.push({ ...valid, id: `${fixture.id}-damaged-${name}`, proofFile: damagedFile, expectValid: false });
    }
    timings.push({ id: fixture.id, proofBytes: proof.length, proveMs, verifyMs, pythonProofVerifyMs });
  }
  const crossManifest = join(temporary, 'js-proofs.json');
  await writeFile(crossManifest, JSON.stringify({ cases }));
  const pythonResults = reference('verify', crossManifest);
  assert.equal(pythonResults.ok, true);
  assert.equal(pythonResults.verified, manifest.cases.length);
  assert.equal(pythonResults.rejected, cases.length - manifest.cases.length);
  console.log(JSON.stringify({ ok: true, runtime: process.version, fixtureFilesReproduced: 2 + manifest.cases.length,
    exactByteParity: manifest.cases.length, pythonToJs: manifest.cases.length, jsToPython: pythonResults, timings }, null, 2));
} finally {
  await rm(temporary, { recursive: true, force: true });
}
