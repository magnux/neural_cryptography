import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';

const root = new URL('../dist/', import.meta.url);
const read = path => readFileSync(new URL(path, root), 'utf8');

function moduleGraph(path, seen = new Set()) {
  const url = path instanceof URL ? path : new URL(path, root);
  if (seen.has(url.href)) return seen;
  seen.add(url.href);
  const source = readFileSync(url, 'utf8');
  for (const match of source.matchAll(/^import\s+(?:[^;]*?\s+from\s+)?['"]([^'"]+)['"]/gm)) {
    assert.ok(match[1].startsWith('./') || match[1].startsWith('../'), `nonlocal runtime import: ${match[1]}`);
    moduleGraph(new URL(match[1], url), seen);
  }
  return seen;
}

test('all workspaces have independent entrypoints, accessible navigation, and complete local assets', () => {
  for (const [page, script] of [['index.html', 'js/app.js'], ['crypto.html', 'js/crypto-lab/lab.js'], ['link.html', 'js/link/lab.js']]) {
    const html = read(page), js = read(script);
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
    assert.equal(new Set(ids).size, ids.length, `${page}: duplicate IDs`);
    for (const match of js.matchAll(/\$\(['"]([^'"]+)['"]\)/g)) {
      assert.ok(ids.includes(match[1]), `${page}: missing control ${match[1]}`);
    }
    for (const match of html.matchAll(/\bfor="([^"]+)"/g)) {
      assert.ok(ids.includes(match[1]), `${page}: missing labelled input ${match[1]}`);
    }
    for (const match of html.matchAll(/(?:href|src)="(\.\/[^"#]+)"/g)) {
      assert.ok(existsSync(new URL(match[1], root)), `${page}: missing asset ${match[1]}`);
    }
    assert.match(html, new RegExp(`href="\\./${page.replace('.', '\\.')}" aria-current="page"`));
    assert.match(html, /aria-label="Application"/);
    assert.match(html, new RegExp(`src="\\./${script.replaceAll('.', '\\.')}"`));
    assert.doesNotMatch(html, /(?:src|href)="https?:/);
    moduleGraph(script);
  }
});

test('optics and crypto UI do not import each other and proof execution loads through the worker', () => {
  const optical = [...moduleGraph('js/app.js')];
  const lab = [...moduleGraph('js/crypto-lab/lab.js')];
  assert.ok(optical.every(path => !path.includes('/crypto/')));
  assert.ok(optical.every(path => !path.includes('/crypto-lab/')));
  assert.ok(lab.every(path => !/\/(?:app|vision|decoder-worker|decoder-client|format)\.js$/.test(path)));
  assert.ok(lab.every(path => !/\/(?:proof|circuit|operations)\.js$/.test(path)), 'heavy crypto must stay off the UI import graph');
  const worker = [...moduleGraph('js/crypto-lab/worker.js')];
  assert.ok(worker.some(path => path.endsWith('/crypto/proof.js')));
  assert.ok(worker.some(path => path.endsWith('/crypto/circuit.js')));
  assert.match(read('js/crypto-lab/client.js'), /new Worker\(new URL\('\.\/worker\.js', import\.meta\.url\)/);
});

// Execute the real controller against a small DOM/worker boundary. The proof
// engine itself is exercised in crypto-worker.test.mjs; these checks cover
// controls invalidating public state without discarding the worker's key.
async function controller() {
  const { createContext, runInContext } = await import('node:vm');
  class Element {
    value = ''; textContent = ''; hidden = false; disabled = false;
    dataset = {}; files = []; listeners = new Map(); children = [];
    addEventListener(type, handler) { this.listeners.set(type, handler); }
    async fire(type) { await this.listeners.get(type)?.({ target: this }); await new Promise(resolve => setImmediate(resolve)); }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    removeAttribute() {}
    remove() {}
    click() { return this.fire('click'); }
  }
  const elements = new Map([...read('crypto.html').matchAll(/\bid="([^"]+)"/g)].map(match => [match[1], new Element()]));
  for (const [id, value] of Object.entries({ profileSide: '8', profileRounds: '16', profileRepetitions: '219', requestedRepetitions: '219', repetitionPreset: '219', lockId: 'demo-door', challengeAction: 'unlock', benchmarkRuns: '3' })) elements.get(id).value = value;
  const requests = []; let worker;
  class Client {
    alive = false; cancelCount = 0; publicKey = null;
    constructor() { worker = this; }
    cancel() { this.alive = false; this.cancelCount++; }
    destroy() { this.cancel(); }
    async request(type, payload) {
      this.alive = true; requests.push({ type, payload });
      if (type === 'createKey') {
        const { side, rounds, repetitions } = payload.profile;
        this.publicKey = { protocol: 'test', circuit: { side, rounds }, minimum_repetitions: repetitions };
        return { publicKey: this.publicKey, keyId: 'ab'.repeat(32), keygenMs: 1 };
      }
      if (type === 'newChallenge') return { challenge: { protocol: 'test', nonce: 'cd'.repeat(32), lock_id: payload.lockId, action: payload.action, required_repetitions: payload.requestedRepetitions } };
      if (type === 'prove') return { publicKey: this.publicKey, challenge: payload.challenge, proof: new Uint8Array(32), usedRepetitions: Math.max(this.publicKey.minimum_repetitions, payload.challenge.required_repetitions), proveMs: 1 };
      if (type === 'verify') return { accepted: true, verifyMs: 1, usedRepetitions: Math.max(payload.publicKey.minimum_repetitions, payload.challenge.required_repetitions) };
      throw new Error(`Unexpected request: ${type}`);
    }
  }
  const context = createContext({
    document: { getElementById: id => elements.get(id), createElement: () => new Element(), body: new Element() },
    CryptoClient: Client, LabReport: class { add() {} clear() {} }, describeEnvironment: () => ({}),
    decodeContext: JSON.parse, encodeContext: JSON.stringify, CONTEXT_MAX_BYTES: 8192,
    startResponsivenessMonitor: () => ({ stop: () => ({ measurementAvailable: false }) }),
    isSecureContext: true, Worker: class {}, crypto: { subtle: {}, getRandomValues() {} }, structuredClone,
    TextEncoder, Uint8Array, DOMException, performance, console, addEventListener() {},
    setInterval: () => 1, clearInterval() {}, setTimeout: () => 1,
  });
  runInContext(read('js/crypto-lab/lab.js').replace(/^import .*;\n/gm, ''), context);
  return { elements, requests, worker, state: () => runInContext('state', context) };
}

test('challenge repetition controls preserve the key while invalidating a completed proof', async () => {
  const { elements, requests, worker, state } = await controller();
  elements.get('profileRepetitions').value = '32';
  await elements.get('runRoundTrip').fire('click');
  const key = state().publicKey, cancelCount = worker.cancelCount;
  assert.equal(state().accepted, true);
  elements.get('repetitionPreset').value = '64';
  await elements.get('repetitionPreset').fire('change');
  assert.equal(elements.get('requestedRepetitions').value, '64');
  assert.equal(state().publicKey, key);
  assert.equal(state().keyAvailable, true);
  assert.equal(worker.cancelCount, cancelCount);
  assert.equal(state().proof, null);
  assert.equal(state().challenge, null);
  assert.equal(elements.get('generateProof').disabled, true);
  await elements.get('runRoundTrip').fire('click');
  assert.equal(requests.filter(request => request.type === 'createKey').length, 1);
  assert.equal(state().challenge.required_repetitions, 64);
  assert.equal(state().usedRepetitions, 64);
  elements.get('requestedRepetitions').value = '87';
  await elements.get('requestedRepetitions').fire('input');
  assert.equal(elements.get('repetitionPreset').value, 'custom');
  assert.equal(state().keyAvailable, true);
  assert.equal(state().proof, null);
  assert.match(elements.get('repetitionSummary').textContent, /87 repetitions/);
});

test('below-floor or invalid challenge requests fail visibly before worker operations', async () => {
  const { elements, requests, state } = await controller();
  await elements.get('runRoundTrip').fire('click');
  const count = requests.length;
  for (const value of ['32', '', '1.5', '513']) {
    elements.get('requestedRepetitions').value = value;
    await elements.get('requestedRepetitions').fire('input');
    await elements.get('runRoundTrip').fire('click');
    assert.equal(requests.length, count);
    assert.equal(elements.get('operationError').hidden, false);
    assert.equal(elements.get('repetitionSummary').dataset.invalid, 'true');
    assert.equal(state().keyAvailable, true);
  }
});

test('enrolled floor changes clear the key; compact estimate is explicit and distinct from security', async () => {
  const { elements, worker, state } = await controller();
  assert.match(elements.get('repetitionSummary').textContent, /119,249 B/);
  assert.match(elements.get('repetitionSummary').textContent, /2\^−128\.1/);
  assert.match(elements.get('repetitionSummary').textContent, /not guaranteed security/);
  await elements.get('runRoundTrip').fire('click');
  const canceled = worker.cancelCount;
  elements.get('profileRepetitions').value = '32';
  await elements.get('profileRepetitions').fire('input');
  assert.equal(worker.cancelCount, canceled + 1);
  assert.equal(state().keyAvailable, false);
  assert.equal(state().publicKey, null);
  assert.equal(state().proof, null);
});

test('context import restores requested repetitions independently of the enrolled key floor', async () => {
  const { elements, state } = await controller();
  await elements.get('runRoundTrip').fire('click');
  const context = { publicKey: { circuit: { side: 4, rounds: 2 }, minimum_repetitions: 32 }, challenge: { nonce: 'ab'.repeat(32), lock_id: 'other-door', action: 'open', required_repetitions: 110 } };
  elements.get('contextFile').files = [{ size: 500, text: async () => JSON.stringify(context) }];
  await elements.get('contextFile').fire('change');
  assert.equal(elements.get('profileRepetitions').value, '32');
  assert.equal(elements.get('requestedRepetitions').value, '110');
  assert.equal(elements.get('repetitionPreset').value, '110');
  assert.equal(state().keyAvailable, false);
  assert.equal(state().proof, null);
  assert.equal(state().challenge.required_repetitions, 110);
});
