/** Small DOM/controller harness; this is not browser or physical-camera QA. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { encodeQr, renderQr, fitDecodeDimensions } from '../dist/js/qr.js';
import { encodePacket, decodePacket } from '../dist/js/protocol.js';
import { SPEED_PRESETS, validateProfile, maxChunkBytes } from '../dist/js/profiles.js';
import { OpticalLink } from '../dist/js/auth/link.js';
import { AuthSession } from '../dist/js/auth/session.js';

const html = await readFile(new URL('../dist/auth.html', import.meta.url), 'utf8');

test('authentication page explains local verification, peer reports, and manual pairing', () => {
  assert.match(html, /confirm[^<]*pairing/i);
  assert.match(html, /peer report, not proof of the verifier’s identity/);
  assert.match(html, /device ID is a local label, not a credential/);
  assert.match(html, /nominal repetition bound is not a security guarantee/);
  assert.match(html, /Swap roles for a separate reverse-direction test/);
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(ids.length, new Set(ids).size, 'HTML IDs are unique');
  for (const id of ['startAttempt', 'resetAttempt', 'confirmPairing', 'pairingCode', 'cameraToggle', 'cameraSelect', 'cameraResolution', 'qrCanvas', 'statusTitle', 'exportReport']) assert.ok(ids.includes(id), `${id} exists`);
});

function makeHarness(source, expose, overrides = {}) {
  let clock = 0, nextTimer = 0;
  const timers = new Map(), frames = new Map(), paints = [], clients = [], downloads = [], storage = new Map(), windowEvents = new Map(), documentEvents = new Map();
  const context2d = {
    putImageData(image) { paints.push(image); }, drawImage() {}, clearRect() {},
    getImageData() { return overrides.capture?.() ?? { width: 1, height: 1, data: new Uint8ClampedArray(4) }; },
  };
  class Element {
    constructor(id = '') {
      this.id = id; this.value = ''; this.checked = false; this.hidden = false; this.disabled = false; this.textContent = ''; this.innerHTML = '';
      this.style = {}; this.dataset = {}; this.files = []; this.options = []; this.attributes = {}; this.events = new Map();
      this.clientWidth = 640; this.clientHeight = 480; this.readyState = 0; this.srcObject = null;
      this.videoWidth = 1280; this.videoHeight = 720;
      const names = new Set();
      this.classList = { add: (...xs) => xs.forEach(x => names.add(x)), remove: (...xs) => xs.forEach(x => names.delete(x)), contains: x => names.has(x), toggle: (x, on) => on ? names.add(x) : names.delete(x) };
    }
    addEventListener(type, fn) { this.events.set(type, fn); }
    removeEventListener(type) { this.events.delete(type); }
    setAttribute(key, value) { this.attributes[key] = value; }
    removeAttribute(key) { delete this.attributes[key]; }
    querySelectorAll() { return (this.id === 'attemptConfig' ? ['repetitionPreset', 'repetitions', 'attemptMinutes'] : ['devicePreset', 'speedPreset', 'chunkBytes', 'dataEcc', 'windowSize', 'dwellMs', 'ackHoldMs', 'adaptiveTransport']).map(id => elements.get(id)); }
    getContext() { return context2d; }
    play() { return Promise.resolve(); }
    pause() {}
    click() { this.events.get('click')?.(); }
    append(...items) { this.options.push(...items); }
    appendChild(item) { this.options.push(item); return item; }
    replaceChildren(...items) { this.options = items; }
    remove() {}
    setCustomValidity(message) { this.validationMessage = message; }
    reportValidity() { return !this.validationMessage; }
    getBoundingClientRect() { return { width: this.clientWidth, height: this.clientHeight }; }
  }
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match => [match[1], new Element(match[1])]));
  for (const match of html.matchAll(/<input\b([^>]*)>/g)) {
    const id = /\bid="([^"]+)"/.exec(match[1])?.[1];
    if (id) { elements.get(id).value = /\bvalue="([^"]*)"/.exec(match[1])?.[1] ?? ''; elements.get(id).checked = /\bchecked\b/.test(match[1]); }
  }
  for (const match of html.matchAll(/<select[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const select = elements.get(match[1]);
    for (const optionMatch of match[2].matchAll(/<option([^>]*)>([^<]*)<\/option>/g)) {
      const option = new Element(); option.value = /value="([^"]*)"/.exec(optionMatch[1])?.[1] ?? optionMatch[2]; option.textContent = optionMatch[2];
      select.options.push(option);
      if (!select.value || /\bselected\b/.test(optionMatch[1])) select.value = option.value;
    }
  }
  const roles = ['verifier', 'prover'].map(role => Object.assign(new Element(), { dataset: { role } }));
  const document = {
    getElementById: id => elements.get(id) ?? null, createElement: () => new Element(),
    querySelectorAll: selector => selector === '[data-role]' ? roles : [], addEventListener: (type, fn) => documentEvents.set(type, fn),
    body: new Element(), documentElement: new Element(), fullscreenElement: null,
  };
  class FakeCryptoClient {
    constructor(options = {}) { this.options = options; this.alive = false; this.busy = false; this.calls = []; clients.push(this); }
    async request(type, payload, options) {
      this.calls.push({ type, payload }); this.alive = true; this.busy = true;
      try {
        if (overrides.cryptoRequest) return await overrides.cryptoRequest(type, payload, options, this);
        if (type === 'createKey') return { publicKey: {
          protocol: 'NeuralPicnic-POC-MPC3-v2', circuit: { name: 'binary-rotconv-feistel-v1', side: 8, rounds: 16, key_bits: 256 },
          minimum_repetitions: payload.profile.repetitions, anchor_outputs: 'a1'.repeat(32),
        } };
        throw Error(`Unexpected crypto request: ${type}`);
      } finally { this.busy = false; }
    }
    cancel() { this.destroy(); }
    destroy() { const active = this.alive; this.alive = false; this.busy = false; if (active) this.options.onInvalidate?.(Object.assign(new Error('Key discarded'), { name: 'AbortError' })); }
    fail() { this.alive = false; this.busy = false; this.options.onInvalidate?.(new Error('Crypto worker failed; the ephemeral key was discarded')); }
  }
  class TestURL extends URL {
    static createObjectURL(blob) { downloads.push(blob); return `blob:test-${downloads.length}`; }
    static revokeObjectURL() {}
  }
  const runtime = vm.createContext({
    document, window: { devicePixelRatio: 2, innerWidth: overrides.innerWidth ?? 1200, innerHeight: 900, addEventListener: (type, fn) => windowEvents.set(type, fn) },
    navigator: { mediaDevices: { getUserMedia: async () => { throw Error('No camera in controller tests'); }, enumerateDevices: async () => [], ...overrides.mediaDevices } },
    localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
    performance: { now: () => clock }, isSecureContext: true,
    TextEncoder, TextDecoder, Uint8Array, Uint8ClampedArray, Blob, URL: TestURL, crypto, console, structuredClone,
    ImageData: class { constructor(data, width, height) { Object.assign(this, { data, width, height }); } },
    setInterval: (fn, delay) => { timers.set(++nextTimer, { fn, delay }); return nextTimer; },
    setTimeout: (fn, delay) => { timers.set(++nextTimer, { fn, delay }); return nextTimer; },
    clearTimeout: id => timers.delete(id), clearInterval: id => timers.delete(id),
    requestAnimationFrame: fn => { frames.set(++nextTimer, fn); return nextTimer; }, cancelAnimationFrame: id => frames.delete(id),
    encodeQr, renderQr, fitDecodeDimensions, decodePacket, SPEED_PRESETS, validateProfile, maxChunkBytes,
    OpticalLink, AuthSession, CryptoClient: FakeCryptoClient,
    QrDecoder: class { constructor() { this.busy = false; } close() {} decode(...args) { return overrides.decode?.(...args) ?? Promise.resolve(null); } },
  });
  vm.runInContext(`${source}\nglobalThis.controller = {${expose}};`, runtime);
  return { ...runtime.controller, elements, roles, clients, downloads, paints, storage, timers, frames, windowEvents, documentEvents, document,
    clock: value => { clock = value; },
    click: id => elements.get(id).events.get('click')?.(),
    change: (id, value) => { const el = elements.get(id); el.value = String(value); return el.events.get('change')?.(); },
  };
}

const appSource = (await readFile(new URL('../dist/js/auth/app.js', import.meta.url), 'utf8'))
  .replace(/^import .+;\n/gm, '').replaceAll('import.meta.url', JSON.stringify(new URL('../dist/js/auth/app.js', import.meta.url).href));
const expose = 'ui, startAttempt, resetAttempt, update, startCamera, stopCamera, scanLoop, displayPacket, drawQr, configSummary, makeReport, selectRole, readProfile, readAckHold, applyTransport, selectDevicePreset';
const harness = overrides => makeHarness(appSource, expose, overrides);
const publicResult = repetitions => ({ publicKey: {
  protocol: 'NeuralPicnic-POC-MPC3-v2', circuit: { name: 'binary-rotconv-feistel-v1', side: 8, rounds: 16, key_bits: 256 },
  minimum_repetitions: repetitions, anchor_outputs: 'a1'.repeat(32),
} });

for (const event of ['stop', 'pagehide', 'hidden']) {
  test(`late camera permission after ${event} releases every track`, async () => {
    const deferred = Promise.withResolvers(); let stopped = 0;
    const h = harness({ mediaDevices: { getUserMedia: () => deferred.promise } });
    const pending = h.startCamera(); assert.equal(h.ui.cameraPending, true);
    if (event === 'stop') h.stopCamera();
    else if (event === 'pagehide') h.windowEvents.get('pagehide')();
    else { h.document.hidden = true; h.documentEvents.get('visibilitychange')(); }
    deferred.resolve({ getTracks: () => [{ stop() { stopped++; } }, { stop() { stopped++; } }] });
    await pending;
    assert.equal(stopped, 2); assert.equal(h.ui.stream, null); assert.equal(h.ui.decoder, null);
    assert.equal(h.elements.get('cameraVideo').srcObject, null);
    assert.equal(h.elements.get('cameraToggle').disabled, false);
  });
}

test('late key generation after reset cannot replace a new attempt or restore discarded keys', async () => {
  const deferred = Promise.withResolvers(); let calls = 0;
  const h = harness({ cryptoRequest: (type, payload) => {
    assert.equal(type, 'createKey'); calls++;
    return calls === 1 ? deferred.promise : Promise.resolve(publicResult(payload.profile.repetitions));
  } });
  const oldStart = h.startAttempt(); assert.equal(calls, 1);
  const oldSession = h.ui.session, oldClient = h.ui.crypto;
  h.resetAttempt(); h.selectRole('prover'); await h.startAttempt();
  const freshSession = h.ui.session, freshClient = h.ui.crypto;
  assert.notEqual(freshSession, oldSession); assert.notEqual(freshClient, oldClient);
  assert.equal(freshSession.snapshot().state, 'awaiting-hello');
  deferred.resolve(publicResult(219)); await oldStart;
  assert.equal(h.ui.session, freshSession); assert.equal(h.ui.crypto, freshClient);
  assert.equal(oldSession.snapshot().state, 'closed'); assert.equal(oldClient.alive, false);
  assert.equal(h.ui.session.snapshot().role, 'prover');
  assert.equal(h.elements.get('startAttempt').disabled, true);
  h.resetAttempt();
});

test('worker failure discards keys, active transport, and the visible outgoing QR', async () => {
  const h = harness(); await h.startAttempt();
  const oldLink = h.ui.link, oldSession = h.ui.session, client = h.ui.crypto;
  h.displayPacket(encodePacket({ kind: 'data', transferId: '12'.repeat(8), index: 0, data: Uint8Array.of(1, 2) }));
  assert.ok(h.ui.qrBytes);
  client.fail();
  assert.equal(h.ui.session, null); assert.equal(h.ui.crypto, null); assert.equal(h.ui.link, null);
  assert.equal(oldSession.snapshot().state, 'closed'); assert.equal(oldLink.snapshot().closed, true);
  assert.equal(h.ui.qrBytes, null); assert.equal(h.elements.get('qrCanvas').hidden, true);
  assert.equal(h.elements.get('startAttempt').disabled, false);
  assert.match(h.elements.get('notice').textContent, /ephemeral key was discarded/);
});

test('QR renewal stays visually stable with four white modules around the data', () => {
  const h = harness(); h.ui.profile = SPEED_PRESETS.find(p => p.id === 'fast');
  const bytes = encodePacket({ kind: 'data', transferId: '34'.repeat(8), index: 1, data: new Uint8Array(1024).fill(0xa5) });
  h.displayPacket(bytes); assert.equal(h.paints.length, 1); assert.equal(h.ui.qrEcc, 'M');
  h.displayPacket(bytes.slice()); assert.equal(h.paints.length, 1, 'Same wire bytes do not trigger a redraw');
  const raster = h.paints[0], scale = raster.width / (h.ui.qr.size + 8), border = scale * 4;
  assert.ok(Number.isInteger(scale)); assert.ok(border > 0);
  for (let y = 0; y < raster.height; y++) for (let x = 0; x < raster.width; x++) {
    if (x >= border && x < raster.width - border && y >= border && y < raster.height - border) continue;
    const i = (y * raster.width + x) * 4;
    assert.equal(raster.data[i], 255); assert.equal(raster.data[i + 1], 255); assert.equal(raster.data[i + 2], 255);
  }
  const ack = encodePacket({ kind: 'ack', transferId: '34'.repeat(8), base: 0, bitmap: 0, status: 'ready', digestPrefix: '11'.repeat(8) });
  h.displayPacket(ack); assert.equal(h.ui.qrEcc, 'Q'); assert.equal(h.paints.length, 2);
});

test('a stale camera decode is not delivered to an attempt created after reset', async () => {
  const deferred = Promise.withResolvers(); const received = [];
  const h = harness();
  h.ui.stream = { getTracks: () => [] }; h.ui.decoder = { busy: false, close() {}, decode: () => deferred.promise };
  h.elements.get('cameraVideo').readyState = 2;
  const scan = h.scanLoop(h.ui.cameraGeneration);
  h.resetAttempt();
  h.ui.link = { receive: bytes => received.push(bytes), close() {} };
  deferred.resolve({ bytes: Uint8Array.of(1, 2, 3) }); await scan;
  assert.deepEqual(received, []);
  h.stopCamera(); h.ui.link = null;
});

test('peer-result UI and reports distinguish received claims from local verification', () => {
  const h = harness(); h.ui.role = 'prover';
  const snapshot = {
    state: 'peer-accepted', status: 'Peer reports acceptance', role: 'prover', confirmed: true, pairingCode: '1'.repeat(24),
    localKeyId: '2'.repeat(64), peerKeyId: '3'.repeat(64), result: true, peerReported: true,
    proofBytes: 17474, proofDigest: '4'.repeat(64), repetitions: 32, secretKey: 'must not appear', proof: new Uint8Array(20),
  };
  h.ui.session = { snapshot: () => snapshot };
  h.update();
  assert.equal(h.elements.get('statusTitle').textContent, 'Peer reports accepted');
  assert.match(h.elements.get('statusDetail').textContent, /not cryptographic authentication/);
  const report = h.makeReport();
  assert.equal(report.authentication.peerReported, true); assert.equal(report.authentication.state, 'peer-accepted');
  assert.equal(report.role, 'prover');
  assert.doesNotMatch(JSON.stringify(report), /must not appear|secretKey|"proof":/);
  assert.equal(h.elements.get('statusBox').className.includes('success'), false);
  h.click('exportReport'); assert.equal(h.downloads.length, 1);
});

test('invalid repetition counts cannot start workers and all controller element IDs exist', async () => {
  const h = harness();
  for (const value of ['0', '513', '1.5', '']) {
    h.elements.get('repetitions').value = value; await h.startAttempt();
    assert.equal(h.clients.length, 0); assert.equal(h.ui.session, null);
    assert.match(h.elements.get('notice').textContent, /1 to 512/);
  }
  for (const match of appSource.matchAll(/\$\('([^']+)'\)/g)) assert.ok(h.elements.has(match[1]), `Controller ID ${match[1]} is present`);
  assert.deepEqual([...h.storage.keys()], ['qr-neural-key-device-id'], 'Only a non-secret device label is persisted');
});

test('attempt expiry stops outgoing transport and clears QR while preserving diagnostic outcome', async () => {
  const h = harness(); h.elements.get('attemptMinutes').value = '10'; await h.startAttempt();
  const link = h.ui.link;
  h.displayPacket(encodePacket({ kind: 'data', transferId: '56'.repeat(8), index: 0, data: Uint8Array.of(3) }));
  h.clock(600001); h.ui.session.tick(); h.update();
  assert.equal(h.ui.session.snapshot().state, 'timed-out'); assert.equal(h.ui.link, null);
  assert.equal(link.snapshot().closed, true); assert.equal(h.ui.crypto.alive, false);
  assert.equal(h.ui.qrBytes, null); assert.equal(h.elements.get('qrCanvas').hidden, true);
  assert.equal(h.makeReport().authentication.state, 'timed-out');
  h.resetAttempt();
});

test('successful authentication keeps the transport alive to finish result acknowledgments', () => {
  const h = harness(); let closed = 0;
  h.ui.session = { snapshot: () => ({ state: 'verified', status: 'Proof accepted locally', confirmed: true, resultDelivered: false }) };
  const link = { snapshot: () => ({ sender: { state: 'awaiting-ack' }, receiver: { state: 'delivered' }, queued: 0 }), close: () => { closed++; } };
  h.ui.link = link; h.update();
  assert.equal(h.ui.link, link); assert.equal(closed, 0);
  assert.match(h.elements.get('statusDetail').textContent, /Keep both screens aligned/);
});

test('display presets are independent of authentication role and default to the local viewport', async () => {
  const phone = harness({ innerWidth: 390 }), desktop = harness({ innerWidth: 1200 });
  assert.equal(phone.readProfile().chunkBytes, 128); assert.equal(phone.readProfile().ecc, 'Q');
  assert.equal(phone.readProfile().windowSize, 1); assert.equal(phone.readProfile().dwellMs, 350);
  assert.equal(phone.readAckHold(), 750); assert.equal(phone.elements.get('qrWidth').value, '100');
  assert.equal(desktop.readProfile().chunkBytes, 1024); assert.equal(desktop.readProfile().ecc, 'M');
  assert.equal(desktop.readAckHold(), 500); assert.equal(desktop.elements.get('qrWidth').value, '65');
  phone.selectRole('prover'); desktop.selectRole('prover');
  assert.equal(phone.readProfile().chunkBytes, 128); assert.equal(desktop.readProfile().chunkBytes, 1024);
  await phone.startAttempt();
  assert.equal(phone.ui.link.snapshot().configuredProfile.chunkBytes, 128, 'Phone auto-upshift ceiling is its local configuration');
  assert.equal(phone.ui.link.snapshot().adaptive, true);
  assert.equal(phone.ui.session.snapshot().deadline, 1200000, 'Conservative display has a fixed twenty-minute default deadline');
  for (const id of ['repetitions', 'repetitionPreset', 'attemptMinutes']) assert.equal(phone.elements.get(id).disabled, true);
  for (const id of ['devicePreset', 'speedPreset', 'chunkBytes', 'dataEcc', 'windowSize', 'dwellMs', 'ackHoldMs', 'adaptiveTransport', 'applyTransport', 'qrWidth']) assert.equal(phone.elements.get(id).disabled, false, `${id} stays editable`);
  phone.resetAttempt();
});

test('outgoing settings validate byte capacity at the selected correction level before applying', async () => {
  const h = harness(); h.elements.get('chunkBytes').value = '2048'; h.elements.get('dataEcc').value = 'H';
  assert.throws(() => h.readProfile(), /chunkBytes at ECC H/);
  await h.startAttempt(); assert.equal(h.clients.length, 0);
  assert.match(h.elements.get('notice').textContent, /chunkBytes at ECC H/);
  h.elements.get('dataEcc').value = 'M'; assert.equal(h.readProfile().chunkBytes, 2048);
  h.elements.get('ackHoldMs').value = '249'; await h.applyTransport();
  assert.match(h.elements.get('notice').textContent, /250 to 2000/);
  h.elements.get('ackHoldMs').value = '750'; h.configSummary();
  assert.equal(h.elements.get('chunkBytes').max, String(maxChunkBytes('M')));
});

test('applying smaller outgoing chunks during an attempt updates the active link without discarding its key', async () => {
  const h = harness(); await h.startAttempt();
  const session = h.ui.session, client = h.ui.crypto, link = h.ui.link;
  const calls = [], actualSetProfile = link.setProfile.bind(link);
  link.setProfile = (...args) => { calls.push(args); return actualSetProfile(...args); };
  h.elements.get('chunkBytes').value = '128'; h.elements.get('dataEcc').value = 'Q';
  h.elements.get('windowSize').value = '1'; h.elements.get('dwellMs').value = '350';
  h.elements.get('ackHoldMs').value = '900'; h.elements.get('adaptiveTransport').checked = false;
  await h.click('applyTransport');
  assert.equal(calls.length, 1); assert.equal(calls[0][0].chunkBytes, 128); assert.equal(calls[0][1].ackHoldMs, 900);
  assert.equal(calls[0][1].adaptive, false);
  assert.equal(h.ui.session, session); assert.equal(h.ui.crypto, client); assert.equal(client.alive, true);
  assert.equal(link.snapshot().configuredProfile.chunkBytes, 128); assert.equal(link.snapshot().ackHoldMs, 900);
  assert.match(h.elements.get('transportApplied').textContent, /Applied/);
  h.resetAttempt();
});

test('physical QR width changes live with integer raster modules and the quiet zone intact', () => {
  const h = harness(); h.ui.profile = SPEED_PRESETS.find(p => p.id === 'fast');
  h.displayPacket(encodePacket({ kind: 'data', transferId: 'ab'.repeat(8), index: 0, data: new Uint8Array(1024) }));
  const original = Number.parseFloat(h.elements.get('qrCanvas').style.width);
  h.elements.get('qrWidth').value = '40'; h.elements.get('qrWidth').events.get('input')();
  const width = Number.parseFloat(h.elements.get('qrCanvas').style.width);
  assert.ok(width < original); assert.ok(width <= 570 * .4);
  assert.equal(h.elements.get('qrWidthValue').textContent, '40%');
  assert.match(h.elements.get('qrMeta').textContent, /modules.*CSS px/);
  const raster = h.paints.at(-1), scale = raster.width / (h.ui.qr.size + 8);
  assert.ok(Number.isInteger(scale));
  for (const y of [0, scale * 4 - 1, raster.height - 1]) for (let x = 0; x < raster.width; x++) {
    const offset = (y * raster.width + x) * 4;
    assert.equal(raster.data[offset], 255); assert.equal(raster.data[offset + 1], 255); assert.equal(raster.data[offset + 2], 255);
  }
  h.elements.get('stage').classList.add('expanded'); h.elements.get('qrWidth').value = '100'; h.drawQr();
  assert.ok(Number.parseFloat(h.elements.get('qrCanvas').style.width) > width, 'The same live slider works in expanded display');
});

test('local effective profile controls QR correction independently from the incoming peer profile', () => {
  const h = harness(); h.ui.profile = SPEED_PRESETS.find(p => p.id === 'fast');
  h.ui.link = { snapshot: () => ({ configuredProfile: h.ui.profile, effectiveProfile: { ...h.ui.profile, chunkBytes: 128, ecc: 'Q', windowSize: 1, dwellMs: 350 }, receiver: { chunkBytes: 1024, windowSize: 4 } }) };
  h.displayPacket(encodePacket({ kind: 'data', transferId: 'ac'.repeat(8), index: 0, data: new Uint8Array(128) })); h.update();
  assert.equal(h.ui.qrEcc, 'Q');
  assert.match(h.elements.get('outgoingProfile').textContent, /128 B\/chunk.*Q correction.*window 1/);
  assert.match(h.elements.get('incomingProfile').textContent, /1,024 B\/chunk.*window 4.*chosen by peer/);
  assert.equal(h.readProfile().chunkBytes, 1024, 'Incoming settings never rewrite local controls');
});

test('directional reports contain bounded adaptation history and no retained message bytes', () => {
  const h = harness({ innerWidth: 390 });
  const configured = h.readProfile(), effective = { ...configured, chunkBytes: 64 };
  h.ui.link = { snapshot: () => ({ configuredProfile: configured, effectiveProfile: effective, adaptive: true, ackHoldMs: 750,
    receiveProfile: { chunkBytes: 1024, windowSize: 4, proof: new Uint8Array(8) },
    adaptation: { restarts: 2, downshifts: 1, upshifts: 0, reason: 'Missing ACKs', events: Array.from({ length: 200 }, (_, ms) => ({ ms, kind: 'downshift', fromChunkBytes: 128, toChunkBytes: 64, proof: 'do-not-export' })) },
  }) };
  const report = h.makeReport();
  assert.equal(report.transport.configuredProfile.chunkBytes, 128); assert.equal(report.transport.effectiveProfile.chunkBytes, 64);
  assert.equal(report.transport.receiveProfile.chunkBytes, 1024); assert.equal(report.configuration.qrWidthPercent, 100);
  assert.equal(report.transport.adaptation.events.length, 64); assert.equal(report.transport.adaptation.events[0].ms, 136);
  assert.equal(report.transport.adaptation.events[0].kind, 'downshift'); assert.equal(report.transport.adaptation.events[0].toChunkBytes, 64);
  assert.doesNotMatch(JSON.stringify(report), /do-not-export|"proof":/);
});
