/** Controller integration only: a small DOM harness does not replace browser/device QA. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { createSender, Receiver } from '../dist/js/session.js';
import { encodePacket, decodePacket } from '../dist/js/protocol.js';
import { encodeQr, renderQr, fitDecodeDimensions } from '../dist/js/qr.js';
import { diagnosticPayload } from '../dist/js/payload.js';
import { QR_BYTE_CAPACITIES, SPEED_PRESETS, DEFAULT_PRESET, maxChunkBytes, validateProfile } from '../dist/js/profiles.js';

const html = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');
const source = (await readFile(new URL('../dist/js/app.js', import.meta.url), 'utf8'))
  .replace(/^import .+;\n/gm, '').replaceAll('import.meta.url', JSON.stringify(new URL('../dist/js/app.js', import.meta.url).href));

function harness(overrides = {}) {
  let clock = 0, timerId = 0, paints = 0;
  const defaults = { payloadKind: 'text', payloadText: 'Light is the link.', chunkBytes: '128', windowSize: '1', retryMs: '2500',
    timeoutMs: '3600000', ecc: 'Q', controlEcc: 'Q', dwellMs: '350', speedPreset: 'slow', simPreset: 'slow', simChunk: '128', simWindow: '1', simEcc: 'Q', simControlEcc: 'Q', simDwell: '350', simRetry: '2500', simBytes: '1024', simDrop: '15', simDuplicate: '15', simCorrupt: '5', simSeed: '1234', scanKind: 'text', scanText: 'QR link check', scanEcc: 'Q', cameraSelect: 'environment', cameraResolution: '1280' };
  const timers = new Map(), timerDelays = [], workers = [];
  const canvasContext = { putImageData() { paints++; }, drawImage() {}, getImageData() { if (overrides.capture) return overrides.capture(); throw Error('Unexpected capture in controller test'); } };
  class Element {
    constructor(id = '') {
      this.id = id; this.value = defaults[id] || ''; this.hidden = false; this.disabled = false; this.textContent = '';
      this.style = {}; this.dataset = {}; this.files = []; this.options = []; this.attributes = {}; this.events = new Map();
      this.clientWidth = 640; this.readyState = 0; this.srcObject = null;
      const names = new Set(); this.classList = { add: x => names.add(x), remove: x => names.delete(x), contains: x => names.has(x), toggle: (x, on) => on ? names.add(x) : names.delete(x) };
    }
    addEventListener(type, fn) { this.events.set(type, fn); }
    setAttribute(k, v) { this.attributes[k] = v; }
    removeAttribute(k) { delete this.attributes[k]; }
    querySelectorAll() { return [...elements.values()].filter(e => Object.hasOwn(defaults, e.id)); }
    getContext() { return canvasContext; }
    play() { return Promise.resolve(); }
    click() {}
    append(e) { this.options.push(e); }
    setCustomValidity(message) { this.validationMessage = message; }
    reportValidity() { this.validityReports = (this.validityReports || 0) + 1; return !this.validationMessage; }
  }
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new Element(m[1])]));
  for (const match of html.matchAll(/<select[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const el = elements.get(match[1]);
    for (const m of match[2].matchAll(/<option([^>]*)>([^<]*)<\/option>/g)) {
      const option = new Element(); option.value = /value="([^"]*)"/.exec(m[1])?.[1] ?? m[2]; option.textContent = m[2]; el.options.push(option);
    }
  }
  const tabs = ['transfer', 'scan', 'simulation'].map(tab => Object.assign(new Element(), { dataset: { tab } }));
  const roles = ['send', 'receive'].map(role => Object.assign(new Element(), { dataset: { role } }));
  const document = { getElementById: id => elements.get(id) || null, createElement: () => new Element(), addEventListener() {},
    fullscreenElement: null, querySelectorAll: query => query === '[data-tab]' ? tabs : query === '[data-role]' ? roles : [] };
  const context = vm.createContext({
    document, window: { devicePixelRatio: 2, innerWidth: 1200, innerHeight: 900, addEventListener() {} },
    navigator: { mediaDevices: { getUserMedia: async () => { throw Error('No physical camera in controller tests'); }, enumerateDevices: async () => [], ...overrides.mediaDevices } },
    performance: { now: () => clock }, isSecureContext: true,
    TextEncoder, TextDecoder, Uint8Array, Uint8ClampedArray, Blob, URL, crypto, console,
    ImageData: class { constructor(data, width, height) { Object.assign(this, { data, width, height }); } },
    setInterval: (fn, delay) => { timers.set(++timerId, fn); timerDelays.push({ kind: 'interval', delay }); return timerId; }, setTimeout: (fn, delay) => { timers.set(++timerId, fn); timerDelays.push({ kind: 'timeout', delay }); return timerId; },
    clearTimeout: id => timers.delete(id), requestAnimationFrame: fn => fn(),
    encodeQr, renderQr, fitDecodeDimensions, createSender: overrides.createSender || createSender, Receiver, decodePacket, diagnosticPayload,
    QR_BYTE_CAPACITIES, SPEED_PRESETS, DEFAULT_PRESET, maxChunkBytes, validateProfile,
    QrDecoder: class { close() {} decode(...args) { return overrides.decode?.(...args) ?? Promise.resolve(null); } },
    Worker: class { constructor() { this.events = new Map(); workers.push(this); } addEventListener(type, fn) { this.events.set(type, fn); } postMessage(message) { this.message = message; } terminate() {} },
  });
  vm.runInContext(`${source}\n globalThis.controller = {ui, startTransfer, cancelTransfer, selectRole, selectTab, display, pump, processScan, startCamera, stopCamera, scanSample, scanLoop, runSimulation};`, context);
  return { ...context.controller, elements, input: (id, value) => { const el = elements.get(id); el.value = String(value); return el.events.get('input')?.(); }, clock: value => { clock = value; }, paints: () => paints, timerDelays, timers, workers, change: (id, value) => { const el = elements.get(id); el.value = String(value); return el.events.get('change')?.(); } };
}

test('idle receiver can be cancelled and started again without locking the controls', async () => {
  const h = harness(); h.selectRole('receive'); await h.startTransfer();
  const first = h.ui.receiver;
  assert.equal(first.snapshot().state, 'idle');
  assert.equal(h.elements.get('startTransfer').disabled, true);
  h.cancelTransfer();
  assert.equal(h.ui.receiver, null);
  assert.equal(h.elements.get('startTransfer').disabled, false);
  await h.startTransfer();
  assert.notEqual(h.ui.receiver, first);
  assert.equal(h.ui.receiver.snapshot().state, 'idle');
});

test('a completed receiver keeps its final ACK and refuses new or expired replayed STARTs', async () => {
  const h = harness(); h.selectRole('receive'); await h.startTransfer();
  const sender = await createSender(Uint8Array.of(1, 2, 3), { transferId: '1122334455667788', now: 0 });
  const start = sender.next(0);
  await h.processScan({ bytes: start, version: 4 }, h.ui.generation);
  sender.receive(h.ui.qrBytes, 0);
  const data = sender.next(1);
  await h.processScan({ bytes: data, version: 4 }, h.ui.generation);
  const finalAck = h.ui.qrBytes.slice();
  assert.equal(decodePacket(finalAck).status, 'complete');
  assert.equal(h.ui.receiver.snapshot().state, 'delivered');
  await h.processScan({ bytes: data, version: 4 }, h.ui.generation);
  assert.deepEqual(h.ui.qrBytes, finalAck);
  const otherStart = encodePacket({ ...decodePacket(start), transferId: '8877665544332211' });
  await h.processScan({ bytes: otherStart, version: 4 }, h.ui.generation);
  h.clock(3600001); await h.pump();
  assert.equal(h.ui.receiver.snapshot().terminalCount, 0);
  await h.processScan({ bytes: start, version: 4 }, h.ui.generation);
  assert.equal(h.ui.receiver.snapshot().state, 'delivered');
  assert.equal(h.ui.receiver.snapshot().completedMessages, 1);
  assert.deepEqual(h.ui.qrBytes, finalAck);
});

test('camera permission resolving after Stop immediately releases its stream', async () => {
  const deferred = Promise.withResolvers(); let stopped = 0;
  const h = harness({ mediaDevices: { getUserMedia: () => deferred.promise } });
  const pending = h.startCamera(); assert.equal(h.ui.cameraPending, true);
  h.stopCamera();
  deferred.resolve({ getTracks: () => [{ stop() { stopped++; } }] });
  await pending;
  assert.equal(stopped, 1); assert.equal(h.ui.stream, null);
  assert.equal(h.elements.get('cameraVideo').srcObject, null);
  assert.equal(h.elements.get('cameraToggle').disabled, false);
});

test('identical outgoing bytes stay visually stable and a stalled sender still expires', async () => {
  const h = harness();
  const bytes = Uint8Array.of(1, 2, 3, 4);
  h.display(bytes, 'Check'); const paints = h.paints();
  h.display(bytes.slice(), 'Check');
  assert.equal(h.paints(), paints);
  assert.equal(h.ui.stats.displayChanges, 1);
  await h.startTransfer();
  assert.equal(h.ui.sender.snapshot().state, 'awaiting-start');
  h.clock(3600001); await h.pump();
  assert.equal(h.ui.sender.snapshot().state, 'timed-out');
  assert.equal(h.elements.get('startTransfer').disabled, false);
});

test('late sender preparation cannot overwrite a new receiver session after cancellation', async () => {
  const deferred = Promise.withResolvers(); let entered = false;
  const h = harness({ createSender: () => { entered = true; return deferred.promise; } });
  const oldStart = h.startTransfer(); await Promise.resolve();
  assert.equal(entered, true);
  h.cancelTransfer(); h.selectRole('receive'); await h.startTransfer();
  const receiver = h.ui.receiver;
  deferred.resolve({ snapshot: () => ({ state: 'awaiting-start' }) });
  await oldStart;
  assert.equal(h.ui.sender, null); assert.equal(h.ui.receiver, receiver);
  assert.equal(h.ui.role, 'receive');
  h.selectTab('scan');
  assert.equal(h.ui.receiver, null);
  assert.equal(h.elements.get('scanPanel').hidden, false);
  assert.equal(h.elements.get('transferPanel').hidden, true);
  assert.ok(h.ui.qrBytes.length > 0);
});

test('presets resolve identically for transfer and simulation; manual edits retain Custom values', () => {
  const h = harness();
  for (const preset of SPEED_PRESETS) {
    h.change('speedPreset', preset.id); h.change('simPreset', preset.id);
    for (const [field, live, sim] of [
      ['chunkBytes', 'chunkBytes', 'simChunk'], ['ecc', 'ecc', 'simEcc'], ['windowSize', 'windowSize', 'simWindow'],
      ['dwellMs', 'dwellMs', 'simDwell'], ['retryMs', 'retryMs', 'simRetry'],
    ]) {
      assert.equal(h.elements.get(live).value, String(preset[field]));
      assert.equal(h.elements.get(sim).value, String(preset[field]));
    }
  }
  h.input('dwellMs', 50); h.change('simChunk', 512);
  assert.equal(h.elements.get('speedPreset').value, 'custom');
  assert.equal(h.elements.get('simPreset').value, 'custom');
  h.change('speedPreset', 'custom'); h.change('simPreset', 'custom');
  assert.equal(h.elements.get('dwellMs').value, '50');
  assert.equal(h.elements.get('simChunk').value, '512');
  assert.equal(h.elements.get('chunkBytes').value, '2935');
});

test('higher correction clamps explicitly and disables oversized chunks in both profiles', () => {
  const h = harness();
  h.change('speedPreset', 'fastest'); h.change('ecc', 'Q');
  assert.equal(h.elements.get('chunkBytes').value, '1645');
  assert.equal(h.elements.get('speedPreset').value, 'custom');
  assert.equal(h.elements.get('profileAdjustment').hidden, false);
  assert.match(h.elements.get('profileAdjustment').textContent, /2,935 to 1,645/);
  for (const option of h.elements.get('chunkBytes').options) assert.equal(option.disabled, Number(option.value) > 1645);
  h.change('simPreset', 'fastest'); h.change('simEcc', 'H');
  assert.equal(h.elements.get('simChunk').value, '1255');
  assert.equal(h.elements.get('simPreset').value, 'custom');
  assert.equal(h.elements.get('simAdjustment').hidden, false);
  for (const option of h.elements.get('simChunk').options) assert.equal(option.disabled, Number(option.value) > 1255);
});

test('maximum DATA starts as version 40 L while START and receiver ACK stay Q', async () => {
  const h = harness(); h.change('speedPreset', 'fastest'); h.change('payloadKind', '4096');
  await h.startTransfer();
  assert.equal(decodePacket(h.ui.qrBytes).kind, 'start');
  assert.equal(h.ui.qrEcc, 'Q');
  const peer = new Receiver(); const accepted = await peer.receive(h.ui.qrBytes, 0);
  await h.processScan({ bytes: accepted.outbound[0], version: 3 }, h.ui.generation);
  assert.equal(h.ui.qrBytes.length, 2953); assert.equal(h.ui.qr.version, 40); assert.equal(h.ui.qrEcc, 'L');
  assert.equal(h.ui.profile.presetId, 'fastest'); assert.equal(h.ui.profile.dataEcc, 'L');
  assert.equal(h.ui.profile.controlEcc, 'Q'); assert.equal(h.ui.profile.dwellMs, 100);
  assert.ok(h.ui.stats.qrEncodeCount >= 2); assert.ok(h.ui.stats.qrRenderCount >= 2);
  const receiver = harness(); receiver.change('speedPreset', 'fastest'); receiver.selectRole('receive'); await receiver.startTransfer();
  const sender = await createSender(Uint8Array.of(9), { now: 0 });
  await receiver.processScan({ bytes: sender.next(0), version: 5 }, receiver.ui.generation);
  assert.equal(receiver.ui.qrBytes.length, 28); assert.equal(receiver.ui.qrEcc, 'Q');
  assert.equal(receiver.ui.profile.controlEcc, 'Q'); assert.equal(receiver.ui.profile.ecc, undefined);
  assert.equal(receiver.ui.profile.presetId, undefined);
});

test('invalid settings fail before sender creation while receiver ignores hidden sender settings', async () => {
  let created = 0;
  const h = harness({ createSender: async () => { created++; throw Error('Must not create'); } });
  h.elements.get('chunkBytes').value = '2935'; h.elements.get('ecc').value = 'H';
  await h.startTransfer();
  assert.equal(created, 0); assert.equal(h.ui.sender, null); assert.equal(h.ui.preparing, false);
  assert.match(h.elements.get('notice').textContent, /chunkBytes at ECC H/);
  h.selectRole('receive'); await h.startTransfer();
  assert.equal(h.ui.receiver.snapshot().state, 'idle'); assert.equal(created, 0);
});

test('cycling respects the minimum interval after partial ACK, including one-at-a-time', async () => {
  const h = harness(); h.change('speedPreset', 'fast'); h.change('payloadKind', '16384'); await h.startTransfer();
  const peer = new Receiver();
  const ready = await peer.receive(h.ui.qrBytes, 0);
  await h.processScan({ bytes: ready.outbound[0], version: 3 }, h.ui.generation);
  assert.equal(decodePacket(h.ui.qrBytes).index, 0);
  const partial = await peer.receive(h.ui.qrBytes, 0);
  for (const time of [200, 400, 600]) { h.clock(time); await h.pump(); }
  assert.equal(decodePacket(h.ui.qrBytes).index, 3);
  h.clock(610); await h.processScan({ bytes: partial.outbound[0], version: 3 }, h.ui.generation);
  assert.equal(h.ui.nextTxAt, 800);
  h.clock(799); await h.pump(); assert.equal(decodePacket(h.ui.qrBytes).index, 3);
  h.clock(800); await h.pump(); assert.equal(decodePacket(h.ui.qrBytes).index, 1);
  assert.equal(h.ui.stats.dataDisplayIntervalCount, 4);
  assert.equal(h.ui.stats.dataDisplayIntervalMs, 800);
  assert.equal(h.ui.stats.dataDisplayIntervalMaxMs, 200);
  const single = harness(); single.change('payloadKind', '1024'); await single.startTransfer();
  const other = new Receiver(); const intro = await other.receive(single.ui.qrBytes, 0);
  await single.processScan({ bytes: intro.outbound[0], version: 3 }, single.ui.generation);
  const ack = await other.receive(single.ui.qrBytes, 10);
  single.clock(10); await single.processScan({ bytes: ack.outbound[0], version: 3 }, single.ui.generation);
  assert.equal(decodePacket(single.ui.qrBytes).index, 0);
  single.clock(349); await single.pump(); assert.equal(decodePacket(single.ui.qrBytes).index, 0);
  single.clock(350); await single.pump(); assert.equal(decodePacket(single.ui.qrBytes).index, 1);
});

test('Scan check supports maximum QR and safely reports incompatible sender-sized samples', () => {
  const h = harness(); h.change('speedPreset', 'fastest'); h.selectTab('scan');
  h.change('scanEcc', 'L'); h.change('scanKind', 'max');
  assert.equal(h.ui.qrBytes.length, 2953); assert.equal(h.ui.qr.version, 40);
  h.change('scanKind', 'current'); assert.equal(h.ui.qrBytes.length, 2953);
  h.change('scanEcc', 'H');
  assert.equal(h.ui.qr, null); assert.match(h.elements.get('notice').textContent, /exceed H capacity/);
  h.change('scanKind', 'ack'); assert.equal(h.ui.qrBytes.length, 28); assert.equal(h.ui.qrEcc, 'H');
});

test('simulation receives the resolved preset including timing and independent control correction', () => {
  const h = harness(); h.change('simPreset', 'fastest'); h.change('simControlEcc', 'H');
  h.runSimulation();
  const options = h.workers[0].message.options;
  assert.equal(options.presetId, 'fastest'); assert.equal(options.chunkBytes, 2935);
  assert.equal(options.ecc, 'L'); assert.equal(options.dataEcc, 'L'); assert.equal(options.controlEcc, 'H');
  assert.equal(options.windowSize, 8); assert.equal(options.dwellMs, 100); assert.equal(options.retryMs, 1500);
});

test('camera cadence and pump intervals support fast profiles without queueing another decode', async () => {
  const pending = Promise.withResolvers(); let decodes = 0;
  const h = harness({ capture: () => ({ width: 1, height: 1, data: new Uint8ClampedArray(4) }) });
  assert.deepEqual(h.timerDelays.find(t => t.kind === 'interval'), { kind: 'interval', delay: 25 });
  const video = h.elements.get('cameraVideo'); Object.assign(video, { readyState: 2, videoWidth: 1280, videoHeight: 720 });
  h.ui.stream = {}; h.ui.decoder = { busy: false, decode: () => { decodes++; return pending.promise; } };
  h.ui.cameraRun = { attemptedFrames: 0, successfulFrames: 0, decodeTimeMs: 0, firstDecodeMs: null };
  h.ui.cameraRunStarted = 0;
  const loop = h.scanLoop(h.ui.cameraGeneration);
  assert.equal(decodes, 1); assert.equal(h.timerDelays.filter(t => t.kind === 'timeout').length, 0);
  pending.resolve(null); await loop;
  assert.equal(decodes, 1); assert.equal(h.timerDelays.at(-1).delay, 50);
});

test('ACK grouping and typed timing controls are prominent and expose every group size', () => {
  const h = harness();
  for (const id of ['windowSize', 'simWindow']) {
    assert.deepEqual(h.elements.get(id).options.map(option => option.value), ['1', '2', '3', '4', '5', '6', '7', '8']);
    assert.equal(h.elements.get(id).options[0].textContent, '1 · every chunk');
  }
  const advanced = html.match(/<details class="advanced">([\s\S]*?)<\/details>/)[1];
  assert.doesNotMatch(advanced, /id="(?:windowSize|dwellMs)"/);
  for (const id of ['dwellMs', 'simDwell']) {
    const tag = html.match(new RegExp(`<input[^>]*id="${id}"[^>]*>`))[0];
    for (const attribute of ['type="number"', 'min="50"', 'max="2000"', 'step="1"', 'required']) assert.ok(tag.includes(attribute));
  }
});

test('typing 375 ms immediately selects Custom and preserves payload, correction, and matching simulation settings', async () => {
  const h = harness();
  h.change('speedPreset', 'faster'); h.change('simPreset', 'faster'); h.change('payloadKind', '4096');
  h.input('dwellMs', 375); h.input('simDwell', 375); h.change('windowSize', 2); h.change('simWindow', 2);
  assert.equal(h.elements.get('speedPreset').value, 'custom'); assert.equal(h.elements.get('simPreset').value, 'custom');
  assert.equal(h.elements.get('payloadKind').value, '4096');
  assert.equal(h.elements.get('chunkBytes').value, '2048'); assert.equal(h.elements.get('ecc').value, 'M');
  await h.startTransfer(); h.runSimulation();
  for (const profile of [h.ui.profile, h.workers[0].message.options]) {
    assert.equal(profile.presetId, 'custom'); assert.equal(profile.dwellMs, 375); assert.equal(profile.windowSize, 2);
    assert.equal(profile.chunkBytes, 2048); assert.equal(profile.ecc, 'M');
  }
});

test('longer typed display time raises only the retry interval and explains the adjustment', () => {
  const h = harness();
  h.change('speedPreset', 'fastest'); h.change('simPreset', 'fastest');
  h.input('dwellMs', 1751); h.input('simDwell', 1751);
  for (const [retry, chunk, ecc, adjustment] of [
    ['retryMs', 'chunkBytes', 'ecc', 'profileAdjustment'], ['simRetry', 'simChunk', 'simEcc', 'simAdjustment'],
  ]) {
    assert.equal(h.elements.get(retry).value, '2000'); assert.equal(h.elements.get(chunk).value, '2935');
    assert.equal(h.elements.get(ecc).value, 'L'); assert.equal(h.elements.get(adjustment).hidden, false);
    assert.match(h.elements.get(adjustment).textContent, /1,500 to 2,000 ms/);
  }
});

test('invalid typed intervals are visible and rejected before sender or simulation creation', async () => {
  let created = 0;
  const h = harness({ createSender: async () => { created++; throw Error('Must not create'); } });
  for (const value of ['', '0', '49', '2001', '375.5', 'bad']) {
    h.input('dwellMs', value); h.input('simDwell', value);
    assert.equal(h.elements.get('profileAdjustment').hidden, false); assert.equal(h.elements.get('simAdjustment').hidden, false);
    assert.match(h.elements.get('profileAdjustment').textContent, /whole number from 50 to 2,000/);
    await h.startTransfer(); h.runSimulation();
    assert.equal(created, 0); assert.equal(h.workers.length, 0); assert.equal(h.ui.sender, null);
    assert.equal(h.elements.get('dwellMs').validityReports, h.elements.get('simDwell').validityReports);
  }
  assert.equal(h.elements.get('dwellMs').validityReports, 6);
});

for (const groupSize of [1, 2, 8]) {
  test(`${groupSize}-chunk grouping pauses for confirmation and honors early ACK display time`, async () => {
    const h = harness(); h.change('payloadKind', '16384'); h.change('windowSize', groupSize); h.input('dwellMs', 375);
    await h.startTransfer(); const peer = new Receiver();
    const ready = await peer.receive(h.ui.qrBytes, 0);
    await h.processScan({ bytes: ready.outbound[0], version: 3 }, h.ui.generation);
    let ack;
    for (let index = 0; index < groupSize; index++) {
      const at = index * 375; h.clock(at); await h.pump();
      assert.equal(decodePacket(h.ui.qrBytes).index, index);
      ack = (await peer.receive(h.ui.qrBytes, at)).outbound[0];
    }
    assert.equal(h.ui.sender.snapshot().state, 'awaiting-ack');
    const displayedAt = (groupSize - 1) * 375;
    h.clock(displayedAt + 50); await h.pump();
    assert.equal(decodePacket(h.ui.qrBytes).index, groupSize - 1);
    h.clock(displayedAt + 100); await h.processScan({ bytes: ack, version: 3 }, h.ui.generation);
    assert.equal(h.ui.sender.snapshot().windowBase, groupSize);
    assert.equal(decodePacket(h.ui.qrBytes).index, groupSize - 1);
    h.clock(displayedAt + 374); await h.pump(); assert.equal(decodePacket(h.ui.qrBytes).index, groupSize - 1);
    h.clock(displayedAt + 375); await h.pump(); assert.equal(decodePacket(h.ui.qrBytes).index, groupSize);
  });
}

test('a late ACK advances immediately and an identical DATA retry never resets the display clock', async () => {
  const h = harness(); h.change('payloadKind', '1024'); h.input('dwellMs', 375); await h.startTransfer();
  const peer = new Receiver(); const ready = await peer.receive(h.ui.qrBytes, 0);
  await h.processScan({ bytes: ready.outbound[0], version: 3 }, h.ui.generation);
  const firstData = h.ui.qrBytes.slice(), displays = h.ui.stats.displayChanges;
  const ack = (await peer.receive(firstData, 0)).outbound[0];
  h.clock(375); await h.pump();
  assert.equal(h.ui.sender.snapshot().windowBase, 0); assert.deepEqual(h.ui.qrBytes, firstData);
  h.clock(2500); await h.pump();
  assert.equal(h.ui.sender.snapshot().retries, 1); assert.equal(h.ui.nextTxAt, 375);
  assert.equal(h.ui.stats.displayChanges, displays);
  h.clock(2501); await h.processScan({ bytes: ack, version: 3 }, h.ui.generation);
  assert.equal(decodePacket(h.ui.qrBytes).index, 1); assert.equal(h.ui.nextTxAt, 2876);
});

test('absolute sender deadline is checked before a longer display interval finishes', async () => {
  const h = harness({ createSender: (bytes, options) => createSender(bytes, { ...options, timeoutMs: 100 }) });
  h.input('dwellMs', 2000); await h.startTransfer();
  assert.equal(h.ui.nextTxAt, 2000);
  h.clock(100); await h.pump();
  assert.equal(h.ui.sender.snapshot().state, 'timed-out');
  assert.equal(decodePacket(h.ui.qrBytes).kind, 'cancel');
});
