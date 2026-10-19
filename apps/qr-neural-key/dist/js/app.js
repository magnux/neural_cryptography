import { encodeQr, renderQr, fitDecodeDimensions } from './qr.js';
import { QrDecoder } from './qr-client.js';
import { createSender, Receiver } from './session.js';
import { decodePacket } from './protocol.js';
import { diagnosticPayload } from './payload.js';
import { QR_BYTE_CAPACITIES, SPEED_PRESETS, DEFAULT_PRESET, maxChunkBytes, validateProfile } from './profiles.js';

const $ = id => document.getElementById(id);
const MAX_BYTES = 8 * 1024 * 1024;
const PUMP_INTERVAL_MS = 25;
const SCAN_INTERVAL_MS = 50;
const TERMINAL = new Set(['delivered', 'failed', 'timed-out', 'cancelled']);
const encoder = new TextEncoder();
const newStats = () => ({ scans: 0, decodeMs: 0, decodeErrors: 0, packetsReceived: 0, acceptedPackets: 0, invalidPackets: 0, ackPackets: 0, displayChanges: 0, qrEncodeCount: 0, qrEncodeMs: 0, qrEncodeMaxMs: 0, qrRenderCount: 0, qrRenderMs: 0, qrRenderMaxMs: 0, dataDisplayIntervalCount: 0, dataDisplayIntervalMs: 0, dataDisplayIntervalMaxMs: 0 });
const ui = {
  tab: 'transfer', role: 'send', generation: 0, preparing: false,
  sender: null, receiver: null, startedAt: null, endedAt: null, wallStartedAt: null,
  nextTxAt: 0, lastDataDisplayedAt: null, lastMetricsAt: -Infinity, receiverBusy: false, lastState: '', profile: null, events: [],
  stats: newStats(),
  received: null, qr: null, qrBytes: null, qrEcc: null, qrDescription: '',
  stream: null, decoder: null, cameraGeneration: 0, cameraPending: false, scanTimer: null, cameraSettings: null,
  cameraRun: null, cameraRunStarted: null, cameraRunEnded: null,
  scanCount: 0, scanStartedAt: null, simWorker: null, simId: 0, simReport: null,
};
const capture = document.createElement('canvas');
const captureContext = capture.getContext('2d', { willReadFrequently: true });
const now = () => performance.now();
const n = id => Number($(id).value);
const fmt = value => Math.max(0, Math.round(value || 0)).toLocaleString();
const elapsed = () => ui.startedAt === null ? 0 : Math.max(0, (ui.endedAt ?? now()) - ui.startedAt);
const sameBytes = (a, b) => a?.length === b.length && b.every((v, i) => v === a[i]);
function event(type, detail = {}) { ui.events.push({ ms: Math.round(elapsed()), type, ...detail }); if (ui.events.length > 100) ui.events.shift(); }
function notice(message) { $('notice').textContent = message || ''; $('notice').hidden = !message; }
function status(title, detail, kind = '') {
  $('statusTitle').textContent = title; $('statusDetail').textContent = detail; $('statusBox').className = `status ${kind}`;
}
function clearQr(title = 'Your outgoing QR appears here', note = 'Choose a role and start a transfer.') {
  ui.qr = null; ui.qrBytes = null; ui.qrEcc = null;
  $('qrCanvas').hidden = true; $('qrPlaceholder').hidden = false;
  $('placeholderTitle').textContent = title; $('placeholderNote').textContent = note;
  $('qrMeta').textContent = 'No code displayed'; $('qrLabel').textContent = 'Outgoing QR';
}
function drawQr() {
  if (!ui.qr) return;
  const renderStarted = now();
  const ratio = window.devicePixelRatio || 1;
  const expanded = $('stage').classList.contains('expanded') || document.fullscreenElement === $('stage');
  const mobile = window.innerWidth <= 500;
  const maxHeight = expanded ? window.innerHeight - (mobile ? 165 : 100) : 640;
  const available = Math.max(80, Math.min($('qrZone').clientWidth, maxHeight));
  const moduleCount = ui.qr.size + 8;
  const scale = Math.min(32, Math.max(1, Math.floor(available * ratio / moduleCount)));
  const raster = renderQr(ui.qr, { scale, quiet: 4 });
  const canvas = $('qrCanvas'); canvas.width = raster.width; canvas.height = raster.height;
  canvas.getContext('2d').putImageData(new ImageData(raster.data, raster.width, raster.height), 0, 0);
  canvas.style.width = `${raster.width / ratio}px`; canvas.style.height = `${raster.height / ratio}px`;
  const renderMs = now() - renderStarted; ui.stats.qrRenderCount++; ui.stats.qrRenderMs += renderMs; ui.stats.qrRenderMaxMs = Math.max(ui.stats.qrRenderMaxMs, renderMs);
}
function display(bytes, description, ecc = $('ecc').value) {
  if (sameBytes(ui.qrBytes, bytes) && ui.qrEcc === ecc) return false;
  const encodeStarted = now();
  ui.qr = encodeQr(bytes, { ecc });
  const encodeMs = now() - encodeStarted; ui.stats.qrEncodeCount++; ui.stats.qrEncodeMs += encodeMs; ui.stats.qrEncodeMaxMs = Math.max(ui.stats.qrEncodeMaxMs, encodeMs);
  ui.qrBytes = bytes.slice(); ui.qrEcc = ecc; ui.qrDescription = description;
  $('qrPlaceholder').hidden = true; $('qrCanvas').hidden = false;
  $('qrLabel').textContent = description;
  $('qrMeta').textContent = `${bytes.length} B · QR ${ui.qr.version} · ${ui.qr.size}² · ${ecc}`;
  ui.stats.displayChanges++; drawQr(); return true;
}
function displayPacket(bytes) {
  const p = decodePacket(bytes);
  const label = p.kind === 'data' ? `Data · chunk ${p.index + 1}` : p.kind === 'ack' ? (p.status === 'complete' ? 'Complete ACK · keep visible' : 'Acknowledgment · keep visible') : p.kind === 'start' ? 'Transfer introduction' : p.kind === 'poll' ? 'Requesting acknowledgment' : 'Transfer cancelled';
  const ecc = p.kind === 'data' ? ui.profile?.ecc || $('ecc').value : ui.profile?.controlEcc || $('controlEcc').value;
  if (!display(bytes, label, ecc)) return false;
  if (p.kind === 'data') {
    const displayedAt = now();
    if (ui.lastDataDisplayedAt !== null) {
      const interval = displayedAt - ui.lastDataDisplayedAt; ui.stats.dataDisplayIntervalCount++; ui.stats.dataDisplayIntervalMs += interval; ui.stats.dataDisplayIntervalMaxMs = Math.max(ui.stats.dataDisplayIntervalMaxMs, interval);
    }
    ui.lastDataDisplayedAt = displayedAt;
  }
  event('display', { kind: p.kind, packetBytes: bytes.length, qrVersion: ui.qr.version });
  return true;
}
function clearReceived() { ui.received = null; $('receivedPanel').hidden = true; $('receivedText').value = ''; }
function showContent(bytes, { scan = false, digest = null, version = null } = {}) {
  ui.received = bytes.slice(); $('receivedPanel').hidden = false;
  $('contentTitle').textContent = scan ? 'Scanned content' : 'Received content';
  $('contentMeta').textContent = `${fmt(bytes.length)} bytes${version ? ` · QR version ${version}` : ''}${digest ? ` · SHA-256 ${digest}` : ''}`;
  try {
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(0, Math.min(bytes.length, 262144)), { stream: bytes.length > 262144 });
    // NUL and most control bytes signal a binary payload even when UTF-8 technically decodes.
    if (/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(text)) throw new Error('Binary content');
    $('receivedText').value = text;
    $('contentNote').textContent = `Decoded as UTF-8${bytes.length > 262144 ? ' · preview limited to 256 KiB; download contains all bytes' : ''}.`;
  } catch {
    $('receivedText').value = [...bytes.subarray(0, 2048)].map(b => b.toString(16).padStart(2, '0')).join(' ');
    $('contentNote').textContent = `Binary content · hexadecimal preview${bytes.length > 2048 ? ' of the first 2,048 bytes' : ''}. Download preserves the exact bytes.`;
  }
}
function download(blob, name) {
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 5000);
}
function snapshot() { return (ui.sender || ui.receiver)?.snapshot() || null; }
function sessionActive() { const s = snapshot(); return !!s && !TERMINAL.has(s.state); }
function freezeConfig(frozen) {
  for (const el of $('transferConfig').querySelectorAll('input,select,textarea')) el.disabled = frozen;
  for (const el of document.querySelectorAll('[data-role]')) el.disabled = frozen;
  $('startTransfer').disabled = frozen;
  $('cancelTransfer').disabled = !frozen;
}
function updateMetrics() {
  const s = snapshot(); if (!s) return;
  ui.lastMetricsAt = now();
  const bytes = ui.sender ? s.ackedBytes : s.receivedBytes;
  $('metricProgress').textContent = `${fmt(bytes)} / ${fmt(s.totalBytes)} B`;
  $('metricSpeed').textContent = `${elapsed() > 0 ? fmt(bytes * 1000 / elapsed()) : '0'} B/s`;
  $('metricAcks').textContent = `${fmt(ui.stats.ackPackets)} / ${fmt(s.retries)}`;
  $('metricElapsed').textContent = `${(elapsed() / 1000).toFixed(1)} s`;
  $('sessionLabel').textContent = s.transferId ? `Session ${s.transferId}` : 'Waiting for an introduction';
  const terminal = TERMINAL.has(s.state);
  if (terminal && ui.endedAt === null) { ui.endedAt = now(); event('terminal', { state: s.state, error: s.error || null }); }
  freezeConfig(ui.preparing || !terminal);
  if (s.state !== ui.lastState) { ui.lastState = s.state; event('state', { state: s.state }); }
  if (s.state === 'delivered') {
    if (ui.sender) {
      status('Transfer confirmed by receiver', 'The receiver checked the complete SHA-256 digest and sent the final acknowledgment.', 'success');
      clearQr('Transfer confirmed', 'The receiving device has acknowledged the complete payload.');
    } else {
      status('Payload verified locally · complete ACK displayed', 'Keep this screen facing the sender until it says “Transfer confirmed.” Local receipt does not tell us whether the final ACK was read.', 'success');
    }
  } else if (s.state === 'failed' || s.state === 'timed-out') {
    status(s.state === 'timed-out' ? 'Transfer timed out' : 'Transfer failed', s.error || 'The transfer could not finish. Export a report, then try a shorter payload or check the return path.', 'error');
    if (!ui.qrBytes || decodePacket(ui.qrBytes).kind !== 'cancel') clearQr('Transfer stopped', 'Export a report or start a fresh transfer.');
  } else if (s.state === 'cancelled') {
    status('Transfer cancelled', 'No further data is being sent. Start again to use a new transfer identifier.');
  } else if (ui.sender) {
    const titles = { 'awaiting-start': 'Waiting for the receiver’s first ACK', sending: 'Sending data', 'awaiting-ack': 'Waiting for this group’s ACK' };
    const first = s.windowBase + 1, last = Math.min(s.windowBase + s.windowSize, s.chunkCount);
    const group = s.state === 'awaiting-ack' ? ` Group ${first}–${last} must be confirmed before the next group; missing chunks are retried.` : '';
    status(titles[s.state] || 'Sending', `${fmt(s.ackedChunks)} / ${fmt(s.chunkCount)} chunks confirmed.${group} Keep both cameras pointed at the opposite display.`, 'working');
  } else {
    status(s.state === 'idle' ? 'Receiver ready' : 'Receiving data', s.state === 'idle' ? 'Scan the sender’s introduction QR. An ACK will appear here as soon as it is accepted.' : `${fmt(s.receivedChunks)} / ${fmt(s.chunkCount)} chunks collected. Keep the ACK visible to the sender.`, 'working');
  }
}
async function payload() {
  const kind = $('payloadKind').value;
  if (kind === 'text') return encoder.encode($('payloadText').value);
  if (kind === 'file') {
    const file = $('fileInput').files[0]; if (!file) throw new Error('Choose a file first.');
    if (file.size > MAX_BYTES) throw new Error('Choose a file of 8 MiB or less.');
    return new Uint8Array(await file.arrayBuffer());
  }
  return diagnosticPayload(Number(kind), 1234);
}
function payloadControls() {
  const kind = $('payloadKind').value;
  $('payloadText').hidden = kind !== 'text'; $('fileField').hidden = kind !== 'file';
  const count = kind === 'text' ? encoder.encode($('payloadText').value).length : kind === 'file' ? $('fileInput').files[0]?.size : Number(kind);
  $('payloadSize').textContent = count === undefined ? 'Choose a local file' : `${fmt(count)} bytes${kind === 'text' ? ' · UTF-8' : ''}`;
  profileSummary('send');
}
const profileFields = {
  send: { preset: 'speedPreset', chunkBytes: 'chunkBytes', windowSize: 'windowSize', ecc: 'ecc', controlEcc: 'controlEcc', dwellMs: 'dwellMs', retryMs: 'retryMs', detail: 'presetDetail', limit: 'chunkLimit', adjustment: 'profileAdjustment' },
  simulation: { preset: 'simPreset', chunkBytes: 'simChunk', windowSize: 'simWindow', ecc: 'simEcc', controlEcc: 'simControlEcc', dwellMs: 'simDwell', retryMs: 'simRetry', detail: 'simPresetDetail', limit: 'simChunkLimit', adjustment: 'simAdjustment' },
};
function profileSummary(mode) {
  const ids = profileFields[mode], ecc = $(ids.ecc).value, windowSize = n(ids.windowSize);
  const label = SPEED_PRESETS.find(p => p.id === $(ids.preset).value)?.label || 'Custom';
  const timing = pacingError(mode) ? 'Choose a whole-number display time from 50 to 2,000 ms.' : `At least ${n(ids.dwellMs)} ms per data QR, even if its ACK arrives earlier.`;
  const description = `${windowSize === 1 ? 'Wait for every chunk’s ACK.' : `Send ${windowSize} chunks, then wait for their ACK.`} ${timing}`;
  $(ids.detail).textContent = `${n(ids.chunkBytes).toLocaleString()} useful bytes / QR · ${ecc} data correction. ${description} ${label === 'Fastest' ? 'Maximum density; use excellent alignment and a large display.' : ''}`;
  $(ids.limit).textContent = `${ecc} allows up to ${maxChunkBytes(ecc).toLocaleString()} useful bytes per QR, after the 18-byte data header. Larger payloads require denser codes. Presets are test targets, not guaranteed speeds.`;
  if (mode === 'send') $('profileSummary').textContent = ui.role === 'receive' ? `Receiver · ${$('controlEcc').value} ACK / control correction` : `${label} · ${n(ids.chunkBytes).toLocaleString()} B · ${ecc} · ${windowSize} before ACK · ${pacingError(mode) ? 'set timing' : `${n(ids.dwellMs)} ms`}`;
}
function enforceChunkOptions(mode, adjust = false) {
  const ids = profileFields[mode], cap = maxChunkBytes($(ids.ecc).value), chunk = $(ids.chunkBytes);
  for (const option of chunk.options) option.disabled = Number(option.value) > cap;
  $(ids.adjustment).hidden = true;
  if (adjust && Number(chunk.value) > cap) {
    const previous = chunk.value; chunk.value = String(cap);
    $(ids.adjustment).textContent = `Chunk adjusted from ${Number(previous).toLocaleString()} to ${cap.toLocaleString()} bytes to fit ${$(ids.ecc).value} correction.`;
    $(ids.adjustment).hidden = false;
  }
  profileSummary(mode);
}
function applyPreset(mode) {
  const ids = profileFields[mode], preset = SPEED_PRESETS.find(p => p.id === $(ids.preset).value);
  if (!preset) { profileSummary(mode); return; }
  for (const key of ['chunkBytes', 'windowSize', 'ecc', 'dwellMs', 'retryMs']) $(ids[key]).value = String(preset[key]);
  enforceChunkOptions(mode); validatePacing(mode);
}
function pacingError(mode) {
  const value = $(profileFields[mode].dwellMs).value;
  return value.trim() === '' || !Number.isSafeInteger(Number(value)) || Number(value) < 50 || Number(value) > 2000
    ? 'Milliseconds between QRs must be a whole number from 50 to 2,000.' : '';
}
function validatePacing(mode, adjustRetry = false) {
  const ids = profileFields[mode], error = pacingError(mode);
  $(ids.dwellMs).setCustomValidity(error);
  if (error) {
    $(ids.adjustment).textContent = error; $(ids.adjustment).hidden = false;
  } else if (adjustRetry && n(ids.retryMs) < n(ids.dwellMs)) {
    const next = [...$(ids.retryMs).options].map(option => Number(option.value)).find(value => value >= n(ids.dwellMs));
    const previous = n(ids.retryMs); $(ids.retryMs).value = String(next);
    $(ids.adjustment).textContent = `Retry interval increased from ${previous.toLocaleString()} to ${next.toLocaleString()} ms to cover the minimum QR display time.`;
    $(ids.adjustment).hidden = false;
  }
  return error;
}
function manualProfile(mode, changed) {
  const ids = profileFields[mode]; $(ids.preset).value = 'custom';
  enforceChunkOptions(mode, changed === 'ecc');
  validatePacing(mode, true);
}
function readProfile(mode) {
  const ids = profileFields[mode];
  const error = validatePacing(mode);
  if (error) { $(ids.dwellMs).reportValidity(); throw new RangeError(error); }
  return validateProfile({ presetId: $(ids.preset).value, chunkBytes: n(ids.chunkBytes), windowSize: n(ids.windowSize), retryMs: n(ids.retryMs), dwellMs: n(ids.dwellMs), ecc: $(ids.ecc).value, dataEcc: $(ids.ecc).value, controlEcc: $(ids.controlEcc).value, maxRetries: 20, timeoutMs: mode === 'send' ? n('timeoutMs') : 3600000 });
}
function initializeProfiles() {
  for (const [mode, ids] of Object.entries(profileFields)) {
    for (const preset of [...SPEED_PRESETS, { id: 'custom', label: 'Custom' }]) {
      const option = document.createElement('option'); option.value = preset.id; option.textContent = preset.label; $(ids.preset).append(option);
    }
    $(ids.preset).value = DEFAULT_PRESET;
    $(ids.preset).addEventListener('change', () => applyPreset(mode));
    for (const field of ['chunkBytes', 'windowSize', 'ecc', 'retryMs']) $(ids[field]).addEventListener('change', () => manualProfile(mode, field));
    $(ids.dwellMs).addEventListener('input', () => manualProfile(mode, 'dwellMs'));
    $(ids.controlEcc).addEventListener('change', () => profileSummary(mode));
    applyPreset(mode);
  }
}
function resetSession() {
  ui.generation++; ui.sender = null; ui.receiver = null; ui.preparing = false;
  ui.startedAt = null; ui.endedAt = null; ui.wallStartedAt = null; ui.lastState = ''; ui.profile = null; ui.events = []; ui.nextTxAt = 0; ui.lastDataDisplayedAt = null;
  ui.stats = newStats();
  freezeConfig(false); $('sessionLabel').textContent = 'Ready for a new transfer';
  $('metricProgress').textContent = '0 / 0 B'; $('metricSpeed').textContent = '— B/s'; $('metricAcks').textContent = '0 / 0'; $('metricElapsed').textContent = '0.0 s';
}
async function startTransfer() {
  if (ui.preparing || sessionActive()) return;
  notice(''); stopCamera(); resetSession(); clearReceived(); clearQr();
  const generation = ui.generation; ui.preparing = true;
  ui.startedAt = now(); ui.wallStartedAt = new Date().toISOString();
  try {
    ui.profile = ui.role === 'send' ? readProfile('send') : { controlEcc: $('controlEcc').value, timeoutMs: n('timeoutMs') };
    if (!Object.hasOwn(QR_BYTE_CAPACITIES, ui.profile.controlEcc)) throw new Error('Choose a valid ACK / control correction level.');
    freezeConfig(true);
    if (ui.role === 'send') {
      const bytes = await payload(); if (generation !== ui.generation) return;
      if (bytes.length > MAX_BYTES) throw new Error('Payload exceeds the 8 MiB limit.');
      const sender = await createSender(bytes, { ...ui.profile, now: now() });
      if (generation !== ui.generation) return;
      ui.sender = sender;
    } else ui.receiver = new Receiver({ timeoutMs: ui.profile.timeoutMs, maxMessageBytes: MAX_BYTES });
    ui.preparing = false; $('transferConfig').open = false;
    event('started', { role: ui.role }); updateMetrics(); pump();
    await startCamera();
  } catch (error) {
    if (generation !== ui.generation) return;
    ui.preparing = false; freezeConfig(false); notice(error.message); status('Unable to start', error.message, 'error');
  }
}
function cancelTransfer() {
  const session = ui.sender || ui.receiver;
  ui.generation++; ui.preparing = false;
  if (session) {
    const packet = session.cancel(now());
    if (packet) displayPacket(packet); else clearQr('Transfer cancelled', 'Start again when both devices are ready.');
    ui.endedAt = now(); event('cancelled');
    if (session.snapshot().state === 'idle') { ui.receiver = null; freezeConfig(false); status('Receiver stopped', 'Start again when both devices are ready.'); }
    else updateMetrics();
  } else { clearQr(); freezeConfig(false); }
  stopCamera();
}
let ticking = false;
async function pump() {
  if (ticking || ui.preparing || ui.tab !== 'transfer') return;
  ticking = true; const generation = ui.generation;
  try {
    if (ui.sender && !TERMINAL.has(ui.sender.snapshot().state) && (now() >= ui.nextTxAt || now() >= ui.sender.deadline)) {
      const packet = ui.sender.next(now());
      // A retry of the same QR does not restart its existing on-screen time.
      if (packet && displayPacket(packet)) ui.nextTxAt = now() + ui.profile.dwellMs;
    }
    // tick() is synchronous and can expire a transfer even while its digest is pending.
    if (ui.receiver) {
      const result = ui.receiver.tick(now());
      if (generation === ui.generation) for (const packet of result?.outbound || []) displayPacket(packet);
    }
    if (generation === ui.generation && (now() - ui.lastMetricsAt >= 100 || snapshot()?.state !== ui.lastState)) updateMetrics();
  } catch (error) { if (generation === ui.generation) { notice(error.message); cancelTransfer(); } }
  finally { ticking = false; }
}
async function processScan(result, generation) {
  if (generation !== ui.generation) return;
  ui.stats.scans++; ui.scanCount++;
  $('scanMeta').textContent = `QR ${result.version || '?'} · ${result.bytes.length} B · ${ui.scanCount} reads`;
  if (ui.tab === 'scan') {
    showContent(result.bytes, { scan: true, version: result.version });
    status('QR decoded', `${result.bytes.length} bytes recovered. Reverse the direction and compare the short ACK-sized sample with the larger data-sized sample.`, 'success');
    return;
  }
  if (ui.tab !== 'transfer' || (!ui.sender && !ui.receiver)) return;
  ui.stats.packetsReceived++;
  let packet;
  try { packet = decodePacket(result.bytes); }
  catch { ui.stats.invalidPackets++; return; }
  if (ui.sender) {
    const wasIntroduction = ui.sender.snapshot().state === 'awaiting-start';
    const received = ui.sender.receive(result.bytes, now());
    if (received.accepted) {
      ui.stats.acceptedPackets++; if (packet.kind === 'ack') ui.stats.ackPackets++;
      // Introduction ACKs can begin data immediately. DATA ACKs still honor
      // the minimum display time; late ACKs add no further delay.
      if (wasIntroduction) ui.nextTxAt = 0;
      pump();
    }
  } else if (ui.receiver && !ui.receiverBusy) {
    const current = ui.receiver.snapshot();
    // One explicit Start receiver action accepts one transfer. Preserve the last
    // receipt for retransmissions without admitting a new identity over old content.
    if (current.transferId && current.transferId !== packet.transferId) return;
    if (TERMINAL.has(current.state) && packet.kind === 'start') return;
    ui.receiverBusy = true;
    try {
      const received = await ui.receiver.receive(result.bytes, now());
      if (generation !== ui.generation) return;
      if (received.accepted) ui.stats.acceptedPackets++;
      for (const response of received.outbound || []) { ui.stats.ackPackets++; displayPacket(response); }
      if (received.delivery) {
        showContent(received.delivery.bytes, { digest: received.delivery.digest });
        event('local-integrity-pass', { bytes: received.delivery.bytes.length });
      }
    } finally { ui.receiverBusy = false; }
  }
  if (generation === ui.generation) updateMetrics();
}
function cameraReport() {
  if (!ui.cameraRun) return null;
  const r = ui.cameraRun;
  return { ...r, elapsedMs: Math.round((ui.cameraRunEnded ?? now()) - ui.cameraRunStarted), decodeRate: r.attemptedFrames ? r.successfulFrames / r.attemptedFrames : 0, meanDecodeMs: r.attemptedFrames ? r.decodeTimeMs / r.attemptedFrames : 0 };
}
function cameraDiagnostics() {
  const r = cameraReport(); if (!r) return;
  const first = r.firstDecodeMs === null ? 'no first read yet' : `first ${(r.firstDecodeMs / 1000).toFixed(2)} s`;
  $('scanMeta').textContent = `${r.successfulFrames}/${r.attemptedFrames} reads · ${Math.round(r.meanDecodeMs)} ms/decode`;
  if (ui.tab === 'scan') {
    const last = r.lastDecoded;
    status(last ? 'QR decoded' : 'Looking for a QR', `${last ? `Last: QR ${last.version} · ${last.bytes} bytes. ` : ''}${(r.decodeRate * 100).toFixed(1)}% of attempted frames decoded · ${first}. Move back slightly or try the smaller ACK-sized sample if reads are scarce.`, last ? 'success' : 'working');
  }
}
function stopCamera() {
  if (ui.cameraRunStarted !== null && ui.cameraRunEnded === null) ui.cameraRunEnded = now();
  ui.cameraGeneration++; clearTimeout(ui.scanTimer); ui.scanTimer = null;
  ui.stream?.getTracks().forEach(track => track.stop()); ui.stream = null;
  ui.decoder?.close(); ui.decoder = null; $('cameraVideo').srcObject = null;
  ui.cameraPending = false; $('cameraToggle').disabled = false; $('cameraToggle').textContent = 'Start camera';
  $('cameraPlaceholder').hidden = false; $('cameraPlaceholder').textContent = 'Camera off';
  $('cameraDot').classList.remove('active');
}
async function enumerateCameras() {
  const selected = $('cameraSelect').value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const select = $('cameraSelect');
  for (const opt of [...select.options].slice(2)) opt.remove();
  for (const [i, device] of devices.filter(d => d.kind === 'videoinput').entries()) {
    const option = document.createElement('option'); option.value = device.deviceId; option.textContent = device.label || `Camera ${i + 1}`; select.append(option);
  }
  if ([...select.options].some(o => o.value === selected)) select.value = selected;
}
async function startCamera() {
  if (ui.cameraPending || ui.stream || ui.tab === 'simulation') return;
  if (!navigator.mediaDevices?.getUserMedia) { notice('Camera access needs HTTPS or localhost in a browser that supports camera capture.'); return; }
  const token = ++ui.cameraGeneration; ui.cameraPending = true; $('cameraToggle').disabled = true;
  const width = n('cameraResolution'), height = width === 640 ? 480 : Math.round(width * 9 / 16);
  const choice = $('cameraSelect').value;
  const video = { width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: 30, max: 30 }, ...(choice === 'environment' || choice === 'user' ? { facingMode: { ideal: choice } } : { deviceId: { exact: choice } }) };
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    if (token !== ui.cameraGeneration) { stream.getTracks().forEach(t => t.stop()); return; }
    ui.stream = stream; $('cameraVideo').srcObject = stream;
    await $('cameraVideo').play();
    if (token !== ui.cameraGeneration) return;
    const settings = stream.getVideoTracks()[0].getSettings();
    ui.cameraSettings = { width: settings.width, height: settings.height, frameRate: settings.frameRate, facingMode: settings.facingMode || null };
    $('cameraMeta').textContent = `${settings.width || '?'} × ${settings.height || '?'} · ${Math.round(settings.frameRate || 0)} fps`;
    $('cameraPlaceholder').hidden = true; $('cameraDot').classList.add('active');
    $('cameraToggle').textContent = 'Stop camera'; ui.decoder = new QrDecoder();
    ui.cameraRunStarted = now(); ui.cameraRunEnded = null; ui.scanCount = 0;
    ui.cameraRun = { startedAt: new Date().toISOString(), requested: { width, height, facingMode: choice === 'user' || choice === 'environment' ? choice : 'selected-device' }, actual: { ...ui.cameraSettings }, attemptedFrames: 0, successfulFrames: 0, decodeTimeMs: 0, firstDecodeMs: null, lastDecoded: null };
    ui.cameraPending = false; $('cameraToggle').disabled = false;
    enumerateCameras().catch(() => {}); scanLoop(token);
  } catch (error) {
    if (token !== ui.cameraGeneration) return;
    stopCamera(); notice(`Camera could not start: ${error.message}. You can keep the transfer open and try Start camera again.`);
  }
}
async function scanLoop(cameraToken) {
  if (cameraToken !== ui.cameraGeneration || !ui.stream || !ui.decoder) return;
  const start = now(), generation = ui.generation, decoder = ui.decoder, video = $('cameraVideo');
  try {
    if (video.readyState >= 2 && video.videoWidth && !decoder.busy) {
      const dimensions = fitDecodeDimensions(video.videoWidth, video.videoHeight, { maxSide: 1920, maxPixels: 2073600 });
      if (capture.width !== dimensions.width || capture.height !== dimensions.height) { capture.width = dimensions.width; capture.height = dimensions.height; }
      captureContext.drawImage(video, 0, 0, capture.width, capture.height);
      ui.cameraRun.attemptedFrames++;
      const decodeStarted = now();
      const result = await decoder.decode(captureContext.getImageData(0, 0, capture.width, capture.height));
      if (cameraToken !== ui.cameraGeneration || generation !== ui.generation) return;
      const decodeMs = now() - decodeStarted;
      ui.stats.decodeMs += decodeMs; ui.cameraRun.decodeTimeMs += decodeMs;
      if (result) {
        ui.cameraRun.successfulFrames++;
        ui.cameraRun.firstDecodeMs ??= now() - ui.cameraRunStarted;
        ui.cameraRun.lastDecoded = { bytes: result.bytes.length, version: result.version };
        await processScan(result, generation);
      }
      if (cameraToken === ui.cameraGeneration) cameraDiagnostics();
    }
  } catch (error) {
    if (cameraToken === ui.cameraGeneration) { ui.stats.decodeErrors++; $('scanMeta').textContent = `Decode error: ${error.message}`; }
  } finally {
    if (cameraToken === ui.cameraGeneration && ui.stream) ui.scanTimer = setTimeout(() => scanLoop(cameraToken), Math.max(20, SCAN_INTERVAL_MS - (now() - start)));
  }
}
function selectRole(role) {
  if (sessionActive() || ui.preparing) return;
  stopCamera(); resetSession(); clearReceived(); clearQr(); ui.role = role;
  for (const button of document.querySelectorAll('[data-role]')) { const selected = button.dataset.role === role; button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected)); }
  $('sendFields').hidden = role === 'receive'; $('receiveHint').hidden = role !== 'receive';
  $('startTransfer').textContent = role === 'send' ? 'Start sender' : 'Start receiver';
  $('roleHint').textContent = role === 'send' ? 'Start the receiver first. Use front cameras with screens facing each other.' : 'Receive the bytes, then display a large acknowledgment for the sender to scan.';
  profileSummary('send');
  $('transferConfig').open = true; status('Ready', role === 'send' ? 'Prepare a short payload for your first transfer.' : 'Start the receiver before starting the sender.');
}
function scanSample() {
  if (ui.tab !== 'scan') return;
  const kind = $('scanKind').value, ecc = $('scanEcc').value; $('scanText').hidden = kind !== 'text';
  try {
    const cap = QR_BYTE_CAPACITIES[ecc];
    if (!cap) throw new Error('Choose a valid QR correction level.');
    const size = kind === 'ack' ? 28 : kind === 'current' ? n('chunkBytes') + 18 : kind === 'max' ? cap : 146;
    const bytes = kind === 'text' ? encoder.encode($('scanText').value || 'QR link check') : diagnosticPayload(size, 7);
    if (bytes.length > cap) throw new Error(`${bytes.length.toLocaleString()} bytes exceed ${ecc} capacity (${cap.toLocaleString()}). Choose lower correction or a smaller sender chunk.`);
    notice('');
    display(bytes, kind === 'ack' ? 'Return-path check · ACK-sized' : kind === 'text' ? 'Independent scan check' : 'Forward-path check · data-sized', ecc);
    $('scanSizeHint').textContent = `${bytes.length.toLocaleString()} QR bytes · maximum ${cap.toLocaleString()} at ${ecc}. Binary samples are diagnostic patterns, not live protocol messages.`;
  } catch (error) { clearQr('Sample exceeds QR capacity', 'Adjust the sample or correction level.'); notice(error.message); }
}
function selectTab(tab) {
  if (tab === ui.tab) return;
  if (ui.sender || ui.receiver || ui.preparing) cancelTransfer();
  stopCamera(); resetSession(); clearReceived(); clearQr(); ui.tab = tab; ui.scanCount = 0;
  if (ui.simWorker) cancelSimulation();
  for (const button of document.querySelectorAll('[data-tab]')) {
    const selected = button.dataset.tab === tab; button.classList.toggle('selected', selected);
    if (selected) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  }
  $('transferPanel').hidden = tab !== 'transfer'; $('scanPanel').hidden = tab !== 'scan';
  $('simulationPanel').hidden = tab !== 'simulation'; $('livePanel').hidden = tab === 'simulation'; $('metrics').hidden = tab !== 'transfer';
  notice(''); $('scanMeta').textContent = 'No QR detected yet';
  if (tab === 'scan') { ui.scanStartedAt = now(); scanSample(); status('Independent QR check', 'Start the camera on the other device and point it at this code. No transfer protocol is running.'); }
  else if (tab === 'transfer') status('Ready', 'Start the receiver first, then the sender.');
}
async function expandStage() {
  const stage = $('stage');
  if (document.fullscreenElement === stage || stage.classList.contains('expanded')) {
    if (document.fullscreenElement === stage) await document.exitFullscreen();
    stage.classList.remove('expanded'); $('closeExpand').hidden = true;
  } else {
    try { if (!stage.requestFullscreen) throw new Error('Fullscreen unavailable'); await stage.requestFullscreen(); }
    catch { stage.classList.add('expanded'); }
    $('closeExpand').hidden = false;
  }
  requestAnimationFrame(drawQr);
}
function simulationSummary(report) {
  const stats = report.stats || {};
  return [
    `Status: ${report.status || 'running'}`,
    `Payload: ${fmt(report.payloadBytes)} bytes`,
    `Exact recovery: ${report.exact ? 'yes' : report.status === 'running' ? 'pending' : 'no'}`,
    `Sender received final ACK: ${report.confirmationReceived ? 'yes' : 'not yet'}`,
    `Simulated time: ${((report.simulatedMs || 0) / 1000).toFixed(1)} s`,
    `Actual computation: ${((report.elapsedMs || 0) / 1000).toFixed(2)} s`,
    `Packets: ${fmt(stats.transmissions)} · dropped ${fmt(stats.dropped)} · duplicated ${fmt(stats.duplicated)} · corrupted ${fmt(stats.corrupted)}`,
    `QR decodes: ${fmt(stats.qrDecoded)} · rejected ${fmt(stats.qrRejected)}`,
    `Data confirmed: ${fmt(report.sender?.ackedBytes)} / ${fmt(report.payloadBytes)} bytes`,
    report.expectedDigest ? `Expected SHA-256: ${report.expectedDigest}` : '',
    report.receivedDigest ? `Received SHA-256: ${report.receivedDigest}` : '',
  ].filter(Boolean).join('\n');
}
function runSimulation() {
  if (ui.simWorker) return;
  notice('');
  for (const id of ['simDrop', 'simDuplicate', 'simCorrupt', 'simSeed']) { if (!$(id).reportValidity()) return; }
  const id = ++ui.simId;
  let profile;
  try { profile = readProfile('simulation'); } catch (error) { notice(error.message); return; }
  const options = { ...profile, bytes: n('simBytes'), dropRate: n('simDrop') / 100, duplicateRate: n('simDuplicate') / 100, corruptRate: n('simCorrupt') / 100, seed: n('simSeed') };
  const worker = new Worker(new URL('./simulation-worker.js', import.meta.url), { type: 'module' }); ui.simWorker = worker;
  ui.simReport = { schema: 'qr-link-report-v1', status: 'running', options };
  $('runSimulation').disabled = true; $('cancelSimulation').disabled = false; $('simResults').hidden = false;
  $('simResults').textContent = 'Encoding and decoding the first standard QR…'; $('simStatus').textContent = 'Running in a worker';
  const finish = () => { if (ui.simWorker === worker) { worker.terminate(); ui.simWorker = null; $('runSimulation').disabled = false; $('cancelSimulation').disabled = true; } };
  worker.addEventListener('message', ({ data }) => {
    if (ui.simWorker !== worker || data?.id !== id) return;
    if (data.type === 'error') { ui.simReport = { ...ui.simReport, status: 'failed', error: String(data.error) }; $('simStatus').textContent = 'Failed'; notice(String(data.error)); finish(); return; }
    if (data.report) { ui.simReport = data.report; $('simResults').textContent = simulationSummary(data.report); }
    if (data.type === 'result') { $('simStatus').textContent = data.report?.exact && data.report?.confirmationReceived ? 'Exact transfer confirmed' : 'Finished without confirmed recovery'; finish(); }
  });
  worker.addEventListener('error', error => { if (ui.simWorker !== worker) return; ui.simReport = { ...ui.simReport, status: 'failed', error: error.message }; $('simStatus').textContent = 'Worker failed'; notice(error.message); finish(); });
  worker.postMessage({ type: 'run', id, options });
}
function cancelSimulation() {
  ui.simWorker?.terminate(); ui.simWorker = null;
  ui.simReport = { ...ui.simReport, status: 'cancelled' };
  $('runSimulation').disabled = false; $('cancelSimulation').disabled = true; $('simStatus').textContent = 'Cancelled';
}
function exportReport() {
  const s = snapshot();
  const report = ui.tab === 'simulation' && ui.simReport ? ui.simReport : {
    schema: 'qr-optical-report-v1', exportedAt: new Date().toISOString(), tab: ui.tab, role: ui.tab === 'transfer' ? ui.role : null,
    startedAt: ui.wallStartedAt, elapsedMs: Math.round(elapsed()), profile: ui.profile,
    status: s?.state || 'idle', session: s, camera: ui.cameraSettings, cameraRun: cameraReport(), stats: { ...ui.stats, qrEncodeMeanMs: ui.stats.qrEncodeCount ? ui.stats.qrEncodeMs / ui.stats.qrEncodeCount : 0, qrRenderMeanMs: ui.stats.qrRenderCount ? ui.stats.qrRenderMs / ui.stats.qrRenderCount : 0, dataDisplayIntervalMeanMs: ui.stats.dataDisplayIntervalCount ? ui.stats.dataDisplayIntervalMs / ui.stats.dataDisplayIntervalCount : 0 },
    lastDisplayed: ui.qr ? { bytes: ui.qrBytes.length, version: ui.qr.version, modules: ui.qr.size, ecc: ui.qrEcc } : null,
    scanCheck: ui.tab === 'scan' ? { kind: $('scanKind').value, ecc: $('scanEcc').value, reads: ui.scanCount, elapsedMs: Math.round(now() - (ui.scanStartedAt ?? now())) } : null,
    events: [...ui.events],
  };
  download(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }), `qr-report-${ui.tab}-${Date.now()}.json`);
}

for (const button of document.querySelectorAll('[data-tab]')) button.addEventListener('click', () => selectTab(button.dataset.tab));
for (const button of document.querySelectorAll('[data-role]')) button.addEventListener('click', () => selectRole(button.dataset.role));
for (const id of ['payloadKind', 'fileInput']) $(id).addEventListener('change', payloadControls);
$('payloadText').addEventListener('input', payloadControls);
$('startTransfer').addEventListener('click', startTransfer); $('cancelTransfer').addEventListener('click', cancelTransfer);
$('cameraToggle').addEventListener('click', () => ui.stream || ui.cameraPending ? stopCamera() : startCamera());
for (const id of ['cameraSelect', 'cameraResolution']) $(id).addEventListener('change', () => { if (ui.stream || ui.cameraPending) { stopCamera(); startCamera(); } });
for (const id of ['scanKind', 'scanEcc']) $(id).addEventListener('change', scanSample);
$('scanText').addEventListener('input', scanSample);
$('expandStage').addEventListener('click', expandStage); $('closeExpand').addEventListener('click', expandStage);
$('downloadReceived').addEventListener('click', () => { if (ui.received) download(new Blob([ui.received], { type: 'application/octet-stream' }), 'qr-received.bin'); });
$('runSimulation').addEventListener('click', runSimulation); $('cancelSimulation').addEventListener('click', cancelSimulation);
$('exportReport').addEventListener('click', exportReport);
window.addEventListener('resize', () => requestAnimationFrame(drawQr));
document.addEventListener('fullscreenchange', () => { $('closeExpand').hidden = !(document.fullscreenElement === $('stage') || $('stage').classList.contains('expanded')); requestAnimationFrame(drawQr); });
document.addEventListener('keydown', event => { if (event.key === 'Escape' && $('stage').classList.contains('expanded')) expandStage(); });
window.addEventListener('pagehide', () => { ui.generation++; stopCamera(); ui.simWorker?.terminate(); });
setInterval(pump, PUMP_INTERVAL_MS); initializeProfiles(); payloadControls();
if (!isSecureContext) notice('Open this page over HTTPS or localhost to enable the camera and integrity checks.');
