import { encodeQr, renderQr, fitDecodeDimensions } from '../qr.js';
import { QrDecoder } from '../qr-client.js';
import { decodePacket } from '../protocol.js';
import { SPEED_PRESETS, validateProfile, maxChunkBytes } from '../profiles.js';
import { CryptoClient } from '../crypto-worker/client.js';
import { OpticalLink } from './link.js';
import { AuthSession } from './session.js';

const $ = id => document.getElementById(id);
const now = () => performance.now();
const fmt = value => Math.max(0, Math.round(value || 0)).toLocaleString();
const DEVICE_PRESETS = Object.freeze({
  phone: { speed: 'slow', chunkBytes: 128, ecc: 'Q', windowSize: 1, dwellMs: 350, ackHoldMs: 750, width: 100 },
  desktop: { speed: 'fast', chunkBytes: 1024, ecc: 'M', windowSize: 4, dwellMs: 200, ackHoldMs: 500, width: 65 },
});
const TERMINAL = new Set(['verified', 'rejected', 'peer-accepted', 'peer-rejected', 'failed', 'timed-out', 'closed']);
const ui = {
  role: 'verifier', generation: 0, session: null, link: null, crypto: null, profile: null,
  startedAt: null, wallStartedAt: null, lastState: '', events: [], lastReport: null, lastLinkSnapshot: null,
  qr: null, qrBytes: null, qrEcc: null, stream: null, decoder: null, cameraGeneration: 0,
  cameraPending: false, scanTimer: null, cameraSettings: null, timeoutMs: 1200000, qrCssWidth: null,
  cameraStats: { attempted: 0, decoded: 0, decodeMs: 0, decodeErrors: 0 },
};
const capture = document.createElement('canvas');
const captureContext = capture.getContext('2d', { willReadFrequently: true });
function deviceId() {
  let value;
  try { value = localStorage.getItem('qr-neural-key-device-id'); } catch { /* Storage is optional. */ }
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value || '')) {
    value = crypto.randomUUID();
    try { localStorage.setItem('qr-neural-key-device-id', value); } catch { /* Ephemeral label is sufficient. */ }
  }
  return value;
}
const localDeviceId = deviceId();
function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function event(state, detail = {}) {
  ui.events.push({ ms: Math.round(ui.startedAt === null ? 0 : now() - ui.startedAt), state, ...detail });
  if (ui.events.length > 100) ui.events.shift();
}
function readCount() {
  const value = Number($('repetitions').value);
  if (!$('repetitions').value.trim() || !Number.isInteger(value) || value < 1 || value > 512) throw new RangeError('Choose a whole-number repetition count from 1 to 512.');
  return value;
}
function readProfile() {
  const preset = SPEED_PRESETS.find(p => p.id === $('speedPreset').value) || SPEED_PRESETS[0];
  return validateProfile({ ...preset, controlEcc: 'Q', chunkBytes: Number($('chunkBytes').value), ecc: $('dataEcc').value, windowSize: Number($('windowSize').value), dwellMs: Number($('dwellMs').value), retryMs: Math.max(preset.retryMs, Number($('dwellMs').value)) });
}
function readAckHold() {
  const value = Number($('ackHoldMs').value);
  if (!Number.isInteger(value) || value < 250 || value > 2000) throw new RangeError('ACK display time must be a whole number from 250 to 2000 ms.');
  return value;
}
function readTimeout() {
  const value = Number($('attemptMinutes').value);
  if (![10, 20, 30].includes(value)) throw new RangeError('Choose an attempt deadline of 10, 20, or 30 minutes.');
  return value * 60000;
}
function profileDescription(profile) {
  return profile ? `${fmt(profile.chunkBytes)} B/chunk · ${profile.ecc} correction · window ${profile.windowSize} · ${profile.dwellMs} ms/QR` : 'Not started';
}
function configSummary() {
  try {
    const count = readCount(), bytes = 58 + Math.ceil(count / 4) + 544 * count;
    $('proofEstimate').textContent = `${fmt(bytes)} proof bytes · about ${(bytes / 1000).toFixed(1)} seconds at 1,000 B/s, plus pairing and ACKs. Phone-safe settings will be slower. Nominal repetition exponent: ${(count * Math.log2(1.5)).toFixed(1)} bits.`;
    $('profileSummary').textContent = `${count} repetitions · ${Number($('attemptMinutes').value)} minute deadline`;
  } catch (error) { $('proofEstimate').textContent = error.message; }
  try {
    $('chunkBytes').max = String(maxChunkBytes($('dataEcc').value));
    const profile = readProfile(), hold = readAckHold();
    $('transportSummary').textContent = `${fmt(profile.chunkBytes)} B/chunk · window ${profile.windowSize} · ${profile.dwellMs} ms`;
    $('presetDetail').textContent = `${profileDescription(profile)} · ACK visible for at least ${hold} ms. Configured maximum: ${fmt(profile.chunkBytes)} bytes per chunk.`;
  } catch (error) { $('presetDetail').textContent = error.message; }
}
function selectDevicePreset(value) {
  const preset = DEVICE_PRESETS[value];
  if (!preset) return;
  $('devicePreset').value = value; $('speedPreset').value = preset.speed;
  for (const key of ['chunkBytes', 'windowSize', 'dwellMs', 'ackHoldMs']) $(key).value = String(preset[key]);
  $('dataEcc').value = preset.ecc; $('qrWidth').value = String(preset.width);
  $('qrWidthValue').textContent = `${preset.width}%`; drawQr(); configSummary();
  $('transportApplied').textContent = ui.link ? 'Outgoing settings changed. Apply to this attempt.' : 'Settings are ready for the next attempt.';
}
async function applyTransport() {
  try {
    const profile = readProfile(), ackHoldMs = readAckHold();
    const link = ui.link;
    if (link) await link.setProfile(profile, { adaptive: $('adaptiveTransport').checked, ackHoldMs });
    if (link && ui.link !== link) return;
    ui.profile = profile; configSummary(); update(); notice('');
    $('transportApplied').textContent = link ? 'Applied. Active settings below show what this message is using.' : 'Outgoing settings ready for the next attempt.';
    event('transport-configured', { chunkBytes: profile.chunkBytes, windowSize: profile.windowSize, dwellMs: profile.dwellMs, ecc: profile.ecc, ackHoldMs, adaptive: $('adaptiveTransport').checked });
  } catch (error) { notice(error.message); $('transportApplied').textContent = error.message; }
}
function qrWidthPercent() {
  return Math.max(40, Math.min(100, Number($('qrWidth').value) || 100));
}
function freezeConfig(frozen) {
  for (const input of $('attemptConfig').querySelectorAll('input,select')) input.disabled = frozen;
  for (const button of document.querySelectorAll('[data-role]')) button.disabled = frozen;
  $('startAttempt').disabled = frozen;
  $('resetAttempt').disabled = !frozen;
  $('swapRole').disabled = frozen;
}
function clearQr() {
  ui.qr = ui.qrBytes = ui.qrEcc = null; ui.qrCssWidth = null;
  $('qrCanvas').hidden = true; $('qrPlaceholder').hidden = false;
  $('qrLabel').textContent = 'Outgoing QR'; $('qrMeta').textContent = 'No code displayed';
}
function drawQr() {
  if (!ui.qr) return;
  const ratio = window.devicePixelRatio || 1;
  const expanded = $('stage').classList.contains('expanded') || document.fullscreenElement === $('stage');
  const room = expanded ? Math.max(90, window.innerHeight - 260) : 570;
  const available = Math.max(40, Math.min($('qrZone').clientWidth, room) * qrWidthPercent() / 100);
  const scale = Math.min(16, Math.floor(2800 / (ui.qr.size + 8)), Math.max(1, Math.floor(available * ratio / (ui.qr.size + 8))));
  const raster = renderQr(ui.qr, { scale, quiet: 4 });
  const canvas = $('qrCanvas'); canvas.width = raster.width; canvas.height = raster.height;
  canvas.getContext('2d').putImageData(new ImageData(raster.data, raster.width, raster.height), 0, 0);
  ui.qrCssWidth = raster.width / ratio;
  canvas.style.width = `${ui.qrCssWidth}px`; canvas.style.height = `${ui.qrCssWidth}px`;
  $('qrMeta').textContent = `${fmt(ui.qrBytes?.length)} B · QR ${ui.qr.version} · ${ui.qr.size} modules · ${ui.qrEcc} · ${Math.round(ui.qrCssWidth)} CSS px`;
}
function displayPacket(bytes) {
  const packet = decodePacket(bytes);
  const ecc = packet.kind === 'data' ? (ui.link?.snapshot().effectiveProfile || ui.profile || readProfile()).ecc : 'Q';
  if (ui.qrEcc === ecc && ui.qrBytes?.length === bytes.length && bytes.every((v, i) => v === ui.qrBytes[i])) return;
  ui.qr = encodeQr(bytes, { ecc }); ui.qrBytes = bytes.slice(); ui.qrEcc = ecc;
  $('qrCanvas').hidden = false; $('qrPlaceholder').hidden = true;
  $('qrLabel').textContent = packet.kind === 'data' ? `Sending chunk ${packet.index + 1}` : packet.kind === 'ack' ? 'Acknowledgment · keep visible' : packet.kind === 'start' || packet.kind === 'restart' ? 'Introducing a message' : packet.kind === 'poll' ? 'Requesting acknowledgment' : 'Transfer control';
  drawQr();
}
function progressLabel(value) {
  if (!value || typeof value !== 'object') return '';
  const current = value.done ?? value.completed ?? value.round ?? value.current;
  const total = value.total ?? value.repetitions;
  return Number.isFinite(current) && Number.isFinite(total) ? ` ${fmt(current)} / ${fmt(total)}.` : '';
}
function update() {
  const snapshot = ui.link?.snapshot() || ui.lastLinkSnapshot;
  const effective = snapshot?.effectiveProfile || ui.profile;
  $('outgoingProfile').textContent = profileDescription(effective);
  const incoming = snapshot?.receiveProfile || snapshot?.receiver;
  $('incomingProfile').textContent = incoming?.chunkBytes ? `${fmt(incoming.chunkBytes)} B/chunk · window ${incoming.windowSize} · chosen by peer` : 'Waiting for peer’s message settings';
  if (snapshot?.adaptation?.reason) $('outgoingProfile').textContent += ` · ${snapshot.adaptation.reason}`;
  const s = ui.session?.snapshot();
  if (!s) return;
  if (s.state !== ui.lastState) { ui.lastState = s.state; event(s.state); }
  if (['failed', 'timed-out', 'closed'].includes(s.state) && ui.link) {
    const stoppedLink = ui.link;
    ui.lastLinkSnapshot = stoppedLink.snapshot(); ui.link = null;
    stoppedLink.close(s.error || s.status); clearQr();
  }
  const link = ui.link?.snapshot() || ui.lastLinkSnapshot;
  $('pairingPanel').hidden = !s.pairingCode;
  $('pairingCode').textContent = (s.pairingCode || '').replace(/\s/g, '').match(/.{1,4}/g)?.join(' ') || '';
  $('pairingState').textContent = s.confirmed ? 'CONFIRMED ON THIS DEVICE' : 'CONFIRM ON EACH DEVICE';
  $('confirmPairing').disabled = !(s.canConfirm ?? (s.pairingCode && !s.confirmed && !TERMINAL.has(s.state)));
  $('confirmPairing').textContent = s.confirmed ? 'Pairing confirmed locally' : 'The codes match · confirm pairing';
  $('localKeyId').textContent = s.localKeyId || 'Not generated';
  $('peerKeyId').textContent = s.peerKeyId || 'Not received';
  let title = s.status || s.state.replaceAll('-', ' '), detail = '';
  if (s.state === 'verified') {
    title = 'Proof accepted locally';
    detail = s.resultDelivered ? 'The key holder received the result and its ACK returned. This verifier completed the authentication.' : s.error || 'This verifier checked the proof against its pinned key and fresh challenge. Keep both screens aligned while the result is delivered.';
  } else if (s.state === 'rejected') {
    title = 'Proof rejected locally'; detail = 'The proof did not verify. Keep the displays aligned to deliver the failure result, then reset both devices.';
  } else if (s.state === 'peer-accepted' || s.state === 'peer-rejected') {
    title = s.state === 'peer-accepted' ? 'Peer reports accepted' : 'Peer reports rejected';
    detail = 'This is the other device’s reported result, not cryptographic authentication of that verifier. Keep this ACK visible until the verifier confirms result delivery. Swap roles for a separate reverse-direction test.';
  } else if (s.state === 'pairing') {
    detail = s.confirmed ? 'Pairing confirmed here. The other device must also confirm before its proof can be generated.' : 'Compare all six groups with the intended device, then confirm the matching code on both screens.';
  } else if (s.state === 'failed' || s.state === 'timed-out' || s.state === 'closed') {
    detail = s.error || 'Reset both devices and start a fresh attempt.';
  } else {
    detail = `${progressLabel(s.progress)} Keep both cameras aimed at the opposite display. ACKs must travel in both directions.`.trim();
  }
  if ($('statusTitle').textContent !== title) $('statusTitle').textContent = title;
  if ($('statusDetail').textContent !== detail) $('statusDetail').textContent = detail;
  $('statusBox').className = `status ${s.state === 'verified' ? 'success' : ['failed', 'timed-out', 'rejected', 'peer-rejected'].includes(s.state) ? 'error' : 'working'}`;
  $('stageStatus').textContent = s.pairingCode && !s.confirmed ? 'Pairing code ready: leave expanded view and compare both screens.' : title;
  const sender = link?.sender, receiver = link?.receiver;
  $('sendProgress').textContent = sender?.totalBytes ? `${fmt(sender.ackedBytes)} / ${fmt(sender.totalBytes)} B` : '—';
  $('receiveProgress').textContent = receiver?.totalBytes ? `${fmt(receiver.receivedBytes)} / ${fmt(receiver.totalBytes)} B` : '—';
  $('proofBytes').textContent = s.proofBytes ? `${fmt(s.proofBytes)} B` : '—';
  const remaining = s.deadline ? Math.max(0, Math.ceil((s.deadline - now()) / 1000)) : null;
  $('timeRemaining').textContent = TERMINAL.has(s.state) ? 'Finished' : remaining === null ? '—' : `${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, '0')}`;
  $('transportState').textContent = `Send: ${sender?.state || 'idle'} · receive: ${receiver?.state || 'idle'} · queued: ${link?.queued || 0} · retries: ${sender?.retries || 0}`;
}
function plainFields(value, keys) {
  if (!value) return null;
  return Object.fromEntries(keys.filter(key => ['number', 'string', 'boolean'].includes(typeof value[key])).map(key => [key, value[key]]));
}
function reportProfile(profile) {
  return plainFields(profile, ['id', 'label', 'chunkBytes', 'ecc', 'controlEcc', 'windowSize', 'dwellMs', 'retryMs']);
}
function reportAdaptation(value) {
  if (!value) return null;
  return { ...plainFields(value, ['restarts', 'downshifts', 'upshifts', 'reason']), pendingProfile: reportProfile(value.pendingProfile), events: (value.events || []).slice(-64).map(item => plainFields(item, ['at', 'ms', 'kind', 'type', 'reason', 'chunkBytes', 'windowSize', 'dwellMs', 'fromChunkBytes', 'toChunkBytes', 'from', 'to', 'retries'])) };
}
function makeReport() {
  const s = ui.session?.snapshot(), link = ui.link?.snapshot() || ui.lastLinkSnapshot;
  return {
    format: 'QR-Neural-Key-auth-report-v2', exportedAt: new Date().toISOString(), startedAt: ui.wallStartedAt,
    elapsedMs: ui.startedAt === null ? 0 : Math.round(now() - ui.startedAt), role: ui.role,
    configuration: { repetitions: s?.repetitions ?? Number($('repetitions').value), profile: ui.profile || readProfile(), circuitSide: 8, circuitRounds: 16, timeoutMs: ui.timeoutMs, qrWidthPercent: qrWidthPercent(), qrCssWidth: ui.qrCssWidth, qrVersion: ui.qr?.version ?? null, qrModules: ui.qr?.size ?? null },
    authentication: plainFields(s, ['state', 'status', 'confirmed', 'repetitions', 'pairingCode', 'localKeyId', 'peerKeyId', 'proofBytes', 'proofDigest', 'resultDelivered', 'peerReported', 'error']),
    timings: plainFields(s?.timings, ['keygenMs', 'proveMs', 'verifyMs', 'elapsedMs']),
    transport: { configuredProfile: reportProfile(link?.configuredProfile || ui.profile || readProfile()), effectiveProfile: reportProfile(link?.effectiveProfile || ui.profile), ackHoldMs: link?.ackHoldMs ?? readAckHold(), adaptive: link?.adaptive ?? $('adaptiveTransport').checked, receiveProfile: plainFields(link?.receiveProfile || link?.receiver, ['chunkBytes', 'windowSize', 'totalBytes']), adaptation: reportAdaptation(link?.adaptation), queued: link?.queued || 0, sender: plainFields(link?.sender, ['state', 'totalBytes', 'ackedBytes', 'chunkCount', 'ackedChunks', 'sentPackets', 'retries', 'error']), receiver: plainFields(link?.receiver, ['state', 'totalBytes', 'receivedBytes', 'receivedChunks', 'completedMessages', 'duplicatePackets', 'malformedPackets', 'error']) },
    camera: { requestedWidth: Number($('cameraResolution').value), actual: ui.cameraSettings, ...ui.cameraStats },
    events: ui.events.map(item => ({ ...item })),
  };
}
function resetAttempt(message = '') {
  if (ui.session) ui.lastReport = makeReport();
  ui.generation++;
  const session = ui.session, link = ui.link, cryptoClient = ui.crypto;
  ui.session = ui.link = ui.crypto = null;
  session?.close(); link?.close(); cryptoClient?.destroy();
  clearQr(); freezeConfig(false); $('pairingPanel').hidden = true;
  $('statusTitle').textContent = 'Ready'; $('statusDetail').textContent = message || 'Start a new attempt on both devices. The previous ephemeral key and pairing have been discarded.';
  $('statusBox').className = 'status'; $('stageStatus').textContent = 'Start an attempt on both devices.';
  for (const id of ['sendProgress', 'receiveProgress', 'proofBytes', 'timeRemaining']) $(id).textContent = '—';
  $('localKeyId').textContent = 'Not generated'; $('peerKeyId').textContent = 'Not received'; $('transportState').textContent = 'Idle';
}
async function startAttempt() {
  if (ui.session) return;
  let repetitions, profile, ackHoldMs, timeoutMs;
  try { repetitions = readCount(); profile = readProfile(); ackHoldMs = readAckHold(); timeoutMs = readTimeout(); }
  catch (error) { notice(error.message); return; }
  notice(''); ui.startedAt = now(); ui.wallStartedAt = new Date().toISOString(); ui.events = []; ui.lastState = ''; ui.lastLinkSnapshot = null; ui.profile = profile; ui.timeoutMs = timeoutMs;
  const generation = ++ui.generation;
  const fail = error => {
    if (generation !== ui.generation) return;
    notice(error.message || String(error)); event('ui-error', { message: error.message || String(error) }); update();
  };
  try {
    ui.crypto = new CryptoClient({ onInvalidate: error => { if (generation === ui.generation && error.name !== 'AbortError') { resetAttempt(error.message); notice(error.message); } } });
    ui.link = new OpticalLink({ profile, adaptive: $('adaptiveTransport').checked, ackHoldMs, now, timeoutMs, onChange: update, onError: fail, onDelivery: bytes => {
      if (generation !== ui.generation) return;
      Promise.resolve(ui.session?.receive(bytes)).catch(fail);
    } });
    ui.session = new AuthSession({ role: ui.role, deviceId: localDeviceId, cryptoClient: ui.crypto, send: bytes => ui.link.send(bytes), now, onChange: update, timeoutMs, repetitions });
    freezeConfig(true); $('attemptConfig').open = false; $('transportConfig').open = false;
    await ui.session.start();
    if (generation === ui.generation) { update(); if (!ui.stream) notice('Start the camera on this device so it can read the other screen.'); }
  } catch (error) { fail(error); if (generation === ui.generation && !ui.session) resetAttempt(error.message); }
}
function selectRole(role) {
  if (ui.session) return;
  ui.role = role;
  for (const button of document.querySelectorAll('[data-role]')) { const selected = button.dataset.role === role; button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected)); }
  $('startAttempt').textContent = role === 'verifier' ? 'Generate key & start verifier' : 'Generate key & wait for verifier';
  $('roleHint').textContent = role === 'verifier' ? 'This device challenges the key holder and verifies its proof locally.' : 'This device proves possession of its local key to the verifier.';
}
function stopCamera() {
  ui.cameraGeneration++; clearTimeout(ui.scanTimer); ui.scanTimer = null;
  ui.stream?.getTracks().forEach(track => track.stop()); ui.stream = null;
  ui.decoder?.close(); ui.decoder = null; $('cameraVideo').srcObject = null;
  ui.cameraPending = false; $('cameraToggle').disabled = false; $('cameraToggle').textContent = 'Start camera';
  $('cameraPlaceholder').hidden = false; $('cameraDot').classList.remove('active');
}
async function enumerateCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const selected = $('cameraSelect').value, devices = await navigator.mediaDevices.enumerateDevices();
  for (const option of [...$('cameraSelect').options].slice(2)) option.remove();
  for (const [index, device] of devices.filter(d => d.kind === 'videoinput').entries()) {
    const option = document.createElement('option'); option.value = device.deviceId; option.textContent = device.label || `Camera ${index + 1}`; $('cameraSelect').append(option);
  }
  if ([...$('cameraSelect').options].some(option => option.value === selected)) $('cameraSelect').value = selected;
}
async function startCamera() {
  if (ui.stream || ui.cameraPending) return;
  if (!navigator.mediaDevices?.getUserMedia) { notice('Camera capture requires HTTPS or localhost and a browser with camera access.'); return; }
  const token = ++ui.cameraGeneration; ui.cameraPending = true; $('cameraToggle').disabled = true;
  const width = Number($('cameraResolution').value), height = width === 640 ? 480 : Math.round(width * 9 / 16), choice = $('cameraSelect').value;
  const video = { width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: 30, max: 30 }, ...(choice === 'user' || choice === 'environment' ? { facingMode: { ideal: choice } } : { deviceId: { exact: choice } }) };
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    if (token !== ui.cameraGeneration) { stream.getTracks().forEach(track => track.stop()); return; }
    ui.stream = stream; $('cameraVideo').srcObject = stream; await $('cameraVideo').play();
    if (token !== ui.cameraGeneration) return;
    const settings = stream.getVideoTracks()[0].getSettings();
    ui.cameraSettings = { width: settings.width || null, height: settings.height || null, frameRate: settings.frameRate || null, facingMode: settings.facingMode || 'selected-device' };
    $('cameraMeta').textContent = `${settings.width || '?'} × ${settings.height || '?'} · ${Math.round(settings.frameRate || 0)} fps`;
    $('cameraPlaceholder').hidden = true; $('cameraDot').classList.add('active'); $('cameraToggle').textContent = 'Stop camera';
    ui.decoder = new QrDecoder(); ui.cameraPending = false; $('cameraToggle').disabled = false;
    ui.cameraStats = { attempted: 0, decoded: 0, decodeMs: 0, decodeErrors: 0 };
    enumerateCameras().catch(() => {}); notice(''); scanLoop(token);
  } catch (error) {
    if (token !== ui.cameraGeneration) return;
    stopCamera(); notice(`Camera could not start: ${error.message}. Check permission or choose another camera.`);
  }
}
async function scanLoop(token) {
  if (token !== ui.cameraGeneration || !ui.stream || !ui.decoder) return;
  const start = now(), decoder = ui.decoder, generation = ui.generation, video = $('cameraVideo');
  try {
    if (video.readyState >= 2 && video.videoWidth && !decoder.busy) {
      const dimensions = fitDecodeDimensions(video.videoWidth, video.videoHeight, { maxSide: 1920, maxPixels: 2073600 });
      if (capture.width !== dimensions.width || capture.height !== dimensions.height) { capture.width = dimensions.width; capture.height = dimensions.height; }
      captureContext.drawImage(video, 0, 0, capture.width, capture.height);
      ui.cameraStats.attempted++;
      const result = await decoder.decode(captureContext.getImageData(0, 0, capture.width, capture.height));
      if (token !== ui.cameraGeneration) return;
      ui.cameraStats.decodeMs += now() - start;
      if (result) {
        ui.cameraStats.decoded++;
        if (generation === ui.generation && ui.link) await ui.link.receive(result.bytes);
      }
      $('scanMeta').textContent = `${ui.cameraStats.decoded}/${ui.cameraStats.attempted} reads · ${Math.round(ui.cameraStats.decodeMs / ui.cameraStats.attempted)} ms`;
    }
  } catch (error) { if (token === ui.cameraGeneration) { ui.cameraStats.decodeErrors++; $('scanMeta').textContent = `Read error: ${error.message}`; } }
  finally { if (token === ui.cameraGeneration && ui.stream) ui.scanTimer = setTimeout(() => scanLoop(token), Math.max(20, 50 - (now() - start))); }
}
async function closeExpanded() {
  if (document.fullscreenElement === $('stage')) await document.exitFullscreen();
  $('stage').classList.remove('expanded'); $('closeExpand').hidden = true; drawQr();
}
for (const preset of SPEED_PRESETS) { const option = document.createElement('option'); option.value = preset.id; option.textContent = preset.label; $('speedPreset').append(option); }
$('devicePreset').addEventListener('change', () => selectDevicePreset($('devicePreset').value));
$('speedPreset').addEventListener('change', () => {
  const p = SPEED_PRESETS.find(p => p.id === $('speedPreset').value);
  for (const key of ['chunkBytes', 'windowSize', 'dwellMs']) $(key).value = String(p[key]);
  $('dataEcc').value = p.ecc; $('devicePreset').value = 'custom'; configSummary();
  $('transportApplied').textContent = 'Outgoing settings changed. Apply to use them during this attempt.';
});
for (const id of ['chunkBytes', 'dataEcc', 'windowSize', 'dwellMs', 'ackHoldMs', 'adaptiveTransport']) $(id).addEventListener('input', () => {
  $('devicePreset').value = 'custom'; configSummary();
  $('transportApplied').textContent = 'Outgoing settings changed. Apply to use them during this attempt.';
});
$('applyTransport').addEventListener('click', applyTransport);
$('qrWidth').addEventListener('input', () => { $('qrWidthValue').textContent = `${qrWidthPercent()}%`; drawQr(); });
$('attemptMinutes').addEventListener('change', configSummary);
selectDevicePreset(window.innerWidth <= 700 ? 'phone' : 'desktop');
$('repetitionPreset').addEventListener('change', () => { if ($('repetitionPreset').value !== 'custom') $('repetitions').value = $('repetitionPreset').value; configSummary(); });
$('repetitions').addEventListener('input', () => { $('repetitionPreset').value = ['32', '64', '110', '219'].includes($('repetitions').value) ? $('repetitions').value : 'custom'; configSummary(); });
for (const button of document.querySelectorAll('[data-role]')) button.addEventListener('click', () => selectRole(button.dataset.role));
$('swapRole').addEventListener('click', () => selectRole(ui.role === 'verifier' ? 'prover' : 'verifier'));
$('startAttempt').addEventListener('click', startAttempt);
$('resetAttempt').addEventListener('click', () => { resetAttempt(); notice('Reset the other device too before starting a new attempt.'); });
$('confirmPairing').addEventListener('click', async () => { const generation = ui.generation; try { await ui.session?.confirmPairing(); } catch (error) { if (generation === ui.generation) notice(error.message); } update(); });
$('cameraToggle').addEventListener('click', () => ui.stream ? stopCamera() : startCamera());
for (const id of ['cameraSelect', 'cameraResolution']) $(id).addEventListener('change', () => { if (ui.stream || ui.cameraPending) { stopCamera(); startCamera(); } });
$('expandStage').addEventListener('click', async () => {
  try { if ($('stage').requestFullscreen) await $('stage').requestFullscreen(); else $('stage').classList.add('expanded'); }
  catch { $('stage').classList.add('expanded'); }
  $('closeExpand').hidden = false; drawQr();
});
$('closeExpand').addEventListener('click', () => closeExpanded().catch(() => {}));
document.addEventListener('fullscreenchange', () => { if (document.fullscreenElement !== $('stage')) $('closeExpand').hidden = !$('stage').classList.contains('expanded'); drawQr(); });
document.addEventListener('keydown', event => { if (event.key === 'Escape' && $('stage').classList.contains('expanded')) closeExpanded().catch(() => {}); });
$('exportReport').addEventListener('click', () => {
  const report = ui.session ? makeReport() : ui.lastReport || makeReport();
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `qr-neural-key-auth-${ui.role}-${Date.now()}.json`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
});
window.addEventListener('resize', drawQr);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { stopCamera(); if (ui.session) resetAttempt('The page was hidden, so the attempt stopped and the key was discarded. Start a fresh attempt on both devices.'); }
});
window.addEventListener('pagehide', () => { stopCamera(); resetAttempt(); });
setInterval(() => {
  try { ui.session?.tick(); const wire = ui.link?.next(); if (wire) displayPacket(wire); update(); }
  catch (error) { notice(error.message); resetAttempt(error.message); }
}, 50);
$('deviceLabel').textContent = `Device ${localDeviceId.slice(0, 8)}`;
configSummary(); enumerateCameras().catch(() => {});
