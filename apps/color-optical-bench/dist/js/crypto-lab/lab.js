import { CryptoClient } from './client.js';
import { CONTEXT_MAX_BYTES, LabReport, decodeContext, describeEnvironment, encodeContext, startResponsivenessMonitor } from './reports.js';

// This controller holds public data only. Secret generation, retention, and
// every circuit/proof operation are confined to the dedicated crypto worker.
const $ = id => document.getElementById(id);
const MAX_PROOF_BYTES = 8 * 1024 * 1024;
const report = new LabReport(describeEnvironment());
const client = new CryptoClient({ onInvalidate(error) {
  state.keyAvailable = false;
  clearVerification();
  if (!busy && error?.name === 'WorkerError') {
    showError(error);
    setStatus('Crypto worker lost', 'The private key was cleared. Run a new round trip to generate a key, or verify the retained public files.', 'error');
  }
  render();
} });
const supported = globalThis.isSecureContext && typeof Worker === 'function' &&
  !!globalThis.crypto?.subtle && typeof globalThis.crypto?.getRandomValues === 'function' &&
  typeof globalThis.structuredClone === 'function';

const state = {
  keyAvailable: false, publicKey: null, keyId: null, keygenMs: null,
  challenge: null, proof: null, proofSha256: null, proofOrigin: null,
  proveMs: null, verifyMs: null, usedRepetitions: null, accepted: null, benchmark: null,
  tamper: null, reference: null, reportCount: 0,
};
let busy = false;
let operationSerial = 0;
let reportEpoch = 0;
let elapsedTimer = null;
let activeMonitor = null;

const actionButtons = ['runRoundTrip', 'generateKey', 'newChallenge', 'generateProof', 'verifyProof',
  'runTamper', 'runReference', 'runBenchmark', 'importContext', 'importProof',
  'downloadContext', 'downloadProof', 'downloadReport'];
const configuredInputs = ['profileSide', 'profileRounds', 'profileRepetitions', 'lockId', 'challengeAction', 'requestedRepetitions', 'repetitionPreset', 'benchmarkRuns'];

function formatMs(value) {
  if (!Number.isFinite(value)) return '—';
  return value < 1000 ? `${value.toFixed(1)} ms` : `${(value / 1000).toFixed(2)} s`;
}
function formatBytes(value) {
  if (!Number.isFinite(value)) return '—';
  return value < 1024 ? `${value} B` : `${(value / 1024).toFixed(1)} KiB`;
}
function integer(id, minimum, maximum, description) {
  const value = Number($(id).value);
  if (!$(id).value.trim() || !Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${description} must be an integer from ${minimum} to ${maximum}.`);
  }
  return value;
}
function configuredProfile() {
  return {
    side: integer('profileSide', 4, 16, 'Circuit side'),
    rounds: integer('profileRounds', 2, 64, 'Circuit rounds'),
    repetitions: integer('profileRepetitions', 1, 512, 'Key minimum repetitions'),
  };
}
function activeProfile() {
  const key = state.publicKey;
  return key ? { side: key.circuit.side, rounds: key.circuit.rounds, repetitions: key.minimum_repetitions } : undefined;
}
function challengeInput() {
  const lockId = $('lockId').value;
  const action = $('challengeAction').value;
  const encoder = new TextEncoder();
  for (const [label, text] of [['Lock identifier', lockId], ['Action', action]]) {
    if (!text.length || encoder.encode(text).length > 256) throw new RangeError(`${label} must contain 1–256 UTF-8 bytes.`);
  }
  const requestedRepetitions = integer('requestedRepetitions', 1, 512, 'Requested proof repetitions');
  const floor = state.publicKey?.minimum_repetitions ?? configuredProfile().repetitions;
  if (requestedRepetitions < floor) throw new RangeError(`Requested proof repetitions must be at least the key minimum (${floor}). To test lower counts, choose a lower key minimum in Advanced profile and generate a new key.`);
  return { lockId, action, requestedRepetitions };
}
function publicContext() {
  if (!state.publicKey || !state.challenge) throw new Error('Load a public key and challenge first.');
  return { publicKey: state.publicKey, challenge: state.challenge };
}
function proofInput() {
  if (!state.proof) throw new Error('Generate or import a proof first.');
  return { ...publicContext(), proof: state.proof };
}
function assertActive(token) {
  if (token !== operationSerial) throw new DOMException('Operation canceled.', 'AbortError');
}
function setStatus(title, detail, status = 'idle') {
  $('operationStatus').textContent = title;
  $('operationDetail').textContent = detail;
  $('operationLight').dataset.state = status;
}
function clearError() { $('operationError').hidden = true; $('operationError').textContent = ''; }
function showError(error) {
  const message = error instanceof Error ? error.message : 'The operation could not be completed.';
  $('operationError').textContent = message;
  $('operationError').hidden = false;
}
function clearVerification() { state.accepted = null; state.verifyMs = null; }
function clearProof() {
  state.proof = null; state.proofSha256 = null; state.proofOrigin = null;
  state.proveMs = null; state.usedRepetitions = null; state.tamper = null; clearVerification();
}
function clearSession() {
  state.keyAvailable = false; state.publicKey = null; state.keyId = null; state.keygenMs = null;
  state.challenge = null; state.benchmark = null; clearProof();
}
function discardWorker() { client.cancel(); state.keyAvailable = false; }
function updateProfileInputs(profile) {
  $('profileSide').value = String(profile.side);
  $('profileRounds').value = String(profile.rounds);
  $('profileRepetitions').value = String(profile.repetitions);
}

function updateRepetitionPreset() {
  const value = $('requestedRepetitions').value;
  $('repetitionPreset').value = ['32', '64', '110', '160', '219'].includes(value) ? value : 'custom';
}
function renderRepetitionEstimate() {
  const element = $('repetitionSummary');
  try {
    const profile = activeProfile() || configuredProfile();
    const requested = integer('requestedRepetitions', 1, 512, 'Requested proof repetitions');
    const effective = Math.max(profile.repetitions, requested);
    const half = profile.side * profile.side;
    const lanes = Math.ceil(256 / (2 * half)) + 1;
    // Wire-layout arithmetic only. Circuit execution stays in the worker.
    const bytes = 42 + half / 4 + Math.ceil(effective / 4) + effective * (160 + profile.rounds * lanes * half / 8);
    element.dataset.invalid = String(requested < profile.repetitions);
    element.textContent = requested < profile.repetitions
      ? `Below key minimum ${profile.repetitions}. Effective lower bound: ${effective}; choose at least ${effective}, or change the key minimum and regenerate the key.`
      : `${effective} repetitions · compact proof ${bytes.toLocaleString()} B (${formatBytes(bytes)}) · nominal term (2/3)^${effective} ≈ 2^−${(effective * Math.log2(1.5)).toFixed(1)}. This is not guaranteed security.`;
  } catch (error) { element.dataset.invalid = 'true'; element.textContent = error.message; }
}

function renderControls() {
  const enabled = supported && !busy;
  for (const id of actionButtons) $(id).disabled = !enabled;
  for (const id of configuredInputs) $(id).disabled = !enabled;
  $('newChallenge').disabled = !enabled || !state.publicKey;
  $('generateProof').disabled = !enabled || !state.keyAvailable || !state.publicKey || !state.challenge;
  $('verifyProof').disabled = !enabled || !state.publicKey || !state.challenge || !state.proof;
  $('runTamper').disabled = !enabled || !state.publicKey || !state.challenge || !state.proof;
  $('importProof').disabled = !enabled || !state.publicKey || !state.challenge;
  $('downloadContext').disabled = !enabled || !state.publicKey || !state.challenge;
  $('downloadProof').disabled = !enabled || !state.proof;
  $('downloadReport').disabled = busy || state.reportCount === 0;
  $('cancelOperation').disabled = !busy;
  document.body.dataset.busy = String(busy);
}

function renderVerification() {
  const accepted = state.accepted;
  $('verificationResult').dataset.result = accepted === true ? 'pass' : accepted === false ? 'fail' : 'waiting';
  $('resultSymbol').textContent = accepted === true ? '✓' : accepted === false ? '×' : '○';
  $('verificationTitle').textContent = accepted === true ? 'Proof accepted' : accepted === false ? 'Proof rejected' : state.proof ? 'Ready to verify' : 'Ready to experiment';
  $('verificationDetail').textContent = accepted === true
    ? 'This proof verifies against the loaded public key and exact challenge.'
    : accepted === false ? 'The proof does not verify against the loaded public context.'
    : state.proof ? 'No verification result for the current proof and context.'
    : 'Run a round trip, or import a public context and binary proof.';
}

function createTable(headers, rows) {
  const wrap = document.createElement('div'); wrap.className = 'crypto-table-wrap';
  const table = document.createElement('table'); table.className = 'crypto-table';
  const head = document.createElement('thead'); const heading = document.createElement('tr');
  for (const text of headers) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = text; heading.append(th); }
  head.append(heading); table.append(head);
  const body = document.createElement('tbody');
  for (const cells of rows) {
    const row = document.createElement('tr');
    for (const value of cells) {
      const cell = document.createElement('td');
      cell.textContent = typeof value === 'object' ? value.text : value;
      if (typeof value === 'object' && value.className) cell.className = value.className;
      row.append(cell);
    }
    body.append(row);
  }
  table.append(body); wrap.append(table); return wrap;
}
function renderChecks(id, title, result, empty) {
  const container = $(id); container.replaceChildren();
  if (!result) { const p = document.createElement('p'); p.className = 'empty-result'; p.textContent = empty; container.append(p); return; }
  const heading = document.createElement('div'); heading.className = 'check-heading';
  const name = document.createElement('strong'); name.textContent = title;
  const outcome = document.createElement('span'); outcome.className = result.passed ? 'check-pass' : 'check-fail';
  outcome.textContent = `${result.passed ? 'All passed' : 'Check failed'} · ${formatMs(result.totalMs)}`;
  heading.append(name, outcome); container.append(heading);
  container.append(createTable(['Check', 'Expected', 'Observed', 'Result'], result.checks.map(check => [
    check.name, check.expected ? 'Accept' : 'Reject', check.accepted ? 'Accepted' : 'Rejected',
    { text: check.passed ? 'Pass' : 'Fail', className: check.passed ? 'check-pass' : 'check-fail' },
  ])));
}
function renderBenchmark() {
  const container = $('benchmarkResults'); container.replaceChildren();
  const result = state.benchmark;
  if (!result) { const p = document.createElement('p'); p.className = 'empty-result'; p.textContent = 'No benchmark yet. Try three runs on your phone and laptop.'; container.append(p); return; }
  const summary = document.createElement('div'); summary.className = 'benchmark-summary';
  for (const [name, value] of [['Median prove', result.summary.proveMedianMs], ['Median verify', result.summary.verifyMedianMs]]) {
    const item = document.createElement('div'); const label = document.createElement('span'); label.textContent = name;
    const time = document.createElement('strong'); time.textContent = formatMs(value); item.append(label, time); summary.append(item);
  }
  container.append(summary, createTable(['Run', 'Prove', 'Verify', 'Result'], result.runs.map((run, index) => [
    String(index + 1), formatMs(run.proveMs), formatMs(run.verifyMs),
    { text: run.accepted ? 'Accepted' : 'Rejected', className: run.accepted ? 'check-pass' : 'check-fail' },
  ])));
}
function render() {
  const profile = activeProfile();
  $('profileSummary').textContent = `${$('profileSide').value} × ${$('profileSide').value} circuit · ${$('profileRounds').value || '—'} circuit rounds · key minimum ${$('profileRepetitions').value || '—'} repetitions`;
  $('keyMode').textContent = state.keyAvailable ? 'LOCAL KEY' : state.publicKey ? 'PUBLIC ONLY' : 'NO KEY';
  $('keyState').textContent = state.keyAvailable ? `Secret in worker${Number.isFinite(state.keygenMs) ? ` · generated in ${formatMs(state.keygenMs)}` : ''}.` : 'No private key in memory.';
  $('keyFingerprint').textContent = state.keyId || (state.publicKey ? 'Imported public key · inspect the context below.' : '—');
  $('challengeSummary').textContent = state.challenge ? `${state.challenge.lock_id} · ${state.challenge.action} · requests ${state.challenge.required_repetitions} repetitions` : 'Create a challenge to bind the proof to a lock and action.';
  $('challengeNonce').textContent = state.challenge?.nonce || '—';
  $('proofOrigin').textContent = state.proof ? `${state.proofOrigin === 'imported' ? 'IMPORTED' : 'GENERATED'} · ${state.proof.byteLength.toLocaleString()} BYTES` : 'WAITING';
  $('proofSize').textContent = state.proof ? formatBytes(state.proof.byteLength) : '—';
  $('proveTime').textContent = formatMs(state.proveMs); $('verifyTime').textContent = formatMs(state.verifyMs);
  $('proofHash').textContent = state.proofSha256 || (state.proof ? 'Calculated when verification runs.' : '—');
  $('contextState').textContent = profile
    ? `${profile.side} × ${profile.side} circuit · ${profile.rounds} rounds · ${profile.repetitions} minimum repetitions${state.challenge ? ` · requests ${state.challenge.required_repetitions}` : ' · no challenge'}${state.usedRepetitions !== null ? ` · proof uses ${state.usedRepetitions}` : ''}.`
    : 'No public context loaded.';
  $('publicContext').textContent = state.publicKey && state.challenge ? encodeContext(publicContext()) : state.publicKey ? JSON.stringify({ publicKey: state.publicKey }, null, 2) : 'No public context yet.';
  $('reportCount').textContent = state.reportCount ? `${Math.min(state.reportCount, 50)} measurement${state.reportCount === 1 ? '' : 's'}${state.reportCount > 50 ? ' retained' : ''}.` : 'No measurements yet.';
  renderRepetitionEstimate(); renderVerification(); renderControls();
  renderChecks('tamperResults', 'Current-proof checks', state.tamper, 'Tamper checks need a proof and its context.');
  renderChecks('referenceResults', 'Python reference', state.reference, 'Python reference check has not run.');
  renderBenchmark();
}

function showResponsiveness(stats) {
  $('responsivenessSummary').textContent = stats.measurementAvailable
    ? `Latest operation: largest UI frame gap ${formatMs(stats.maxFrameGapMs)}; 95th percentile ${formatMs(stats.p95FrameGapMs)}.${stats.hiddenDuringRun ? ' The tab was hidden for part of the run; this is a partial observation.' : ''}`
    : `Latest operation was too short or had too few visible frame callbacks to measure UI gaps.${stats.hiddenDuringRun ? ' The tab was hidden during the run.' : ''}`;
}
async function request(type, payload, token) {
  assertActive(token);
  const monitor = startResponsivenessMonitor(); activeMonitor = monitor;
  const epoch = reportEpoch;
  const metadata = { profile: type === 'createKey' ? payload.profile : activeProfile(), keyId: state.keyId,
    keyMinimumRepetitions: type === 'createKey' ? payload.profile.repetitions : state.publicKey?.minimum_repetitions,
    requestedRepetitions: payload.requestedRepetitions ?? payload.challenge?.required_repetitions };
  let result, failure;
  try {
    result = await client.request(type, payload, { onProgress(progress) {
      if (token !== operationSerial) return;
      const phase = typeof progress?.phase === 'string' ? progress.phase.replaceAll('-', ' ') : 'Computing';
      const complete = Number(progress?.completed); const total = Number(progress?.total);
      $('operationDetail').textContent = Number.isFinite(complete) && total > 0 ? `${phase} · ${complete} / ${total}` : phase;
      if (Number.isFinite(complete) && total > 0) { $('operationProgress').max = total; $('operationProgress').value = complete; }
      else $('operationProgress').removeAttribute('value');
    } });
    assertActive(token);
  } catch (error) {
    failure = error;
    throw error;
  } finally {
    const stats = monitor.stop();
    if (activeMonitor === monitor) activeMonitor = null;
    // Cancellation can finish after a new job starts. Keep its measurement,
    // but never repopulate a report the user has explicitly reset.
    if (epoch === reportEpoch) {
      const status = failure ? failure.name === 'AbortError' ? 'canceled' : failure.name === 'TimeoutError' ? 'timeout' : 'failed' : 'completed';
      report.add(type, { ...metadata, ...result, status, ...(failure ? { errorName: failure.name } : {}) }, stats);
      state.reportCount++;
      if (token === operationSerial) showResponsiveness(stats);
      render();
    }
  }
  return result;
}
async function run(title, task) {
  if (busy || !supported) return;
  const token = ++operationSerial;
  busy = true; clearError();
  const started = performance.now();
  $('operationElapsed').textContent = '0.0 s';
  $('operationProgress').removeAttribute('value');
  setStatus(title, 'Running locally in the crypto worker. You can cancel at any time.', 'busy');
  renderControls();
  elapsedTimer = setInterval(() => { $('operationElapsed').textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`; }, 100);
  try {
    const detail = await task(token); assertActive(token);
    setStatus(`${title} complete`, detail || 'Measurements are available in the report.', 'success');
    $('operationProgress').max = 1; $('operationProgress').value = 1;
  } catch (error) {
    if (token !== operationSerial) return;
    clearVerification();
    if (!client.alive) state.keyAvailable = false;
    showError(error);
    setStatus(`${title} failed`, !client.alive ? 'The worker is closed and its private key has been cleared. Start a new round trip to generate a key.' : 'Review the message above and try again.', 'error');
    $('operationProgress').max = 1; $('operationProgress').value = 0;
  } finally {
    if (token === operationSerial) {
      clearInterval(elapsedTimer); elapsedTimer = null;
      $('operationElapsed').textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`;
      busy = false; render();
    }
  }
}
async function generateKey(token) {
  const profile = configuredProfile();
  discardWorker(); clearSession(); render();
  const result = await request('createKey', { profile }, token);
  state.publicKey = result.publicKey; state.keyId = result.keyId; state.keygenMs = result.keygenMs; state.keyAvailable = true;
  $('importStatus').textContent = 'Local key generated. Exported files contain public data only.';
  render(); return result;
}
async function ensureKey(token) {
  if (state.keyAvailable && !client.alive) {
    state.keyAvailable = false; clearVerification(); render();
  }
  if (!state.keyAvailable) await generateKey(token);
}
async function newChallenge(token) {
  const input = challengeInput();
  clearProof(); state.challenge = null; render();
  const result = await request('newChallenge', input, token);
  state.challenge = result.challenge; render(); return result;
}
function applyProof(result, origin = 'generated') {
  state.proof = result.proof; state.proofSha256 = result.proofSha256 || null;
  state.proofOrigin = origin; state.proveMs = result.proveMs ?? null; state.usedRepetitions = result.usedRepetitions ?? null;
  if (result.publicKey) state.publicKey = result.publicKey;
  if (result.keyId) state.keyId = result.keyId;
  if (result.challenge) state.challenge = result.challenge;
  state.tamper = null; clearVerification();
}
async function generateProof(token) {
  if (!state.keyAvailable || !client.alive) throw new Error('The private key is no longer available. Generate a key or run a new round trip.');
  clearProof(); render();
  const result = await request('prove', { challenge: state.challenge }, token);
  applyProof(result); render(); return result;
}
async function verifyProof(token) {
  clearVerification(); render();
  const result = await request('verify', proofInput(), token);
  state.accepted = result.accepted; state.verifyMs = result.verifyMs; state.proofSha256 = result.proofSha256; state.usedRepetitions = result.usedRepetitions ?? null;
  render(); return result;
}
function stop(title = 'Operation canceled', detail = 'Private key cleared. Public files can still be verified; a new round trip creates a new key.') {
  operationSerial++; clearInterval(elapsedTimer); elapsedTimer = null;
  activeMonitor?.stop(); activeMonitor = null;
  discardWorker(); busy = false; clearVerification(); clearError();
  $('operationProgress').max = 1; $('operationProgress').value = 0;
  setStatus(title, detail); render();
}
function download(content, type, filename) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a'); link.href = url; link.download = filename;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$('runRoundTrip').addEventListener('click', () => run('Round trip', async token => {
  challengeInput(); await ensureKey(token); await newChallenge(token); await generateProof(token);
  const result = await verifyProof(token);
  return result.accepted ? 'Fresh challenge → proof → accepted. Download a report or compare another device.' : 'The generated proof was rejected. Keep the report for investigation.';
}));
$('generateKey').addEventListener('click', () => run('Key generation', async token => {
  await generateKey(token); return 'Private key retained only in worker memory. Create a new challenge next.';
}));
$('newChallenge').addEventListener('click', () => run('New challenge', async token => {
  await newChallenge(token); return 'A fresh challenge is loaded. Any previous proof has been cleared.';
}));
$('generateProof').addEventListener('click', () => run('Proof generation', async token => {
  await generateProof(token); return 'Proof generated. Select Verify proof to check it against the public context.';
}));
$('verifyProof').addEventListener('click', () => run('Verification', async token => {
  const result = await verifyProof(token); return result.accepted ? 'The proof was accepted for this public context.' : 'The proof was rejected for this public context.';
}));
$('runTamper').addEventListener('click', () => run('Tamper checks', async token => {
  state.tamper = null; render(); state.tamper = await request('tamper', proofInput(), token);
  return state.tamper.passed ? 'Every check produced its expected accept or reject result.' : 'One or more checks did not produce the expected result.';
}));
$('runReference').addEventListener('click', () => run('Python reference check', async token => {
  state.reference = null; render(); state.reference = await request('selfTest', {}, token);
  return state.reference.passed ? 'The bundled Python reference checks passed. Your current key and proof are unchanged.' : 'A Python reference check failed. Your current key and proof are unchanged.';
}));
$('runBenchmark').addEventListener('click', () => run('Benchmark', async token => {
  const runs = integer('benchmarkRuns', 1, 10, 'Benchmark runs'); const input = challengeInput();
  await ensureKey(token); clearProof(); state.benchmark = null; render();
  const result = await request('benchmark', { ...input, runs }, token);
  state.benchmark = { runs: result.runs, summary: result.summary, totalMs: result.totalMs };
  applyProof(result.latest); state.accepted = result.latest.accepted; state.verifyMs = result.latest.verifyMs;
  return `${result.runs.length} round trips measured. The last proof and its context are ready to export.`;
}));
$('cancelOperation').addEventListener('click', () => stop());
$('resetLab').addEventListener('click', () => {
  stop('Lab reset', 'Private key and public session cleared. Ready for a new round trip.');
  clearSession(); state.reference = null; reportEpoch++; report.clear(); state.reportCount = 0;
  $('responsivenessSummary').textContent = 'UI frame gaps will be recorded during each operation.';
  $('importStatus').textContent = 'Context ≤ 8 KiB · proof ≤ 8 MiB. Files are read locally.';
  $('operationElapsed').textContent = '0.0 s'; render();
});
for (const id of ['profileSide', 'profileRounds', 'profileRepetitions']) {
  $(id).addEventListener('input', () => {
    if (busy) return;
    discardWorker(); clearSession(); clearError();
    setStatus('Profile changed', 'Local key and proof cleared. Generate a key or run a new round trip.'); render();
  });
}
for (const id of ['lockId', 'challengeAction', 'requestedRepetitions']) {
  $(id).addEventListener('input', () => {
    if (busy) return;
    state.challenge = null; clearProof(); clearError();
    if (id === 'requestedRepetitions') updateRepetitionPreset();
    setStatus('Challenge fields changed', 'Create a new challenge before generating or verifying a proof.'); render();
  });
}

$('repetitionPreset').addEventListener('change', () => {
  if (busy || $('repetitionPreset').value === 'custom') return;
  $('requestedRepetitions').value = $('repetitionPreset').value;
  state.challenge = null; clearProof(); clearError();
  setStatus('Proof repetitions changed', 'The key is unchanged. Create a new challenge before generating a proof.'); render();
});

$('downloadContext').addEventListener('click', () => {
  try { download(encodeContext(publicContext()), 'application/json', 'neural-key-public-context.json'); }
  catch (error) { showError(error); }
});
$('downloadProof').addEventListener('click', () => { if (state.proof) download(state.proof, 'application/octet-stream', 'neural-key-proof.bin'); });
$('downloadReport').addEventListener('click', () => download(JSON.stringify(report.export($('testNotes').value), null, 2), 'application/json', 'neural-key-crypto-report.json'));
$('importContext').addEventListener('click', () => { $('contextFile').value = ''; $('contextFile').click(); });
$('importProof').addEventListener('click', () => { $('proofFile').value = ''; $('proofFile').click(); });
$('contextFile').addEventListener('change', () => {
  const file = $('contextFile').files?.[0]; if (!file) return;
  run('Context import', async token => {
    discardWorker(); clearSession(); render();
    $('importStatus').textContent = 'Reading public context. Any local key and proof have been cleared.';
    if (file.size > CONTEXT_MAX_BYTES) throw new RangeError('Public context file exceeds 8 KiB.');
    const text = await file.text(); assertActive(token);
    const context = decodeContext(text);
    state.publicKey = context.publicKey; state.challenge = context.challenge;
    updateProfileInputs(activeProfile());
    $('lockId').value = state.challenge.lock_id; $('challengeAction').value = state.challenge.action;
    $('requestedRepetitions').value = String(state.challenge.required_repetitions); updateRepetitionPreset();
    $('importStatus').textContent = 'Public context loaded. Import its binary proof, then select Verify proof.';
    return 'Public-only context loaded. No private key is required for verification.';
  });
});
$('proofFile').addEventListener('change', () => {
  const file = $('proofFile').files?.[0]; if (!file) return;
  run('Proof import', async token => {
    publicContext(); clearProof(); render();
    $('importStatus').textContent = 'Reading binary proof. Previous verification result cleared.';
    if (file.size === 0 || file.size > MAX_PROOF_BYTES) throw new RangeError('Proof file must contain 1 byte to 8 MiB.');
    const buffer = await file.arrayBuffer(); assertActive(token);
    applyProof({ proof: new Uint8Array(buffer) }, 'imported');
    $('importStatus').textContent = 'Binary proof loaded but not yet verified. Select Verify proof.';
    return 'Proof bytes loaded. Verification will check the format and public context.';
  });
});

addEventListener('pagehide', () => {
  operationSerial++; clearInterval(elapsedTimer); elapsedTimer = null;
  activeMonitor?.stop(); activeMonitor = null;
  client.destroy(); state.keyAvailable = false; busy = false; clearVerification();
});
addEventListener('pageshow', event => {
  if (event.persisted) {
    $('operationProgress').max = 1; $('operationProgress').value = 0;
    setStatus('Page restored', 'The private key was cleared when you left this page. Public files can still be verified.'); render();
  }
});

if (!supported) {
  $('environmentNotice').textContent = 'This lab needs a secure HTTPS page, Web Workers, and Web Crypto. Open the hosted page in a current browser; localhost is also supported for development.';
  $('environmentNotice').hidden = false;
  setStatus('Browser requirements unavailable', 'Computation is disabled in this environment.');
} else {
  setStatus('Lab ready', 'Run a round trip to create an ephemeral key and test the default profile.');
}
render();
