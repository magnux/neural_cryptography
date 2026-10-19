/** Small, non-cryptographic helpers for Crypto Lab context exchange and reports. */
import { PROTOCOL, exactKeys } from '../crypto/bytes.js';

export const CONTEXT_MAX_BYTES = 8192;
const CONTEXT_FORMAT = 'NeuralPicnicLab-context-v2';
const REPORT_FORMAT = 'NeuralPicnicLab-report-v2';
const MAX_ENTRIES = 50;
const MAX_FRAME_SAMPLES = 4096;
const encoder = new TextEncoder();

function boundedInteger(value, minimum, maximum) {
  return Number.isSafeInteger(value) && value >= minimum && value <= maximum;
}

function canonicalHex(value, byteCount) {
  return typeof value === 'string' && value.length === byteCount * 2 && /^[0-9a-f]+$/.test(value);
}

function validText(value) {
  if (typeof value !== 'string' || !value.length || value.length > 256) return false;
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return false;
    } else if (code >= 0xdc00 && code <= 0xdfff) return false;
  }
  return encoder.encode(value).length <= 256;
}

// Structural validation only: the worker remains responsible for evaluating
// the circuit and verifying a proof. Never import the proof engine here.
function contextSnapshot(document) {
  if (!exactKeys(document, ['publicKey', 'challenge'])) throw new TypeError('Invalid public context fields.');
  const { publicKey, challenge } = document;
  if (!exactKeys(publicKey, ['protocol', 'circuit', 'minimum_repetitions', 'anchor_outputs']) ||
      publicKey.protocol !== PROTOCOL) throw new TypeError('Invalid public key.');
  const descriptor = publicKey.circuit;
  if (!exactKeys(descriptor, ['name', 'side', 'rounds', 'key_bits']) ||
      descriptor.name !== 'binary-rotconv-feistel-v1' || descriptor.key_bits !== 256 ||
      ![4, 8, 16].includes(descriptor.side) || !boundedInteger(descriptor.rounds, 2, 64) ||
      !boundedInteger(publicKey.minimum_repetitions, 1, 512)) {
    throw new TypeError('Invalid circuit profile.');
  }
  const blockBits = 2 * descriptor.side * descriptor.side;
  const anchorBytes = Math.ceil(256 / blockBits) * blockBits / 8;
  if (!canonicalHex(publicKey.anchor_outputs, anchorBytes)) throw new TypeError('Invalid anchor outputs.');
  if (!exactKeys(challenge, ['protocol', 'nonce', 'lock_id', 'action', 'required_repetitions']) || challenge.protocol !== PROTOCOL ||
      !boundedInteger(challenge.required_repetitions, 1, 512) || !canonicalHex(challenge.nonce, 32) || !validText(challenge.lock_id) || !validText(challenge.action)) {
    throw new TypeError('Invalid challenge.');
  }
  return {
    publicKey: {
      protocol: PROTOCOL,
      circuit: { name: descriptor.name, side: descriptor.side, rounds: descriptor.rounds, key_bits: 256 },
      minimum_repetitions: publicKey.minimum_repetitions,
      anchor_outputs: publicKey.anchor_outputs,
    },
    challenge: { protocol: PROTOCOL, nonce: challenge.nonce, lock_id: challenge.lock_id, action: challenge.action, required_repetitions: challenge.required_repetitions },
  };
}

export function encodeContext(document) {
  const text = JSON.stringify({ format: CONTEXT_FORMAT, ...contextSnapshot(document) }, null, 2);
  if (encoder.encode(text).length > CONTEXT_MAX_BYTES) throw new RangeError('Public context exceeds 8 KiB.');
  return text;
}

export function decodeContext(text) {
  // Bound UTF-16 length before creating a UTF-8 copy or parsing JSON.
  if (typeof text !== 'string' || text.length > CONTEXT_MAX_BYTES ||
      encoder.encode(text).length > CONTEXT_MAX_BYTES) throw new RangeError('Public context exceeds 8 KiB.');
  let document;
  try { document = JSON.parse(text); } catch { throw new TypeError('Public context is not valid JSON.'); }
  if (!exactKeys(document, ['format', 'publicKey', 'challenge']) || document.format !== CONTEXT_FORMAT) {
    throw new TypeError('Unsupported public context format.');
  }
  return contextSnapshot({ publicKey: document.publicKey, challenge: document.challenge });
}

function record(value) {
  if (value === null || typeof value !== 'object') return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function ownData(value, key) {
  if (!record(value)) return undefined;
  const property = Object.getOwnPropertyDescriptor(value, key);
  return property && Object.hasOwn(property, 'value') ? property.value : undefined;
}

function pickNumbers(value, names) {
  const result = {};
  for (const name of names) {
    const candidate = ownData(value, name);
    if (typeof candidate === 'number' && Number.isFinite(candidate) && candidate >= 0) result[name] = candidate;
  }
  return result;
}

function pickBooleans(value, names) {
  const result = {};
  for (const name of names) {
    const candidate = ownData(value, name);
    if (typeof candidate === 'boolean') result[name] = candidate;
  }
  return result;
}

function profileSnapshot(value) {
  const result = pickNumbers(value, ['side', 'rounds', 'repetitions']);
  for (const name of ['id', 'name', 'label']) {
    const candidate = ownData(value, name);
    if (typeof candidate === 'string') result[name] = candidate.slice(0, 80);
  }
  return result;
}

function resultSnapshot(value) {
  const result = {
    ...pickNumbers(value, ['keygenMs', 'proveMs', 'verifyMs', 'totalMs', 'proofBytes', 'keyMinimumRepetitions', 'requestedRepetitions', 'usedRepetitions']),
    ...pickBooleans(value, ['accepted', 'passed']),
  };
  const status = ownData(value, 'status');
  if (['completed', 'failed', 'canceled', 'timeout'].includes(status)) result.status = status;
  const errorName = ownData(value, 'errorName');
  if (['Error', 'TypeError', 'RangeError', 'AbortError', 'TimeoutError', 'WorkerError', 'InvalidStateError'].includes(errorName)) {
    result.errorName = errorName;
  }
  for (const name of ['keyId', 'proofSha256']) {
    const candidate = ownData(value, name);
    if (typeof candidate === 'string' && /^[0-9a-f]{8,64}$/.test(candidate)) result[name] = candidate;
  }
  const profile = ownData(value, 'profile');
  if (record(profile)) result.profile = profileSnapshot(profile);
  const timings = ownData(value, 'timings');
  if (record(timings)) result.timings = pickNumbers(timings, ['keygenMs', 'proveMs', 'verifyMs', 'totalMs']);
  const checks = ownData(value, 'checks');
  if (Array.isArray(checks)) {
    result.checks = checks.slice(0, 32).map(check => {
      const name = ownData(check, 'name');
      return {
        ...(typeof name === 'string' ? { name: name.slice(0, 96) } : {}),
        ...pickBooleans(check, ['expected', 'accepted', 'passed']),
        ...pickNumbers(check, ['ms']),
      };
    });
  }
  const runs = ownData(value, 'runs');
  if (Array.isArray(runs)) {
    result.runs = runs.slice(0, 20).map(run => ({
      ...pickNumbers(run, ['proveMs', 'verifyMs', 'proofBytes', 'keyMinimumRepetitions', 'requestedRepetitions', 'usedRepetitions']),
      ...pickBooleans(run, ['accepted']),
    }));
  }
  const summary = ownData(value, 'summary');
  if (record(summary)) result.summary = pickNumbers(summary, ['proveMedianMs', 'verifyMedianMs']);
  // Deliberately do not traverse benchmark.latest, proof, challenge, publicKey,
  // secrets, seeds, raw error messages, or arbitrary objects.
  return result;
}

function responsivenessSnapshot(value) {
  const result = {
    ...pickNumbers(value, ['durationMs', 'visibleFrames', 'maxFrameGapMs', 'p95FrameGapMs', 'longTaskCount', 'longTaskMaxMs']),
    ...pickBooleans(value, ['hiddenDuringRun', 'longTasksSupported', 'measurementAvailable']),
  };
  for (const name of ['maxFrameGapMs', 'p95FrameGapMs']) {
    if (ownData(value, name) === null) result[name] = null;
  }
  return result;
}

function environmentSnapshot(value) {
  const result = pickNumbers(value, ['hardwareConcurrency', 'deviceMemoryGiB', 'devicePixelRatio']);
  for (const name of ['userAgent', 'platform', 'language']) {
    const candidate = ownData(value, name);
    if (typeof candidate === 'string') result[name] = candidate.slice(0, name === 'userAgent' ? 512 : 128);
  }
  for (const name of ['screen', 'viewport']) {
    const candidate = ownData(value, name);
    if (record(candidate)) result[name] = pickNumbers(candidate, ['width', 'height']);
  }
  return result;
}

export function describeEnvironment() {
  const navigator = globalThis.navigator;
  return environmentSnapshot({
    userAgent: navigator?.userAgent,
    platform: navigator?.platform,
    language: navigator?.language,
    hardwareConcurrency: navigator?.hardwareConcurrency,
    deviceMemoryGiB: navigator?.deviceMemory,
    devicePixelRatio: globalThis.devicePixelRatio,
    screen: { width: globalThis.screen?.width, height: globalThis.screen?.height },
    viewport: { width: globalThis.innerWidth, height: globalThis.innerHeight },
  });
}

export class LabReport {
  #environment;
  #entries = [];

  constructor(environment = {}) { this.#environment = environmentSnapshot(environment); }

  add(operation, result, responsiveness = {}) {
    if (typeof operation !== 'string' || !operation.length || operation.length > 64) {
      throw new TypeError('Report operation must contain 1..64 characters.');
    }
    const entry = {
      at: new Date().toISOString(), operation,
      result: resultSnapshot(result), responsiveness: responsivenessSnapshot(responsiveness),
    };
    this.#entries.push(entry);
    if (this.#entries.length > MAX_ENTRIES) this.#entries.shift();
    return structuredClone(entry);
  }

  export(notes = '') {
    if (typeof notes !== 'string') throw new TypeError('Report notes must be text.');
    return structuredClone({
      format: REPORT_FORMAT, appVersion: '0.9.0', protocol: PROTOCOL, createdAt: new Date().toISOString(),
      environment: this.#environment, notes: notes.slice(0, 2048), entries: this.#entries,
    });
  }

  clear() { this.#entries = []; }
}

/** Nearest-rank p95 of the most recent bounded frame interval sample. */
export function frameGapStatistics(samples) {
  if (!Array.isArray(samples) || samples.length > MAX_FRAME_SAMPLES ||
      samples.some(value => typeof value !== 'number' || !Number.isFinite(value) || value < 0)) {
    throw new TypeError('Invalid frame interval samples.');
  }
  if (!samples.length) return { maxFrameGapMs: null, p95FrameGapMs: null };
  const sorted = [...samples].sort((a, b) => a - b);
  return { maxFrameGapMs: sorted.at(-1), p95FrameGapMs: sorted[Math.ceil(sorted.length * 0.95) - 1] };
}

/** Observe rendering opportunities; never treat a hidden tab as a smooth UI. */
export function startResponsivenessMonitor() {
  const clock = () => globalThis.performance?.now() ?? Date.now();
  const started = clock();
  const document = globalThis.document;
  const samples = [];
  let sampleCursor = 0;
  let visibleFrames = 0;
  let lastFrame = null;
  let maximumGap = 0;
  let hiddenDuringRun = Boolean(document?.hidden);
  let requestId = null;
  let observer = null;
  let longTasksSupported = false;
  let longTaskCount = 0;
  let longTaskMaxMs = 0;
  let stopped = null;
  const consumeTasks = entries => {
    for (const entry of entries) {
      if (entry.startTime < started || !Number.isFinite(entry.duration) || entry.duration < 0) continue;
      longTaskCount++;
      longTaskMaxMs = Math.max(longTaskMaxMs, entry.duration);
    }
  };
  const visibilityChanged = () => {
    if (document?.hidden) hiddenDuringRun = true;
    lastFrame = null; // A background pause is not a measured rendering gap.
  };
  const frame = timestamp => {
    if (stopped) return;
    if (document?.hidden) { hiddenDuringRun = true; lastFrame = null; }
    else {
      visibleFrames++;
      if (lastFrame !== null && timestamp >= lastFrame) {
        const gap = timestamp - lastFrame;
        maximumGap = Math.max(maximumGap, gap);
        if (samples.length < MAX_FRAME_SAMPLES) samples.push(gap);
        else { samples[sampleCursor] = gap; sampleCursor = (sampleCursor + 1) % MAX_FRAME_SAMPLES; }
      }
      lastFrame = timestamp;
    }
    requestId = globalThis.requestAnimationFrame?.(frame) ?? null;
  };
  document?.addEventListener?.('visibilitychange', visibilityChanged);
  if (typeof globalThis.requestAnimationFrame === 'function') requestId = globalThis.requestAnimationFrame(frame);
  try {
    if (globalThis.PerformanceObserver?.supportedEntryTypes?.includes('longtask')) {
      observer = new globalThis.PerformanceObserver(list => consumeTasks(list.getEntries()));
      observer.observe({ type: 'longtask', buffered: false });
      longTasksSupported = true;
    }
  } catch { observer?.disconnect(); observer = null; }

  return {
    stop() {
      if (stopped) return { ...stopped };
      if (requestId !== null) globalThis.cancelAnimationFrame?.(requestId);
      document?.removeEventListener?.('visibilitychange', visibilityChanged);
      if (observer) { consumeTasks(observer.takeRecords()); observer.disconnect(); }
      const stats = frameGapStatistics(samples);
      stopped = {
        durationMs: Math.max(0, clock() - started), visibleFrames, hiddenDuringRun,
        maxFrameGapMs: samples.length ? maximumGap : null,
        p95FrameGapMs: stats.p95FrameGapMs,
        longTaskCount, longTaskMaxMs, longTasksSupported,
        measurementAvailable: samples.length > 0,
      };
      return { ...stopped };
    },
  };
}
