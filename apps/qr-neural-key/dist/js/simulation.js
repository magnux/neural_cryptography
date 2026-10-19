/** Real QR encode/decode with synthetic packet faults; not a camera performance model. */
import { createSender, Receiver } from './session.js';
import { decodePacket } from './protocol.js';
import { equalBytes } from './bytes.js';
import { diagnosticPayload, seededRandom } from './payload.js';
import { encodeQr, renderQr, decodeQr } from './qr.js';
import { maxChunkBytes, SPEED_PRESETS } from './profiles.js';

export const SIMULATION_LIMITS = Object.freeze({ maxBytes: 1024 * 1024, maxTransmissions: 200000, maxSteps: 500000 });
const TERMINAL = new Set(['delivered', 'failed', 'timed-out', 'cancelled']);
const clock = () => performance.now();
const yieldTask = () => new Promise(resolve => setTimeout(resolve, 0));
function integer(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw Error(`Invalid ${name}: expected ${min}..${max}`);
  return value;
}
function rate(value, name) {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw Error(`Invalid ${name}: expected 0..1`);
  return value;
}
function settings(input) {
  const ecc = input.ecc ?? 'Q', controlEcc = input.controlEcc ?? 'Q';
  // Validate the selected symbol capacity before copying or generating payloads.
  const maxChunk = maxChunkBytes(ecc);
  maxChunkBytes(controlEcc);
  if (input.presetId !== undefined && input.presetId !== 'custom' && !SPEED_PRESETS.some(p => p.id === input.presetId)) {
    throw Error('Invalid presetId');
  }
  const options = {
    bytes: integer(input.payload?.length ?? input.bytes ?? 1024, 0, SIMULATION_LIMITS.maxBytes, 'payload bytes'),
    chunkBytes: integer(input.chunkBytes ?? 128, 1, maxChunk, `chunkBytes at ECC ${ecc}`),
    windowSize: integer(input.windowSize ?? 1, 1, 8, 'windowSize'),
    seed: integer(input.seed ?? 1234, 0, 0xffffffff, 'seed'),
    dropRate: rate(input.dropRate ?? 0.15, 'dropRate'),
    duplicateRate: rate(input.duplicateRate ?? 0.15, 'duplicateRate'),
    corruptRate: rate(input.corruptRate ?? 0.05, 'corruptRate'),
    dwellMs: integer(input.dwellMs ?? 350, 1, 60000, 'dwellMs'),
    retryMs: integer(input.retryMs ?? 2500, 1, 86400000, 'retryMs'),
    timeoutMs: integer(input.timeoutMs ?? 3600000, 1, 86400000, 'timeoutMs'),
    maxRetries: integer(input.maxRetries ?? 20, 0, 10000, 'maxRetries'),
    ecc, controlEcc,
    ...(input.presetId !== undefined ? { presetId: input.presetId } : {}),
  };
  if (input.payload !== undefined && !(input.payload instanceof Uint8Array)) throw Error('Payload must be Uint8Array');
  return options;
}

/** The decoder gets only rendered pixels. Original bytes are used solely for the final comparison. */
export async function runSimulation(input = {}, { onProgress } = {}) {
  const options = settings(input), startedAt = clock();
  const payload = input.payload ? input.payload.slice() : diagnosticPayload(options.bytes, options.seed);
  const random = seededRandom(options.seed);
  const transferId = Array.from({ length: 8 }, () => Math.floor(random() * 256).toString(16).padStart(2, '0')).join('');
  const sender = await createSender(payload, { ...options, transferId, now: 0 });
  const receiver = new Receiver({ timeoutMs: options.timeoutMs, maxMessageBytes: SIMULATION_LIMITS.maxBytes });
  const expectedDigest = sender.snapshot().digest;
  const stats = { transmissions: 0, dataTransmissions: 0, controlTransmissions: 0, dropped: 0, duplicated: 0,
    corrupted: 0, qrDecoded: 0, qrRejected: 0, protocolRejected: 0, deliveries: 0 };
  const qr = { ecc: options.ecc, controlEcc: options.controlEcc, scale: 3, quiet: 4, data: null, control: null };
  const events = [];
  let simulatedMs = 0, receivedDigest = null, exact = false, guardError = null;
  function event(entry) {
    if (events.length === 100) events.shift();
    events.push({ simulatedMs, ...entry });
  }
  function report(final = false) {
    const s = sender.snapshot(), r = receiver.snapshot();
    return {
      schema: 'qr-link-report-v1', mode: 'simulation',
      model: 'Rendered standard QR images; deterministic packet loss, duplicate observations and pre-encoding CRC damage. No lens, motion, moire or camera model.',
      status: guardError ? 'failed' : final ? s.state : 'running', exact,
      confirmationReceived: s.state === 'delivered', payloadBytes: payload.length,
      elapsedMs: Math.round((clock() - startedAt) * 100) / 100, simulatedMs,
      stats: { ...stats }, sender: s, receiver: r, expectedDigest, receivedDigest,
      options: { ...options }, qr: { ...qr, data: qr.data && { ...qr.data }, control: qr.control && { ...qr.control } },
      events: events.slice(), ...(guardError ? { error: guardError } : {}),
    };
  }
  function inspectCode(code, byteLength, kind) {
    const group = kind === 'data' ? 'data' : 'control';
    if (!qr[group] || code.version > qr[group].version) qr[group] = {
      ecc: code.ecc, version: code.version, modules: code.size, imagePixels: (code.size + 8) * qr.scale, packetBytes: byteLength,
    };
  }
  async function transmit(packet, direction) {
    if (stats.transmissions >= SIMULATION_LIMITS.maxTransmissions) {
      guardError = 'Simulation transmission limit reached'; return;
    }
    const kind = decodePacket(packet).kind;
    stats.transmissions++;
    if (kind === 'data') stats.dataTransmissions++; else stats.controlTransmissions++;
    simulatedMs += options.dwellMs;
    if (random() < options.dropRate) {
      stats.dropped++; event({ direction, kind, outcome: 'dropped' }); return;
    }
    let wire = packet;
    const corrupted = random() < options.corruptRate;
    if (corrupted) {
      wire = packet.slice();
      wire[Math.floor(random() * wire.length)] ^= 1 << Math.floor(random() * 8);
      stats.corrupted++;
    }
    // Damage the wire packet before encoding it as a valid QR: QR decoding must
    // succeed, then the transport CRC must reject the modified packet.
    const code = encodeQr(wire, { ecc: kind === 'data' ? options.ecc : options.controlEcc });
    inspectCode(code, wire.length, kind);
    const image = renderQr(code, { scale: qr.scale, quiet: qr.quiet });
    const copies = random() < options.duplicateRate ? 2 : 1;
    if (copies === 2) stats.duplicated++;
    for (let copy = 0; copy < copies; copy++) {
      const decoded = decodeQr(image);
      if (!decoded) { stats.qrRejected++; event({ direction, kind, outcome: 'qr-rejected' }); continue; }
      stats.qrDecoded++;
      const result = direction === 'sender-to-receiver'
        ? await receiver.receive(decoded.bytes, simulatedMs)
        : sender.receive(decoded.bytes, simulatedMs);
      if (!result.accepted) stats.protocolRejected++;
      event({ direction, kind, outcome: result.accepted ? 'accepted' : result.reason ?? 'rejected', corrupted, duplicate: copy > 0 });
      if (result.delivery) {
        stats.deliveries++;
        receivedDigest = result.delivery.digest;
        exact = equalBytes(payload, result.delivery.bytes) && receivedDigest === expectedDigest;
      }
      for (const reply of result.outbound ?? []) await transmit(reply, 'receiver-to-sender');
    }
  }

  onProgress?.(report());
  let lastProgress = 0, steps = 0;
  while (!TERMINAL.has(sender.snapshot().state) && !guardError) {
    if (++steps > SIMULATION_LIMITS.maxSteps) { guardError = 'Simulation step limit reached'; break; }
    const packet = sender.next(simulatedMs);
    if (packet) await transmit(packet, 'sender-to-receiver');
    else simulatedMs += options.dwellMs;
    const expiry = receiver.tick(simulatedMs);
    for (const reply of expiry.outbound) await transmit(reply, 'receiver-to-sender');
    if (stats.transmissions - lastProgress >= 8) {
      lastProgress = stats.transmissions; onProgress?.(report()); await yieldTask();
    }
  }
  if (guardError) { sender.cancel(simulatedMs); receiver.cancel(simulatedMs); }
  const final = report(true);
  if (final.status === 'delivered' && !exact) { final.status = 'failed'; final.error = 'Confirmation arrived without exact payload recovery'; }
  return final;
}
