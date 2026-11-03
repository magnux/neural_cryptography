/** Human grouping for current recording exports; not a pairing/authentication key. */
import {validateSessionId} from './contract.js';

function kind(bundle) {
  if(!bundle||bundle.version!==1||!['optical-decoder-sender','optical-decoder-captures'].includes(bundle.format))
    throw Error('Expected a current sender or receiver recording.');
  return bundle.format==='optical-decoder-sender'?'sender':'captures';
}

/** The payload CRC is unchanged by palette phases, unlike the header CRC.
 * Receiver exports without accepted records have no matching matrix yet.
 * A mixed-sequence receiver export is named for its first retained record;
 * offline matching still checks every complete header, never this prefix.
 */
export function recordingPrefix(bundle) {
  const sender=kind(bundle)==='sender',rows=sender?bundle.frames:bundle.records;
  if(!Array.isArray(rows)||sender&&!rows.length)throw Error('Recording has no valid frame list.');
  if(!rows.length)return 'unpaired';
  const crc=rows[0]?.info?.chunkCRC;
  if(!Number.isInteger(crc)||crc<0||crc>0xffffffff)throw Error('Recording has no valid first-matrix CRC32.');
  return crc.toString(16).padStart(8,'0');
}

export function recordingFilename(bundle) {
  const type=kind(bundle),prefix=recordingPrefix(bundle);
  validateSessionId(bundle.sessionId);
  if(type==='sender') {
    if(typeof bundle.sequenceId!=='string'||! /^[a-f0-9]{16}$/.test(bundle.sequenceId))throw Error('Recording has no valid sequence ID.');
    return `${prefix}_decoder-sender-${bundle.sequenceId}.json`;
  }
  if(typeof bundle.created!=='string'||!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/.test(bundle.created)||
      !Number.isFinite(Date.parse(bundle.created))||new Date(bundle.created).toISOString()!==bundle.created)
    throw Error('Recording has no valid creation timestamp.');
  const created=bundle.created.replace(/[:.]/g,'-');
  return `${prefix}_decoder-captures-${bundle.sessionId}-${created}.json`;
}
