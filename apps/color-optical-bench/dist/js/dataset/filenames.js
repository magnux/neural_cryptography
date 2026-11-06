/** Public collection-code grouping, never an optical authentication key. */
import {validateSessionId,NEURAL_QR_VERSION} from './contract.js';

function kind(bundle) {
  if(!bundle||bundle.version!==NEURAL_QR_VERSION||!['neural-qr-sender','neural-qr-captures'].includes(bundle.format))
    throw Error('Expected a current Neural-QR sender or receiver recording.');
  return bundle.format==='neural-qr-sender'?'sender':'captures';
}

/** Header-free acquisition cannot recover an identity before model inference.
 * The receiver's explicitly entered code pairs the recording offline. It does
 * not certify that the camera was pointed at the intended physical matrix.
 */
export function recordingPrefix(bundle) {
  kind(bundle);
  if(typeof bundle.collectionCode!=='string'||! /^[a-f0-9]{8}$/.test(bundle.collectionCode))
    throw Error('Recording has no valid public collection code.');
  const rows=bundle.format==='neural-qr-sender'?bundle.frames:bundle.records;
  if(!Array.isArray(rows)||bundle.format==='neural-qr-sender'&&rows.length!==4)throw Error('Recording has no valid frame list.');
  if(rows.some(row=>row?.info?.collectionCode!==bundle.collectionCode))throw Error('Recording mixes collection codes.');
  return bundle.collectionCode;
}

export function recordingFilename(bundle) {
  const type=kind(bundle),prefix=recordingPrefix(bundle);
  validateSessionId(bundle.sessionId);
  if(type==='sender')return `${prefix}_neural-qr-sender-${bundle.sessionId}.json`;
  if(typeof bundle.created!=='string'||!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/.test(bundle.created)||
      !Number.isFinite(Date.parse(bundle.created))||new Date(bundle.created).toISOString()!==bundle.created)
    throw Error('Recording has no valid creation timestamp.');
  const created=bundle.created.replace(/[:.]/g,'-');
  return `${prefix}_neural-qr-captures-${bundle.sessionId}-${created}.json`;
}
