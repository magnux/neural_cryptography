import {sameBytes} from '../../apps/color-optical-bench/dist/js/format.js';
import {packetLayout,encodePacket,decodePacket,DEFAULT_PARITY} from '../../apps/color-optical-bench/dist/js/neural-qr/fec-packet.js';
export {packetLayout,encodePacket,decodePacket,DEFAULT_PARITY,MAX_WIRE_BYTES,CRC_BYTES} from '../../apps/color-optical-bench/dist/js/neural-qr/fec-packet.js';

/** Diagnostic recorded-error replay, not decoding of raw collection matrices.
 *
 * Collection targets have no RS redundancy or packet CRC. After classification,
 * replayCase measures their byte XOR errors and applies that channel to a valid
 * diagnostic codeword of the SAME total capacity. Data, CRC and parity all suffer
 * the measured errors. The blind decodePacket function has no ground truth,
 * error locations or erasures. Expected payload bytes are compared only after
 * it returns. Transferring an error pattern to different transmitted symbols is
 * an approximation; these numbers are not measured optical packet recovery.
 * CRC detects accidental corruption and is not authentication.
 */
export const REPLAY_METRIC='recorded-error-replay';
/** Measure channel errors after inference, construct replay, then score blindly. */
export function replayCase(expected,predicted,{parity=DEFAULT_PARITY}={}){
  if(!(expected instanceof Uint8Array)||!(predicted instanceof Uint8Array))throw Error('Expected and predicted must be Uint8Array.');
  if(expected.length!==predicted.length)throw Error('Replay expected/predicted byte lengths differ.');
  const layout=packetLayout(expected.length,parity);
  const {blockLengths,...publicLayout}=layout;
  const base={metric:REPLAY_METRIC,...publicLayout,rawCRCValid:0,rsDecoded:0,crcAccepted:0,
    exact:0,wrongAccepted:0,correctedBytes:0,byteErrors:0};
  for(let i=0;i<expected.length;i++)base.byteErrors+=Number(expected[i]!==predicted[i]);
  if(!layout.eligible)return {...base,payloadBytes:0};
  const payload=expected.slice(0,layout.payloadBytes),received=encodePacket(payload,expected.length,parity);
  for(let i=0;i<received.length;i++)received[i]^=expected[i]^predicted[i];
  const decoded=decodePacket(received,parity),exact=Number(decoded.crcAccepted&&sameBytes(decoded.payload,payload));
  const {payload:unused,...result}=decoded;
  return {...base,...result,exact,wrongAccepted:Number(decoded.crcAccepted&&!exact)};
}
