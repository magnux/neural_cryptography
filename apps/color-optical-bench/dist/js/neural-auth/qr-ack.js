/** Static standard QR ACKs. The OLM1 packet and turn matching remain unchanged. */
import {encodeQr,renderQr,decodeQr} from '../qr-core/js/qr.js';
import {decodePacket} from '../link/protocol.js';

function requireACK(bytes) {
  const packet=decodePacket(bytes);
  if(packet.kind!=='ack')throw Error('Expected an OLM1 ACK packet.');
  return packet;
}

/** 84 binary bytes: version 5-M, 37 modules plus a four-module quiet zone. */
export function renderACK(bytes) {
  requireACK(bytes);
  const qr=encodeQr(bytes,{ecc:'M'});
  return {...renderQr(qr,{scale:16,quiet:4}),qrVersion:qr.version,qrModules:qr.size,ecc:qr.ecc};
}

/** Run in the capture worker; transport acceptance still matches the outstanding packet. */
export function readACK(image) {
  const start=performance.now();
  const result={ok:false,stage:'detect',error:'Looking for a static ACK QR.',corners:[],markers:[],method:'qr-ack',stats:{}};
  try {
    const qr=decodeQr(image);
    if(qr) {
      result.corners=['topLeftCorner','topRightCorner','bottomRightCorner','bottomLeftCorner']
        .map(name=>[qr.location[name].x,qr.location[name].y]);
      result.stats.qrVersion=qr.version;result.stage='payload';
      requireACK(qr.bytes);
      result.ok=true;result.error=null;result.chunk=qr.bytes;
    }
  }catch(error){result.error=error.message;}
  result.stats.ms=performance.now()-start;
  return result;
}
