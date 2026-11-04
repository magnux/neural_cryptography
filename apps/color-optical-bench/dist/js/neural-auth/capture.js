import {extractCapture} from '../vision.js';
import {readACK} from './qr-ack.js';

/** One camera's short-lived geometry hint. Every exposure must pass its own
 * header CRC. A periodic full search and payload-failure invalidation limit drift.
 */
export class TransportCapture {
  constructor(){this.reset();}
  reset(){this.context=null;this.geometry=null;this.detectedAt=-Infinity;this.lastTime=-Infinity;}
  invalidate(){this.geometry=null;this.detectedAt=-Infinity;}
  scan(image,{expected,context,mediaTime}){
    if(!['data','ack'].includes(expected)||typeof context!=='string'||!context||!Number.isFinite(mediaTime))throw Error('Invalid transport capture context.');
    const key=`${context}/${expected}`;
    if(key!==this.context||mediaTime<this.lastTime){this.reset();this.context=key;}
    this.lastTime=mediaTime;
    if(expected==='ack')return readACK(image);
    if(this.geometry&&(this.geometry.width!==image.width||this.geometry.height!==image.height||mediaTime-this.detectedAt>=.5))this.invalidate();
    const capture=extractCapture(image,{preview:false,spatial:'cross',previousGeometry:this.geometry});
    if(capture.geometry){
      this.geometry=capture.geometry;
      if(capture.stats.geometrySource==='detected')this.detectedAt=mediaTime;
    }
    return capture;
  }
}
