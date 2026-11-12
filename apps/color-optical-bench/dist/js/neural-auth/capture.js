import {extractNeuralQR} from '../neural-qr/vision.js';

/** Geometry is only a short-lived hint. Every exposure remeasures orientation,
 * grid-size codes, palette references and both phase-cue pairs. No packet is
 * accepted until neural correction, RS/CRC and transfer checks succeed.
 */
export class TransportCapture {
  constructor(){this.reset();}
  reset(){this.context=null;this.geometry=null;this.lastTime=-Infinity;}
  invalidate(){this.geometry=null;}
  scan(image,{expected,context,mediaTime}){
    if(!['data','ack'].includes(expected)||typeof context!=='string'||!context||!Number.isFinite(mediaTime)||mediaTime<0)
      throw Error('Invalid transport capture context.');
    const key=`${context}/${expected}`;
    if(key!==this.context||mediaTime<this.lastTime){this.reset();this.context=key;}
    this.lastTime=mediaTime;
    const capture=extractNeuralQR(image,{previousGeometry:this.geometry,now:mediaTime*1000});
    if(capture.geometry)this.geometry=capture.geometry;
    return capture;
  }
}
