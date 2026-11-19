/** Shared by the prepared live worker and the timing lab's model-free worker.
 * A receipt is read from one real exposure. The only optical state retained
 * between reads is a short-lived geometry hint, never cells or a fingerprint.
 */
import {MAX_CAPTURE_PIXELS} from '../dataset/camera.js';
import {StageProfiler} from '../neural-auth/timing.js';
import {decodeReceipt16Capture} from './receipt16.js';

export class ReceiptCaptureReader {
  constructor(){this.reset();}
  reset(){this.runId=0;this.geometry=null;this.mediaTime=-Infinity;this.presentedFrames=0;}
  scan({image,runId,captureIntervalMs,mediaTime,presentedFrames,statistics=false}){
    if(!Number.isSafeInteger(runId)||runId<1||runId<this.runId||
       !Number.isFinite(captureIntervalMs)||captureIntervalMs<50||captureIntervalMs>2000||
       !Number.isFinite(mediaTime)||mediaTime<0||!Number.isSafeInteger(presentedFrames)||presentedFrames<1)
      throw Error('Invalid G16 receipt run, exposure or capture interval.');
    const {width,height,data}=image??{};
    if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<32||height<32||width*height>MAX_CAPTURE_PIXELS||
       !(data instanceof Uint8ClampedArray||data instanceof Uint8Array)||data.length!==4*width*height)
      throw Error('Expected one bounded RGBA exposure.');
    if(this.runId!==runId){this.reset();this.runId=runId;}
    if(mediaTime<=this.mediaTime||presentedFrames<=this.presentedFrames)
      return {ok:false,stage:'duplicate-exposure',skipped:true,duplicateExposure:true,method:'receipt16',digest:null};
    this.mediaTime=mediaTime;this.presentedFrames=presentedFrames;
    const profiler=statistics===true?new StageProfiler(true):null;
    const r=decodeReceipt16Capture(image,{previousGeometry:this.geometry,now:performance.now(),captureIntervalMs,profiler,collectTimings:!!profiler});
    this.geometry=r.geometry??null;
    // Keep rasters, transforms, sample buffers and expected-packet data out of the RPC.
    const {ok,stage,error,info,corners,tentativeCorners,digest}=r;
    return {ok,stage,error,info,corners,tentativeCorners,digest,method:'receipt16',
      inputCaptures:info?.size===16?1:null,...(profiler?{stats:r.stats,timings:profiler.snapshot()}:{})};
  }
}
