import {CorrectorClient} from './client.js';
import {validateOptions, summarizeTimings, validFrameCount, MAX_INPUT_FRAMES} from './runtime.js';

const $ = id => document.getElementById(id);
const records = [];
const CAMERA_INTERVAL_MS = 100;
const client = new CorrectorClient({onProgress: text => { $('labStatus').textContent = text; }});
const supported = isSecureContext && typeof Worker === 'function' && Boolean(crypto?.subtle);
let stream = null, cameraStarting = false, cameraEpoch = 0, busy = false, operation = 0, lastResult = null, activeKind=null, cancelFrameWait=null;

function error(message) { $('labError').textContent = message; $('labError').hidden = !message; }
function settings() { return validateOptions({size: Number($('modelSize').value), backend: $('backend').value, iterations: Number($('iterations').value), captureFrames:Number($('captureFrames').value)}); }
function cameraInfo() {
  const value = stream?.getVideoTracks()[0]?.getSettings();
  return value ? {active: true, width: value.width, height: value.height, frameRate: value.frameRate,
    facingMode: value.facingMode, requestedHeight: Number($('resolution').value)} : {active: false};
}
function renderControls() {
  $('runBenchmark').disabled = !supported || busy;
  $('runCapture').disabled = !supported || busy || !stream;
  $('cancel').disabled = !busy;
  for (const id of ['modelSize', 'backend', 'iterations', 'captureFrames']) $(id).disabled = busy;
  $('startCamera').disabled = !supported || cameraStarting || Boolean(stream);
  $('stopCamera').disabled = !stream && !cameraStarting;
  $('camera').disabled = Boolean(stream) || cameraStarting;
  $('resolution').disabled = Boolean(stream) || cameraStarting;
  $('downloadReport').disabled = records.length === 0;
  $('reportCount').textContent = `${records.length} test${records.length === 1 ? '' : 's'} recorded (latest 50 retained).`;
}
function record(value) { records.push(value); if (records.length > 50) records.shift(); renderControls(); }
function monitorUI() {
  const started = performance.now(), gaps = [], heapStart = performance.memory?.usedJSHeapSize; let previous = started, active = true, handle;
  const tick = now => { if (!active) return; if (gaps.length < 10000) gaps.push(now-previous); previous=now; handle=requestAnimationFrame(tick); };
  handle=requestAnimationFrame(tick);
  return () => { active=false; cancelAnimationFrame(handle); return {durationMs: performance.now()-started,
    frameGaps: gaps.length ? summarizeTimings(gaps) : null, gapsOver50Ms: gaps.filter(x => x > 50).length,
    hiddenAtEnd: document.hidden, jsHeapBytes: {start: heapStart, end: performance.memory?.usedJSHeapSize},
    note: 'Animation-frame spacing, not touch latency. Optional browser JS heap readings exclude GPU/worker/native runtime memory.'}; };
}
function showResult(result) {
  $('modelNoticeTitle').textContent = result.modelStatus === 'trained' ? 'Trained checkpoint · numerical checks' : 'Untrained model · numerical checks';
  $('modelNoticeText').textContent = result.modelStatus === 'trained'
    ? 'The prepared checkpoint is loaded. Numerical agreement tests runtime correctness; evaluate recovery on held-out captures before using this model for transport.'
    : 'These random weights test runtime feasibility. Collect labeled captures, then train and evaluate a checkpoint before measuring recovery.';
  const box=$('results'); box.replaceChildren();
  const title=document.createElement('div'); title.className='corrector-parity';
  title.dataset.pass=String(result.operation === 'benchmark' ? result.parity.passed : result.logitsFinite);
  title.textContent=result.operation === 'benchmark' ? (result.parity.passed ? '✓ Python parity passed' : '× Python parity failed') : '✓ Camera inference completed';
  box.append(title);
  const metrics=document.createElement('dl'); metrics.className='corrector-metrics';
  const fields=result.operation === 'benchmark'
    ? [['Warm p50',`${result.timings.p50Ms.toFixed(1)} ms`],['Warm p95',`${result.timings.p95Ms.toFixed(1)} ms`],['Backend',result.effectiveBackend.toUpperCase()]]
    : [['Preprocess total',`${result.inputStats.preprocessMs.toFixed(1)} ms`],['Inference',`${result.inferenceMs.toFixed(1)} ms`],['Backend',result.effectiveBackend.toUpperCase()]];
  for (const [label,value] of fields) { const item=document.createElement('div'), dt=document.createElement('dt'), dd=document.createElement('dd'); dt.textContent=label; dd.textContent=value; item.append(dt,dd); metrics.append(item); }
  box.append(metrics);
  const detail=document.createElement('p'); detail.className='muted';
  detail.textContent=result.operation === 'benchmark'
    ? `${result.parity.checked.toLocaleString()} logits checked · ${result.parity.failures} outside tolerance · maximum absolute error ${result.parity.maxAbsoluteError?.toExponential(2) ?? 'non-finite'}. Two warm-ups; ${result.timings.count} measured runs. Load ${result.loadMs.toFixed(0)} ms${result.cacheReused ? ' (cached session)' : `, including ${result.initMs.toFixed(0)} ms initialization`}.`
    : `A ${result.size}×${result.size}, ${result.paletteColors}-color matrix supplied ${result.inputCaptures} distinct exposure${result.inputCaptures===1?'':'s'} in ${result.framesPerInput} model slots. ${result.note}`;
  box.append(detail);
  if (result.fallbackReason) { const fallback=document.createElement('p'); fallback.className='muted'; fallback.textContent=`Auto fallback: ${result.fallbackReason}`; box.append(fallback); }
  const responsive=document.createElement('p'); responsive.className='muted';
  responsive.textContent=`UI frame gaps: p95 ${result.ui?.frameGaps?.p95Ms.toFixed(1) ?? '—'} ms; ${result.ui?.gapsOver50Ms ?? 0} over 50 ms. Camera ${result.camera.active ? `${result.camera.width}×${result.camera.height}` : 'off'}.`;
  box.append(responsive);
}
function waitForCameraFrame(previous,token,epoch,notBefore=-Infinity) {
  const video=$('video');
  if (typeof video.requestVideoFrameCallback!=='function'||typeof video.cancelVideoFrameCallback!=='function') {
    return Promise.reject(Error('Camera burst tests require requestVideoFrameCallback. Use a current browser.'));
  }
  return new Promise((resolve,reject)=>{
    let callbackId=null,settled=false;
    const finish=(error,result)=>{
      if(settled)return;settled=true;clearTimeout(timeout);
      if(callbackId!==null)video.cancelVideoFrameCallback(callbackId);
      if(cancelFrameWait===abort)cancelFrameWait=null;
      if(error)reject(error);else resolve(result);
    };
    const abort=()=>finish(new DOMException('Camera burst canceled.','AbortError'));
    const timeout=setTimeout(()=>finish(Error('No new camera exposure arrived. Restart the camera and try again.')),3000);
    cancelFrameWait=abort;
    const request=()=>{
      if(token!==operation||epoch!==cameraEpoch||!stream){abort();return;}
      callbackId=video.requestVideoFrameCallback((now,metadata)=>{
        callbackId=null;
        if(token!==operation||epoch!==cameraEpoch||!stream){abort();return;}
        if(!Number.isFinite(metadata?.mediaTime)||metadata.mediaTime<0||
            !Number.isSafeInteger(metadata.presentedFrames)||metadata.presentedFrames<1){
          finish(Error('The browser did not identify the presented camera frame.'));return;
        }
        if(metadata.presentedFrames<=previous||now<notBefore){request();return;}
        if(!video.videoWidth||video.readyState<2){finish(Error('Wait for a live camera image first.'));return;}
        finish(null,{video,mediaTime:metadata.mediaTime,presentedFrames:metadata.presentedFrames,captureStartedMs:now});
      });
    };
    request();
  });
}
async function run(kind) {
  if (busy) return;
  error(''); let options;
  try { options=settings(); } catch (e) { error(e.message); return; }
  busy=true; activeKind=kind; const token=++operation, epoch=cameraEpoch; renderControls();
  $('labStatus').textContent='Starting the local corrector worker…';
  const stopMonitor=monitorUI(), camera=cameraInfo(), start=new Date().toISOString();
  try {
    let result;
    if (kind === 'capture') {
      let previous=0,notBefore=-Infinity,maxAttempts=4*MAX_INPUT_FRAMES;
      camera.requestedSampleIntervalMs=CAMERA_INTERVAL_MS;
      // Bound attempts if the sender changes phase while gathering the stack.
      for (let attempt=0;attempt<maxAttempts;attempt++) {
        const exposure=await waitForCameraFrame(previous,token,epoch,notBefore),video=exposure.video;
        previous=exposure.presentedFrames;
        notBefore=exposure.captureStartedMs+CAMERA_INTERVAL_MS;
        const scale=Math.min(1,Math.sqrt(2073600/(video.videoWidth*video.videoHeight)));
        const canvas=document.createElement('canvas'); canvas.width=Math.max(1,Math.floor(video.videoWidth*scale)); canvas.height=Math.max(1,Math.floor(video.videoHeight*scale));
        const context=canvas.getContext('2d',{willReadFrequently:true}); context.drawImage(video,0,0,canvas.width,canvas.height);
        const image=context.getImageData(0,0,canvas.width,canvas.height);
        camera.captureWidth=image.width; camera.captureHeight=image.height;
        result=await client.call(kind,{...options,captureIntervalMs:CAMERA_INTERVAL_MS,runId:token,mediaTime:exposure.mediaTime,presentedFrames:exposure.presentedFrames,
          image:{width:image.width,height:image.height,data:image.data}},[image.data.buffer]);
        if (!validFrameCount(result.framesPerInput)||!validFrameCount(result.requiredCaptures)||result.requiredCaptures>result.framesPerInput) throw Error('Worker returned an invalid camera frame count.');
        maxAttempts=result.phaseGrouped?8*MAX_INPUT_FRAMES:4*result.requiredCaptures;
        if (!result.waiting) break;
        $('labStatus').textContent=result.phaseGrouped?`Collected phases ${(result.phaseCounts??[]).join('/')}. Cycle all four palette phases of the same matrix.`:`Collected ${result.collected}/${result.requiredCaptures} exposures. Keep the same matrix and palette phase visible.`;
      }
      if (result?.waiting) { client.cancel(); throw Error(result.phaseGrouped?'Could not collect all four phases of the same matrix. Cycle the phases and retry.':'The matrix or phase kept changing before a complete burst. Hold one phase steady and retry.'); }
    } else result=await client.call(kind,options);
    if (token !== operation) return;
    lastResult={...result, ui:stopMonitor(), camera, start};
    record({ok:true,...lastResult}); showResult(lastResult);
    $('labStatus').textContent=result.operation === 'benchmark'
      ? (result.parity.passed ? 'Completed. Save this report, then test another supported grid/backend.' : 'Completed with a parity mismatch. Save the report; do not train against this export yet.')
      : 'Completed. Camera test confirms the input/inference path only.';
  } catch (e) {
    const ui=stopMonitor();
    record({ok:false, operation:kind, start, size:options.size, requestedBackend:options.backend, iterations:options.iterations,captureFrames:options.captureFrames,
      error:e.message, canceled:e.name === 'AbortError', ui, camera});
    if (token === operation) { error(e.message); $('labStatus').textContent=e.name === 'AbortError' ? 'Canceled. The worker model was released.' : 'Test failed. The failure is included in the report.'; }
  } finally {
    stopMonitor();
    if (token === operation) { busy=false; activeKind=null; renderControls(); }
  }
}
async function enumerateCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const selected=$('camera').value;
  try {
    const cameras=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
    $('camera').replaceChildren(new Option('Default camera',''));
    cameras.forEach((d,i)=>$('camera').add(new Option(d.label || `Camera ${i+1}`,d.deviceId)));
    if (cameras.some(d=>d.deviceId===selected)) $('camera').value=selected;
  } catch { /* Device names are optional until camera permission is granted. */ }
}
function stopCamera() {
  if (busy&&activeKind==='capture') cancel();
  cameraEpoch++;
  if (stream) for (const track of stream.getTracks()) track.stop();
  stream=null; cameraStarting=false; $('video').srcObject=null;
  $('cameraPlaceholder').hidden=false; $('cameraStatus').textContent='Camera is off. No frames were saved or uploaded.';
  renderControls();
}
async function startCamera() {
  if (cameraStarting || stream) return;
  error(''); cameraStarting=true; const epoch=++cameraEpoch; renderControls();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires a supported browser on trusted HTTPS or localhost.');
    const height=Number($('resolution').value), width=height===480?640:Math.round(height*16/9);
    const acquired=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:width},height:{ideal:height},
      ...($('camera').value ? {deviceId:{exact:$('camera').value}} : {facingMode:{ideal:'environment'}})}});
    if (epoch !== cameraEpoch) { acquired.getTracks().forEach(t=>t.stop()); return; }
    stream=acquired; const video=$('video'); video.srcObject=stream; await video.play();
    if (epoch !== cameraEpoch) return;
    $('cameraPlaceholder').hidden=true;
    const actual=cameraInfo(); $('cameraStatus').textContent=`Actual capture: ${actual.width} × ${actual.height}, ${actual.frameRate?.toFixed(0) ?? 'unknown'} fps. Keep the full matrix and border visible.`;
    await enumerateCameras();
  } catch (e) { if (epoch === cameraEpoch) { stopCamera(); error(e.message); } }
  finally { if (epoch === cameraEpoch) { cameraStarting=false; renderControls(); } }
}
function cancel() { operation++; cancelFrameWait?.(); client.cancel(); busy=false; activeKind=null; renderControls(); $('labStatus').textContent='Canceled. The worker and pending camera burst were released.'; }
function shutdown() { cancel(); stopCamera(); }
$('runBenchmark').addEventListener('click',()=>run('benchmark'));
$('runCapture').addEventListener('click',()=>run('capture'));
$('cancel').addEventListener('click',cancel);
$('startCamera').addEventListener('click',startCamera);
$('stopCamera').addEventListener('click',stopCamera);
$('downloadReport').addEventListener('click',()=>{
  const report={format:'neural-corrector-device-report',version:1,created:new Date().toISOString(),
    environment:{userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency,deviceMemoryGiB:navigator.deviceMemory,
      secureContext:isSecureContext,webgpuExposed:Boolean(navigator.gpu)},notes:$('notes').value,records};
  const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url; link.download=`corrector-${new Date().toISOString().replace(/[:.]/g,'-')}.json`; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
window.addEventListener('pagehide',shutdown);
document.addEventListener('visibilitychange',()=>{ if (document.hidden) shutdown(); });
if (!supported) error('Use a current browser on trusted HTTPS or localhost, with Web Workers and Web Crypto available.');
renderControls(); enumerateCameras();
