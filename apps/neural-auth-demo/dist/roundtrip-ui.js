/** Camera diagnostic only. A timing trial is one held physical phase, never an authentication packet. */
import {createTimingTrial,compareTimingEcho,TimingPeer} from './roundtrip-core.js';
import {renderNeuralQR} from './js/neural-qr/layout.js';
import {openCaptureCamera,captureDimensions} from './js/dataset/camera.js';

const fmt=n=>Number.isFinite(n)?Math.round(n)+' ms':'—';
const pct=n=>Number.isFinite(n)?(100*n).toFixed(2)+'%':'—';
export function timingSummary(samples){
  const good=samples.filter(row=>row.status==='matched');
  const ms=good.map(row=>row.rttMs),cells=good.reduce((s,x)=>s+x.cells,0);
  return {attempts:samples.length,matched:good.length,timeouts:samples.filter(row=>row.status==='timeout').length,
    minMs:ms.length?Math.min(...ms):null,meanMs:ms.length?ms.reduce((a,b)=>a+b,0)/ms.length:null,
    maxMs:ms.length?Math.max(...ms):null,cer:cells?good.reduce((s,x)=>s+x.errors,0)/cells:null};
}
function record(dict,name,value){
  if(!Number.isFinite(value)||value<0)return;const s=dict[name]??={n:0,minMs:value,meanMs:0,maxMs:value};
  s.n++;s.meanMs+=(value-s.meanMs)/s.n;s.minMs=Math.min(s.minMs,value);s.maxMs=Math.max(s.maxMs,value);
}
function workerClient(onProgress,onFatal){
  const w=new Worker(new URL('./roundtrip-worker.js',import.meta.url),{type:'module'});
  let pending,serial=0;
  function reject(reason){if(!pending)return;clearTimeout(pending.timer);pending.reject(reason);pending=null;}
  w.onmessage=({data:m})=>{
    if(!pending||m.id!==pending.id)return;
    if(m.progress){onProgress(m.progress);return;}
    const p=pending;pending=null;clearTimeout(p.timer);if(m.error)p.reject(Error(m.error));else p.resolve(m.value);
  };
  w.onerror=e=>{const error=Error(e.message||'Timing worker failed.');reject(error);onFatal(error);};
  w.onmessageerror=()=>{const error=Error('Unreadable timing worker reply.');reject(error);onFatal(error);};
  return {call(op,fields={},transfer=[]){
    if(pending)throw Error('A timing scan is already running.');
    return new Promise((resolve,rejectCall)=>{
      const id=++serial;pending={id,resolve,reject:rejectCall,timer:setTimeout(()=>reject(Error('Timing worker timed out.')),op==='init'?120000:45000)};
      try{w.postMessage({id,op,...fields},transfer);}catch(e){reject(e);}
    });
  },close(){w.terminate();reject(Error('Timing run stopped.'));}};
}

export function initRoundtrip(host,{onActive=()=>{}}={}){
  host.innerHTML=`
    <div class="rt-head"><div><span class="eyebrow rt-label">OPTICAL DIAGNOSTIC</span><h2>Matrix roundtrip timing</h2></div><div class="rt-view viewfinder-rim" data-lock="search"><video muted playsinline autoplay aria-label="Camera alignment"></video><canvas aria-hidden="true"></canvas></div></div>
    <p class="rt-msg" role="status" aria-live="polite">Start the receiver first. Face the two cameras toward the opposite screens.</p>
    <div class="rt-toolbar"><label>Role<select name="role"><option value="sender">Sender</option><option value="receiver">Receiver</option></select></label>
      <button type="button" class="rt-go">Start sender</button><details class="rt-details"><summary>Settings</summary><div class="rt-drawer">
      <fieldset>
      <label>Grid<select name="size"><option>16</option><option>32</option><option selected>64</option><option>128</option></select></label>
      <label>Camera<select name="facing"><option value="user">Front</option><option value="environment">Rear</option></select></label>
      <label class="wide">Camera device<select name="device"><option value="">Use front/rear preference</option></select></label>
      <label>Resolution<select name="resolution"><option>720</option><option selected>1080</option><option>1440</option><option>2160</option><option value="max">Maximum</option></select></label>
      <label>Capture interval (ms)<input name="sample" type="number" value="100" min="30" max="2000" step="1"></label>
      <label>Sender trials<input name="count" type="number" value="12" min="1" max="128" step="1"></label>
      <label>Echo timeout (ms)<input name="deadline" type="number" value="6000" min="500" max="60000" step="1"></label>
      <label class="wide">Probe exposure (ms; 0 = hold until echo)<input name="exposure" type="number" value="0" min="0" max="30000" step="1"></label>
      <label class="toggle wide"><span>Time a dummy inference before echo<small>Receiver only · untrained VW/WASM · G32/G64</small></span><input name="inference" type="checkbox"></label>
      </fieldset>
      <p>One phase per trial. RTT starts at the sender’s canvas draw and ends when it decodes the matching echo. CER compares payload cells over both legs. The correlation tag must match; payload errors are counted without error correction.</p>
      <p>Restart the receiver before starting another sender run. To choose a phase interval, try shorter probe exposures and compare matched trials and CER. RTT includes the return path and is not a one-way phase duration.</p>
      <button class="rt-save" type="button" disabled>Save measurements</button>
      </div></details></div>
    <dl class="rt-stats"><div><dt>Matched / tried</dt><dd data-stat="count">—</dd></div><div><dt>RTT min / mean / max</dt><dd data-stat="rtt">—</dd></div><div><dt>Cell error</dt><dd data-stat="cer">—</dd></div><div><dt>Timeouts</dt><dd data-stat="timeouts">—</dd></div></dl>
    <div class="rt-slot"><canvas class="rt-matrix" width="1" height="1" hidden aria-label="Single-phase timing probe or echo"></canvas><span class="rt-empty">Single-phase probe appears here</span></div>`;
  const one=s=>host.querySelector(s),el=name=>one('[name="'+name+'"]'),video=one('video'),view=one('.rt-view'),overlay=one('.rt-view canvas'),
    output=one('.rt-matrix'),placeholder=one('.rt-empty'),msg=one('.rt-msg'),go=one('.rt-go'),details=one('details'),save=one('.rt-save');
  const stat=(key,text)=>{one('[data-stat="'+key+'"]').textContent=text;};
  const cap=document.createElement('canvas');let active=false,epoch=0,worker,stream,callback,watchdog,deadlineTimer,exposureTimer,frameDraw;
  let report=null,pending=null,peer,scanBusy=false,sequence=0,runId=0,lastFrame=-1,lastSample=-Infinity,lastFresh=0,wrongGrid=0;
  function clearMatrix(){output.hidden=true;placeholder.hidden=false;}
  function selected(){const sender=el('role').value==='sender';
    for(const key of ['count','deadline','exposure'])el(key).disabled=!sender;
    el('inference').disabled=sender||!['32','64'].includes(el('size').value);
    if(!active)go.textContent=sender?'Start sender':'Start receiver';}
  function settings(){
    const integer=(name,min,max)=>{const node=el(name),v=node.valueAsNumber;if(!Number.isSafeInteger(v)||v<min||v>max)throw Error('Check '+node.parentElement.textContent.trim()+': enter '+min+'–'+max+'.');return v;};
    const role=el('role').value,sender=role==='sender',size=Number(el('size').value);
    const captureIntervalMs=integer('sample',30,2000),trials=sender?integer('count',1,128):null,
      timeoutMs=sender?integer('deadline',500,60000):null,exposureMs=sender?integer('exposure',0,30000):null;
    if(sender&&exposureMs>=timeoutMs)throw Error('Set the probe exposure below the echo timeout or use 0 to hold.');
    return {role,size,captureIntervalMs,trials,timeoutMs,exposureMs,inference:!sender&&!el('inference').disabled&&el('inference').checked,
      resolution:el('resolution').value,deviceId:el('device').value,facingMode:el('facing').value};
  }
  function stats(){
    if(!report)return;const s=timingSummary(report.trials);report.summary=s;
    if(report.configuration.role==='receiver'){stat('count',report.echoes+' echoes');stat('rtt','On sender');stat('cer','On sender');stat('timeouts','—');}
    else{stat('count',s.matched+' / '+report.probesDrawn);stat('rtt',[s.minMs,s.meanMs,s.maxMs].map(fmt).join(' / '));stat('cer',pct(s.cer));stat('timeouts',s.timeouts);}
    save.disabled=false;
  }
  function stop(reason='Stopped. Measurements are in Settings.'){
    if(!active)return;active=false;epoch++;
    clearInterval(watchdog);clearTimeout(deadlineTimer);clearTimeout(exposureTimer);cancelAnimationFrame(frameDraw);
    if(callback!=null)video.cancelVideoFrameCallback?.(callback);callback=null;
    worker?.close();worker=null;stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;
    if(pending?.at!=null)report.trials.push({sequence,phase:pending.frame.phase,status:'cancelled'});pending=null;
    overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);view.dataset.lock='search';wrongGrid=0;one('.rt-label').textContent='OPTICAL DIAGNOSTIC';
    clearMatrix();go.disabled=false;el('role').disabled=false;one('fieldset').disabled=false;selected();
    scanBusy=false;stats();onActive(false);msg.textContent=reason;
  }
  const current=t=>active&&epoch===t;
  function draw(frame,t,painted){
    const image=renderNeuralQR(frame,frame.size===32?512:1024);
    const bitmap=new ImageData(image.data,image.width,image.height);
    if(!current(t))return;
    // RAF schedules the canvas submission close to display refresh. It cannot
    // timestamp physical panel illumination or a remote device's exposure.
    frameDraw=requestAnimationFrame(()=>{
      if(!current(t))return;
      if(output.width!==image.width)output.width=output.height=image.width;
      output.hidden=false;placeholder.hidden=true;output.getContext('2d').putImageData(bitmap,0,0);
      painted(performance.now());
    });
  }
  function advance(t){
    if(!current(t))return;
    if(sequence>=report.configuration.trials){stop('Timing complete. Open Settings to save the report.');return;}
    const frame=createTimingTrial({size:report.configuration.size,run:runId,sequence:++sequence});
    const id=sequence;pending={frame,at:null,hiddenAt:null};
    draw(frame,t,at=>{
      pending.at=at;report.probesDrawn++;msg.textContent='Showing probe '+id+' / '+report.configuration.trials+' · phase '+(frame.phase+1)+'/4';stats();
      if(report.configuration.exposureMs)exposureTimer=setTimeout(()=>{
        if(current(t)&&sequence===id&&pending){clearMatrix();pending.hiddenAt=performance.now();placeholder.textContent='Waiting for echo';}
      },report.configuration.exposureMs);
      deadlineTimer=setTimeout(()=>{
        if(!current(t)||sequence!==id||!pending)return;
        report.trials.push({sequence:id,phase:frame.phase,status:'timeout',elapsedMs:performance.now()-at,
          visibleMs:(pending.hiddenAt??performance.now())-at});pending=null;clearTimeout(exposureTimer);stats();advance(t);
      },report.configuration.timeoutMs);
    });
  }
  function feedback(observation,width,height){
    const corners=observation.corners?.length===4?observation.corners:observation.tentativeCorners;
    const seenSize=observation.info?.size,expected=report.configuration.size;
    const mismatch=seenSize&&seenSize!==expected;
    wrongGrid=mismatch?wrongGrid+1:0;
    if(wrongGrid>=3)one('.rt-label').textContent='SEEING G'+seenSize+' · SELECT G'+seenSize+' ON BOTH DEVICES';
    else if(!mismatch)one('.rt-label').textContent='OPTICAL DIAGNOSTIC';
    view.dataset.lock=observation.ok&&!mismatch?'read':observation.info&&corners?.length===4?'acquired':corners?.length===4?'tentative':'search';
    if(overlay.width!==width||overlay.height!==height){overlay.width=width;overlay.height=height;}
    const g=overlay.getContext('2d');g.clearRect(0,0,width,height);
    if(corners?.length===4&&corners.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite))){
      g.beginPath();for(let i=0;i<4;i++)g[i?'lineTo':'moveTo'](...corners[i]);g.closePath();g.fillStyle={read:'#1fc88248',acquired:'#4089df48',tentative:'#ee9c3648'}[view.dataset.lock];g.fill();
    }
  }
  async function scan(t){
    const c=report.configuration,d=captureDimensions(video.videoWidth,video.videoHeight),begin=performance.now();
    if(cap.width!==d.width||cap.height!==d.height){cap.width=d.width;cap.height=d.height;}
    const g=cap.getContext('2d',{willReadFrequently:true});g.drawImage(video,0,0,d.width,d.height);
    const image=g.getImageData(0,0,d.width,d.height);
    record(report.stages,'readbackMs',performance.now()-begin);const before=performance.now();report.scans++;
    const found=await worker.call('scan',{image,captureIntervalMs:c.captureIntervalMs},[image.data.buffer]);
    const decodedAt=performance.now();if(!current(t))return;
    record(report.stages,'workerScanMs',decodedAt-before);
    for(const name of ['detectionMs','decodeMs'])record(report.stages,name,found.timings?.[name]);
    feedback(found,d.width,d.height);if(!found.ok||found.frame.size!==c.size)return;
    if(c.role==='receiver'){
      const echo=peer.accept(found);if(!echo)return;
      if(c.inference){const timing=await worker.call('infer');if(!current(t))return;record(report.stages,'dummyInferenceMs',timing.inferenceMs);}
      draw(echo,t,at=>{report.echoes++;record(report.stages,'captureToEchoDrawMs',at-begin);
        msg.textContent='Echoing probe '+found.sequence+' · phase '+(echo.phase+1)+'/4. Keep steady.';stats();});
    }else if(pending?.at!=null&&decodedAt-pending.at<c.timeoutMs){
      const cmp=compareTimingEcho(pending.frame,found);if(!cmp)return;
      const {frame,at,hiddenAt}=pending;pending=null;clearTimeout(deadlineTimer);clearTimeout(exposureTimer);
      report.trials.push({sequence,phase:frame.phase,status:'matched',...cmp,rttMs:decodedAt-at,visibleMs:(hiddenAt??decodedAt)-at});
      stats();advance(t);
    }
  }
  function schedule(t){
    if(!current(t)||scanBusy)return;
    callback=video.requestVideoFrameCallback((now,m)=>{
      callback=null;if(!current(t))return;
      if(!Number.isSafeInteger(m.presentedFrames)){stop('Camera frame timestamps are unavailable.');return;}
      if(m.presentedFrames<=lastFrame||now-lastSample<report.configuration.captureIntervalMs||video.readyState<2||!video.videoWidth){schedule(t);return;}
      lastFresh=performance.now();lastFrame=m.presentedFrames;lastSample=now;scanBusy=true;
      void scan(t).catch(error=>{if(current(t))stop(error.message);}).finally(()=>{if(current(t)){scanBusy=false;schedule(t);}});
    });
  }
  async function start(){
    let c;try{c=settings();}catch(e){msg.textContent=e.message;details.open=true;return;}
    if(!navigator.mediaDevices?.getUserMedia||!isSecureContext||!video.requestVideoFrameCallback){msg.textContent='Use a browser with camera frame callbacks on trusted HTTPS or localhost.';return;}
    const t=++epoch;active=true;onActive(true);sequence=0;pending=null;scanBusy=false;lastSample=-Infinity;lastFrame=-1;
    details.open=false;go.textContent='Stop';el('role').disabled=true;one('fieldset').disabled=true;
    const recordedConfig={...c};delete recordedConfig.deviceId; // Reports do not need a persistent browser camera identifier.
    report={format:'neural-qr-single-phase-roundtrip',version:1,started:new Date().toISOString(),configuration:recordedConfig,
      scans:0,echoes:0,probesDrawn:0,stages:{},trials:[],summary:null};stats();
    try{
      peer=new TimingPeer();runId=crypto.getRandomValues(new Uint32Array(1))[0]||1;
      worker=workerClient(text=>{if(current(t))msg.textContent=text;},e=>{if(current(t))stop(e.message);});
      msg.textContent='Preparing optical reader…';await worker.call('init',{size:c.size,inference:c.inference});if(!current(t))return;
      msg.textContent='Opening camera…';const opened=await openCaptureCamera(navigator.mediaDevices,{resolution:c.resolution,deviceId:c.deviceId,facingMode:c.facingMode});
      if(!current(t)){opened.stream.getTracks().forEach(track=>track.stop());return;}stream=opened.stream;
      for(const track of stream.getVideoTracks())track.addEventListener('ended',()=>{if(current(t))stop('Camera disconnected. Start again.');},{once:true});
      void listCameras();video.srcObject=stream;await video.play();if(!current(t))return;
      const actual=stream.getVideoTracks()[0]?.getSettings()??{};report.camera={width:actual.width,height:actual.height,frameRate:actual.frameRate,facingMode:actual.facingMode};
      placeholder.textContent='Waiting for a timing probe';msg.textContent='Listening for a G'+c.size+' probe. Start the sender now.';
      if(c.role==='sender')advance(t);lastFresh=performance.now();schedule(t);
      watchdog=setInterval(()=>{if(current(t)&&!scanBusy&&performance.now()-lastFresh>15000)stop('Camera stopped delivering frames.');},1000);
    }catch(e){if(current(t))stop(e.message);}
  }
  async function listCameras(){
    if(!navigator.mediaDevices?.enumerateDevices)return;
    try{
      const inputs=(await navigator.mediaDevices.enumerateDevices()).filter(x=>x.kind==='videoinput'&&x.deviceId);
      const device=el('device'),old=device.value;device.replaceChildren(new Option('Use front/rear preference',''),
        ...inputs.map((x,i)=>new Option(x.label||'Camera '+(i+1),x.deviceId)));
      if(inputs.some(x=>x.deviceId===old))device.value=old;
    }catch{/* Discovery is optional. Explicit selection is never silently changed on open. */}
  }
  details.addEventListener('toggle',()=>{if(details.open&&!active)void listCameras();});
  selected();el('role').addEventListener('change',selected);el('size').addEventListener('change',selected);
  go.addEventListener('click',()=>{if(active)stop();else void start();});
  save.addEventListener('click',()=>{if(!report)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='matrix-timing-'+report.configuration.role+'-'+report.started.replace(/[:.]/g,'-')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&active)stop('Stopped because this page was hidden.');});
  window.addEventListener('pagehide',()=>{if(active)stop();});
  return {stop,get active(){return active;}};
}
