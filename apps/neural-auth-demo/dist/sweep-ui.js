// Two physical devices, one probe and one ACK per operator-controlled step.
// Known probe bytes never enter camera acquisition, tensor packing or inference.
import {openCaptureCamera,captureDimensions} from './js/dataset/camera.js';
import {renderNeuralQR} from './js/neural-qr/layout.js';
import {SweepReader} from './sweep-reader.js';
import {SweepRun,CapturePacer} from './sweep-run.js';
import {newSweepRunId,createSweepProbe,acknowledgeSweepProbe,matchesSweepAck} from './sweep-protocol.js';

const fmt=v=>Number.isFinite(v)?Math.round(v)+' ms':'—';
const now=()=>performance.now();

function markup(){return `
  <div class="sw-head"><div><span class="eyebrow">TRAINED CORRECTOR · FOUR PHASES</span><h2>Phase timing sweep</h2></div>
    <div class="rt-view viewfinder-rim" data-lock="search"><video muted playsinline autoplay aria-label="Camera alignment"></video><canvas aria-hidden="true"></canvas></div></div>
  <p class="sw-status" role="status" aria-live="polite">Choose a sender and a receiver. Use the same grid and timing settings on both.</p>
  <div class="sw-toolbar"><label>Role<select name="role"><option value="sender">Sender</option><option value="receiver">Receiver</option></select></label>
    <button type="button" class="sw-start">Start camera</button><button type="button" class="sw-next" hidden>Start step 1</button></div>
  <details class="sw-settings"><summary>Settings &amp; measurements</summary><div class="sw-body"><fieldset>
    <label>Data grid<select name="size"><option>32</option><option selected>64</option><option>128</option></select></label>
    <label>Corrector backend<select name="backend"><option value="auto">Auto</option><option value="wasm">WASM</option><option value="webgpu">WebGPU</option></select></label>
    <label>Initial phase (ms)<input name="phase" type="number" min="50" max="2000" value="500" step="1"></label>
    <label>Initial capture (ms)<input name="capture" type="number" min="50" max="2000" value="450" step="1"></label>
    <label class="toggle wide"><span>Vary capture spacing (±20%)<small>Tests repeated alignment with display transitions on this camera</small></span><input name="varyCapture" type="checkbox"></label>
    <label>Decrease per step (ms)<input name="decrement" type="number" min="1" max="500" value="50" step="1"></label>
    <label>Step timeout (s)<input name="timeout" type="number" min="3" max="120" value="30" step="1"></label>
    <label>Camera<select name="facing"><option value="user">Front</option><option value="environment">Rear</option></select></label>
    <label>Resolution<select name="resolution"><option>720</option><option selected>1080</option><option>1440</option><option>2160</option><option value="max">Maximum</option></select></label>
    <label class="wide">Camera device<select name="device"><option value="">Use front/rear preference</option></select></label></fieldset>
    <p>DATA and its G32 ACK each use four phases. Wait for “ACK read” on the sender, then tap Next on the receiver first: this clears the old ACK while the next DATA is being read. Save a report on each device.</p>
    <div class="sw-history"></div><button class="sw-save rt-save" type="button" disabled>Download sweep report</button></div></details>
  <p class="sw-progress">Ready · 500 ms display / 450 ms capture</p>
  <div class="sw-slot"><canvas width="1" height="1" hidden aria-label="Four-phase sweep matrix"></canvas><span>Timing matrix appears here</span></div>`;}

function paintOverlay(view,overlay,found,w,h){
  if(overlay.width!==w||overlay.height!==h){overlay.width=w;overlay.height=h;}
  const ctx=overlay.getContext('2d');ctx.clearRect(0,0,w,h);
  const corners=found.corners?.length===4?found.corners:found.tentativeCorners;
  const lock=found.ok?'read':found.info?'acquired':corners?.length===4?'tentative':'search';
  view.dataset.lock=lock;
  if(corners?.length!==4||!corners.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)))return;
  ctx.beginPath();corners.forEach(([x,y],i)=>ctx[i?'lineTo':'moveTo'](x,y));ctx.closePath();
  ctx.lineWidth=Math.max(2,w/200);ctx.strokeStyle=lock==='tentative'?'#d3964d':'#32b77b';ctx.stroke();
}

// Only paints a new phase at a refresh boundary. Holds the same four rendered
// images, including their padding, across every cycle of a single message.
function display(output,empty){
  let timer,raf,pendingRAF,revision=0,switchId=0;
  function cancel(){++revision;clearTimeout(timer);cancelAnimationFrame(raf);}
  function clear(){
    ++switchId;cancelAnimationFrame(pendingRAF);cancel();output.hidden=true;empty.hidden=false;
    output.getContext('2d').clearRect(0,0,output.width,output.height);
  }
  function show(frames,phaseMs,onFirst,onPhase,allow){
    const begin=now(),bitmaps=frames.map(frame=>{
      const image=renderNeuralQR(frame,frame.size*8);return new ImageData(image.data,image.width,image.height);
    });
    const renderMs=now()-begin;const ticket=++switchId;cancelAnimationFrame(pendingRAF);
    // A late first paint must never revive an ACK after Next, timeout or Stop.
    pendingRAF=requestAnimationFrame(()=>{
      if(ticket!==switchId||!allow())return;
      cancel();const id=revision,origin=now();let previous=-1;
      function paint(){
        if(id!==revision)return;
        const ticks=Math.floor((now()-origin)/phaseMs);
        if(ticks!==previous){
          const bit=bitmaps[ticks%4];
          if(output.width!==bit.width||output.height!==bit.height){output.width=bit.width;output.height=bit.height;}
          output.hidden=false;empty.hidden=true;output.getContext('2d').putImageData(bit,0,0);
          const paintedAt=now();if(previous<0)onFirst(paintedAt,renderMs);previous=ticks;onPhase(paintedAt,ticks);
        }
        if(id!==revision)return;
        timer=setTimeout(()=>{raf=requestAnimationFrame(paint);},Math.max(1,origin+(ticks+1)*phaseMs-now()));
      }
      paint();
    });
  }
  return {show,clear};
}

export function initSweep(host,{onActive=()=>{}}={}){
  host.classList.add('sw-panel');host.innerHTML=markup();
  const q=s=>host.querySelector(s),el=n=>q('[name="'+n+'"]'),video=q('video'),overlay=q('.rt-view canvas'),view=q('.rt-view'),
    output=q('.sw-slot canvas'),empty=q('.sw-slot span'),dsp=display(output,empty),cap=document.createElement('canvas'),
    status=q('.sw-status'),progress=q('.sw-progress'),go=q('.sw-start'),next=q('.sw-next'),save=q('.sw-save'),details=q('details');
  const integer=(name,min,max)=>{const n=el(name).valueAsNumber;
    if(!Number.isSafeInteger(n)||n<min||n>max)throw Error('Check '+el(name).parentElement.textContent.trim()+' ('+min+'–'+max+').');
    return n;};
  let active=false,ready=false,epoch=0,run,c,reader,stream,model,camera,runId,probe,busy=false,
    frameCb,deadline,lastVideo,pacer,arming=false;
  const current=t=>active&&t===epoch;
  const measuring=(t,row)=>current(t)&&row===run.active;
  function settings(){return {role:el('role').value,size:Number(el('size').value),
    varyCapture:el('varyCapture').checked===true,startPhaseMs:integer('phase',50,2000),startCaptureMs:integer('capture',50,2000),decrementMs:integer('decrement',1,500),
    timeoutMs:1000*integer('timeout',3,120),resolution:el('resolution').value,facingMode:el('facing').value,
    deviceId:el('device').value,backend:el('backend').value};}
  function buttons(){
    const upcoming=run?.next;
    go.textContent=active?'Stop run':'Start camera';next.hidden=!active;next.disabled=!ready||Boolean(run?.active)||!upcoming||busy||arming;
    next.textContent=run?.active?'Reading step '+run.active.step:upcoming?(upcoming.step===1?'Start step 1':'Next: step '+upcoming.step):'Sweep complete';
    el('role').disabled=active;q('fieldset').disabled=active;save.disabled=!run?.rows.length;
  }
  function summary(){
    buttons();const row=run?.rows.at(-1);
    if(!row)return;
    const detailsText=run.rows.map(r=>'Step '+r.step+' · '+r.phaseMs+'/'+r.captureMs+' ms · '+r.status+' · '+fmt(r.rttMs??r.armToAckMs??r.elapsedMs));
    const stageRows=Object.entries(row.reader?.stages??{});
    const substeps=stageRows.filter(([,s])=>s.count>0).map(([key,s])=>key+': '+fmt(s.avgMsPerMeasuredScan)+' / measured scan · '+s.measuredScans+' scans');
    const hits=row.reader?.phases,rejects=Object.entries(row.reader?.phaseRejections??{}).filter(([,n])=>n>0);
    if(hits)substeps.unshift('Admitted captures by phase (0–3): '+hits.join(' / '));
    if(rejects.length)substeps.push('Rejected phase cues: '+rejects.map(([reason,n])=>reason+' '+n).join(', '));
    q('.sw-history').textContent=detailsText.join('\n')+(substeps.length?'\n\nLast step · phase counts and stage times\n'+substeps.join('\n'):'');
    progress.textContent='Step '+row.step+' · '+row.status+' · '+fmt(row.rttMs??row.armToAckMs??row.elapsedMs)+
      (row.reader?.stages?.inference?.count?' · inference '+fmt(row.reader.stages.inference.avgMs):'');
  }
  function detachFrame(){if(frameCb!=null)video.cancelVideoFrameCallback?.(frameCb);frameCb=null;}
  function endRow(outcome,at=now()){
    if(!run?.active)return;clearTimeout(deadline);detachFrame();arming=false;
    run.finish(outcome,at,reader?.snapshot());summary();
  }
  function timedOut(t,row){
    if(!measuring(t,row))return true;
    if(now()-row.startedAt<c.timeoutMs)return false;
    endRow('timeout');
    dsp.clear();
    status.textContent='Step '+row.step+' timed out. Align both devices and advance both, or restart to repeat this pace.';
    return true;
  }
  function fail(t,error){
    if(!current(t))return;
    endRow('error');stop(error?.message||String(error));
  }
  function stop(reason='Run stopped. Save the measurements from each device.'){
    if(!active)return;
    endRow('cancelled');active=false;++epoch;ready=false;busy=false;arming=false;
    detachFrame();clearTimeout(deadline);dsp.clear();reader?.close();reader=null;
    stream?.getTracks().forEach(track=>track.stop());stream=null;video.srcObject=null;
    view.dataset.lock='search';overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
    probe=null;status.textContent=reason;buttons();onActive(false);
  }
  function schedule(t,row){
    if(!measuring(t,row)||busy||frameCb!=null)return;
    frameCb=video.requestVideoFrameCallback((stamp,m)=>{
      frameCb=null;if(timedOut(t,row))return;
      if(!Number.isSafeInteger(m.presentedFrames)||!Number.isFinite(m.mediaTime)){fail(t,Error('Distinct camera timestamps are required.'));return;}
      if(m.presentedFrames<=lastVideo||video.readyState<2||!video.videoWidth){schedule(t,row);return;}
      const slot=pacer.take(stamp);if(!slot){schedule(t,row);return;}
      if(slot.interval){run.sample('captureTargetSpacing',slot.interval.requested);run.sample('captureSpacing',slot.interval.actual);}
      lastVideo=m.presentedFrames;busy=true;
      void scan(t,row,m).catch(e=>{if(measuring(t,row))fail(t,e);}).finally(()=>{
        if(!current(t))return;busy=false;buttons();if(measuring(t,row)&&!arming)schedule(t,row);
      });
    });
  }
  async function scan(t,row,m){
    const start=now(),d=captureDimensions(video.videoWidth,video.videoHeight);
    if(cap.width!==d.width||cap.height!==d.height){cap.width=d.width;cap.height=d.height;}
    const g=cap.getContext('2d',{willReadFrequently:true});g.drawImage(video,0,0,d.width,d.height);
    const image=g.getImageData(0,0,d.width,d.height);run.sample('readback',now()-start);
    const found=await reader.scan(image,{mediaTime:m.mediaTime,presentedFrames:m.presentedFrames,captureIntervalMs:row.captureMs});
    if(timedOut(t,row))return;
    const decodedAt=now(),want=c.role==='sender'?32:c.size;
    paintOverlay(view,overlay,found,d.width,d.height);
    if(found.info?.size===want&&found.corners?.length===4)run.mark('firstLock',start);
    const counts=found.phaseCounts??[0,0,0,0];
    if(found.info?.size===want&&(counts.length===4&&counts.every(Boolean)||found.inputCaptures>=4))run.mark('allPhases',start);
    progress.textContent='Step '+row.step+' · '+row.phaseMs+'/'+row.captureMs+' ms · phases '+counts.join(' / ');
    if(!found.ok||found.info?.size!==want)return;
    const check=now();
    if(c.role==='sender'){
      const match=matchesSweepAck(found.chunk,probe);run.sample('packetCheck',now()-check);
      if(!match||timedOut(t,row))return;
      run.mark('packetDecoded',decodedAt);run.mark('ackDecoded',now());endRow('ack-read');dsp.clear();
      status.textContent=run.next?'ACK read. Both devices can start step '+run.next.step+'.':'Final ACK read. Save both reports.';
    }else{
      const ack=await acknowledgeSweepProbe(found.chunk,{runId,step:row.step,size:c.size,phaseMs:row.phaseMs,captureMs:row.captureMs});
      if(timedOut(t,row))return;run.sample('checkDataAndBuildAck',now()-check);
      if(!ack)return;
      run.mark('packetDecoded',decodedAt);arming=true;
      dsp.show(ack.frames,row.phaseMs,(at,renderMs)=>{
        if(timedOut(t,row))return;
        runId=ack.runId;
        run.sample('render',renderMs);run.mark('ackDraw',at);endRow('ack-shown',at);
        status.textContent='ACK is cycling. Wait for “ACK read” on the sender.'+(run.next?' Then tap Next here to clear it and arm step '+run.next.step+'.':' Save both reports.');
      },(at,tick)=>{if(current(t))run.painted(row.step,'ack',tick,at);},()=>!timedOut(t,row));
    }
  }
  async function begin(){
    if(!active||!ready||busy||arming||run.active||!run.next)return;
    const t=epoch;
    try{
      const row=run.begin(now());
      if(c.role==='receiver')dsp.clear(); // End every old phase before starting this step's reader.
      reader.beginStep();pacer=new CapturePacer(row.captureMs,c.varyCapture);arming=false;probe=null;
      progress.textContent='Step '+row.step+' · '+row.phaseMs+' ms display / '+row.captureMs+' ms capture';
      status.textContent=c.role==='sender'?'Showing DATA; waiting for its G32 ACK.':'Armed for new DATA · keep both screens aligned.';
      const armDeadline=()=>{deadline=setTimeout(()=>{if(!timedOut(t,row))armDeadline();},Math.max(1,Math.ceil(c.timeoutMs-(now()-row.startedAt))));};
      armDeadline();buttons();
      if(c.role==='receiver'){schedule(t,row);return;}
      const buildStart=now();const fresh=await createSweepProbe({runId,step:row.step,size:c.size,phaseMs:row.phaseMs,captureMs:row.captureMs});
      if(timedOut(t,row))return;run.sample('buildData',now()-buildStart);probe=fresh;
      dsp.show(probe.frames,row.phaseMs,(at,renderMs)=>{
        if(timedOut(t,row))return;
        run.sample('render',renderMs);run.mark('probeDraw',at);schedule(t,row);
      },(at,tick)=>{if(current(t))run.painted(row.step,'data',tick,at);},()=>!timedOut(t,row));
    }catch(e){fail(t,e);}
  }
  async function start(){
    let opts;try{opts=settings();}catch(e){status.textContent=e.message;details.open=true;return;}
    if(!navigator.mediaDevices?.getUserMedia||!globalThis.isSecureContext||!video.requestVideoFrameCallback){
      status.textContent='Open on trusted HTTPS or localhost with camera frame callbacks.';return;}
    let candidate;try{candidate=new SweepRun(opts);}catch(e){status.textContent=e.message;details.open=true;return;}
    const t=++epoch;run=candidate;c=opts;runId=c.role==='sender'?newSweepRunId():null;
    active=true;onActive(true);ready=false;model=null;camera=null;busy=false;lastVideo=-1;
    details.open=false;q('.sw-history').textContent='';progress.textContent='Preparing trained reader…';status.textContent='Loading and checking the trained corrector for this device.';buttons();
    try{
      reader=new SweepReader({role:c.role,size:c.size,backend:c.backend,onProgress:s=>{if(current(t))status.textContent=String(s);}});
      const loaded=await reader.init();if(!current(t))return;model=loaded;
      status.textContent='Opening camera…';const opened=await openCaptureCamera(navigator.mediaDevices,{resolution:c.resolution,deviceId:c.deviceId,facingMode:c.facingMode});
      if(!current(t)){opened.stream.getTracks().forEach(track=>track.stop());return;}
      stream=opened.stream;for(const track of stream.getVideoTracks())track.addEventListener('ended',()=>{if(current(t))fail(t,Error('Camera disconnected.'));},{once:true});
      video.srcObject=stream;await video.play();if(!current(t))return;
      const actual=stream.getVideoTracks()[0]?.getSettings()??{};camera={width:actual.width,height:actual.height,frameRate:actual.frameRate,facingMode:actual.facingMode};
      ready=true;progress.textContent='Ready · '+c.startPhaseMs+' ms display / '+c.startCaptureMs+' ms capture';
      status.textContent='Align the cameras. Start step 1 on both devices; arm the receiver first for a clean RTT.';buttons();
    }catch(e){fail(t,e);}
  }
  async function cameras(){
    if(!navigator.mediaDevices?.enumerateDevices)return;
    try{const inputs=(await navigator.mediaDevices.enumerateDevices()).filter(x=>x.kind==='videoinput'&&x.deviceId),node=el('device'),old=node.value;
      node.replaceChildren(new Option('Use front/rear preference',''),...inputs.map((x,i)=>new Option(x.label||'Camera '+(i+1),x.deviceId)));
      if(inputs.some(x=>x.deviceId===old))node.value=old;
    }catch{/* User can select a camera direction without enumeration. */}
  }
  details.addEventListener('toggle',()=>{if(details.open&&!active)void cameras();});
  go.addEventListener('click',()=>active?stop():void start());next.addEventListener('click',()=>void begin());
  save.addEventListener('click',()=>{
    if(!run?.rows.length)return;
    const data=run.report({runId,model,camera,requestedBackend:c.backend,requestedResolution:c.resolution,facingMode:c.facingMode});
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');
    a.href=url;a.download='neural-qr-sweep-'+c.role+'-'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';
    a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&active)stop('Stopped because the page was hidden.');});
  window.addEventListener('pagehide',()=>{if(active)stop();});
  buttons();return {stop,get active(){return active;}};
}
