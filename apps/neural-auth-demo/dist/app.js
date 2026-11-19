import {initRoundtrip} from './roundtrip-ui.js';
import {initSweep} from './sweep-ui.js';
import {AuthSession} from './js/auth-core/session.js';
import {exchangeStep} from './auth-progress.js';
import {ExchangeStepClock,buildExchangeSummary} from './exchange-summary.js';
import {createFinishCard} from './finish-card.js';
import {decodeEnvelope,normalizeDisplayName,validDeviceId} from './js/auth-core/envelope.js';
import {CameraPreparation} from './camera-preparation.js';
import {sameBytes} from './js/format.js';
import {renderNeuralQR} from './js/neural-qr/layout.js';
import {openCaptureCamera,captureDimensions} from './js/dataset/camera.js';
import {decodePacket} from './js/link/protocol.js';
import {TurnChannel} from './js/neural-auth/channel.js';
import {packetFrames,unpackOptical} from './js/neural-auth/optical.js';
import {POCAutoPairing} from './js/neural-auth/poc-pairing.js';
import {PacketTelemetry} from './js/neural-auth/telemetry.js';
import {AcceptedProbeGate} from './js/neural-auth/frame-gate.js';
import {PhaseWatch,createPhaseSampler} from './phase-watch.js';
import {FirstPresentation} from './first-presentation.js';
import {bindDisplaySettings,readDisplaySettings} from './display-settings.js';

// A narrow desktop window remains a framed desktop preview. iPadOS can present
// a desktop user agent, so touch-enabled MacIntel is also treated as mobile.
export function detectDevice({userAgent='',platform='',maxTouchPoints=0,userAgentData}={}) {
  return userAgentData?.mobile===true||/Android|iPhone|iPad|iPod|Mobile/i.test(userAgent)||platform==='MacIntel'&&maxTouchPoints>1?'mobile':'desktop';
}

// Display preferences never enter camera capture or the transport configuration.
export function displayPreferences(saved) {
  let value;try{value=typeof saved==='string'&&saved.length<512?JSON.parse(saved):null;}catch{/* Use the safe display defaults. */}
  return {mirrored:value?.mirrored!==false,qrPosition:value?.qrPosition==='top'?'top':'bottom'};
}

export function matrixEdge(width,height) {
  return Number.isFinite(width)&&Number.isFinite(height)?Math.max(0,Math.floor(Math.min(width,height))):0;
}

export function sessionMessage(state,link) {
  const peer=state?.peerDisplayName?`“${state.peerDisplayName}”`:'the other device';
  if(state?.peerReported)return {title:'Peer reports completion.',instruction:`${peer} reports ${state.result?'acceptance':'rejection'} of your proof. Keep this code visible until that device finishes.`,result:`Peer reports ${state.result?'accepted':'rejected'} · ${peer}.`,complete:true};
  if(state?.resultDelivered)return {title:state.result?'Authentication complete.':'Proof rejected.',instruction:`${peer} received the result. You can finish this exchange.`,result:`${state.result?'Verified':'Rejected'} ${peer} locally.`,complete:true};
  if(state?.state==='verified'||state?.state==='rejected')return {title:state.result?'Proof verified locally.':'Proof rejected locally.',instruction:`Hold steady. Sending the result to ${peer}.`,result:`${state.result?'Verified':'Rejected'} ${peer} locally · result delivery pending.`,complete:false};
  if(state?.state==='proving')return {title:'Creating your proof.',instruction:'Hold steady. Your device is doing the work locally.',result:'',complete:false};
  if(state?.state==='verifying')return {title:'Checking the proof.',instruction:`Hold steady while this device checks ${peer}.`,result:'',complete:false};
  if(state?.state==='generating-key')return {title:'Creating your local key.',instruction:'Your key stays on this device. The exchange will start automatically.',result:'',complete:false};
  if(!state)return {title:'Preparing your connection.',instruction:'Preparing the local reader and opening your front camera.',result:'',complete:false};
  return {title:link?.mode==='receive'?'Aim at the other screen.':'Hold steady.',instruction:link?.mode==='receive'?`Point your camera at ${peer}. Keep the complete matrix in view.`:`Keep both screens facing the other device’s camera. Exchanging with ${peer}.`,result:'',complete:false};
}

/** Constant-memory aggregation: no raw frame, input tensor or key material. */
export function addTiming(operations,name,value) {
  if(typeof name!=='string'||name.length>80||!Number.isFinite(value)||value<0)return;
  if(!Object.hasOwn(operations,name)&&Object.keys(operations).length>=48)return;
  const row=operations[name]??={count:0,totalMs:0,minMs:Infinity,maxMs:0};
  row.count++;row.totalMs+=value;row.minMs=Math.min(row.minMs,value);row.maxMs=Math.max(row.maxMs,value);
}

export function timingSummary(operations) {
  return Object.fromEntries(Object.entries(operations).map(([name,row])=>[name,{...row,meanMs:row.totalMs/row.count}]));
}

export function mergeTimings(operations,updates) {
  for(const [name,value] of Object.entries(updates??{})){
    if(Number.isFinite(value)){addTiming(operations,name,value);continue;}
    if(!value||!Number.isSafeInteger(value.count)||value.count<1||![value.totalMs,value.minMs,value.maxMs].every(n=>Number.isFinite(n)&&n>=0)||name.length>80)continue;
    if(!Object.hasOwn(operations,name)&&Object.keys(operations).length>=48)continue;
    const row=operations[name]??={count:0,totalMs:0,minMs:Infinity,maxMs:0};
    row.count+=value.count;row.totalMs+=value.totalMs;row.minMs=Math.min(row.minMs,value.minMs);row.maxMs=Math.max(row.maxMs,value.maxMs);
  }
}

if(typeof document!=='undefined')initialize();

function initialize() {
  const $=id=>document.getElementById(id),setText=(id,text)=>{if($(id).textContent!==text)$(id).textContent=text;};
  bindDisplaySettings($);
  const finishCard=createFinishCard({byId:$,createElement:tag=>document.createElement(tag)});
  const device=detectDevice(navigator);document.body.dataset.device=device;
  let display=displayPreferences();
  try{display=displayPreferences(localStorage.getItem('neural-key-display'));}catch{/* Storage is optional. */}
  function applyDisplayPreferences() {
    $('app').dataset.previewMirrored=String(display.mirrored);$('app').dataset.qrPosition=display.qrPosition;
    $('mirrorPreview').setAttribute('aria-pressed',String(display.mirrored));setText('mirrorPreview',display.mirrored?'Mirrored':'Normal');
    $('mirrorPreview').setAttribute('aria-label',`Camera preview: ${display.mirrored?'mirrored':'normal'}. Switch to ${display.mirrored?'normal':'mirrored'}.`);
    $('qrPosition').setAttribute('aria-pressed',String(display.qrPosition==='top'));setText('qrPosition',`QR: ${display.qrPosition}`);
    $('qrPosition').title=`Move the outgoing matrix to the ${display.qrPosition==='top'?'bottom':'top'} of the screen.`;
    try{localStorage.setItem('neural-key-display',JSON.stringify(display));}catch{/* Preferences still last for this page. */}
  }
  function sizeMatrix() {
    const slot=$('matrixSlot'),edge=matrixEdge(slot.clientWidth,slot.clientHeight);
    if(slot.style.getPropertyValue('--matrix-edge')!==`${edge}px`)slot.style.setProperty('--matrix-edge',`${edge}px`);
  }
  function resizeViewport() {
    // visualViewport shrinks with the on-screen keyboard; Settings remains the
    // only scroll surface, and closing it restores the complete exchange view.
    const height=Math.min(window.innerHeight,window.visualViewport?.height??window.innerHeight);
    if(Number.isFinite(height)&&height>0)document.documentElement.style.setProperty('--viewport-height',`${height}px`);
    sizeMatrix();
  }
  applyDisplayPreferences();
  if(typeof ResizeObserver==='function')new ResizeObserver(sizeMatrix).observe($('matrixSlot'));
  window.addEventListener('resize',resizeViewport);window.visualViewport?.addEventListener('resize',resizeViewport);
  $('mirrorPreview').addEventListener('click',()=>{display.mirrored=!display.mirrored;applyDisplayPreferences();});
  $('qrPosition').addEventListener('click',()=>{display.qrPosition=display.qrPosition==='top'?'bottom':'top';applyDisplayPreferences();sizeMatrix();});
  $('settings').addEventListener('toggle',()=>{if(!$('settings').open)$('app').scrollTop=0;resizeViewport();});
  resizeViewport();
  let localIdentity={deviceId:null,displayName:device==='mobile'?'My phone':'My computer'};
  try{
    const saved=localStorage.getItem('neural-key-device');
    if(saved&&saved.length<1024){const value=JSON.parse(saved);if(validDeviceId(value.deviceId))localIdentity={deviceId:value.deviceId,displayName:normalizeDisplayName(value.displayName)};}
  }catch{/* Private browsing may deny local storage. */}
  if(!localIdentity.deviceId&&typeof globalThis.crypto?.randomUUID==='function'){
    localIdentity.deviceId=crypto.randomUUID();localIdentity.displayName=`${device==='mobile'?'Phone':'Computer'} ${localIdentity.deviceId.slice(-4).toUpperCase()}`;
  }
  function saveIdentity(){try{if(localIdentity.deviceId)localStorage.setItem('neural-key-device',JSON.stringify(localIdentity));}catch{/* The identity still lasts for this page. */}}
  function showLocalName(){setText('deviceName',localIdentity.displayName);$('deviceName').title=`This device: ${localIdentity.displayName}`;}
  $('displayName').value=localIdentity.displayName;showLocalName();saveIdentity();
  const supported=globalThis.isSecureContext&&Boolean(globalThis.crypto?.subtle)&&typeof Worker==='function'&&Boolean(navigator.mediaDevices?.getUserMedia);
  let singleTiming=null,sweepTiming=null,roundtripActive=false;
  let stepClock=null,shortReport=null;
  let mode='camera',active=false,epoch=0,stream=null,corrector=null,cryptoClient=null,session=null,channel=null;
  let config=null,authState=null,report=null,startedAt=0,packetTelemetry=null,autoPairing=null,simulationAbort=null;
  const displayPhases=new FirstPresentation();
  let presentationId=0,cameraTicket=0,readerIdle=false,displayStats=null,displayRound=0;
  let captureCallback=null,heartbeat=null,lastCapture=-Infinity,lastPresented=0,frameWaitAt=0,scanning=false;
  let phaseWatch=null,phaseSampler=null,watchGeneration=0,watchExpected=null,sourceWidth=0,sourceHeight=0,lastMediaTime=-Infinity;
  function resetPhaseWatch(){phaseWatch?.reset();watchGeneration++;}
  let shown=null,rendered=[],lastFeedbackAt=-Infinity,lastInference=null,scanCount=0,acceptedCount=0;
  let operations=Object.create(null),renderCache=[],events=[],lastAnnouncement='',probeGate=new AcceptedProbeGate(),mainThreadSkipped=0;
  const integer=(value,min,max,label)=>{const n=Number(value);if(!Number.isSafeInteger(n)||n<min||n>max)throw Error(`${label} must be ${min}–${max}.`);return n;};
  const error=(message='')=>{setText('error',message);$('error').hidden=!message;};
  const isCurrent=token=>active&&token===epoch;
  const timing=(name,ms)=>{if(config?.statistics)addTiming(operations,name,ms);};
  const event=message=>{if(!config?.statistics)return;events.push({afterMs:Math.round(performance.now()-startedAt),message:String(message).slice(0,240)});if(events.length>100)events.shift();};

  const preparation=new CameraPreparation({onState:(state,detail)=>{
    const notice=$('preloadNotice');notice.dataset.status=state;notice.hidden=!supported;
    const message=state==='ready'?'Ready for flight mode. Keep this page open on both devices.':
      state==='loading'?'Downloading neural corrector and checking this device…':
      state==='error'?'Not ready for flight mode. '+detail:'Prepare on Wi-Fi before another offline exchange.';
    setText('preloadText',message);notice.title=state==='loading'?String(detail??''):'';
    $('retryPreload').hidden=!['idle','error'].includes(state);
  },onLost:cause=>{if(active&&mode==='camera')fail(cause,epoch);}});
  function preloadCamera(){
    if(supported&&mode==='camera'&&!active)void preparation.prepare({size:Number($('incomingSize').value),backend:$('backend').value}).catch(()=>{});
  }

  function readSettings() {
    const role=$('role').value,backend=$('backend').value,repetitions=Number($('repetitions').value),resolution=$('resolution').value;
    const outgoingSize=Number($('outgoingSize').value),incomingMaxSize=Number($('incomingSize').value);
    if(!['verifier','prover'].includes(role)||!['auto','wasm','webgpu'].includes(backend)||![32,64,110,160,219].includes(repetitions)||!['480','720','1080','1440','2160','max'].includes(resolution)||![64,128].includes(outgoingSize)||![64,128].includes(incomingMaxSize))throw Error('Choose supported connection settings.');
    const cadence=readDisplaySettings($);
    return {displayName:normalizeDisplayName($('displayName').value),role,backend,repetitions,resolution,outgoingSize,incomingMaxSize,controlSize:32,
      ...cadence,firstCycle:$('firstCycle').checked,captureMs:integer($('captureMs').value,50,2000,'Capture interval'),cameraId:$('camera').value,facingMode:'user',
      phaseWatch:$('phaseWatch').checked,
      statistics:$('statistics').checked,optimized:$('optimized').checked,automaticTurns:true,handoff:'request',pairingConfirmation:'automatic-poc-unchecked'};
  }

  function buttons() {
    $('settingsFields').disabled=active;
    for(const id of ['cameraTab','simulationTab','roundtripTab'])$(id).disabled=active||roundtripActive;
    $('start').hidden=active;$('start').disabled=mode==='camera'&&!supported;
    $('start').innerHTML=mode==='camera'?'Start authentication <span aria-hidden="true">↗</span>':'Run local simulation <span aria-hidden="true">↗</span>';
    $('stop').hidden=!active;setText('stop',authState?.resultDelivered||authState?.peerReported?'Finish exchange':mode==='simulation'?'Cancel simulation':'Cancel exchange');
    $('statisticsPanel').hidden=!report||!config?.statistics;
    $('downloadReport').disabled=!report;
  }

  function showExchangeStep(link) {
    if (link?.closed) return; // Preserve the last observed step on failure/cancel.
    const {number,total,label}=exchangeStep(config.role,link);
    $('exchangeStep').hidden=false;
    setText('exchangeStepLabel',`Step ${number}/${total} · ${label}`);
    $('exchangeStepProgress').value=number;
    stepClock?.observe(number,performance.now());
  }

  function showIdle() {
    $('exchangeStep').hidden=true;
    setText('title',mode==='camera'?'Connect with a look.':'Try the complete exchange.');
    setText('instruction',mode==='camera'?'Open this page on both devices. Choose one verifier and one key holder in Settings, then start.':'See two virtual devices exchange identities, a challenge, a proof and a result.');
    setText('stepLabel',mode==='camera'?'NEURAL-ZKBOO · NEURAL-QR':'LOCAL SOFTWARE SIMULATION');
    $('peerName').hidden=true;$('app').dataset.state='idle';buttons();
  }

  function update() {
    buttons();if(!active||mode!=='camera')return;
    const link=channel?.snapshot(),message=sessionMessage(authState,link);
    if(link&&link.expected!==watchExpected){resetPhaseWatch();watchExpected=link.expected;}
    showExchangeStep(link);
    setText('title',message.title);setText('instruction',message.instruction);
    $('peerName').hidden=!message.complete||!authState?.peerDisplayName;setText('peerName',authState?.peerDisplayName??'');$('peerName').title=authState?.peerDisplayName??'';
    $('app').dataset.state=message.complete?'complete':'active';
    setText('stepLabel',message.complete?'EXCHANGE COMPLETE':authState?.peerDisplayName?`CONNECTED TO ${authState.peerDisplayName}`:'BROWSER-LOCAL EXCHANGE');
    $('result').hidden=!message.result;setText('result',message.result);$('result').dataset.accepted=String(authState?.result!==false);
    if(lastAnnouncement!==message.title){lastAnnouncement=message.title;setText('liveStatus',message.title);}
    const row=link?.outgoing??link?.incoming;
    $('transfer').hidden=!row?.chunkCount;
    if(row?.chunkCount){
      const progress=Math.min(1,Math.max(0,row.progress??0));$('transferProgress').value=progress;
      setText('transferPercent',`${Math.round(progress*100)}%`);
      setText('transferLabel',link.outgoing?`Sending ${Math.max(0,row.index+1)} / ${row.chunkCount} packets`:`Received ${row.receivedChunks} / ${row.chunkCount} packets`);
    }
    if(report){
      report.authentication=authState;report.transport=link;
      if(!report.completion&&message.complete){
        const now=performance.now();stepClock?.finish(now);
        report.completion={afterMs:now-startedAt,kind:authState.resultDelivered?'result-acknowledged':'peer-result-received'};
      }
    }
    // Once REQUEST grants the turn, any repeated REQUEST is idle until the
    // local reply is ready. A completed prover still displays the final ACK.
    readerIdle=Boolean(message.complete||(link?.mode==='receive'&&link.expected==='request'&&!link.waitingForRequest));
    renderCurrent();syncCamera(epoch);
    // A receive can notify the session before the final ACK is recorded.
    // Let that scan close the counters; later scans while holding the ACK
    // and the human delay before Finish do not change the short report.
    if(report?.completion&&!scanning)freezeShortReport();
  }

  function freezeShortReport(){
    if(shortReport||mode!=='camera'||!report?.completion)return;
    shortReport=buildExchangeSummary({settings:config,auth:authState,started:report.started,clock:stepClock,
      packetRtt:packetTelemetry?.snapshot().roundTrips.overall,acceptedMatrices:acceptedCount,
      captureAttempts:scanCount,inference:config.statistics?timingSummary(operations).inference:null,model:report.model,camera:report.camera});
  }

  function renderCurrent() {
    if(!channel||!active)return;
    const bytes=channel.current();
    if(!bytes){
      if(shown){endPresentation();resetPhaseWatch();shown=null;rendered=[];lastCapture=-Infinity;$('matrixZone').dataset.visible='false';$('matrixPlaceholder').hidden=false;}
      setText('matrixLabel','Looking for the other device');return;
    }
    if(shown&&sameBytes(shown,bytes))return;
    // A new outgoing packet expects a new reply even when both replies are
    // G32 ACKs. A phase repaint of the same packet must not clear the lock.
    endPresentation();resetPhaseWatch();
    const began=config.statistics?performance.now():0,packet=decodePacket(bytes),size=['ack','poll'].includes(packet.kind)?32:config.outgoingSize;
    // Rendering adds secure random padding. Cache the exact first four phases
    // for repeated display of the same packet instead of generating new pixels.
    let cached=renderCache.find(item=>item.size===size&&sameBytes(item.bytes,bytes));
    if(!cached){
      cached={bytes:bytes.slice(),size,frames:packetFrames(bytes,{size,phaseMs:config.phaseMs}).map(frame=>renderNeuralQR(frame,size*8))};
      renderCache.push(cached);if(renderCache.length>4)renderCache.shift();
    }
    shown=bytes.slice();rendered=cached.frames;
    $('matrixZone').dataset.visible='true';$('matrixPlaceholder').hidden=true;
    setText('matrixLabel',packet.kind==='ack'?'Acknowledgment · keep visible':packet.kind==='poll'?'Requesting next message · keep visible':`G${size} · ${packet.kind==='start'?'starting message':'sending data'}`);
    const token=epoch;
    let firstDraw=null;
    if(displayStats)displayStats.started++;
    if(config.firstCycle&&!readerIdle)setText('cameraStatus','Showing four phases before listening');
    displayPhases.start({phaseMs:config.phaseMs,incrementMs:config.phaseStepMs,maxPhaseMs:config.phaseMaxMs,
      paint:(phase,first,dwell,round)=>{
        const image=rendered[phase],canvas=$('matrix');
        if(canvas.width!==image.width)canvas.width=image.width;if(canvas.height!==image.height)canvas.height=image.height;
        canvas.getContext('2d').putImageData(new ImageData(image.data,image.width,image.height),0,0);
        if(first&&config.statistics){firstDraw=performance.now();packetTelemetry?.displayed(bytes,{grid:size,atMs:firstDraw});}
        if(displayStats){
          displayRound=round;
          displayStats.peakDwellMs=Math.max(dwell,displayStats.peakDwellMs);
          displayStats.maxRound=Math.max(round,displayStats.maxRound);
        }
      },
      onReady:()=>{
        if(!isCurrent(token))return;
        if(displayStats){
          displayStats.completed++;
          addTiming(displayStats.timings,'firstDrawToFourthPresented',performance.now()-firstDraw);
        }
        if(config.firstCycle){
          if(!readerIdle)setText('cameraStatus','Listening for the reply');
          lastCapture=-Infinity;syncCamera(token);
        }
      },onError:e=>fail(e,token)});
    if(config.statistics)timing('packetRender',performance.now()-began);
  }

  function endPresentation(){
    if(displayStats&&displayPhases.holding)displayStats.interrupted++;
    displayPhases.clear();presentationId++;displayRound=0;
  }

  function presentationStats(){
    if(!displayStats)return null;
    const {timings,...counts}=displayStats;
    return {...counts,ackReadByRound:counts.ackReadByRound.slice(),
      firstDrawToFourthPresentedMs:timingSummary(timings).firstDrawToFourthPresented??null};
  }

  function mayRead(){return Boolean(channel&&session&&stream&&!readerIdle&&!(config.firstCycle&&displayPhases.holding));}
  function cancelCapture(){
    ++cameraTicket;
    if(captureCallback!==null)$('video').cancelVideoFrameCallback(captureCallback);
    captureCallback=null;
  }
  function syncCamera(token){
    if(!isCurrent(token))return;
    if(mayRead())scheduleCapture(token);
    else if(captureCallback!==null)cancelCapture();
  }

  function feedback(result,width,height) {
    const now=performance.now();if(now-lastFeedbackAt<300&&!result.ok)return;lastFeedbackAt=now;
    const canvas=$('overlay');if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;
    const ctx=canvas.getContext('2d');ctx.clearRect(0,0,width,height);
    const corners=result.corners?.length===4?result.corners:result.tentativeCorners;
    if(!result.skipped)$('app').dataset.lock=result.ok?'read':result.info?'acquired':corners?.length===4?'tentative':'none';
    if(corners?.length===4&&corners.every(point=>Array.isArray(point)&&point.length===2&&point.every(Number.isFinite))){
      ctx.strokeStyle=result.ok?'#6be2a3':result.info?'#8babff':'#edac49';ctx.fillStyle=result.ok?'rgba(107,226,163,.17)':result.info?'rgba(139,171,255,.14)':'rgba(237,172,73,.14)';ctx.lineWidth=Math.max(4,width/200);ctx.beginPath();corners.forEach((point,i)=>ctx[i?'lineTo':'moveTo'](...point));ctx.closePath();ctx.fill();ctx.stroke();
    }
    setText('cameraStatus',result.ok?'Matrix read · checking packet':result.skipped?'Holding steady · watching for changes':result.info?'Matrix found · hold steady':'Find all four corners in the preview');
  }

  function scheduleCapture(token) {
    if(!isCurrent(token)||scanning||captureCallback!==null||!mayRead())return;frameWaitAt=performance.now();
    const ticket=++cameraTicket;
    const camera=$('video');
    captureCallback=camera.requestVideoFrameCallback((now,metadata)=>{
      if(ticket!==cameraTicket||!isCurrent(token))return;
      captureCallback=null;if(!mayRead())return;
      if(displayStats&&displayPhases.holding)displayStats.cameraCallbacksBeforeFourth++;
      if(!Number.isFinite(metadata.mediaTime)||!Number.isSafeInteger(metadata.presentedFrames)){fail(Error('This camera does not provide distinct frame timestamps. Try a current browser.'),token);return;}
      if(!camera.videoWidth||!camera.videoHeight||camera.readyState<2){scheduleCapture(token);return;}
      if(camera.videoWidth!==sourceWidth||camera.videoHeight!==sourceHeight||metadata.mediaTime<lastMediaTime){
        resetPhaseWatch();probeGate.reset();sourceWidth=camera.videoWidth;sourceHeight=camera.videoHeight;
        lastMediaTime=-Infinity;lastPresented=0;lastCapture=-Infinity;
      }
      if(metadata.presentedFrames<=lastPresented||metadata.mediaTime<=lastMediaTime){scheduleCapture(token);return;}
      lastPresented=metadata.presentedFrames;lastMediaTime=metadata.mediaTime;
      const due=()=>now-lastCapture>=config.captureMs;
      const trigger=phaseWatch?phaseWatch.decide(now,lastPresented,roi=>phaseSampler.read(camera,roi),due):due()?'timed':null;
      if(!trigger){scheduleCapture(token);return;}
      lastCapture=now;scanning=true;
      void scan(metadata,token,trigger,watchGeneration,presentationId).catch(e=>{if(isCurrent(token))fail(e,token);}).finally(()=>{if(isCurrent(token)){scanning=false;if(report?.completion)freezeShortReport();scheduleCapture(token);}});
    });
  }

  async function scan(metadata,token,trigger,generation,scanPresentation) {
    const camera=$('video'),canvas=$('captureCanvas'),dimensions=captureDimensions(camera.videoWidth,camera.videoHeight);
    const expected=channel?.snapshot().expected??'data';
    if(config.statistics)scanCount++;
    let probe;
    function thumbnail(source){
      const began=config.statistics?performance.now():0;
      const tiny=$('gateCanvas'),tinyContext=tiny.getContext('2d',{willReadFrequently:true});
      tinyContext.drawImage(source,0,0,32,32);const thumb=tinyContext.getImageData(0,0,32,32);
      if(config.statistics)timing(source===camera?'cameraProbe':'canvasProbe',performance.now()-began);
      return thumb;
    }
    // Read from the video only if a previously accepted template could skip
    // the full scan. A phase change or watchdog must always get a full read.
    if(config.optimized&&trigger!=='change'&&trigger!=='watchdog'&&
      probeGate.shouldSkip(()=>probe=thumbnail(camera),{runId:token,expected,mediaTime:metadata.mediaTime})){
      if(config.statistics)mainThreadSkipped++;
      setText('cameraStatus','Holding steady · watching for changes');return;
    }
    const readbackStart=config.statistics?performance.now():0;
    if(displayStats&&displayPhases.holding)displayStats.fullReadbacksBeforeFourth++;
    if(canvas.width!==dimensions.width)canvas.width=dimensions.width;if(canvas.height!==dimensions.height)canvas.height=dimensions.height;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(camera,0,0,canvas.width,canvas.height);
    const image=ctx.getImageData(0,0,canvas.width,canvas.height);
    if(config.statistics){timing('cameraReadback',performance.now()-readbackStart);report.processedCamera=dimensions;}
    // A collecting worker still needs a thumbnail so that accepted exposures
    // can later suppress all four held phases. Reuse the captured pixels.
    if(config.optimized&&!probe)probe=thumbnail(canvas);
    const worker=corrector;
    const options={size:config.incomingMaxSize,backend:config.backend,expected,runId:token,presentationId:scanPresentation,statistics:config.statistics,optimized:config.optimized};
    const workerStart=config.statistics?performance.now():0;
    const result=await worker.call('transport-scan',{...options,image,...(probe?{probe,forceAcquisition:true}:{}),captureIntervalMs:config.captureMs,mediaTime:metadata.mediaTime,presentedFrames:metadata.presentedFrames},[image.data.buffer]);
    if(!isCurrent(token))return;
    if(config.statistics)timing('workerRoundTrip',performance.now()-workerStart);
    // Never apply a read submitted before this outgoing packet/turn. The
    // worker's next scan also starts a fresh phase pool for this presentation.
    if(scanPresentation!==presentationId||readerIdle||expected!==watchExpected){
      if(displayStats)displayStats.staleScanResults++;
      return;
    }
    // The serial capture canvas still holds the exposure the worker examined;
    // taking a baseline from the live video now could silently skip a phase.
    if(phaseWatch&&generation===watchGeneration){
      // All phases of an already accepted packet are duplicates. Keep the
      // cheap accepted-packet gate in charge until there is new optical data.
      if(result.skipped&&result.stage==='unchanged')resetPhaseWatch();
      else phaseWatch.observe(result,canvas.width,canvas.height,roi=>phaseSampler.read(canvas,roi));
    }
    if(config.statistics){
      mergeTimings(operations,result.timings);
      if(Number.isFinite(result.inferenceMs))lastInference=result.inferenceMs;
      if(result.statistics)report.receiverStatistics=result.statistics;
      else if(result.stats)report.lastScanStatistics=result.stats;
    }
    feedback(result,canvas.width,canvas.height);if(!result.ok)return;
    let bytes;try{bytes=unpackOptical(result.chunk,expected);}catch{event('Optical packet rejected by container checks.');return;}
    const receivedAt=performance.now(),replyRound=displayRound,acceptance=await channel.receive(bytes);if(!isCurrent(token))return;
    packetTelemetry?.accepted(bytes,acceptance,{grid:result.info?.size,atMs:receivedAt});
    if(config.statistics)timing('transportReceive',performance.now()-receivedAt);
    if(acceptance.accepted){
      if(displayStats&&expected==='ack'&&replyRound>0)displayStats.ackReadByRound[Math.min(replyRound,8)-1]++;
      resetPhaseWatch();
      if(config.statistics)acceptedCount++;
      if(report?.completion)freezeShortReport();
      // Only an accepted transport packet may suppress its repeated pixels.
      // This serial call finishes before another scan can use the worker.
      if(result.acceptToken!==undefined){
        const accepted=await worker.call('transport-accept',{...options,token:result.acceptToken});
        if(isCurrent(token)&&config.optimized)probeGate.accept(accepted.probeGate);
      }
      if(!isCurrent(token))return;
      setText('cameraStatus',expected==='ack'?'Acknowledgment received · continuing':expected==='request'?'Request received · continuing':'Packet received · hold steady');
    }
  }

  function statisticsView() {
    if(!config?.statistics)return;
    setText('elapsed',`${Math.round((performance.now()-startedAt)/1000)}s`);
    setText('acceptedPackets',String(acceptedCount));setText('scanAttempts',String(scanCount));
    setText('inferenceMs',lastInference===null?'—':`${lastInference.toFixed(0)} ms`);
    setText('proofBytes',authState?.proofBytes?`${authState.proofBytes.toLocaleString()} B`:'—');
  }

  async function populateCameras(token=null) {
    if(!navigator.mediaDevices?.enumerateDevices)return;
    const selected=$('camera').value,devices=await navigator.mediaDevices.enumerateDevices();
    if(token!==null&&!isCurrent(token))return;
    $('camera').replaceChildren(new Option('Front camera · default',''),...devices.filter(d=>d.kind==='videoinput').map((d,i)=>new Option(d.label||`Camera ${i+1}`,d.deviceId)));
    if([...$('camera').options].some(option=>option.value===selected))$('camera').value=selected;
  }

  async function startCamera(token) {
    if(typeof $('video').requestVideoFrameCallback!=='function')throw Error('This demo needs a browser with camera frame callbacks. Try a current Chrome, Safari or Firefox.');
    packetTelemetry=config.statistics?new PacketTelemetry({originMs:startedAt,maxRows:2048}):null;
    autoPairing=new POCAutoPairing({isActive:()=>isCurrent(token),onError:e=>fail(e,token)});
    update();const initStart=config.statistics?performance.now():0;
    const ready=await preparation.prepare({size:config.incomingMaxSize,backend:config.backend});
    if(!isCurrent(token))return;
    corrector=ready.corrector;cryptoClient=ready.cryptoClient;const model=ready.model;
    if(config.statistics){timing('waitForPreparedWorkers',performance.now()-initStart);report.model=model;}
    setText('modelStatus',`Trained ${model.modelProfile??'corrector'} · ${(model.effectiveBackend??config.backend).toUpperCase()} · on-device`);
    const opened=await openCaptureCamera(navigator.mediaDevices,{resolution:config.resolution,deviceId:config.cameraId,facingMode:'user'});
    if(!isCurrent(token)){opened.stream.getTracks().forEach(track=>track.stop());return;}
    stream=opened.stream;for(const track of stream.getVideoTracks())track.addEventListener('ended',()=>{if(isCurrent(token))fail(Error('The camera disconnected. Start a fresh exchange on both devices.'),token);},{once:true});
    $('video').srcObject=stream;await $('video').play();if(!isCurrent(token))return;
    $('cameraPlaceholder').hidden=true;
    if(config.statistics){const camera=stream.getVideoTracks()[0]?.getSettings()??{};report.camera={width:camera.width,height:camera.height,frameRate:camera.frameRate,facingMode:camera.facingMode,...opened.request};}
    await populateCameras(token);if(!isCurrent(token))return;
    channel=new TurnChannel({initiator:config.role==='verifier',outgoingSize:config.outgoingSize,automaticTurns:true,
      onChange:()=>{if(isCurrent(token))update();},onMessage:async bytes=>{if(!isCurrent(token))return;const result=await session.receive(bytes);if(isCurrent(token))event(result.accepted?'Authentication message received.':`Authentication message ignored: ${result.reason}`);},onError:e=>fail(e,token)});
    session=new AuthSession({role:config.role,deviceId:localIdentity.deviceId,displayName:config.displayName,cryptoClient,repetitions:config.repetitions,timeoutMs:60*60*1000,
      send:bytes=>channel.enqueue(bytes,{expectReply:decodeEnvelope(bytes).header.type!=='RESULT'}),onChange:state=>{if(!isCurrent(token))return;const previous=authState?.state;authState=state;if(previous!==state.state)event(state.status);autoPairing.update(session,state);update();}});
    heartbeat=setInterval(()=>{
      if(!isCurrent(token))return;
      if(captureCallback!==null&&!scanning&&performance.now()-frameWaitAt>15000){fail(Error('The camera stopped producing frames. Start again on both devices.'),token);return;}
      channel.tick();session.tick();statisticsView();
      if(['failed','timed-out'].includes(authState?.state))fail(Error(authState.error||'The exchange expired.'),token);
    },1000);
    scheduleCapture(token);update();stepClock?.ready(performance.now());await session.start();
  }

  async function startSimulation(token) {
    simulationAbort=new AbortController();setText('simulationStatus','Preparing two local devices…');
    const {runSimulation}=await import('./simulation.js');if(!isCurrent(token))return;
    const result=await runSimulation({onProgress:message=>{if(isCurrent(token))setText('simulationStatus',String(message));},statistics:config.statistics,optimized:config.optimized,signal:simulationAbort.signal});
    if(!isCurrent(token))return;
    report={...result,settings:report.settings,started:report.started,finished:new Date().toISOString()};
    const baseline=result.baseline,optimized=result.optimized,successful=(optimized??baseline)?.success??result.success;
    setText('title',successful===false?'Simulation did not complete.':'Simulation complete.');
    setText('instruction','A local software exercise with synthetic captures and assumed decoder output.');
    setText('simulationStatus',result.assumedOutput?'Finished with target-assisted decoder output. This does not validate model accuracy or camera performance.':'Finished. Download the report for the measured results.');
    $('app').dataset.state=successful===false?'idle':'complete';
    active=false;simulationAbort=null;buttons();
    if(config.statistics){
      const measured=optimized??baseline,inference=measured?.stages?.find(row=>row.stage==='inference');
      setText('modelStatus',result.assumedOutput?'Untrained-model timing with assumed decoder output.':'Local simulation measurements.');
      setText('elapsed',`${((performance.now()-startedAt)/1000).toFixed(1)}s`);setText('proofBytes',measured?.proofBytes?`${measured.proofBytes.toLocaleString()} B`:'—');
      setText('scanAttempts',String(measured?.scans??'—'));setText('acceptedPackets',String(measured?.acceptedPackets??'—'));
      setText('inferenceLabel','Mean inference');setText('inferenceMs',Number.isFinite(inference?.avgMs)?`${inference.avgMs.toFixed(0)} ms`:'—');
      if(baseline&&optimized){
        $('simulationComparison').hidden=false;setText('baselineTime',`${(baseline.wallMs/1000).toFixed(1)}s`);setText('optimizedTime',`${(optimized.wallMs/1000).toFixed(1)}s`);
        setText('baselineInferences',String(baseline.inferenceCalls));setText('optimizedInferences',String(optimized.inferenceCalls));
      }
    }
  }

  async function start() {
    if(active||mode==='roundtrip')return;error();$('app').dataset.lock='none';try{config=readSettings();}catch(e){error(e.message);$('settings').open=true;return;}
    localIdentity.displayName=config.displayName;saveIdentity();showLocalName();$('peerName').hidden=true;$('simulationComparison').hidden=true;setText('inferenceLabel','Latest inference');
    active=true;const token=++epoch;startedAt=performance.now();authState=null;session=null;channel=null;corrector=null;cryptoClient=null;
    finishCard.reset();shortReport=null;
    stepClock=mode==='camera'&&config.statistics?new ExchangeStepClock(startedAt):null;
    if(mode==='camera')showExchangeStep(null);
    presentationId=0;readerIdle=false;
    displayStats=mode==='camera'&&config.statistics?{started:0,completed:0,interrupted:0,cameraCallbacksBeforeFourth:0,
      fullReadbacksBeforeFourth:0,staleScanResults:0,peakDwellMs:0,maxRound:0,ackReadByRound:Array(8).fill(0),timings:Object.create(null)}:null;
    events=[];operations=Object.create(null);renderCache=[];shown=null;rendered=[];lastInference=null;acceptedCount=0;scanCount=0;mainThreadSkipped=0;probeGate.reset();
    phaseWatch=mode==='camera'&&config.phaseWatch?new PhaseWatch({captureMs:config.captureMs,statistics:config.statistics}):null;
    phaseSampler=phaseWatch?createPhaseSampler():null;watchExpected=null;sourceWidth=sourceHeight=0;lastMediaTime=-Infinity;watchGeneration=0;
    lastCapture=-Infinity;lastPresented=0;lastFeedbackAt=-Infinity;scanning=false;packetTelemetry=null;lastAnnouncement='';
    report={format:'neural-auth-demo-report',version:1,kind:mode,started:new Date().toISOString(),settings:{...config,cameraId:undefined},completion:null};
    $('settings').open=false;$('result').hidden=true;$('transfer').hidden=true;$('matrixZone').dataset.visible='false';$('matrixPlaceholder').hidden=false;
    $('app').dataset.state='active';buttons();statisticsView();
    try{if(mode==='camera')await startCamera(token);else await startSimulation(token);}catch(e){if(isCurrent(token))fail(e,token);}
  }

  function finish(reason='Finished by the user.') {
    if(!active)return;
    const savedAuth=authState,savedTransport=channel?.snapshot(),wasComplete=authState?.resultDelivered||authState?.peerReported;
    if(wasComplete)freezeShortReport();
    // Invalidate callbacks before closing workers, channels or camera tracks.
    active=false;$('app').dataset.lock='none';epoch++;simulationAbort?.abort();simulationAbort=null;autoPairing?.close();autoPairing=null;
    endPresentation();cancelCapture();clearInterval(heartbeat);
    if(report&&displayStats)report.firstPresentation=presentationStats();
    displayStats=null;
    if(mode==='camera')preparation.clear();
    channel?.close();session?.close(reason);cryptoClient=null;corrector=null;
    stream?.getTracks().forEach(track=>track.stop());stream=null;$('video').srcObject=null;$('cameraPlaceholder').hidden=false;
    if(report&&config.statistics&&phaseWatch)report.phaseWatch=phaseWatch.snapshot();
    phaseWatch=null;phaseSampler=null;
    $('overlay').getContext('2d').clearRect(0,0,$('overlay').width,$('overlay').height);
    if(report){report.finished=new Date().toISOString();report.elapsedMs=performance.now()-startedAt;report.stopReason=reason;if(mode==='camera'){report.authentication=savedAuth;report.transport=savedTransport;}}
    shown=null;rendered=[];renderCache=[];scanning=false;probeGate.reset();authState=savedAuth;event(reason);statisticsView();
    $('matrixZone').dataset.visible='false';$('matrixPlaceholder').hidden=false;setText('matrixLabel','Exchange finished');setText('cameraStatus','Camera is off');
    if(!wasComplete){setText('title',mode==='camera'?'Exchange stopped.':'Simulation stopped.');setText('instruction','You can start a fresh exchange when you are ready.');setText('simulationStatus',reason);}
    $('app').dataset.state=wasComplete?'complete':'idle';buttons();
    if(mode==='camera'&&wasComplete&&shortReport){
      setText('instruction','Review this device’s result or download the short report.');
      finishCard.show(shortReport);
    }
  }

  function fail(cause,token=epoch) {
    if(!isCurrent(token))return;const message=session?.snapshot().error||cause?.message||String(cause);finish(message);error(message);
  }

  function download() {
    if(!report)return;
    const value=mode==='simulation'?report:{...report,elapsedMs:report.elapsedMs??performance.now()-startedAt,
      ...(config.statistics?{...(displayStats?{firstPresentation:presentationStats()}:{}),...(phaseWatch?{phaseWatch:phaseWatch.snapshot()}:{}),packetTelemetry:packetTelemetry?.snapshot()??null,operations:timingSummary(operations),captureAttempts:scanCount,mainThreadSkipped,transportAcceptedMatrices:acceptedCount,events:[...events]}:{})};
    const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),anchor=document.createElement('a');
    anchor.href=url;anchor.download=`neural-key-${report.kind}-${new Date().toISOString().replace(/[:.]/g,'-')}.json`;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }

  function setMode(next) {
    if(active||roundtripActive||next===mode)return;
    if(mode==='roundtrip'){singleTiming?.stop();sweepTiming?.stop();}
    finishCard.reset();shortReport=null;stepClock=null;
    if(mode==='camera')preparation.clear();
    mode=next;report=null;config=null;authState=null;error();
    $('settings').open=false;
    $('roundtripTab').setAttribute('aria-pressed',String(next==='roundtrip'));
    $('roundtripPanel').hidden=next!=='roundtrip';
    if(next==='roundtrip')showTimingMethod();
    $('cameraTab').setAttribute('aria-pressed',String(next==='camera'));$('simulationTab').setAttribute('aria-pressed',String(next==='simulation'));
    $('app').dataset.lock='none';$('cameraPanel').hidden=next!=='camera';$('simulationPanel').hidden=next!=='simulation';$('app').dataset.mode=next;showIdle();sizeMatrix();preloadCamera();
  }

  function showTimingMethod(){
    if(roundtripActive)return;
    const sweep=$('timingMethod').value==='trained',options={onActive:value=>{
      roundtripActive=value;$('timingMethod').disabled=value;buttons();
    }};
    $('singleTimingPanel').hidden=sweep;$('sweepTimingPanel').hidden=!sweep;
    if(sweep){sweepTiming??=initSweep($('sweepTimingPanel'),options);}
    else{singleTiming??=initRoundtrip($('singleTimingPanel'),options);}
  }
  $('timingMethod').addEventListener('change',()=>{if(mode==='roundtrip')showTimingMethod();});
  $('roundtripTab').addEventListener('click',()=>setMode('roundtrip'));
  $('cameraTab').addEventListener('click',()=>setMode('camera'));$('simulationTab').addEventListener('click',()=>setMode('simulation'));
  $('start').addEventListener('click',()=>void start());$('stop').addEventListener('click',()=>finish(authState?.peerReported||authState?.resultDelivered?'Finished by the user.':'Cancelled by the user.'));
  $('downloadReport').addEventListener('click',download);
  $('displayName').addEventListener('change',()=>{try{localIdentity.displayName=normalizeDisplayName($('displayName').value);$('displayName').value=localIdentity.displayName;saveIdentity();showLocalName();}catch{/* Start displays input validation errors. */}});
  for(const id of ['role','outgoingSize'])$(id).addEventListener('change',()=>{$('settingsHint').replaceChildren(document.createTextNode(`${$('role').value==='verifier'?'Verifier':'Key holder'} · G${$('outgoingSize').value} `),Object.assign(document.createElement('b'),{textContent:'⌄'}));});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&active)finish('Stopped because the page became hidden. Keep both pages visible during an exchange.');});
  $('retryPreload').addEventListener('click',preloadCamera);
  for(const id of ['incomingSize','backend'])$(id).addEventListener('change',preloadCamera);
  window.addEventListener('pagehide',()=>{finish('Page closed.');preparation.clear();});
  window.addEventListener('pageshow',e=>{if(e.persisted)preloadCamera();});
  if(!supported)error('Camera exchange requires trusted HTTPS or localhost, camera access and browser workers. You can still try the local simulation.');
  void populateCameras().catch(()=>{});showIdle();preloadCamera();
}
