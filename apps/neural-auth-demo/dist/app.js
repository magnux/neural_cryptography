import {AuthSession} from './js/auth-core/session.js';
import {normalizeDisplayName,validDeviceId} from './js/auth-core/envelope.js';
import {CryptoClient} from './js/crypto-lab/client.js';
import {CorrectorClient} from './js/corrector-lab/client.js';
import {sameBytes} from './js/format.js';
import {renderNeuralQR} from './js/neural-qr/layout.js';
import {openCaptureCamera,captureDimensions} from './js/dataset/camera.js';
import {decodePacket} from './js/link/protocol.js';
import {TurnChannel} from './js/neural-auth/channel.js';
import {packetFrames,unpackOptical} from './js/neural-auth/optical.js';
import {POCAutoPairing} from './js/neural-auth/poc-pairing.js';
import {PacketTelemetry} from './js/neural-auth/telemetry.js';
import {AcceptedProbeGate} from './js/neural-auth/frame-gate.js';

// A narrow desktop window remains a framed desktop preview. iPadOS can present
// a desktop user agent, so touch-enabled MacIntel is also treated as mobile.
export function detectDevice({userAgent='',platform='',maxTouchPoints=0,userAgentData}={}) {
  return userAgentData?.mobile===true||/Android|iPhone|iPad|iPod|Mobile/i.test(userAgent)||platform==='MacIntel'&&maxTouchPoints>1?'mobile':'desktop';
}

export function sessionMessage(state,link) {
  const peer=state?.peerDisplayName?`“${state.peerDisplayName}”`:'the other device';
  if(state?.peerReported)return {title:'Peer reports completion.',instruction:`${peer} reports ${state.result?'acceptance':'rejection'} of your proof. Keep this code visible until that device finishes.`,result:`Peer reports ${state.result?'accepted':'rejected'} · ${peer}.`,complete:true};
  if(state?.resultDelivered)return {title:state.result?'Authentication complete.':'Proof rejected.',instruction:`${peer} received the result. You can finish this exchange.`,result:`${state.result?'Verified':'Rejected'} ${peer} locally.`,complete:true};
  if(state?.state==='verified'||state?.state==='rejected')return {title:state.result?'Proof verified locally.':'Proof rejected locally.',instruction:`Hold steady. Sending the result to ${peer}.`,result:`${state.result?'Verified':'Rejected'} ${peer} locally · result delivery pending.`,complete:false};
  if(state?.state==='proving')return {title:'Creating your proof.',instruction:'Hold steady. Your device is doing the work locally.',result:'',complete:false};
  if(state?.state==='verifying')return {title:'Checking the proof.',instruction:`Hold steady while this device checks ${peer}.`,result:'',complete:false};
  if(state?.state==='generating-key')return {title:'Creating your local key.',instruction:'Your key stays on this device. The exchange will start automatically.',result:'',complete:false};
  if(!state)return {title:'Preparing your connection.',instruction:'Loading the trained corrector and opening your front camera.',result:'',complete:false};
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
  const device=detectDevice(navigator);document.body.dataset.device=device;
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
  let mode='camera',active=false,epoch=0,stream=null,corrector=null,cryptoClient=null,session=null,channel=null;
  let config=null,authState=null,report=null,startedAt=0,packetTelemetry=null,autoPairing=null,simulationAbort=null;
  let captureCallback=null,displayTimer=null,heartbeat=null,lastCapture=-Infinity,lastPresented=0,frameWaitAt=0,scanning=false;
  let shown=null,rendered=[],phaseStart=0,phaseIndex=-1,lastFeedbackAt=-Infinity,lastInference=null,scanCount=0,acceptedCount=0;
  let operations=Object.create(null),renderCache=[],events=[],lastAnnouncement='',probeGate=new AcceptedProbeGate(),mainThreadSkipped=0;
  const integer=(value,min,max,label)=>{const n=Number(value);if(!Number.isSafeInteger(n)||n<min||n>max)throw Error(`${label} must be ${min}–${max}.`);return n;};
  const error=(message='')=>{setText('error',message);$('error').hidden=!message;};
  const isCurrent=token=>active&&token===epoch;
  const timing=(name,ms)=>{if(config?.statistics)addTiming(operations,name,ms);};
  const event=message=>{if(!config?.statistics)return;events.push({afterMs:Math.round(performance.now()-startedAt),message:String(message).slice(0,240)});if(events.length>100)events.shift();};

  function readSettings() {
    const role=$('role').value,backend=$('backend').value,repetitions=Number($('repetitions').value),resolution=$('resolution').value;
    const outgoingSize=Number($('outgoingSize').value),incomingMaxSize=Number($('incomingSize').value);
    if(!['verifier','prover'].includes(role)||!['auto','wasm','webgpu'].includes(backend)||![32,64,110,160,219].includes(repetitions)||!['480','720','1080','1440','2160','max'].includes(resolution)||![64,128].includes(outgoingSize)||![64,128].includes(incomingMaxSize))throw Error('Choose supported connection settings.');
    const phaseMs=integer($('phaseMs').value,50,10000,'Phase interval');
    return {displayName:normalizeDisplayName($('displayName').value),role,backend,repetitions,resolution,outgoingSize,incomingMaxSize,controlSize:32,
      phaseMs,captureMs:integer($('captureMs').value,50,2000,'Capture interval'),cameraId:$('camera').value,facingMode:'user',
      statistics:$('statistics').checked,optimized:$('optimized').checked,automaticTurns:true,handoffHoldMs:Math.max(3000,phaseMs*8),pairingConfirmation:'automatic-poc-unchecked'};
  }

  function buttons() {
    $('settingsFields').disabled=active;$('cameraTab').disabled=active;$('simulationTab').disabled=active;
    $('start').hidden=active;$('start').disabled=mode==='camera'&&!supported;
    $('start').innerHTML=mode==='camera'?'Start authentication <span aria-hidden="true">↗</span>':'Run local simulation <span aria-hidden="true">↗</span>';
    $('stop').hidden=!active;setText('stop',authState?.resultDelivered||authState?.peerReported?'Finish exchange':mode==='simulation'?'Cancel simulation':'Cancel exchange');
    $('statisticsPanel').hidden=!report||!config?.statistics;
    $('downloadReport').disabled=!report;
  }

  function showIdle() {
    setText('title',mode==='camera'?'Connect with a look.':'Try the complete exchange.');
    setText('instruction',mode==='camera'?'Open this page on both devices. Choose one verifier and one key holder in Settings, then start.':'See two virtual devices exchange identities, a challenge, a proof and a result.');
    setText('stepLabel',mode==='camera'?'NEURAL-ZKBOO · NEURAL-QR':'LOCAL SOFTWARE SIMULATION');
    $('peerName').hidden=true;$('app').dataset.state='idle';buttons();
  }

  function update() {
    buttons();if(!active||mode!=='camera')return;
    const link=channel?.snapshot(),message=sessionMessage(authState,link);
    setText('title',message.title);setText('instruction',message.instruction);
    $('peerName').hidden=!message.complete||!authState?.peerDisplayName;setText('peerName',authState?.peerDisplayName??'');
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
      if(!report.completion&&message.complete)report.completion={afterMs:performance.now()-startedAt,kind:authState.resultDelivered?'result-acknowledged':'peer-result-received'};
    }
    renderCurrent();
  }

  function renderCurrent() {
    if(!channel||!active)return;
    const bytes=channel.current();
    if(!bytes){
      if(shown){shown=null;rendered=[];clearTimeout(displayTimer);$('matrixZone').dataset.visible='false';$('matrixPlaceholder').hidden=false;}
      setText('matrixLabel','Looking for the other device');return;
    }
    if(shown&&sameBytes(shown,bytes))return;
    const began=config.statistics?performance.now():0,packet=decodePacket(bytes),size=packet.kind==='ack'?32:config.outgoingSize;
    // Rendering adds secure random padding. Cache the exact first four phases
    // for retained ACK/START alternation instead of generating new pixels.
    let cached=renderCache.find(item=>item.size===size&&sameBytes(item.bytes,bytes));
    if(!cached){
      cached={bytes:bytes.slice(),size,frames:packetFrames(bytes,{size,phaseMs:config.phaseMs}).map(frame=>renderNeuralQR(frame,size*8))};
      renderCache.push(cached);if(renderCache.length>4)renderCache.shift();
    }
    shown=bytes.slice();rendered=cached.frames;phaseStart=performance.now();phaseIndex=-1;
    $('matrixZone').dataset.visible='true';$('matrixPlaceholder').hidden=true;
    setText('matrixLabel',packet.kind==='ack'?'Acknowledgment · keep visible':`G${size} · ${packet.kind==='start'?'starting message':'sending data'}`);
    paintPhase();packetTelemetry?.displayed(bytes,{grid:size,atMs:performance.now()});
    if(config.statistics)timing('packetRender',performance.now()-began);
    schedulePhase(epoch);
  }

  function paintPhase() {
    if(!active||!rendered.length)return;
    const next=Math.floor((performance.now()-phaseStart)/config.phaseMs)%rendered.length;if(next===phaseIndex)return;
    phaseIndex=next;const image=rendered[next],canvas=$('matrix');
    if(canvas.width!==image.width)canvas.width=image.width;if(canvas.height!==image.height)canvas.height=image.height;
    canvas.getContext('2d').putImageData(new ImageData(image.data,image.width,image.height),0,0);
  }

  function schedulePhase(token) {
    clearTimeout(displayTimer);if(!isCurrent(token)||rendered.length<2)return;
    const remaining=config.phaseMs-((performance.now()-phaseStart)%config.phaseMs);
    displayTimer=setTimeout(()=>{if(isCurrent(token)){paintPhase();schedulePhase(token);}},Math.max(1,remaining));
  }

  function feedback(result,width,height) {
    const now=performance.now();if(now-lastFeedbackAt<300&&!result.ok)return;lastFeedbackAt=now;
    const canvas=$('overlay');if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;
    const ctx=canvas.getContext('2d');ctx.clearRect(0,0,width,height);
    const corners=result.corners?.length===4?result.corners:result.tentativeCorners;
    if(corners?.length===4&&corners.every(point=>Array.isArray(point)&&point.length===2&&point.every(Number.isFinite))){
      ctx.strokeStyle=result.ok?'#6be2a3':'#8babff';ctx.fillStyle=result.ok?'rgba(107,226,163,.17)':'rgba(139,171,255,.14)';ctx.lineWidth=Math.max(4,width/200);ctx.beginPath();corners.forEach((point,i)=>ctx[i?'lineTo':'moveTo'](...point));ctx.closePath();ctx.fill();ctx.stroke();
    }
    setText('cameraStatus',result.ok?'Matrix read · checking packet':result.skipped?'Holding steady · watching for changes':result.info?'Matrix found · hold steady':'Find all four corners in the preview');
  }

  function scheduleCapture(token) {
    if(!isCurrent(token)||scanning)return;frameWaitAt=performance.now();
    captureCallback=$('video').requestVideoFrameCallback((now,metadata)=>{
      captureCallback=null;if(!isCurrent(token))return;
      if(!Number.isFinite(metadata.mediaTime)||!Number.isSafeInteger(metadata.presentedFrames)){fail(Error('This camera does not provide distinct frame timestamps. Try a current browser.'),token);return;}
      if(now-lastCapture<config.captureMs||metadata.presentedFrames<=lastPresented||!$('video').videoWidth||$('video').readyState<2){scheduleCapture(token);return;}
      lastCapture=now;lastPresented=metadata.presentedFrames;scanning=true;
      void scan(metadata,token).catch(e=>{if(isCurrent(token))fail(e,token);}).finally(()=>{if(isCurrent(token)){scanning=false;scheduleCapture(token);}});
    });
  }

  async function scan(metadata,token) {
    const camera=$('video'),canvas=$('captureCanvas'),dimensions=captureDimensions(camera.videoWidth,camera.videoHeight);
    const expected=channel?.snapshot().outgoing?'ack':'data';
    const began=config.statistics?performance.now():0;
    if(config.statistics)scanCount++;
    let probe;
    if(config.optimized){
      const tiny=$('gateCanvas'),tinyContext=tiny.getContext('2d',{willReadFrequently:true});
      tinyContext.drawImage(camera,0,0,32,32);probe=tinyContext.getImageData(0,0,32,32);
      const skip=probeGate.shouldSkip(probe,{runId:token,expected,mediaTime:metadata.mediaTime});
      if(config.statistics)timing('cameraProbe',performance.now()-began);
      if(skip){
        if(config.statistics)mainThreadSkipped++;
        setText('cameraStatus','Holding steady · watching for changes');return;
      }
    }
    const readbackStart=config.statistics?performance.now():0;
    if(canvas.width!==dimensions.width)canvas.width=dimensions.width;if(canvas.height!==dimensions.height)canvas.height=dimensions.height;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(camera,0,0,canvas.width,canvas.height);
    const image=ctx.getImageData(0,0,canvas.width,canvas.height);
    if(config.statistics){timing('cameraReadback',performance.now()-readbackStart);report.processedCamera=dimensions;}
    const worker=corrector;
    const options={size:config.incomingMaxSize,backend:config.backend,expected,runId:token,statistics:config.statistics,optimized:config.optimized};
    const workerStart=config.statistics?performance.now():0;
    const result=await worker.call('transport-scan',{...options,image,...(probe?{probe,forceAcquisition:true}:{}),captureIntervalMs:config.captureMs,mediaTime:metadata.mediaTime,presentedFrames:metadata.presentedFrames},[image.data.buffer]);
    if(!isCurrent(token))return;
    if(config.statistics){
      timing('workerRoundTrip',performance.now()-workerStart);
      mergeTimings(operations,result.timings);
      if(Number.isFinite(result.inferenceMs))lastInference=result.inferenceMs;
      if(result.statistics)report.receiverStatistics=result.statistics;
      else if(result.stats)report.lastScanStatistics=result.stats;
    }
    feedback(result,canvas.width,canvas.height);if(!result.ok)return;
    let bytes;try{bytes=unpackOptical(result.chunk,expected);}catch{event('Optical packet rejected by container checks.');return;}
    const receivedAt=performance.now(),acceptance=await channel.receive(bytes);if(!isCurrent(token))return;
    packetTelemetry?.accepted(bytes,acceptance,{grid:result.info?.size,atMs:receivedAt});
    if(config.statistics)timing('transportReceive',performance.now()-receivedAt);
    if(acceptance.accepted){
      if(config.statistics)acceptedCount++;
      // Only an accepted transport packet may suppress its repeated pixels.
      // This serial call finishes before another scan can use the worker.
      if(result.acceptToken!==undefined){
        const accepted=await worker.call('transport-accept',{...options,token:result.acceptToken});
        if(isCurrent(token)&&config.optimized)probeGate.accept(accepted.probeGate);
      }
      if(!isCurrent(token))return;
      setText('cameraStatus',expected==='ack'?'Acknowledgment received · continuing':'Packet received · hold steady');
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
    corrector=new CorrectorClient({onProgress:message=>{if(isCurrent(token)&&config.statistics)setText('modelStatus',String(message));}});
    cryptoClient=new CryptoClient({onInvalidate:e=>{if(isCurrent(token))fail(e,token);}});
    update();const initStart=config.statistics?performance.now():0;
    const model=await corrector.call('transport-init',{size:config.incomingMaxSize,backend:config.backend,statistics:config.statistics,optimized:config.optimized});
    if(!isCurrent(token))return;
    if(config.statistics){timing('modelInitialization',performance.now()-initStart);report.model=model;}
    setText('modelStatus',`Trained ${model.modelProfile??'corrector'} · ${(model.effectiveBackend??config.backend).toUpperCase()} · on-device`);
    const opened=await openCaptureCamera(navigator.mediaDevices,{resolution:config.resolution,deviceId:config.cameraId,facingMode:'user'});
    if(!isCurrent(token)){opened.stream.getTracks().forEach(track=>track.stop());return;}
    stream=opened.stream;for(const track of stream.getVideoTracks())track.addEventListener('ended',()=>{if(isCurrent(token))fail(Error('The camera disconnected. Start a fresh exchange on both devices.'),token);},{once:true});
    $('video').srcObject=stream;await $('video').play();if(!isCurrent(token))return;
    $('cameraPlaceholder').hidden=true;
    if(config.statistics){const camera=stream.getVideoTracks()[0]?.getSettings()??{};report.camera={width:camera.width,height:camera.height,frameRate:camera.frameRate,facingMode:camera.facingMode,...opened.request};}
    await populateCameras(token);if(!isCurrent(token))return;
    channel=new TurnChannel({initiator:config.role==='verifier',outgoingSize:config.outgoingSize,automaticTurns:true,handoffHoldMs:config.handoffHoldMs,
      onChange:()=>{if(isCurrent(token))update();},onMessage:async bytes=>{if(!isCurrent(token))return;const result=await session.receive(bytes);if(isCurrent(token))event(result.accepted?'Authentication message received.':`Authentication message ignored: ${result.reason}`);},onError:e=>fail(e,token)});
    session=new AuthSession({role:config.role,deviceId:localIdentity.deviceId,displayName:config.displayName,cryptoClient,repetitions:config.repetitions,timeoutMs:60*60*1000,
      send:bytes=>channel.enqueue(bytes),onChange:state=>{if(!isCurrent(token))return;const previous=authState?.state;authState=state;if(previous!==state.state)event(state.status);autoPairing.update(session,state);update();}});
    heartbeat=setInterval(()=>{
      if(!isCurrent(token))return;
      if(captureCallback!==null&&!scanning&&performance.now()-frameWaitAt>15000){fail(Error('The camera stopped producing frames. Start again on both devices.'),token);return;}
      channel.tick();session.tick();statisticsView();
      if(['failed','timed-out'].includes(authState?.state))fail(Error(authState.error||'The exchange expired.'),token);
    },1000);
    scheduleCapture(token);update();await session.start();
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
    if(active)return;error();try{config=readSettings();}catch(e){error(e.message);$('settings').open=true;return;}
    localIdentity.displayName=config.displayName;saveIdentity();showLocalName();$('peerName').hidden=true;$('simulationComparison').hidden=true;setText('inferenceLabel','Latest inference');
    active=true;const token=++epoch;startedAt=performance.now();authState=null;session=null;channel=null;corrector=null;cryptoClient=null;
    events=[];operations=Object.create(null);renderCache=[];shown=null;rendered=[];lastInference=null;acceptedCount=0;scanCount=0;mainThreadSkipped=0;probeGate.reset();
    lastCapture=-Infinity;lastPresented=0;lastFeedbackAt=-Infinity;scanning=false;packetTelemetry=null;lastAnnouncement='';
    report={format:'neural-auth-demo-report',version:1,kind:mode,started:new Date().toISOString(),settings:{...config,cameraId:undefined},completion:null};
    $('settings').open=false;$('result').hidden=true;$('transfer').hidden=true;$('matrixZone').dataset.visible='false';$('matrixPlaceholder').hidden=false;
    $('app').dataset.state='active';buttons();statisticsView();
    try{if(mode==='camera')await startCamera(token);else await startSimulation(token);}catch(e){if(isCurrent(token))fail(e,token);}
  }

  function finish(reason='Finished by the user.') {
    if(!active)return;
    const savedAuth=authState,savedTransport=channel?.snapshot(),wasComplete=authState?.resultDelivered||authState?.peerReported;
    // Invalidate callbacks before closing workers, channels or camera tracks.
    active=false;epoch++;simulationAbort?.abort();simulationAbort=null;autoPairing?.close();autoPairing=null;
    clearTimeout(displayTimer);clearInterval(heartbeat);
    if(captureCallback!==null)$('video').cancelVideoFrameCallback(captureCallback);captureCallback=null;
    channel?.close();session?.close(reason);cryptoClient?.destroy();corrector?.cancel();
    stream?.getTracks().forEach(track=>track.stop());stream=null;$('video').srcObject=null;$('cameraPlaceholder').hidden=false;
    $('overlay').getContext('2d').clearRect(0,0,$('overlay').width,$('overlay').height);
    if(report){report.finished=new Date().toISOString();report.elapsedMs=performance.now()-startedAt;report.stopReason=reason;if(mode==='camera'){report.authentication=savedAuth;report.transport=savedTransport;}}
    shown=null;rendered=[];renderCache=[];scanning=false;probeGate.reset();authState=savedAuth;event(reason);statisticsView();
    $('matrixZone').dataset.visible='false';$('matrixPlaceholder').hidden=false;setText('matrixLabel','Exchange finished');setText('cameraStatus','Camera is off');
    if(!wasComplete){setText('title',mode==='camera'?'Exchange stopped.':'Simulation stopped.');setText('instruction','You can start a fresh exchange when you are ready.');setText('simulationStatus',reason);}
    $('app').dataset.state=wasComplete?'complete':'idle';buttons();
  }

  function fail(cause,token=epoch) {
    if(!isCurrent(token))return;const message=session?.snapshot().error||cause?.message||String(cause);finish(message);error(message);
  }

  function download() {
    if(!report)return;
    const value=mode==='simulation'?report:{...report,elapsedMs:report.elapsedMs??performance.now()-startedAt,
      ...(config.statistics?{packetTelemetry:packetTelemetry?.snapshot()??null,operations:timingSummary(operations),captureAttempts:scanCount,mainThreadSkipped,transportAcceptedMatrices:acceptedCount,events:[...events]}:{})};
    const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),anchor=document.createElement('a');
    anchor.href=url;anchor.download=`neural-key-${report.kind}-${new Date().toISOString().replace(/[:.]/g,'-')}.json`;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }

  function setMode(next) {
    if(active||next===mode)return;mode=next;report=null;config=null;authState=null;error();
    $('cameraTab').setAttribute('aria-pressed',String(next==='camera'));$('simulationTab').setAttribute('aria-pressed',String(next==='simulation'));
    $('cameraPanel').hidden=next!=='camera';$('simulationPanel').hidden=next!=='simulation';showIdle();
  }

  $('cameraTab').addEventListener('click',()=>setMode('camera'));$('simulationTab').addEventListener('click',()=>setMode('simulation'));
  $('start').addEventListener('click',()=>void start());$('stop').addEventListener('click',()=>finish(authState?.peerReported||authState?.resultDelivered?'Finished by the user.':'Cancelled by the user.'));
  $('downloadReport').addEventListener('click',download);
  $('displayName').addEventListener('change',()=>{try{localIdentity.displayName=normalizeDisplayName($('displayName').value);$('displayName').value=localIdentity.displayName;saveIdentity();showLocalName();}catch{/* Start displays input validation errors. */}});
  for(const id of ['role','outgoingSize'])$(id).addEventListener('change',()=>{$('settingsHint').replaceChildren(document.createTextNode(`${$('role').value==='verifier'?'Verifier':'Key holder'} · G${$('outgoingSize').value} `),Object.assign(document.createElement('b'),{textContent:'⌄'}));});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&active)finish('Stopped because the page became hidden. Keep both pages visible during an exchange.');});
  window.addEventListener('pagehide',()=>finish('Page closed.'));
  if(!supported)error('Camera exchange requires trusted HTTPS or localhost, camera access and browser workers. You can still try the local simulation.');
  void populateCameras().catch(()=>{});showIdle();
}
