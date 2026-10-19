import {renderFrame, sameBytes} from '../format.js';
import {DecoderClient} from '../decoder-client.js';
import {createSender, Receiver} from './session.js';
import {decodePacket, sha256, LIMITS} from './protocol.js';
import {CONTROL_PROFILE, normalizeProfile, getChunkBytes, packetToFrames} from './optical.js';
import {diagnosticPayload} from './payload.js';

const $ = id => document.getElementById(id);
const number = n => new Intl.NumberFormat().format(n);
const terminal = new Set(['delivered', 'failed', 'timed-out', 'cancelled']);
const DECODER_OPTIONS=Object.freeze({streaming:true,filter:'robust',window:5,overhead:2,minPhases:4,recovery:'erasures',spatial:'cross',settleMs:300,preview:false});
const state = {mode:'simulation', generation:0, active:false, chain:Promise.resolve(), role:null, sender:null, receiver:null,
  worker:null, decoder:null, stream:null, displayTimer:null, tickTimer:null, captureTimer:null, elapsedTimer:null,
  frames:[], phase:0, currentPacket:null, pendingPacket:null, outgoingKind:null, delivery:null, report:null,
  started:0, endedAt:null, completedAt:null, senderDeadline:null, stopReason:null, lastError:null, options:null, camera:null, captures:0, validPackets:0, ignoredPackets:0, events:[], lastSnapshot:null};

function fail(message) {state.lastError=message||null;$('linkError').textContent=message; $('linkError').hidden=!message;}
function status(title, detail, kind='') {
  $('statusTitle').textContent=title; $('statusDetail').textContent=detail; $('statusLight').dataset.state=kind;
}
function integer(id, low, high) {
  const value=Number($(id).value);
  if (!Number.isSafeInteger(value)||value<low||value>high) throw Error(`Check ${$(id).closest('label')?.firstChild.textContent.trim()||id}: use ${low}–${high}.`);
  return value;
}
function decimal(id, low, high) {
  const value=Number($(id).value);
  if (!Number.isFinite(value)||value<low||value>high) throw Error(`Check ${$(id).closest('label')?.firstChild.textContent.trim()||id}: use ${low}–${high}.`);
  return value;
}
function options(receiver=false) {
  if(receiver)return {timeoutMs:integer('timeoutMinutes',1,240)*60000};
  const profile=normalizeProfile({size:integer('matrixSize',64,256),colors:16,gridBlock:8,parity:32,
    phases:integer('phaseCount',4,16),permute:true,holdMs:Math.round(decimal('holdSeconds',.25,10)*1000)});
  // The sender starts its retry clock after POLL, after every DATA cycle has
  // been displayed. The window's transmission time is not part of this wait.
  const windowSize=integer('windowSize',1,8),
    retryMs=integer('retryMargin',10,600)*1000, timeoutMs=integer('timeoutMinutes',1,240)*60000;
  if(timeoutMs<=retryMs) throw Error(`Session limit must exceed the ${Math.ceil(retryMs/1000)}-second retry interval.`);
  return {profile,windowSize,chunkBytes:getChunkBytes(profile),retryMs,maxRetries:integer('retryLimit',1,50),timeoutMs};
}
function refreshProfile() {
  try {
    const o=options(), bytes=$('payloadSource').value==='file'?($('payloadFile').files[0]?.size??0):Number($('byteCount').value||165000),count=Math.max(1,Math.ceil(bytes/o.chunkBytes));
    const minimumMs=count*o.profile.phases*o.profile.holdMs;
    $('profileSummary').textContent=`${number(o.chunkBytes)} content B/chunk · ${number(count)} chunks for the test message · ${(o.profile.phases*o.profile.holdMs/1000).toFixed(1)} s/data cycle · ${(minimumMs/60000).toFixed(1)} min minimum data time, before acknowledgments and retries. ACK retry ${(o.retryMs/1000).toFixed(0)} s after the window finishes. Eight-cell references, 32 parity bytes; controls: 32 × 32 / four colors.${minimumMs>=o.timeoutMs?' Increase the session limit: the data cycles alone exceed it.':''}`;
  } catch(error) {$('profileSummary').textContent=error.message;}
}
function refreshControls() {
  const live=state.mode==='optical', receive=live&&$('deviceRole').value==='receiver';
  $('simulationSettings').hidden=live; $('opticalSettings').hidden=!live; $('opticalPanel').hidden=!live;
  $('payloadFields').hidden=receive; $('payloadHint').hidden=receive;
  $('profileFields').hidden=receive;
  $('profileSummary').hidden=receive;
  $('payloadFile').closest('label').hidden=$('payloadSource').value!=='file';
  $('byteCountLabel').hidden=$('payloadSource').value==='file';
  $('modeLabel').textContent=live?'OPTICAL TRANSFER':'LOCAL SIMULATION';
  $('byteCount').max=String(live?LIMITS.maxMessageBytes:1024*1024);
  $('payloadHint').textContent=`A repeatable 165,000-byte message exercises proof-sized transfers. ${live?'Optical messages allow up to 8 MiB.':'Simulation is limited to 1 MiB to bound browser work.'} Files stay in this browser.`;
  $('startLink').textContent=live?`Start ${receive?'receiver':'sender'} →`:'Run simulation →';
  $('startLink').disabled=state.active;
  $('stopLink').disabled=!state.active;
  for(const node of document.querySelectorAll('.link-setup input,.link-setup select')) node.disabled=state.active;
  $('simulationTab').classList.toggle('selected',!live); $('opticalTab').classList.toggle('selected',live);
  $('simulationTab').setAttribute('aria-pressed',String(!live)); $('opticalTab').setAttribute('aria-pressed',String(live));
}
function event(message) {state.events.push({atMs:Math.round(performance.now()-state.started),message}); if(state.events.length>60)state.events.shift();}
function baseReport() {
  return {schema:'neural-key-link-report-v1',mode:state.mode,createdAt:new Date().toISOString(),role:state.role,
    status:state.stopReason?'stopped':(state.sender?.snapshot().state??state.receiver?.snapshot().state??'preparing'),stopReason:state.stopReason,lastError:state.lastError,
    options:state.options,decoderSettings:DECODER_OPTIONS,elapsedMs:Math.round((state.completedAt??state.endedAt??performance.now())-state.started),
    acknowledgmentWatchMs:state.role==='receiver'&&state.completedAt!==null?Math.round((state.endedAt??performance.now())-state.completedAt):0,sender:state.sender?.snapshot()??null,
    receiver:state.receiver?.snapshot()??null,camera:state.camera,receivedProfiles:state.receivedProfiles??{},captures:state.captures,validPackets:state.validPackets,
    ignoredPackets:state.ignoredPackets,delivery:state.delivery?{bytes:state.delivery.bytes.length,digest:state.delivery.digest}:null,events:state.events.slice()};
}
function reportWithNotes() {
  return {...(state.mode==='optical'&&state.started?baseReport():state.report),stopReason:state.stopReason,lastError:state.lastError,notes:$('testNotes').value,
    browser:{userAgent:navigator.userAgent,language:navigator.language,devicePixelRatio:window.devicePixelRatio}};
}
function renderReport() {
  $('downloadReport').disabled=!state.report&&!state.started;
  $('reportPreview').textContent=JSON.stringify(reportWithNotes(),null,2);
}
function resetResults() {
  state.delivery=null;state.report=null;state.captures=0;state.validPackets=0;state.ignoredPackets=0;state.events=[];state.lastSnapshot=null;state.receivedProfiles={};
  $('downloadPayload').disabled=true; $('downloadReport').disabled=true;
  $('senderDigest').textContent='—';$('receivedDigest').textContent='—';$('resultLabel').textContent='IN PROGRESS';
  $('resultText').textContent='Waiting for the complete message and its SHA-256 integrity check.';
  $('reportPreview').textContent='Transfer in progress.'; $('integrityMetric').textContent='Waiting';
  $('byteMetric').textContent='0 B';$('chunkMetric').textContent='—';$('retryMetric').textContent='0';$('transferProgress').value=0;
}
function clearDisplay() {
  state.frames=[];state.currentPacket=null;state.pendingPacket=null;state.phase=0;
  $('linkMatrix').hidden=true;$('matrixPlaceholder').hidden=false;
  $('packetLabel').textContent='NO OUTGOING PACKET';$('phaseLabel').textContent='—';
}
function stopCamera() {
  clearInterval(state.captureTimer);state.captureTimer=null;
  state.stream?.getTracks().forEach(track=>track.stop());state.stream=null;
  if(state.decoder){state.decoder.fail(Error('Camera stopped.'));state.decoder=null;}
  $('cameraVideo').pause();$('cameraVideo').srcObject=null;$('cameraVideo').hidden=true;$('cameraPlaceholder').hidden=false;
  $('cameraLight').className='';$('cameraOverlay').getContext('2d').clearRect(0,0,$('cameraOverlay').width,$('cameraOverlay').height);
}
function clearTimers() {
  clearTimeout(state.displayTimer);clearInterval(state.tickTimer);clearInterval(state.elapsedTimer);
  state.displayTimer=null;state.tickTimer=null;state.elapsedTimer=null;
}
function stop(message='Stopped. Start again to create a new transfer.', sendCancel=true) {
  const wasActive=state.active, sender=state.sender,receiver=state.receiver,oldChain=state.chain;
  const token=++state.generation;
  if(state.started&&state.endedAt===null)state.endedAt=performance.now();
  if(wasActive)state.stopReason=message;
  state.active=false;state.worker?.terminate();state.worker=null;clearTimers();stopCamera();
  state.report=state.mode==='optical'&&state.started?baseReport():state.report??(wasActive?{schema:'neural-key-link-report-v1',mode:state.mode,status:'stopped',options:state.options,elapsedMs:Math.round(state.endedAt-state.started),stopReason:message}:null);
  clearDisplay();refreshControls();
  if(wasActive){event('Stopped locally.');status('Transfer stopped',message);$('decodeStatus').textContent='Camera stopped';$('resultLabel').textContent=state.delivery?'RECEIVED':'STOPPED';}
  if(sendCancel&&(sender||receiver)) {
    state.chain=oldChain.catch(()=>{}).then(async()=>{
      const bytes=sender?sender.cancel(performance.now()):receiver.cancel(performance.now());
      if(!bytes||token!==state.generation)return;
      const frames=await packetToFrames(bytes,CONTROL_PROFILE);
      if(token!==state.generation)return;
      state.frames=frames;state.phase=0;state.outgoingKind='cancel';paintFrame();
      $('displayDetail').textContent='Cancellation is displayed for the other device. Camera and timers are stopped.';
    }).catch(error=>{if(token===state.generation)fail(error.message);});
  }
  if(state.report||state.started)renderReport();
}
function serial(task, token=state.generation) {
  const run=state.chain.catch(()=>{}).then(async()=>{if(!state.active||token!==state.generation)return;await task();});
  state.chain=run.catch(error=>{if(token===state.generation&&state.active){stop('An operation failed. Check the error and restart.',false);fail(error.message);}});
  return state.chain;
}
async function payload(token) {
  if($('payloadSource').value==='file') {
    const file=$('payloadFile').files[0];if(!file)throw Error('Choose a binary file first.');
    const limit=state.mode==='simulation'?1024*1024:LIMITS.maxMessageBytes;
    if(file.size>limit)throw Error(`The file exceeds the ${state.mode==='simulation'?'1 MiB simulation':'8 MiB optical message'} limit.`);
    const bytes=new Uint8Array(await file.arrayBuffer());
    if(token!==state.generation)return null;return bytes;
  }
  const length=integer('byteCount',1,state.mode==='simulation'?1024*1024:LIMITS.maxMessageBytes);
  return diagnosticPayload(length,1234);
}
function startClock(token) {
  state.elapsedTimer=setInterval(()=>{if(token===state.generation)$('elapsed').textContent=`${((performance.now()-state.started)/1000).toFixed(1)} s`;},200);
}
async function start() {
  if(state.active)return;
  const token=++state.generation;
  try {
    fail('');state.role=state.mode==='optical'?$('deviceRole').value:'simulation';state.options=options(state.role==='receiver');
    state.active=true;state.sender=null;state.receiver=null;state.started=performance.now();state.chain=Promise.resolve();
    state.completedAt=null;state.endedAt=null;state.senderDeadline=null;state.stopReason=null;state.camera=null;
    clearDisplay();resetResults();refreshControls();startClock(token);
    status('Preparing transfer','Reading local content and preparing the transport.','busy');
    const bytes=state.role==='receiver'?null:await payload(token);
    if(token!==state.generation||!state.active)return;
    if(state.mode==='simulation') {startSimulation(bytes,token);return;}
    if(state.role==='sender') {
      const now=performance.now(),sender=await createSender(bytes,{...state.options,messageType:'raw',now});
      if(token!==state.generation||!state.active)return;
      state.sender=sender;state.senderDeadline=now+state.options.timeoutMs;
    } else state.receiver=new Receiver({timeoutMs:state.options.timeoutMs,completedTtlMs:state.options.timeoutMs,maxCompleted:1});
    if(token!==state.generation||!state.active)return;
    event(`Started optical ${state.role}.`);
    await openCamera(token);
    if(token!==state.generation||!state.active)return;
    await serial(async()=>{await advanceDisplay(token);updateSnapshot();},token);
    let tickPending=false;
    state.tickTimer=setInterval(()=>{
      if(tickPending)return;tickPending=true;
      serial(async()=>{
        if(state.receiver){const result=state.receiver.tick(performance.now());queueOutbound(result.outbound);}
        // A long phase cycle must not postpone the sender's absolute deadline.
        if(state.sender&&performance.now()>=state.senderDeadline&&!terminal.has(state.sender.snapshot().state)) {
          const cancelled=state.sender.next(performance.now());if(cancelled)state.pendingPacket=cancelled;
        }
        updateSnapshot();
        if(!state.frames.length&&state.active)await advanceDisplay(token);
      },token).finally(()=>{tickPending=false;});
    },1000);
  } catch(error) {
    if(token!==state.generation)return;
    stop('Could not start. Check the error and try again.',false);fail(error.message);
  }
}
function startSimulation(bytes, token) {
  const opts={...state.options,bytes:bytes.length,payload:bytes,source:$('payloadSource').value,dropRate:decimal('dropRate',0,.95),duplicateRate:decimal('duplicateRate',0,.95),
    corruptRate:decimal('corruptRate',0,.95),seed:integer('simulationSeed',1,4294967295)};
  state.options={...state.options,bytes:bytes.length,dropRate:opts.dropRate,duplicateRate:opts.duplicateRate,corruptRate:opts.corruptRate,seed:opts.seed};
  state.worker=new Worker(new URL('./simulation-worker.js',import.meta.url),{type:'module'});
  state.worker.onmessage=({data})=>{
    if(token!==state.generation||data.id!==token)return;
    if(data.type==='progress') {
      const p=data.progress;
      if(p){$('transferProgress').value=p.sender.progress;$('statusDetail').textContent=`${number(p.sender.ackedBytes)} bytes acknowledged · ${(p.simulatedMs/60000).toFixed(1)} simulated minutes · ${(p.elapsedMs/1000).toFixed(2)} s worker time.`;}
      return;
    }
    if(data.type==='error'){stop('Simulation failed.',false);fail(typeof data.error==='string'?data.error:data.error?.message||'Simulation failed.');return;}
    if(data.type!=='result')return;
    state.worker.terminate();state.worker=null;state.active=false;clearTimers();state.report=data.report;refreshControls();
    const r=data.report,passed=r.exact&&r.confirmationReceived;
    $('transferProgress').value=passed?1:(r.sender?.progress??0);
    $('byteMetric').textContent=`${number(r.receiver.receivedBytes)} / ${number(bytes.length)} B`;
    $('chunkMetric').textContent=`${r.sender.ackedChunks} / ${r.sender.chunkCount}`;
    $('retryMetric').textContent=number(r.sender.retries);
    $('integrityMetric').textContent=r.exact?'Exact match':'Incomplete';
    $('senderDigest').textContent=r.expectedDigest;
    $('receivedDigest').textContent=r.receivedDigest??'—';
    $('resultLabel').textContent=passed?'EXACT BYTES · CONFIRMED':r.exact?'EXACT BYTES · UNCONFIRMED':'NOT DELIVERED';
    $('resultText').textContent=passed?'The simulated receiver recovered the original bytes and the sender received its final acknowledgment.':r.exact?'The receiver recovered the exact bytes, but the sender did not receive the final acknowledgment before its retry or time limit.':(r.sender.error??'The transfer did not finish within its retry or time limit. Inspect the report.');
    status(passed?'Simulation passed':'Simulation did not complete',`${(r.simulatedMs/60000).toFixed(1)} simulated minutes · ${(r.elapsedMs/1000).toFixed(2)} s actual worker time. Camera performance must be measured on devices.`,passed?'success':'error');
    // Worker reports exclude message bytes. A successful exact comparison makes
    // the original local byte copy a useful downloadable test fixture.
    if(r.exact){state.delivery={bytes,digest:r.receivedDigest};$('downloadPayload').disabled=false;}
    renderReport();
  };
  state.worker.onerror=error=>{if(token!==state.generation)return;stop('Simulation worker failed.',false);fail(error.message||'Simulation worker unavailable.');};
  state.worker.postMessage({type:'run',id:token,options:opts});
  status('Simulating transport','The worker is exercising multipart delivery with deterministic packet faults.','busy');
}
function queueOutbound(packets=[]) {
  // ACKs describe current state. Retain only the freshest one; no packet backlog
  // can grow while a camera is looking away from the receiver's display.
  for(const packet of packets)state.pendingPacket=packet;
}
function paintFrame() {
  const frame=state.frames[state.phase];if(!frame)return;
  const pixels=Math.min(4096,Math.max(1024,Math.ceil(frame.size*6.4/256)*256)),image=renderFrame(frame,pixels),canvas=$('linkMatrix');
  canvas.width=image.width;canvas.height=image.height;canvas.getContext('2d').putImageData(new ImageData(image.data,image.width,image.height),0,0);
  $('matrixPlaceholder').hidden=true;canvas.hidden=false;
  $('packetLabel').textContent=`${state.outgoingKind.toUpperCase()} · ${frame.size} × ${frame.size} / ${frame.colors}`;
  $('phaseLabel').textContent=`PHASE ${state.phase+1} / ${state.frames.length}`;
}
async function setPacket(bytes,token) {
  if(state.currentPacket&&sameBytes(state.currentPacket,bytes)){state.phase=0;return;}
  const kind=decodePacket(bytes).kind,profile=kind==='data'?state.options.profile:CONTROL_PROFILE;
  const frames=await packetToFrames(bytes,profile);
  if(token!==state.generation||!state.active)return;
  state.frames=frames;state.currentPacket=bytes.slice();state.phase=0;state.outgoingKind=kind;
}
async function advanceDisplay(token) {
  if(token!==state.generation||!state.active)return;
  if(state.phase>=state.frames.length||!state.frames.length) {
    const next=state.sender?state.sender.next(performance.now()):state.pendingPacket;
    if(state.receiver)state.pendingPacket=null;
    if(next)await setPacket(next,token);
    else state.phase=0;
  }
  if(token!==state.generation||!state.active)return;
  updateSnapshot();if(!state.active)return;
  if(!state.frames.length)return;
  paintFrame();
  clearTimeout(state.displayTimer);
  state.displayTimer=setTimeout(()=>serial(async()=>{state.phase++;await advanceDisplay(token);},token),state.frames[state.phase].holdMs);
}
async function acceptPacket(bytes,token,info) {
  if(token!==state.generation||!state.active)return;
  let decoded;try{decoded=decodePacket(bytes);}catch{state.ignoredPackets++;return;}
  if(decoded.messageType!=='raw'){state.ignoredPackets++;return;}
  if(state.delivery&&(decoded.sessionId!==state.delivery.sessionId||decoded.messageId!==state.delivery.messageId)){state.ignoredPackets++;return;}
  const now=performance.now(),result=state.sender?await state.sender.receive(bytes,now):await state.receiver.receive(bytes,now);
  if(token!==state.generation||!state.active)return;
  queueOutbound(result.outbound);
  if(!result.accepted){state.ignoredPackets++;updateSnapshot();return;}
  state.validPackets++;
  if(info)state.receivedProfiles[decoded.kind==='data'?'data':'control']=Object.fromEntries(
    ['size','colors','gridBlock','parity','phases','permute','holdMs'].map(key=>[key,info[key]]));
  if(result.delivery) {
    const receivedDigest=await sha256(result.delivery.bytes);
    if(token!==state.generation||!state.active)return;
    if(receivedDigest!==result.delivery.digest)throw Error('The received bytes disagree with the verified digest.');
    state.delivery=result.delivery;state.completedAt=performance.now();$('senderDigest').textContent=result.delivery.digest;$('receivedDigest').textContent=receivedDigest;
    $('downloadPayload').disabled=false;$('resultLabel').textContent='RECEIVED · DIGEST MATCH';
    $('resultText').textContent=`Recovered ${number(result.delivery.bytes.length)} bytes and matched the sender’s SHA-256. Keep the acknowledgment visible until the sender reports delivery.`;
    event('Received complete message; SHA-256 matched.');
  }
  if(!state.frames.length)await advanceDisplay(token);
  updateSnapshot();
}
function updateSnapshot() {
  const snapshot=state.sender?.snapshot()??state.receiver?.snapshot();if(!snapshot)return;
  state.lastSnapshot=snapshot;
  const receive=!!state.receiver,done=receive?snapshot.receivedBytes:snapshot.ackedBytes,chunks=receive?snapshot.receivedChunks:snapshot.ackedChunks;
  $('byteMetric').textContent=`${number(done??0)} / ${number(snapshot.totalBytes??0)} B`;
  $('chunkMetric').textContent=`${number(chunks??0)} / ${number(snapshot.chunkCount??0)}`;
  $('retryMetric').textContent=number(snapshot.retries??0);$('transferProgress').value=snapshot.progress??0;
  if(snapshot.digest)$('senderDigest').textContent=snapshot.digest;
  if(snapshot.state==='delivered') {
    $('integrityMetric').textContent=receive?'SHA-256 match':'Acknowledged';
    status(receive?'Message received':'Delivery acknowledged',receive?'Full-message digest passed. Keep this acknowledgment visible until the sender finishes.':'The receiver acknowledged the complete message and its digest.','success');
    if(!receive){$('resultLabel').textContent='DELIVERY ACKNOWLEDGED';$('resultText').textContent='The receiver acknowledged the full message and its expected digest. This confirms transport completion, not peer identity.';
      state.completedAt??=performance.now();state.endedAt??=performance.now();state.active=false;clearTimers();stopCamera();refreshControls();}
    state.report=baseReport();renderReport();return;
  }
  if(terminal.has(snapshot.state)) {
    const message=snapshot.error||`Session ${snapshot.state}.`;
    state.endedAt??=performance.now();state.report=baseReport();state.active=false;clearTimers();stopCamera();refreshControls();
    const candidate=state.pendingPacket??(state.outgoingKind==='cancel'?state.currentPacket:null),token=state.generation;
    const cancellation=candidate&&decodePacket(candidate).kind==='cancel'?candidate:null;
    clearDisplay();
    if(cancellation)packetToFrames(cancellation,CONTROL_PROFILE).then(frames=>{
      if(token!==state.generation)return;
      state.frames=frames;state.phase=0;state.outgoingKind='cancel';paintFrame();
      $('displayDetail').textContent='Cancellation is displayed for the other device. Camera and timers are stopped.';
    }).catch(error=>{if(token===state.generation)fail(error.message);});
    status('Transfer ended',message,'error');$('integrityMetric').textContent='Incomplete';$('resultLabel').textContent=snapshot.state.toUpperCase();$('resultText').textContent=message;renderReport();return;
  }
  const descriptions={idle:['Receiver ready','Point this camera at the sender’s matrix. An acknowledgment will appear here after the introduction.'],
    'awaiting-start':['Waiting for receiver','Keep the introduction visible while your camera reads the receiver’s acknowledgment.'],
    sending:['Sending data','Displaying complete palette cycles, then asking which chunks need another attempt.'],
    'awaiting-ack':['Waiting for acknowledgment','Point the sender’s camera at the receiver’s small acknowledgment matrix.'],
    receiving:['Receiving data','Chunks are stored independently. Missing chunks will be requested again.']};
  const [title,detail]=descriptions[snapshot.state]??['Transferring',snapshot.state];status(title,detail,'busy');
  if($('reportDetails').open){state.report=baseReport();renderReport();}
}
function drawCorners(result,width,height) {
  const canvas=$('cameraOverlay');canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d');
  if(result.corners){ctx.strokeStyle='#c9eb9b';ctx.lineWidth=Math.max(2,width/400);ctx.beginPath();result.corners.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();}
}
async function openCamera(token) {
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error('Camera access requires HTTPS or localhost in a browser with camera support.');
  const [width,height]=$('cameraResolution').value.split('x').map(Number),selected=$('cameraSelect').value;
  const constraints={width:{ideal:width,max:4096},height:{ideal:height,max:4096},frameRate:{ideal:15,max:30}};
  if(selected==='user'||selected==='environment')constraints.facingMode={ideal:selected};else constraints.deviceId={exact:selected};
  status('Opening camera','Allow access, then align the other device’s entire matrix.','busy');
  const stream=await navigator.mediaDevices.getUserMedia({video:constraints,audio:false});
  if(token!==state.generation||!state.active){stream.getTracks().forEach(t=>t.stop());return;}
  state.stream=stream;const video=$('cameraVideo');video.srcObject=stream;await video.play();
  if(token!==state.generation||!state.active){stream.getTracks().forEach(t=>t.stop());return;}
  video.hidden=false;$('cameraPlaceholder').hidden=true;$('cameraLight').className='active';
  const actual=stream.getVideoTracks()[0].getSettings();
  state.camera={requested:{width,height},actual:{width:actual.width,height:actual.height,frameRate:actual.frameRate}};
  $('cameraNote').textContent=`Requested ${width} × ${height} · actual ${video.videoWidth} × ${video.videoHeight}.`;
  try {
    const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
    if(token!==state.generation||!state.active)return;
    if(devices.length)$('cameraSelect').replaceChildren(...devices.map((device,index)=>new Option(device.label||`Camera ${index+1}`,device.deviceId,false,device.deviceId===actual.deviceId)));
  } catch {/* Camera enumeration is optional. */}
  state.decoder=new DecoderClient();const decoder=state.decoder,scratch=document.createElement('canvas'),ctx=scratch.getContext('2d',{willReadFrequently:true});let lastTime=-1,capturePending=false;
  state.captureTimer=setInterval(async()=>{
    if(token!==state.generation||!state.active||capturePending||decoder.busy||video.readyState<2||document.hidden||video.currentTime===lastTime)return;
    capturePending=true;
    lastTime=video.currentTime;scratch.width=video.videoWidth;scratch.height=video.videoHeight;
    try {
      ctx.drawImage(video,0,0,scratch.width,scratch.height);const raw=ctx.getImageData(0,0,scratch.width,scratch.height);
      const result=await decoder.decode({width:raw.width,height:raw.height,data:raw.data},DECODER_OPTIONS);
      if(token!==state.generation||!state.active)return;
      state.captures++;drawCorners(result,scratch.width,scratch.height);
      $('decodeStatus').textContent=result.ok?'Packet decoded · CRC passed':result.error||`Reading ${result.stage??'camera'}…`;
      if(result.ok&&result.info?.count===1&&result.info?.content?.type==='raw'&&result.chunk)await serial(()=>acceptPacket(result.chunk,token,result.info),token);
    } catch(error){if(token===state.generation&&state.active){$('decodeStatus').textContent=error.message;fail(`Camera decoding: ${error.message}`);}}
    finally{capturePending=false;}
  },250);
}
function download(bytes,type,name) {
  const url=URL.createObjectURL(new Blob([bytes],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('startLink').addEventListener('click',start);
$('stopLink').addEventListener('click',()=>stop());
for(const [id,mode] of [['simulationTab','simulation'],['opticalTab','optical']])$(id).addEventListener('click',()=>{
  if(state.mode===mode)return;stop('Mode changed. Start a new transfer.',false);state.mode=mode;state.sender=null;state.receiver=null;state.started=0;state.camera=null;resetResults();clearDisplay();refreshControls();status('Ready to transfer',mode==='simulation'?'Run a deterministic fault simulation.':'Start the receiver first, then the sender.');
});
$('payloadSource').addEventListener('change',()=>{refreshControls();refreshProfile();});$('payloadFile').addEventListener('change',refreshProfile);$('deviceRole').addEventListener('change',refreshControls);
for(const id of ['matrixSize','phaseCount','holdSeconds','windowSize','byteCount','retryMargin','retryLimit','timeoutMinutes'])$(id).addEventListener('input',refreshProfile);
$('downloadPayload').addEventListener('click',()=>{if(state.delivery)download(state.delivery.bytes,'application/octet-stream','optical-link-received.bin');});
$('downloadReport').addEventListener('click',()=>{renderReport();download(JSON.stringify(reportWithNotes(),null,2),'application/json','optical-link-report.json');});
$('testNotes').addEventListener('input',()=>{if(state.report||state.started)renderReport();});
$('expandMatrix').addEventListener('click',async()=>{
  const stage=$('linkStage');
  if(document.fullscreenElement){await document.exitFullscreen();return;}
  if(stage.classList.contains('expanded')){stage.classList.remove('expanded');return;}
  try{await stage.requestFullscreen();}catch{stage.classList.add('expanded');}
});
document.addEventListener('keydown',event=>{if(event.key==='Escape')$('linkStage').classList.remove('expanded');});
// Double tapping the display exits the CSS fallback on mobile browsers.
$('linkStage').addEventListener('dblclick',()=>$('linkStage').classList.remove('expanded'));
document.addEventListener('visibilitychange',()=>{if(document.hidden&&state.active)stop('The page was hidden. Restart the transfer with both devices in the foreground.',false);});
window.addEventListener('pagehide',()=>stop('Page closed.',false));
refreshControls();refreshProfile();
if(!window.isSecureContext||!globalThis.crypto?.subtle)fail('Open this app over HTTPS or localhost. SHA-256 and camera access require a secure context.');
