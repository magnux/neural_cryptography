import {AuthSession} from '../auth-core/session.js';
import {CryptoClient} from '../crypto-lab/client.js';
import {CorrectorClient} from '../corrector-lab/client.js';
import {sameBytes} from '../format.js';
import {renderNeuralQR} from '../neural-qr/layout.js';
import {openCaptureCamera,captureDimensions} from '../dataset/camera.js';
import {CameraControls,acquisitionDiagnostics} from '../dataset/camera-controls.js';
import {deviceProfile,transportProfile,transferMetrics} from './profiles.js';
import {decodePacket} from '../link/protocol.js';
import {TurnChannel} from './channel.js';
import {packetFrames,unpackOptical} from './optical.js';
import {captureFeedback} from './feedback.js';

const $=id=>document.getElementById(id);
const supported=isSecureContext&&Boolean(globalThis.crypto?.subtle)&&typeof Worker==='function'&&Boolean(navigator.mediaDevices?.getUserMedia);
let active=false,epoch=0,stream=null,corrector=null,cryptoClient=null,session=null,channel=null,config=null;
let frameCallback=null,displayTimer=null,heartbeat=null,lastCapture=-Infinity,lastPresented=0,scanning=false,frameWaitStarted=0;
let shown=null,rendered=[],phaseStarted=0,phaseIndex=-1,authState=null,report=null,startedAt=0;
const events=[],transfers=new Map();
let lastIncomingSize=null;
let cameraControls=null,changingCameraControls=false;
const int=(value,min,max,label)=>{const n=Number(value);if(!Number.isSafeInteger(n)||n<min||n>max)throw Error(`${label} must be ${min}–${max}.`);return n;};
function settings(){
  const role=$('role').value,backend=$('backend').value,repetitions=Number($('repetitions').value),resolution=$('resolution').value;
  if(!['verifier','prover'].includes(role)||!['auto','wasm','webgpu'].includes(backend)||![32,64,110,160,219].includes(repetitions)||!['480','720','1080','1440','2160','max'].includes(resolution))throw Error('Unsupported test settings.');
  return {role,backend,repetitions,resolution,...transportProfile($('deviceProfile').value,$('outgoingSize').value),
    phaseMs:int($('phaseMs').value,50,10000,'Phase interval'),
    captureMs:int($('captureMs').value,50,2000,'Capture interval'),cameraId:$('camera').value};
}
function populateProfiles(){
  const profile=deviceProfile($('deviceProfile').value);
  for(const id of ['outgoingSize','nextGeometry']){
    $(id).replaceChildren(...profile.outgoingSizes.map(size=>new Option(`G${size}${size===profile.defaultOutgoingSize?' · maximum':''}`,String(size))));
    $(id).value=String(profile.defaultOutgoingSize);
  }
  $('directionProfile').textContent=`${profile.name}: receive up to G${profile.incomingMaxSize} · G32 controls`;
}
function observeTransfers(link){
  if(!report||!link)return;
  const now=performance.now();
  for(const [direction,row] of [['sent',link.outgoing],['received',link.incoming]]){
    if(!row?.messageId||transfers.has(row.messageId))continue;
    if(transfers.size>=64)break;
    transfers.set(row.messageId,{direction,grid:direction==='sent'?link.outgoingSize:lastIncomingSize,startedMs:now});
  }
  const row=link.lastTransfer,entry=row?.messageId?transfers.get(row.messageId):null;
  if(row?.state==='delivered'&&entry&&!entry.finished){
    entry.finished=true;
    const completed={direction:entry.direction,grid:entry.grid,...transferMetrics(row.totalBytes,now-entry.startedMs),
      timingScope:entry.direction==='sent'?'outgoing-message-start-through-final-ACK':'first-valid-START-through-message-assembly'};
    report.transfers.push(completed);
    $('goodput').textContent=`${completed.goodputBytesPerSecond.toFixed(0)} B/s`;
  }
}
function error(text=''){$('error').textContent=text;$('error').hidden=!text;}
function event(text){
  events.push({seconds:Math.round((performance.now()-startedAt)/1000),text});if(events.length>100)events.shift();
  $('eventLog').replaceChildren(...events.slice(-20).map(item=>{const li=document.createElement('li');li.textContent=`${item.seconds}s · ${item.text}`;return li;}));
}
function cameraSettings(){const v=stream?.getVideoTracks()[0]?.getSettings()??{};return {width:v.width,height:v.height,frameRate:v.frameRate,facingMode:v.facingMode};}
const controlLabels={exposureMode:'Exposure mode',exposureCompensation:'Exposure compensation (EV)',exposureTime:'Exposure time (camera units)',focusMode:'Focus mode',focusDistance:'Focus distance (camera units)'};
const modeLabels={continuous:'Auto · continuous','single-shot':'Auto · single shot',manual:'Manual'};
const controlValue=value=>typeof value==='number'?Number(value.toPrecision(5)).toString():modeLabels[value]??'Not reported';
function renderCameraControls(message=''){
  const panel=$('cameraControls'),fields=$('cameraControlFields');panel.hidden=!cameraControls||!active;
  fields.replaceChildren();if(panel.hidden)return;
  const snapshot=cameraControls.snapshot();if(report)report.cameraControls=snapshot;
  for(const key of Object.keys(controlLabels)){
    const cap=snapshot.capabilities[key];if(!cap)continue;
    const label=document.createElement('label'),title=document.createElement('span'),actual=snapshot.actual[key];
    label.htmlFor=`control-${key}`;title.textContent=controlLabels[key];label.append(title);
    let input;
    if(Array.isArray(cap)){
      input=document.createElement('select');
      if(!cap.includes(actual))input.append(new Option('Current mode not reported','',true,true));
      for(const mode of cap)input.append(new Option(modeLabels[mode],mode));
      if(cap.includes(actual))input.value=actual;
    }else{
      input=document.createElement('input');input.type='range';input.min=String(cap.min);input.max=String(cap.max);input.step=String(cap.step??'any');
      input.value=String(Number.isFinite(actual)?Math.min(cap.max,Math.max(cap.min,actual)):(cap.min+cap.max)/2);
      const values=document.createElement('output');values.htmlFor=`control-${key}`;
      values.textContent=`Camera: ${controlValue(actual)}`;label.append(values);
      input.addEventListener('input',()=>{values.textContent=`Request: ${controlValue(Number(input.value))} · camera: ${controlValue(actual)}`;});
    }
    input.id=`control-${key}`;
    const mode=key==='exposureTime'?'exposureMode':key==='focusDistance'?'focusMode':null;
    const needsManual=mode&&snapshot.capabilities[mode]?.includes('manual')&&snapshot.actual[mode]!=='manual';
    input.disabled=changingCameraControls||Boolean(needsManual);
    if(needsManual)input.title='Choose Manual mode first.';
    input.addEventListener('change',()=>void changeCameraControl(key,Array.isArray(cap)?input.value:Number(input.value)));
    label.append(input);fields.append(label);
  }
  $('cameraControlStatus').textContent=message||(fields.children.length
    ?'Only controls exposed by this camera are shown. No settings are changed automatically. Values report what the camera actually applied.'
    :'This browser/camera exposes no adjustable exposure or focus controls. Use the webcam’s own settings if available.');
}
async function changeCameraControl(key,value){
  if(!active||!cameraControls||changingCameraControls)return;
  const controller=cameraControls,token=epoch;changingCameraControls=true;renderCameraControls('Applying camera setting…');
  let message;
  try{
    await controller.apply({[key]:value});
    if(!active||token!==epoch||controller!==cameraControls)return;
    const snapshot=controller.snapshot(),actual=snapshot.actual[key];
    message=`${controlLabels[key]} · requested ${controlValue(snapshot.requested[key])}; camera reports ${controlValue(actual)}.`;
    event(message);
  }catch(e){
    if(!active||token!==epoch||controller!==cameraControls)return;
    message=`Camera setting was not applied: ${e.message||String(e)}`;
  }finally{
    if(active&&token===epoch&&controller===cameraControls){changingCameraControls=false;renderCameraControls(message);}
  }
}
function update(){
  const link=channel?.snapshot();
  $('authApp').dataset.mode=active?(link?.mode??'receive'):'idle';
  $('settingsFields').disabled=active;$('start').disabled=active||!supported;$('stop').disabled=!active;
  $('downloadReport').disabled=!report;$('nextTurn').disabled=!active||!link?.canSendNext;
  $('nextGeometry').disabled=!active||!channel||!!link?.outgoing||!!link?.preparing;
  if(link)$('nextGeometry').value=String(link.outgoingSize);
  observeTransfers(link);
  $('confirmPairing').disabled=!active||!authState?.canConfirm;
  $('proofBytes').textContent=authState?.proofBytes?`${authState.proofBytes.toLocaleString()} B`:'—';
  $('pairingPanel').hidden=!authState?.pairingCode;
  $('pairingCode').textContent=authState?.pairingCode?.match(/.{1,4}/g).join(' ')??'—';
  $('pairingNote').textContent=authState?.confirmed?'Confirmed on this device. The other device must also confirm.':'Compare both codes before confirming. A mismatch means stop and restart.';
  if(!active)return;
  $('status').textContent=authState?.status??'Preparing local model and camera…';
  const mode=link?.mode??'receive';
  $('instruction').textContent=mode==='send'
    ?'Keep this matrix visible. When the receiver shows ACK, turn the phone so this camera can read it. Each chunk waits for its ACK.'
    :mode==='ack'
      ?link?.canSendNext?'ACK ready. Let the other device read it and confirm delivery, then choose “Send next turn”.'
        :'ACK ready. Turn the phone to show this ACK to the sender, then aim back at its next matrix. Keep this page open.'
      :'Aim the camera at the other screen. All four corners and the complete matrix must be visible.';
  if(authState?.resultDelivered)$('instruction').textContent='Authentication and result delivery completed. You can save the report and stop.';
  else if(authState?.peerReported)$('instruction').textContent='The verifier reported its result. Keep this final ACK visible until the verifier confirms result delivery, then save the report and stop.';
  if(authState?.canConfirm)$('instruction').textContent+=' Compare the pairing codes and confirm on both devices.';
  const state=link?.outgoing??link?.incoming;
  $('transferStatus').textContent=link?.outgoing
    ?`Sending ${state.index<0?'message start':`chunk ${state.index+1}/${state.chunkCount}`} · ${state.ackedBytes.toLocaleString()}/${state.totalBytes.toLocaleString()} B acknowledged`
    :state?.chunkCount?`Received ${state.receivedChunks}/${state.chunkCount} chunks · ${state.receivedBytes.toLocaleString()} B${link?.queued?' · next message ready':''}`:'Waiting for a message';
  if(authState?.result!==null&&authState?.result!==undefined){
    $('result').textContent=authState.peerReported?(authState.result?'Peer reports accepted':'Peer reports rejected'):(authState.result?'Verified locally':'Rejected locally');
  }
  if(report){report.authentication=authState;report.transport=link;}
  renderCurrent();
}
function renderCurrent(){
  if(!active||!channel)return;
  const bytes=channel.current();
  if(!bytes){shown=null;rendered=[];phaseIndex=-1;clearTimeout(displayTimer);return;}
  if(shown&&sameBytes(shown,bytes))return;
  shown=bytes.slice();
  const packet=decodePacket(bytes);
  const size=packet.kind==='ack'?32:channel.snapshot().outgoingSize;
  rendered=packetFrames(bytes,{size,phaseMs:config.phaseMs}).map(frame=>renderNeuralQR(frame,1024));
  $('matrixStatus').textContent=packet.kind==='ack'?'G32 ACK · keep visible':packet.kind==='start'?`G${size} · message start`:`G${size} · data chunk ${packet.index+1}`;
  phaseStarted=performance.now();phaseIndex=-1;paintPhase();schedulePhase(epoch);
}
function paintPhase(){
  if(!active||!rendered.length)return;
  const next=Math.floor((performance.now()-phaseStarted)/config.phaseMs)%rendered.length;
  if(next===phaseIndex)return;phaseIndex=next;
  const image=rendered[next],canvas=$('matrix');
  if(canvas.width!==image.width||canvas.height!==image.height){canvas.width=image.width;canvas.height=image.height;}
  canvas.getContext('2d').putImageData(new ImageData(image.data,image.width,image.height),0,0);
  $('phaseStatus').textContent=`Phase ${next+1}/4 · ${config.phaseMs} ms`;
}
function schedulePhase(token){
  clearTimeout(displayTimer);
  if(!active||token!==epoch||rendered.length<2)return;
  const remaining=config.phaseMs-((performance.now()-phaseStarted)%config.phaseMs);
  displayTimer=setTimeout(()=>{if(active&&token===epoch){paintPhase();schedulePhase(token);}},Math.max(1,remaining));
}
async function populateCameras(){
  if(!navigator.mediaDevices?.enumerateDevices)return;
  const selected=$('camera').value,devices=await navigator.mediaDevices.enumerateDevices();
  $('camera').replaceChildren(new Option('Default camera',''),...devices.filter(d=>d.kind==='videoinput').map((d,i)=>new Option(d.label||`Camera ${i+1}`,d.deviceId)));
  if([...$('camera').options].some(option=>option.value===selected))$('camera').value=selected;
}
function overlay(feedback,width,height){
  const canvas=$('overlay');canvas.width=width;canvas.height=height;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,width,height);
  const {corners,markers,validated}=feedback;
  ctx.fillStyle=validated?'rgba(87,214,138,.20)':'rgba(255,153,47,.24)';ctx.strokeStyle=validated?'#79e9a1':'#ff992f';ctx.lineWidth=3;
  if(Array.isArray(corners)&&corners.length===4&&corners.every(point=>Array.isArray(point)&&point.length===2&&point.every(Number.isFinite))){
    ctx.beginPath();for(let i=0;i<4;i++)ctx[i?'lineTo':'moveTo'](...corners[i]);
    ctx.closePath();ctx.fill();ctx.stroke();
  }
  if(!validated)for(const marker of markers){
    if(!Number.isFinite(marker.x)||!Number.isFinite(marker.y))continue;
    const radius=Number.isFinite(marker.module)?Math.max(5,marker.module*3.5):8;
    ctx.beginPath();ctx.arc(marker.x,marker.y,radius,0,2*Math.PI);ctx.fill();ctx.stroke();
  }
}
function scheduleCapture(token){
  if(!active||token!==epoch||scanning)return;
  frameWaitStarted=performance.now();
  frameCallback=$('video').requestVideoFrameCallback((now,metadata)=>{
    frameCallback=null;if(!active||token!==epoch)return;
    if(now-lastCapture<config.captureMs||metadata.presentedFrames<=lastPresented){scheduleCapture(token);return;}
    if(!Number.isFinite(metadata.mediaTime)||!Number.isSafeInteger(metadata.presentedFrames)){
      fail(Error('The camera did not provide distinct frame timestamps. Try a current browser.'));return;
    }
    if(!$('video').videoWidth||$('video').readyState<2){scheduleCapture(token);return;}
    lastCapture=now;lastPresented=metadata.presentedFrames;scanning=true;
    void scan(metadata,token).catch(e=>{if(token===epoch)fail(e);}).finally(()=>{
      if(token!==epoch)return;scanning=false;scheduleCapture(token);
    });
  });
}
async function scan(metadata,token){
  const video=$('video'),canvas=$('captureCanvas'),dimensions=captureDimensions(video.videoWidth,video.videoHeight);
  canvas.width=dimensions.width;canvas.height=dimensions.height;
  report.processedCamera={width:canvas.width,height:canvas.height,scale:dimensions.scale};
  const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(video,0,0,canvas.width,canvas.height);
  const image=ctx.getImageData(0,0,canvas.width,canvas.height),expected=channel?.snapshot().mode==='send'?'ack':'data';
  report.captureAttempts++;
  const result=await corrector.call('transport-scan',{size:config.incomingMaxSize,backend:config.backend,image,expected,
    captureIntervalMs:config.captureMs,runId:token,mediaTime:metadata.mediaTime,presentedFrames:metadata.presentedFrames},[image.data.buffer]);
  if(!active||token!==epoch)return;
  const feedback=captureFeedback(result);overlay(feedback,canvas.width,canvas.height);
  $('cameraStatus').textContent=feedback.label;
  const stage=result.ok?'decoded':result.stage??'unknown',counts=report.captureStages[expected];
  counts[stage]=(counts[stage]??0)+1;
  report.lastAcquisition=acquisitionDiagnostics(result);
  if(!result.ok&&result.error)report.lastAcquisitionFailure=acquisitionDiagnostics(result);
  if(result.stats?.geometrySource==='previous')report.geometryReuses++;
  if(expected==='data'&&result.info?.size)lastIncomingSize=result.info.size;
  if(Number.isFinite(result.stats?.pixelsPerCell))$('cameraDensity').textContent=`${result.stats.pixelsPerCell.toFixed(2)} px/cell`;
  const grid=result.info?.size;
  if(grid){
    const row=report.geometries[String(grid)]??={captureAttempts:0,inferences:0,inferenceTotalMs:0,crcAccepted:0,transportAccepted:0};
    row.captureAttempts++;
    if(result.method==='neural-corrector'&&Number.isFinite(result.inferenceMs)){row.inferences++;row.inferenceTotalMs+=result.inferenceMs;}
    if(result.ok)row.crcAccepted++;
  }
  if(result.method==='neural-corrector'&&Number.isFinite(result.inferenceMs)){
    $('inferenceMs').textContent=`${result.inferenceMs.toFixed(0)} ms`;report.inferenceCount++;
    report.inferenceMs.push(result.inferenceMs);if(report.inferenceMs.length>256)report.inferenceMs.shift();
  }
  if(!result.ok){report.captureMisses++;return;}
  let bytes;try{bytes=unpackOptical(result.chunk,expected);}catch{report.rejectedContainers++;$('cameraStatus').textContent='Code read · packet rejected';return;}
  report.crcAcceptedMatrices++;$('receivedCount').textContent=String(report.crcAcceptedMatrices);
  const accepted=await channel.receive(bytes);
  if(!active||token!==epoch)return;
  if(accepted.accepted){report.transportAcceptedMatrices++;if(grid)report.geometries[String(grid)].transportAccepted++;$('cameraStatus').textContent=expected==='ack'?'Neural-QR ACK matched · continuing':`G${grid} data packet accepted`;if(accepted.delivered)event('Outgoing message fully acknowledged.');}
  else {report.transportRejectedMatrices++;$('cameraStatus').textContent='Code read · waiting for the matching transfer packet';}
}
function startLoops(token){
  heartbeat=setInterval(()=>{
    if(!active||token!==epoch)return;
    if(frameCallback!==null&&!scanning&&performance.now()-frameWaitStarted>15000){fail(Error('Camera stopped producing frames. Restart the camera on both devices.'));return;}
    channel?.tick();session?.tick();
    $('elapsed').textContent=`${Math.round((performance.now()-startedAt)/1000)} s`;
    if(['failed','timed-out'].includes(authState?.state))fail(Error(authState.error||'Authentication expired.'));
  },1000);
  scheduleCapture(token);
}
async function start(){
  if(active)return;error('');
  try{config=settings();}catch(e){error(e.message);return;}
  if(typeof $('video').requestVideoFrameCallback!=='function'){error('This test needs a browser with requestVideoFrameCallback.');return;}
  active=true;const token=++epoch;startedAt=performance.now();events.length=0;transfers.clear();lastIncomingSize=null;authState=null;shown=null;rendered=[];
  channel=null;session=null;cryptoClient=null;corrector=null;
  lastCapture=-Infinity;lastPresented=0;scanning=false;$('result').textContent='';$('receivedCount').textContent='0';$('inferenceMs').textContent='—';$('elapsed').textContent='0 s';$('goodput').textContent='—';$('cameraDensity').textContent='—';
  report={format:'neural-zkboo-auth-report',version:2,started:new Date().toISOString(),settings:{...config,cameraId:undefined},
    captureAttempts:0,captureMisses:0,crcAcceptedMatrices:0,transportAcceptedMatrices:0,transportRejectedMatrices:0,
    rejectedContainers:0,inferenceCount:0,inferenceMs:[],geometries:{},transfers:[],captureStages:{data:{},ack:{}},geometryReuses:0,model:null,camera:null,cameraControls:null,lastAcquisition:null,lastAcquisitionFailure:null,authentication:null,transport:null};
  corrector=new CorrectorClient({onProgress:text=>{if(active&&token===epoch)$('modelStatus').textContent=text;}});
  cryptoClient=new CryptoClient({onInvalidate:e=>{if(active&&token===epoch)fail(e);}});
  update();
  try{
    report.model=await corrector.call('transport-init',{size:config.incomingMaxSize,backend:config.backend});
    if(!active||token!==epoch)return;
    $('modelStatus').textContent=`Trained ${report.model.modelProfile} · ${report.model.effectiveBackend.toUpperCase()} · loaded locally`;
    const opened=await openCaptureCamera(navigator.mediaDevices,{resolution:config.resolution,deviceId:config.cameraId}),media=opened.stream;
    if(!active||token!==epoch){media.getTracks().forEach(track=>track.stop());return;}
    stream=media;
    for(const track of stream.getVideoTracks())track.addEventListener('ended',()=>{if(active&&token===epoch)fail(Error('Camera disconnected. Restart both devices.'));},{once:true});
    $('video').srcObject=stream;await $('video').play();
    if(!active||token!==epoch)return;
    report.camera={...cameraSettings(),...opened.request};$('cameraPlaceholder').hidden=true;
    cameraControls=new CameraControls(stream.getVideoTracks()[0]);changingCameraControls=false;renderCameraControls();
    $('cameraStatus').textContent=`${report.camera.width} × ${report.camera.height} · ${Math.round(report.camera.frameRate??0)} fps`;
    await populateCameras();if(!active||token!==epoch)return;
    const deviceId=crypto.randomUUID();
    channel=new TurnChannel({initiator:config.role==='verifier',outgoingSize:config.outgoingSize,onChange:update,
      onMessage:async bytes=>{
        const result=await session.receive(bytes);
        if(token!==epoch)return;
        event(result.accepted?'Authentication message received.':`Message not used: ${result.reason}`);
      },onError:fail});
    session=new AuthSession({role:config.role,deviceId,cryptoClient,send:bytes=>channel.enqueue(bytes),
      repetitions:config.repetitions,timeoutMs:30*60*1000,onChange:state=>{
        if(token!==epoch)return;
        const previous=authState?.state;authState=state;if(previous!==state.state)event(state.status);
        update();
      }});
    $('settings').open=false;event('Camera and trained model ready.');startLoops(token);update();
    report.authenticationStartedAfterMs=performance.now()-startedAt;
    await session.start();
  }catch(e){if(token===epoch)fail(e);}
}
function stop(reason='Stopped by the user.'){
  if(!active)return;
  const saved=authState;active=false;epoch++;
  clearTimeout(displayTimer);clearInterval(heartbeat);
  if(frameCallback!==null)$('video').cancelVideoFrameCallback(frameCallback);frameCallback=null;
  channel?.close();session?.close(reason);cryptoClient?.destroy();corrector?.cancel();
  if(cameraControls){if(report)report.cameraControls=cameraControls.snapshot();cameraControls.close();cameraControls=null;}
  changingCameraControls=false;$('cameraControls').hidden=true;$('cameraControlFields').replaceChildren();
  stream?.getTracks().forEach(track=>track.stop());stream=null;$('video').srcObject=null;
  $('cameraPlaceholder').hidden=false;$('cameraStatus').textContent='Camera is off';
  shown=null;rendered=[];scanning=false;
  if(report){report.authentication=saved;report.finished=new Date().toISOString();report.stopReason=reason;report.elapsedMs=performance.now()-startedAt;}
  authState=saved;event(reason);$('status').textContent=reason;$('instruction').textContent='Download the report, or start a fresh exchange on both devices.';update();
}
function fail(e){error(e?.message||String(e));stop(e?.message||String(e));}
function download(){
  if(!report)return;
  if(cameraControls)report.cameraControls=cameraControls.snapshot();
  const value={...report,elapsedMs:report.elapsedMs??performance.now()-startedAt,notes:$('reportNotes').value.slice(0,1000),events:[...events]};
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),a=document.createElement('a');
  a.href=url;a.download=`neural-zkboo-${config.role}-${new Date().toISOString().replace(/[:.]/g,'-')}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('start').addEventListener('click',()=>void start());
$('stop').addEventListener('click',()=>stop());
$('confirmPairing').addEventListener('click',()=>{if(active)void session?.confirmPairing().catch(fail);});
$('nextTurn').addEventListener('click',()=>{if(active)void channel?.nextTurn().catch(fail);});
$('downloadReport').addEventListener('click',download);
$('deviceProfile').addEventListener('change',populateProfiles);
$('nextGeometry').addEventListener('change',()=>{
  if(!active||!channel)return;
  try{
    const selected=transportProfile(config.device,$('nextGeometry').value);
    channel.setOutgoingSize(selected.outgoingSize);config.outgoingSize=selected.outgoingSize;
    if(report)report.settings.outgoingSize=selected.outgoingSize;
    event(`Next outgoing message uses G${selected.outgoingSize}.`);update();
  }catch(e){error(e.message);update();}
});
$('mirrorCamera').addEventListener('click',()=>{const value=$('stage').dataset.mirror!=='true';$('stage').dataset.mirror=String(value);$('mirrorCamera').setAttribute('aria-pressed',String(value));});
$('displaySize').addEventListener('input',()=>{$('stage').style.setProperty('--matrix-scale',`${$('displaySize').value}%`);$('displaySizeValue').textContent=`${$('displaySize').value}%`;});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&active)stop('Stopped because the page became hidden. Restart both devices.');});
window.addEventListener('pagehide',()=>stop('Page closed.'));
if(!supported)error('Open this app on trusted HTTPS (or localhost) with camera, Web Crypto and worker support.');
populateProfiles();void populateCameras().catch(()=>{});update();
