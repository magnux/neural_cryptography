import {AuthSession} from '../auth-core/session.js';
import {CryptoClient} from '../crypto-lab/client.js';
import {CorrectorClient} from '../corrector-lab/client.js';
import {renderFrame,sameBytes} from '../format.js';
import {decodePacket} from '../link/protocol.js';
import {TurnChannel} from './channel.js';
import {packetFrames,unpackOptical} from './optical.js';

const $=id=>document.getElementById(id);
const supported=isSecureContext&&Boolean(globalThis.crypto?.subtle)&&typeof Worker==='function'&&Boolean(navigator.mediaDevices?.getUserMedia);
let active=false,epoch=0,stream=null,corrector=null,cryptoClient=null,session=null,channel=null,config=null;
let frameCallback=null,displayTimer=null,heartbeat=null,lastCapture=-Infinity,lastPresented=0,scanning=false,frameWaitStarted=0;
let shown=null,rendered=[],phaseStarted=0,phaseIndex=-1,authState=null,report=null,startedAt=0;
const events=[];
const int=(value,min,max,label)=>{const n=Number(value);if(!Number.isSafeInteger(n)||n<min||n>max)throw Error(`${label} must be ${min}–${max}.`);return n;};
function settings(){
  const role=$('role').value,backend=$('backend').value,repetitions=Number($('repetitions').value),resolution=Number($('resolution').value);
  if(!['verifier','prover'].includes(role)||!['auto','wasm','webgpu'].includes(backend)||![32,64,110,160,219].includes(repetitions)||![480,720,1080].includes(resolution))throw Error('Unsupported test settings.');
  return {role,backend,repetitions,resolution,phaseMs:int($('phaseMs').value,50,10000,'Phase interval'),
    captureMs:int($('captureMs').value,50,2000,'Capture interval'),cameraId:$('camera').value};
}
function error(text=''){$('error').textContent=text;$('error').hidden=!text;}
function event(text){
  events.push({seconds:Math.round((performance.now()-startedAt)/1000),text});if(events.length>100)events.shift();
  $('eventLog').replaceChildren(...events.slice(-20).map(item=>{const li=document.createElement('li');li.textContent=`${item.seconds}s · ${item.text}`;return li;}));
}
function cameraSettings(){const v=stream?.getVideoTracks()[0]?.getSettings()??{};return {width:v.width,height:v.height,frameRate:v.frameRate,facingMode:v.facingMode};}
function update(){
  const link=channel?.snapshot();
  $('authApp').dataset.mode=active?(link?.mode??'receive'):'idle';
  $('settingsFields').disabled=active;$('start').disabled=active||!supported;$('stop').disabled=!active;
  $('downloadReport').disabled=!report;$('nextTurn').disabled=!active||!link?.canSendNext;
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
  rendered=packetFrames(bytes,{phaseMs:config.phaseMs}).map(frame=>renderFrame(frame,1024));
  const packet=decodePacket(bytes);
  $('matrixStatus').textContent=packet.kind==='ack'?'ACK · keep visible':packet.kind==='start'?'Message start':`Data chunk ${packet.index+1}`;
  phaseStarted=performance.now();phaseIndex=-1;paintPhase();schedulePhase(epoch);
}
function paintPhase(){
  if(!active||!rendered.length)return;
  const next=Math.floor((performance.now()-phaseStarted)/config.phaseMs)%rendered.length;
  if(next===phaseIndex)return;phaseIndex=next;
  const image=rendered[next];$('matrix').getContext('2d').putImageData(new ImageData(image.data,image.width,image.height),0,0);
  $('phaseStatus').textContent=rendered.length===1?'32 × 32 / 4 · ACK':`Phase ${next+1}/4 · ${config.phaseMs} ms`;
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
function overlay(corners,width,height){
  const canvas=$('overlay');canvas.width=width;canvas.height=height;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,width,height);
  if(!Array.isArray(corners)||corners.length!==4)return;
  ctx.beginPath();for(let i=0;i<4;i++){const point=corners[i];if(!Array.isArray(point)||point.length!==2||!point.every(Number.isFinite))return;ctx[i?'lineTo':'moveTo'](...point);}
  ctx.closePath();ctx.fillStyle='rgba(87,214,138,.20)';ctx.strokeStyle='#79e9a1';ctx.lineWidth=3;ctx.fill();ctx.stroke();
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
  const video=$('video'),canvas=$('captureCanvas'),scale=Math.min(1,1920/video.videoWidth,1080/video.videoHeight);
  canvas.width=Math.round(video.videoWidth*scale);canvas.height=Math.round(video.videoHeight*scale);
  const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(video,0,0,canvas.width,canvas.height);
  const image=ctx.getImageData(0,0,canvas.width,canvas.height),expected=channel?.snapshot().mode==='send'?'ack':'data';
  report.captureAttempts++;
  const result=await corrector.call('transport-scan',{size:64,backend:config.backend,image,expected,
    runId:token,mediaTime:metadata.mediaTime,presentedFrames:metadata.presentedFrames},[image.data.buffer]);
  if(!active||token!==epoch)return;
  overlay(result.corners,canvas.width,canvas.height);
  if(result.phaseCounts)$('cameraStatus').textContent=`Phases ${result.phaseCounts.join(' / ')} · ${result.inputCaptures??result.collected??0} captures`;
  if(result.method==='neural-corrector'&&Number.isFinite(result.inferenceMs)){
    $('inferenceMs').textContent=`${result.inferenceMs.toFixed(0)} ms`;report.inferenceCount++;
    report.inferenceMs.push(result.inferenceMs);if(report.inferenceMs.length>256)report.inferenceMs.shift();
  }
  if(!result.ok){report.captureMisses++;return;}
  let bytes;try{bytes=unpackOptical(result.chunk,expected);}catch{report.rejectedContainers++;return;}
  report.crcAcceptedMatrices++;$('receivedCount').textContent=String(report.crcAcceptedMatrices);
  const accepted=await channel.receive(bytes);
  if(!active||token!==epoch)return;
  if(accepted.accepted){report.transportAcceptedMatrices++;if(accepted.delivered)event('Outgoing message fully acknowledged.');}
  else report.transportRejectedMatrices++;
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
  active=true;const token=++epoch;startedAt=performance.now();events.length=0;authState=null;shown=null;rendered=[];
  channel=null;session=null;cryptoClient=null;corrector=null;
  lastCapture=-Infinity;lastPresented=0;scanning=false;$('result').textContent='';$('receivedCount').textContent='0';$('inferenceMs').textContent='—';$('elapsed').textContent='0 s';
  report={format:'neural-zkboo-color-auth-report',version:1,started:new Date().toISOString(),settings:{...config,cameraId:undefined},
    captureAttempts:0,captureMisses:0,crcAcceptedMatrices:0,transportAcceptedMatrices:0,transportRejectedMatrices:0,
    rejectedContainers:0,inferenceCount:0,inferenceMs:[],model:null,camera:null,authentication:null,transport:null};
  corrector=new CorrectorClient({onProgress:text=>{if(active&&token===epoch)$('modelStatus').textContent=text;}});
  cryptoClient=new CryptoClient({onInvalidate:e=>{if(active&&token===epoch)fail(e);}});
  update();
  try{
    report.model=await corrector.call('transport-init',{size:64,backend:config.backend});
    if(!active||token!==epoch)return;
    $('modelStatus').textContent=`Trained ${report.model.modelProfile} · ${report.model.effectiveBackend.toUpperCase()} · loaded locally`;
    const media=await navigator.mediaDevices.getUserMedia({audio:false,video:{height:{ideal:config.resolution},width:{ideal:Math.round(config.resolution*16/9)},
      ...(config.cameraId?{deviceId:{exact:config.cameraId}}:{facingMode:{ideal:'environment'}})}});
    if(!active||token!==epoch){media.getTracks().forEach(track=>track.stop());return;}
    stream=media;
    for(const track of stream.getVideoTracks())track.addEventListener('ended',()=>{if(active&&token===epoch)fail(Error('Camera disconnected. Restart both devices.'));},{once:true});
    $('video').srcObject=stream;await $('video').play();
    if(!active||token!==epoch)return;
    report.camera=cameraSettings();$('cameraPlaceholder').hidden=true;
    $('cameraStatus').textContent=`${report.camera.width} × ${report.camera.height} · ${Math.round(report.camera.frameRate??0)} fps`;
    await populateCameras();if(!active||token!==epoch)return;
    const deviceId=crypto.randomUUID();
    channel=new TurnChannel({initiator:config.role==='verifier',onChange:update,
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
    await session.start();
  }catch(e){if(token===epoch)fail(e);}
}
function stop(reason='Stopped by the user.'){
  if(!active)return;
  const saved=authState;active=false;epoch++;
  clearTimeout(displayTimer);clearInterval(heartbeat);
  if(frameCallback!==null)$('video').cancelVideoFrameCallback(frameCallback);frameCallback=null;
  channel?.close();session?.close(reason);cryptoClient?.destroy();corrector?.cancel();
  stream?.getTracks().forEach(track=>track.stop());stream=null;$('video').srcObject=null;
  $('cameraPlaceholder').hidden=false;$('cameraStatus').textContent='Camera is off';
  shown=null;rendered=[];scanning=false;
  if(report){report.authentication=saved;report.finished=new Date().toISOString();report.stopReason=reason;report.elapsedMs=performance.now()-startedAt;}
  authState=saved;event(reason);$('status').textContent=reason;$('instruction').textContent='Download the report, or start a fresh exchange on both devices.';update();
}
function fail(e){error(e?.message||String(e));stop(e?.message||String(e));}
function download(){
  if(!report)return;
  const value={...report,elapsedMs:report.elapsedMs??performance.now()-startedAt,notes:$('reportNotes').value.slice(0,1000),events:[...events]};
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),a=document.createElement('a');
  a.href=url;a.download=`neural-zkboo-${config.role}-${new Date().toISOString().replace(/[:.]/g,'-')}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('start').addEventListener('click',()=>void start());
$('stop').addEventListener('click',()=>stop());
$('confirmPairing').addEventListener('click',()=>{if(active)void session?.confirmPairing().catch(fail);});
$('nextTurn').addEventListener('click',()=>{if(active)void channel?.nextTurn().catch(fail);});
$('downloadReport').addEventListener('click',download);
$('mirrorCamera').addEventListener('click',()=>{const value=$('stage').dataset.mirror!=='true';$('stage').dataset.mirror=String(value);$('mirrorCamera').setAttribute('aria-pressed',String(value));});
$('displaySize').addEventListener('input',()=>{$('stage').style.setProperty('--matrix-scale',`${$('displaySize').value}%`);$('displaySizeValue').textContent=`${$('displaySize').value}%`;});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&active)stop('Stopped because the page became hidden. Restart both devices.');});
window.addEventListener('pagehide',()=>stop('Page closed.'));
if(!supported)error('Open this app on trusted HTTPS (or localhost) with camera, Web Crypto and worker support.');
void populateCameras().catch(()=>{});update();
