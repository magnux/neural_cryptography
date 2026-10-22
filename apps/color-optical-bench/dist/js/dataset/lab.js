import {CaptureClient} from './client.js';
import {CaptureBuffer} from './capture.js';
import {project} from '../vision.js';

const $=id=>document.getElementById(id),client=new CaptureClient(),buffer=new CaptureBuffer();
const supported=isSecureContext&&typeof Worker==='function'&&Boolean(globalThis.crypto?.subtle);
const session=prefix=>`${prefix}-${globalThis.crypto?.randomUUID?.()??'unavailable'}`;
let senderSession=session('sender'),receiverSession=session('receiver'),sender=null,phase=0,role='sender';
let busy=false,epoch=0,stream=null,cameraStarting=false,cameraEpoch=0,delayCancel=null;
let lastMediaTime=-1,snapshotCount=0;
let previousFocus=null;

function showError(message=''){$('captureError').textContent=message;$('captureError').hidden=!message;}
function status(message){$('receiverStatus').textContent=message;}
function cameraInfo(){
  const settings=stream?.getVideoTracks()[0]?.getSettings();
  return settings?{active:true,width:settings.width,height:settings.height,frameRate:settings.frameRate,
    facingMode:settings.facingMode,requestedHeight:Number($('resolution').value)}:{active:false};
}
function refresh(){
  const counts=buffer.summary();
  $('senderRole').disabled=busy;$('receiverRole').disabled=busy;
  $('senderRole').setAttribute('aria-pressed',String(role==='sender'));
  $('receiverRole').setAttribute('aria-pressed',String(role==='receiver'));
  $('senderPanel').hidden=role!=='sender';$('receiverPanel').hidden=role!=='receiver';
  for(const id of ['senderSize','senderColors','senderPhases','createSender','newSenderSession'])$(id).disabled=!supported||busy;
  $('downloadSender').disabled=!sender||busy;$('nextPhase').disabled=!sender||sender.frames.length<2||busy;
  $('focusMatrix').disabled=!sender||busy||$('senderCanvasWrap').hidden;
  $('senderSession').textContent=`Session: ${senderSession}`;$('receiverSession').textContent=`Session: ${receiverSession}`;
  $('startCamera').disabled=!supported||cameraStarting||Boolean(stream)||busy;
  $('stopCamera').disabled=!stream&&!cameraStarting;
  $('camera').disabled=Boolean(stream)||cameraStarting;$('resolution').disabled=Boolean(stream)||cameraStarting;
  $('probeCamera').disabled=!stream||busy;
  $('collectBurst').disabled=!stream||busy||counts.records>=counts.maxRecords||counts.bytes>=counts.maxBytes;
  $('cancelBurst').disabled=!busy||role!=='receiver';$('clearCaptures').disabled=busy;
  $('newReceiverSession').disabled=busy;
  for(const id of ['burstCount','burstInterval','saveOriginal'])$(id).disabled=busy;
  $('downloadCaptures').disabled=busy||(!counts.records&&!counts.diagnostics);
  $('sampleCount').textContent=`${counts.records} / ${counts.maxRecords}`;
  $('diagnosticCount').textContent=String(counts.diagnostics);
  $('captureBytes').textContent=`${(counts.bytes/1048576).toFixed(1)} / ${(counts.maxBytes/1048576).toFixed(0)} MiB`;
}
function saveJSON(value,name){
  const blob=new Blob([JSON.stringify(value)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function focusMatrix(){
  if(busy||!sender||$('senderCanvasWrap').hidden)return;
  previousFocus=document.activeElement;document.body.classList.add('matrix-focused');
  $('senderCanvasWrap').setAttribute('role','dialog');$('senderCanvasWrap').setAttribute('aria-modal','true');$('senderCanvasWrap').setAttribute('aria-label','Static matrix focus view');
  $('closeFocus').hidden=false;$('focusMatrix').setAttribute('aria-expanded','true');$('closeFocus').focus();
}
function closeFocus(restore=true){
  document.body.classList.remove('matrix-focused');$('closeFocus').hidden=true;$('focusMatrix').setAttribute('aria-expanded','false');
  for(const attribute of ['role','aria-modal','aria-label'])$('senderCanvasWrap').removeAttribute(attribute);
  if(restore)previousFocus?.focus();previousFocus=null;
}
async function displayPhase(index,token){
  const frame=sender.frames[index],image=await client.call('render',{frame,pixels:frame.size===128?2048:1024});
  if(token!==epoch)return;
  phase=index;
  const canvas=$('senderCanvas');canvas.width=image.width;canvas.height=image.height;
  canvas.getContext('2d').putImageData(new ImageData(image.data,image.width,image.height),0,0);
  $('senderCanvasWrap').hidden=false;
  $('senderStatus').textContent=`${frame.size}×${frame.size}, ${frame.colors} colors · phase ${phase+1}/${sender.frames.length} · sequence ${frame.id}. Keep this image unchanged while the receiver collects.`;
}
async function createSender(){
  if(busy)return;showError();busy=true;const token=++epoch;$('senderCanvasWrap').hidden=true;refresh();
  try{
    const created=await client.call('create',{size:Number($('senderSize').value),colors:Number($('senderColors').value),
      phases:Number($('senderPhases').value),sessionId:senderSession});
    if(token!==epoch)return;
    sender=created;phase=0;await displayPhase(0,token);
  }catch(error){if(token===epoch)showError(error.message);}
  finally{if(token===epoch){busy=false;refresh();}}
}
async function nextPhase(){
  if(busy||!sender)return;showError();busy=true;const token=++epoch;refresh();
  try{await displayPhase((phase+1)%sender.frames.length,token);}
  catch(error){if(token===epoch)showError(error.message);}finally{if(token===epoch){busy=false;refresh();}}
}
function overlay(record,width,height){
  const canvas=$('alignmentOverlay');canvas.width=width;canvas.height=height;
  const context=canvas.getContext('2d');context.clearRect(0,0,width,height);
  if(!record.transform)return;
  const points=[[8,8],[248,8],[248,248],[8,248]].map(([x,y])=>project(record.transform,x,y));
  if(points.some(point=>point.some(value=>!Number.isFinite(value))))return;
  context.strokeStyle='#8ff3ad';context.lineWidth=Math.max(2,width/400);context.beginPath();
  points.forEach(([x,y],i)=>i?context.lineTo(x,y):context.moveTo(x,y));context.closePath();context.stroke();
}
function showCapture(record){
  $('latestCapture').textContent=record.ok
    ? `Header valid · ${record.info.size}×${record.info.size} / ${record.info.colors} colors · phase ${record.info.phase+1}/${record.info.phases}. Classical payload ${record.baseline?.accepted?'passed CRC':'failed CRC — sample still useful'}.`
    : `${record.stage??'Capture'}: ${record.error??'No usable image.'}`;
  const stats=record.stats??{};
  $('latestDetails').textContent=record.ok
    ? `${Number(stats.pixelsPerCell??0).toFixed(2)} camera pixels/cell · color RMSE ${Number(stats.colorRMSE??0).toFixed(1)} · preprocessing ${Number(stats.preprocessMs??0).toFixed(0)} ms.${record.originalNote?` ${record.originalNote}`:''}`
    : 'Keep the full border visible. Enlarge the displayed matrix or move closer; try 720p if focus or moiré is unstable.';
}
function currentImage(){
  const video=$('video');
  if(!stream||!video.videoWidth||video.readyState<2)throw Error('Wait for a live camera image first.');
  const mediaTime=video.currentTime;
  if(!Number.isFinite(mediaTime)||mediaTime<=lastMediaTime)throw Error('No new camera frame yet. The camera may be paused.');
  lastMediaTime=mediaTime;
  const scale=Math.min(1,Math.sqrt(2073600/(video.videoWidth*video.videoHeight)));
  const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.floor(video.videoWidth*scale));canvas.height=Math.max(1,Math.floor(video.videoHeight*scale));
  const context=canvas.getContext('2d',{willReadFrequently:true});context.drawImage(video,0,0,canvas.width,canvas.height);
  const pixels=context.getImageData(0,0,canvas.width,canvas.height),camera={...cameraInfo(),captureWidth:canvas.width,captureHeight:canvas.height,mediaTime};
  return {image:{width:pixels.width,height:pixels.height,data:pixels.data},timestamp:Date.now(),camera};
}
async function captureOne(original=false){
  const options=currentImage();options.original=original;
  const result=await client.call('capture',options,[options.image.data.buffer]);
  overlay(result,options.image.width,options.image.height);showCapture(result);
  return result;
}
function pause(ms){
  return new Promise(resolve=>{const finish=()=>{clearTimeout(timer);delayCancel=null;resolve();};const timer=setTimeout(finish,ms);delayCancel=finish;});
}
function cancel(){
  epoch++;client.cancel();delayCancel?.();buffer.resetStability();busy=false;
  status('Stopped. Previously collected samples are retained. Save the receiver file before leaving.');refresh();
}
async function probe(){
  if(busy||!stream)return;showError();busy=true;const token=++epoch;refresh();
  try{
    const record=await captureOne();if(token!==epoch)return;
    if(!record.ok)buffer.add(record);
    status(record.ok?'Alignment is ready. This check did not add a training sample. Keep the sender static, then collect a burst.':'Alignment failed. The diagnostic is retained; adjust the angle or display size and check again.');
  }catch(error){if(token===epoch)showError(error.message);}
  finally{if(token===epoch){busy=false;refresh();}}
}
async function collectBurst(){
  if(busy||!stream)return;showError();
  const requested=Number($('burstCount').value),spacing=Number($('burstInterval').value);
  if(!Number.isInteger(requested)||requested<1||requested>20||!Number.isInteger(spacing)||spacing<500||spacing>5000){showError('Choose 1–20 samples and 500–5000 ms spacing.');return;}
  const initial=buffer.summary().records,target=Math.min(requested,20-initial),token=++epoch;
  busy=true;buffer.resetStability();refresh();
  let saved=0,attempts=0,originalAttempted=false,full=false;
  const started=performance.now(),maxAttempts=target+5;
  try{
    while(token===epoch&&saved<target&&attempts<maxAttempts&&performance.now()-started<120000){
      status(`Collecting ${saved}/${target} samples · observation ${attempts+1}. Keep the displayed phase unchanged.`);
      const original=$('saveOriginal').checked&&!originalAttempted&&attempts>=1&&snapshotCount<2;
      if(original)originalAttempted=true;
      attempts++;
      let record;
      try{record=await captureOne(original);}catch(error){
        if(token!==epoch)break;
        if(error.name==='AbortError')break;
        record={ok:false,timestamp:Date.now(),stage:'camera',error:error.message,stats:{}};showCapture(record);
      }
      if(token!==epoch)break;
      const outcome=buffer.add(record,{alreadyValidated:true});
      if(record.original)snapshotCount++;
      if(outcome.accepted)saved++;
      full=outcome.stage==='full';refresh();
      if(full)break;
      if(saved<target&&attempts<maxAttempts)await pause(spacing);
    }
    if(token===epoch)status(full
      ? `Capture budget reached after ${saved} new samples. Save the receiver file and start a new batch.`
      : `Burst finished: ${saved}/${target} samples saved from ${attempts} observations. ${saved<target?'Some captures did not pass header stability; review the latest diagnostic. ':'Only stable, header-validated samples were retained. '}You may now advance the sender.`);
  }catch(error){if(token===epoch)showError(error.message);}
  finally{if(token===epoch){busy=false;buffer.resetStability();refresh();}}
}
async function enumerateCameras(){
  if(!navigator.mediaDevices?.enumerateDevices)return;
  const selected=$('camera').value;
  try{const devices=(await navigator.mediaDevices.enumerateDevices()).filter(device=>device.kind==='videoinput');
    $('camera').replaceChildren(new Option('Default camera',''));devices.forEach((device,i)=>$('camera').add(new Option(device.label||`Camera ${i+1}`,device.deviceId)));
    if(devices.some(device=>device.deviceId===selected))$('camera').value=selected;
  }catch{/* Camera names become available after permission. */}
}
function stopCamera(){
  cameraEpoch++;if(busy&&role==='receiver')cancel();
  stream?.getTracks().forEach(track=>track.stop());stream=null;cameraStarting=false;lastMediaTime=-1;buffer.resetStability();
  $('video').srcObject=null;$('cameraPlaceholder').hidden=false;$('cameraStatus').textContent='Camera is off. Collected samples remain available for download.';
  const canvas=$('alignmentOverlay');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);refresh();
}
async function startCamera(){
  if(cameraStarting||stream||busy)return;showError();cameraStarting=true;const token=++cameraEpoch;refresh();
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw Error('Camera access requires trusted HTTPS or localhost.');
    const height=Number($('resolution').value),width=height===480?640:Math.round(height*16/9);
    const acquired=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:width},height:{ideal:height},
      ...($('camera').value?{deviceId:{exact:$('camera').value}}:{facingMode:{ideal:'environment'}})}});
    if(token!==cameraEpoch){acquired.getTracks().forEach(track=>track.stop());return;}
    stream=acquired;$('video').srcObject=stream;await $('video').play();if(token!==cameraEpoch)return;
    lastMediaTime=-1;$('cameraPlaceholder').hidden=true;const actual=cameraInfo();
    $('cameraStatus').textContent=`Actual capture ${actual.width}×${actual.height}, ${actual.frameRate?.toFixed(0)??'unknown'} fps. Fit all four markers and the top header in view.`;
    await enumerateCameras();
  }catch(error){if(token===cameraEpoch){stopCamera();showError(error.message);}}
  finally{if(token===cameraEpoch){cameraStarting=false;refresh();}}
}
function chooseRole(value){if(busy)return;if(value==='sender')stopCamera();role=value;showError();refresh();}
function shutdown(){closeFocus(false);cancel();stopCamera();}
$('senderRole').addEventListener('click',()=>chooseRole('sender'));
$('receiverRole').addEventListener('click',()=>chooseRole('receiver'));
$('createSender').addEventListener('click',createSender);
$('nextPhase').addEventListener('click',nextPhase);
$('focusMatrix').addEventListener('click',focusMatrix);$('closeFocus').addEventListener('click',()=>closeFocus());
document.addEventListener('keydown',event=>{
  if(!document.body.classList.contains('matrix-focused'))return;
  if(event.key==='Escape')closeFocus();
  else if(event.key==='Tab'){event.preventDefault();$('closeFocus').focus();}
});
$('downloadSender').addEventListener('click',()=>{if(sender)saveJSON(sender.manifest,`decoder-sender-${sender.manifest.sequenceId}.json`);});
$('newSenderSession').addEventListener('click',()=>{senderSession=session('sender');sender=null;phase=0;$('senderCanvasWrap').hidden=true;$('senderStatus').textContent='New physical session. Create and save a fresh sequence.';refresh();});
$('startCamera').addEventListener('click',startCamera);$('stopCamera').addEventListener('click',stopCamera);
$('probeCamera').addEventListener('click',probe);$('collectBurst').addEventListener('click',collectBurst);$('cancelBurst').addEventListener('click',cancel);
$('clearCaptures').addEventListener('click',()=>{buffer.reset();snapshotCount=0;status('Batch cleared. The physical session ID is retained for the next capture file.');refresh();});
$('newReceiverSession').addEventListener('click',()=>{buffer.reset();receiverSession=session('receiver');snapshotCount=0;status('Samples cleared. A new physical receiver session is ready.');refresh();});
$('downloadCaptures').addEventListener('click',()=>saveJSON(buffer.export({notes:$('captureNotes').value,sessionId:receiverSession,camera:cameraInfo()}),`decoder-captures-${receiverSession}-${new Date().toISOString().replace(/[:.]/g,'-')}.json`));
window.addEventListener('pagehide',shutdown);document.addEventListener('visibilitychange',()=>{if(document.hidden)shutdown();});
if(!supported)showError('Use a current browser with Web Workers and Web Crypto on trusted HTTPS or localhost.');
refresh();enumerateCameras();
