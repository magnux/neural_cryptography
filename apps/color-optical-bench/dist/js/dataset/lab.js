import {CaptureClient} from './client.js';
import {CaptureBuffer} from './capture.js';
import {ScanLoop} from './scan-loop.js';
import {project} from '../vision.js';

const $=id=>document.getElementById(id),client=new CaptureClient(),buffer=new CaptureBuffer();
const supported=isSecureContext&&typeof Worker==='function'&&Boolean(globalThis.crypto?.subtle);
const session=prefix=>`${prefix}-${globalThis.crypto?.randomUUID?.()??'unavailable'}`;
let senderSession=session('sender'),receiverSession=session('receiver'),sender=null,phase=0,role='sender';
let busy=false,epoch=0,stream=null,cameraStarting=false,cameraEpoch=0;
let lastMediaTime=-1,collecting=false,budgetFull=false,receiverEpoch=0,activeKey=null,activeInfo=null;
const distances=new Map(),originalAttempts=new Set(),phaseActivity=new Map();
let previousMaximum=32;
const phaseMaximum=()=>Number($('phaseMaximum').value);
const scanner=new ScanLoop({sample:scanOne,receive:receiveScan,cancel:()=>client.cancel(),
  interval:()=>Number($('scanInterval').value),onError:error=>{
    buffer.resetStability();clearOverlay();
    $('latestCapture').textContent=error.message;
    status(collecting?'Waiting for a fresh camera frame; collection remains armed.':'Waiting for a fresh camera frame.');
  }});
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
  const phaseCount=buffer.phaseRecords(activeKey).length;
  $('collectBurst').disabled=!stream||collecting||budgetFull;
  $('cancelBurst').disabled=!collecting;
  $('clearCaptures').disabled=collecting;$('newReceiverSession').disabled=collecting;
  $('downloadCaptures').disabled=collecting||(!counts.records&&!counts.diagnostics);
  $('phaseMaximum').disabled=collecting;
  $('phaseCount').textContent=`${phaseCount} / ${phaseMaximum()}`;
  const activity=phaseActivity.get(activeKey);
  $('seenCount').textContent=String(activity?.seen??0);
  $('replacementCount').textContent=String(activity?.replaced??0);
  $('activePhase').textContent=activeInfo?`${activeInfo.id} · phase ${activeInfo.phase+1}/${activeInfo.phases}`:'No matrix detected yet';
  const metric=distances.get(activeKey),mean=metric?.result.mean;
  $('meanDistance').textContent=mean==null?'—':`${mean.toFixed(2)}%`;
  $('distanceDetails').textContent=metric
    ? `${metric.result.samples} samples · ${metric.result.pairs} pairs${metric.revision!==buffer.revision(activeKey)?' · updating…':''}`
    : 'Needs at least two retained captures of this phase.';
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
  const points=[[0,0],[256,0],[256,256],[0,256]].map(([x,y])=>project(record.transform,x,y));
  if(points.some(point=>point.some(value=>!Number.isFinite(value))))return;
  context.strokeStyle='#8ff3ad';context.lineWidth=Math.max(2,width/400);context.beginPath();
  points.forEach(([x,y],i)=>i?context.lineTo(x,y):context.moveTo(x,y));context.closePath();context.fillStyle=record.ok?'rgba(50,235,135,0.26)':'rgba(255,190,50,0.25)';context.fill();context.stroke();
}
function showCapture(record){
  $('latestCapture').textContent=record.ok
    ? `Header valid · ${record.info.size}×${record.info.size} / ${record.info.colors} colors · phase ${record.info.phase+1}/${record.info.phases}. Classical payload ${record.probe?'not tested during alignment':record.baseline?.accepted?'passed CRC':'failed CRC — sample still useful'}.`
    : `${record.stage??'Capture'}: ${record.error??'No usable image.'}`;
  const stats=record.stats??{};
  $('latestDetails').textContent=record.ok
    ? `${Number(stats.pixelsPerCell??0).toFixed(2)} camera pixels/cell · color RMSE ${stats.colorRMSE==null?'—':Number(stats.colorRMSE).toFixed(1)} · processing ${Number(stats.preprocessMs??stats.ms??0).toFixed(0)} ms.${record.originalNote?` ${record.originalNote}`:''}`
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
function clearOverlay(){const canvas=$('alignmentOverlay');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);}
function restartScanner(){receiverEpoch++;scanner.stop();buffer.resetStability();if(stream)scanner.start();}
async function scanOne(){
  const options=currentImage();
  const full=collecting;
  options.original=full&&$('saveOriginal').checked&&activeKey&&!originalAttempts.has(activeKey)&&buffer.originals<2;
  if(options.original)originalAttempts.add(activeKey);
  const record=await client.call(full?'capture':'probe',options,[options.image.data.buffer]);
  return {record,width:options.image.width,height:options.image.height};
}
async function receiveScan({record,width,height}){
  const token=receiverEpoch;
  overlay(record,width,height);showCapture(record);
  if(record.ok){activeKey=record.key;activeInfo=record.info;}else buffer.resetStability();
  if(collecting&&!record.probe){
    let replaceIndex=null;
    if(record.ok){
      let activity=phaseActivity.get(record.key);
      if(!activity){
        activity={seen:0,replaced:0};phaseActivity.set(record.key,activity);
        if(phaseActivity.size>256)phaseActivity.delete(phaseActivity.keys().next().value);
      }
      activity.seen++;
      if(buffer.phaseRecords(record.key).length>=phaseMaximum()){
        const selected=await client.call('select',{signatures:buffer.phaseSignatures(record.key),incomingSignature:record.signature});
        if(token!==receiverEpoch)return;
        replaceIndex=selected.replaceIndex;
      }
    }
    const outcome=buffer.add(record,{alreadyValidated:true,maxPerPhase:phaseMaximum(),replaceIndex});
    if(outcome.stage==='replaced')phaseActivity.get(record.key).replaced++;
    if(outcome.stage==='full'){
      collecting=false;budgetFull=true;
      status('Export budget reached. Pause is automatic: save receiver captures, clear the downloaded batch, then resume.');
    }else if(!record.ok)status('Looking for a valid matrix. Collection is armed; failed payload CRCs are still useful.');
    else if(outcome.stage==='settling')status('Matrix detected. Waiting for a stable header before saving samples…');
    else status(buffer.phaseRecords(activeKey).length>=phaseMaximum()
      ? 'Pool full: continuing to compare new captures and replace redundant samples. Pause when ready to save.'
      : 'Collecting this phase. Keep the matrix static; small changes in angle or focus can add variety.');
  }else if(collecting){
    status(record.ok&&buffer.phaseRecords(activeKey).length>=phaseMaximum()
      ? 'Pool full: recording continues with automatic diversity selection.'
      : 'Collection armed. Waiting for stable captures of the detected phase…');
  }else if(budgetFull)status('Export budget reached. Save receiver captures, clear the downloaded batch, then resume.');
  else if(record.ok)status('Matrix detected. Start / resume collection when ready.');
  else status('Looking for a matrix. Fit all four markers and the header in view.');
  if(record.probe&&!record.ok)buffer.add(record,{alreadyValidated:true});
  refresh();
  const revision=buffer.revision(activeKey),previous=distances.get(activeKey);
  if(activeKey&&buffer.phaseRecords(activeKey).length&&previous?.revision!==revision){
    const key=activeKey;
    const result=await client.call('compare',{signatures:buffer.phaseSignatures(key)});
    if(token!==receiverEpoch)return;
    distances.set(key,{result,revision,previous:previous?.previous??null});refresh();
  }
}
function collectBurst(){
  if(!stream||collecting||budgetFull)return;showError();collecting=true;restartScanner();
  status('Collection armed. Samples will be saved after the detected header stays stable.');refresh();
}
function cancel(){
  collecting=false;restartScanner();
  status('Collection paused. Alignment continues; retained samples can be downloaded.');refresh();
}
function changePoolSize(){
  const maximum=phaseMaximum(),counts=new Map();
  for(const record of buffer.records)counts.set(record.key,(counts.get(record.key)??0)+1);
  if([...counts.values()].some(count=>count>maximum)){
    $('phaseMaximum').value=String(previousMaximum);
    showError('The new maximum is smaller than a saved phase pool. Save and clear this batch before lowering it.');
    return;
  }
  previousMaximum=maximum;showError();
  if(stream)restartScanner();refresh();
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
  cameraEpoch++;receiverEpoch++;collecting=false;scanner.stop();
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
    if(token===cameraEpoch)scanner.start();
  }catch(error){if(token===cameraEpoch){stopCamera();showError(error.message);}}
  finally{if(token===cameraEpoch){cameraStarting=false;refresh();}}
}
function chooseRole(value){if(busy)return;if(value==='sender')stopCamera();role=value;showError();refresh();}
function shutdown(){closeFocus(false);epoch++;busy=false;stopCamera();client.cancel();}
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
$('collectBurst').addEventListener('click',collectBurst);$('cancelBurst').addEventListener('click',cancel);
$('phaseMaximum').addEventListener('change',changePoolSize);
$('scanInterval').addEventListener('change',()=>{if(stream)restartScanner();});
$('clearCaptures').addEventListener('click',()=>{receiverEpoch++;scanner.stop();buffer.reset();budgetFull=false;distances.clear();originalAttempts.clear();phaseActivity.clear();activeKey=null;activeInfo=null;if(stream)scanner.start();status('Batch cleared. The physical session ID is retained for the next capture file.');refresh();});
$('newReceiverSession').addEventListener('click',()=>{receiverEpoch++;scanner.stop();buffer.reset();budgetFull=false;distances.clear();originalAttempts.clear();phaseActivity.clear();activeKey=null;activeInfo=null;receiverSession=session('receiver');if(stream)scanner.start();status('Samples cleared. A new physical receiver session is ready.');refresh();});
$('downloadCaptures').addEventListener('click',()=>saveJSON({...buffer.export({notes:$('captureNotes').value,sessionId:receiverSession,camera:cameraInfo()}),
  collection:{maxPerPhase:phaseMaximum(),keepRandom:0,scanIntervalMs:Number($('scanInterval').value),
    distances:[...distances].map(([key,value])=>({key,...value.result,current:value.revision===buffer.revision(key),beforeThinning:value.previous}))}},`decoder-captures-${receiverSession}-${new Date().toISOString().replace(/[:.]/g,'-')}.json`));
window.addEventListener('pagehide',shutdown);document.addEventListener('visibilitychange',()=>{if(document.hidden)shutdown();});
if(!supported)showError('Use a current browser with Web Workers and Web Crypto on trusted HTTPS or localhost.');
refresh();enumerateCameras();
