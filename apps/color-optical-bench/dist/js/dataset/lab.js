import {CaptureClient} from './client.js';
import {CaptureBuffer} from './capture.js';
import {ScanLoop} from './scan-loop.js';
import {recordingFilename} from './filenames.js';
import {captureProfile,recommendedPhaseMaximum} from './contract.js';
import {collectionIdentity,environmentMetadata,cameraMetadata,saveDeviceLabel} from './metadata.js';
import {recordingUploadConfig,uploadRecording} from './upload.js';
import {project} from '../vision.js';

const $=id=>document.getElementById(id),client=new CaptureClient(),buffer=new CaptureBuffer();
const supported=isSecureContext&&typeof Worker==='function'&&Boolean(globalThis.crypto?.subtle);
const session=prefix=>`${prefix}-${globalThis.crypto?.randomUUID?.()??'unavailable'}`;
let senderSession=session('sender'),receiverSession=session('receiver'),sender=null,phase=0,role='sender';
let busy=false,epoch=0,stream=null,cameraStarting=false,cameraEpoch=0;
let previewMirrored=false;
const identity=collectionIdentity();
let uploadConfig={enabled:false},uploading=false,uploadController=null,uploadEpoch=0;
let senderSplit=null,receiverSplit=null,senderSaved=false,receiverRevision=0,receiverSavedRevision=0;
let receiverSnapshot=null,receiverEnvironment=null,lastCamera={},sessionCameraId=null;
let lastMediaTime=-1,collecting=false,budgetFull=false,receiverEpoch=0,activeKey=null,activeInfo=null;
const distances=new Map(),originalAttempts=new Set(),phaseActivity=new Map(),phaseReferences=new Map();
let previousMaximum=32,receiverAssignment=null;
function selectedAssignment(){
  const collectionCode=$('receiverCode').value.trim().toLowerCase(),size=Number($('receiverSize').value);
  if(!/^[a-f0-9]{8}$/.test(collectionCode))return null;
  return {collectionCode,profile:captureProfile(size,16,4)};
}
function assignment(){return receiverAssignment??selectedAssignment();}
function changeAssignment(){
  if(collecting||uploading||buffer.records.length||buffer.diagnostics.length){
    const current=receiverAssignment;if(current){$('receiverCode').value=current.collectionCode;$('receiverSize').value=String(current.profile.size);}
    showError('Save and clear this batch before changing its collection code or grid.');return;
  }
  receiverAssignment=null;activeKey=null;activeInfo=null;clearOverlay();
  const selected=selectedAssignment();
  if(selected){const maximum=recommendedPhaseMaximum(selected.profile.size,4);if(phaseMaximum()>maximum){$('phaseMaximum').value=String(maximum);previousMaximum=maximum;}}
  if(stream)restartScanner();showError();refresh();
}
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
  const track=stream?.getVideoTracks()[0];
  const info=cameraMetadata(track,{requestedHeight:Number($('resolution').value),previous:lastCamera});
  if(track)lastCamera=info;
  return info;
}
function environment(){return environmentMetadata({...identity,deviceLabel:$('deviceLabel').value});}
function invalidateReceiver(){receiverRevision++;receiverSnapshot=null;}
function receiverDirty(){return (buffer.records.length||buffer.diagnostics.length)&&receiverRevision!==receiverSavedRevision;}
function canDiscardReceiver(){return !receiverDirty()||confirm('This batch has not been uploaded or downloaded. Discard the retained recordings?');}
function canDiscardSender(){return !sender||senderSaved||confirm('This sender manifest has not been uploaded or downloaded. Discard it?');}
function snapshotReceiver(){
  if(!receiverSnapshot)receiverSnapshot={...buffer.export({notes:$('captureNotes').value,sessionId:receiverSession,camera:cameraInfo()}),
    datasetSplit:receiverSplit??$('datasetSplit').value,environment:receiverEnvironment??environment(),
    ...assignment(),
    collection:{maxPerPhase:phaseMaximum(),keepRandom:0,scanIntervalMs:Number($('scanInterval').value),
      distances:[...distances].map(([key,value])=>({key,...value.result,current:value.revision===buffer.revision(key),beforeThinning:value.previous}))}};
  return receiverSnapshot;
}
async function checkUploads(){
  if(uploading)return;
  $('refreshUploads').disabled=true;uploadConfig=await recordingUploadConfig();
  $('uploadAvailability').textContent=uploadConfig.enabled
    ? 'Recording server ready. Upload saves to the selected split and matrix-size folder. Files stay on this device until you choose Upload.'
    :uploadConfig.reason;
  $('refreshUploads').disabled=false;refresh();
}
async function upload(kind){
  if(uploading||busy)return;
  if(kind==='sender'&&!sender)return;
  if(kind==='receiver'&&(!buffer.records.length&&!buffer.diagnostics.length))return;
  collecting=false;receiverEpoch++;scanner.stop();
  const recording=kind==='sender'?sender.manifest:snapshotReceiver(),revision=receiverRevision;
  const target=$(kind==='sender'?'senderUploadStatus':'receiverUploadStatus');
  uploading=true;const token=++uploadEpoch;uploadController=new AbortController();showError();refresh();
  $('uploadProgress').hidden=false;target.textContent='Preparing recording upload…';
  try{
    const result=await uploadRecording(recording,{split:recording.datasetSplit,config:uploadConfig,signal:uploadController.signal,
      onProgress:({loaded,total})=>{if(token!==uploadEpoch)return;$('uploadProgress').value=total?loaded/total*100:0;
        target.textContent=loaded<total?`Uploading ${(loaded/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MiB…`:'Upload sent. Waiting for the server to confirm storage…';}});
    if(token!==uploadEpoch)return;
    if(kind==='sender')senderSaved=true;else receiverSavedRevision=revision;
    target.textContent=`${result.duplicate?'Already saved':'Saved'}: ${result.storedPath}`;
    if(kind==='receiver')status('Recording saved on the server. Clear this saved batch to collect another matrix or size.');
  }catch(error){if(token===uploadEpoch){target.textContent=error.message;showError(error.message);}}
  finally{if(token===uploadEpoch){uploading=false;uploadController=null;$('uploadProgress').hidden=true;refresh();if(stream)scanner.start();}}
}
function refresh(){
  const counts=buffer.summary(),locked=busy||uploading,split=role==='sender'?senderSplit:receiverSplit;
  $('senderRole').disabled=locked;$('receiverRole').disabled=locked;
  $('datasetSplit').disabled=locked||Boolean(split);if(split)$('datasetSplit').value=split;
  $('deviceLabel').disabled=locked||Boolean(split);$('captureNotes').disabled=uploading;
  $('environmentSummary').textContent=`Device: ${$('deviceLabel').value||identity.deviceId} · ${environment().deviceClass} (browser estimate). Camera labels and actual settings are recorded after camera permission.`;
  $('senderRole').setAttribute('aria-pressed',String(role==='sender'));
  $('receiverRole').setAttribute('aria-pressed',String(role==='receiver'));
  $('senderPanel').hidden=role!=='sender';$('receiverPanel').hidden=role!=='receiver';
  for(const id of ['senderSize','createSender','newSenderSession'])$(id).disabled=!supported||locked;
  $('downloadSender').disabled=!sender||locked;$('uploadSender').disabled=!sender||locked||!uploadConfig.enabled;
  $('nextPhase').disabled=!sender||sender.frames.length<2||locked;
  $('focusMatrix').disabled=!sender||locked||$('senderCanvasWrap').hidden;
  $('senderSession').textContent=`Session: ${senderSession}`;$('receiverSession').textContent=`Session: ${receiverSession}`;
  const assigned=assignment(),assignmentLocked=locked||collecting||Boolean(counts.records||counts.diagnostics);
  $('receiverCode').disabled=assignmentLocked;$('receiverSize').disabled=assignmentLocked;
  $('senderCollectionCode').textContent=sender?.manifest.collectionCode??'Create a matrix first';
  $('startCamera').disabled=!supported||cameraStarting||Boolean(stream)||locked||!assigned;
  $('stopCamera').disabled=!stream&&!cameraStarting;
  $('camera').disabled=Boolean(stream)||cameraStarting||uploading;$('resolution').disabled=Boolean(stream)||cameraStarting||uploading;
  $('cameraPreview').setAttribute('data-mirrored',String(previewMirrored));
  $('mirrorPreview').setAttribute('aria-pressed',String(previewMirrored));
  const phaseCount=buffer.phaseRecords(activeKey).length;
  $('collectBurst').disabled=!stream||collecting||budgetFull||uploading||!assigned;
  $('cancelBurst').disabled=!collecting;
  $('clearCaptures').disabled=collecting||uploading;$('newReceiverSession').disabled=collecting||uploading;
  $('downloadCaptures').disabled=collecting||uploading||(!counts.records&&!counts.diagnostics);
  $('uploadCaptures').disabled=uploading||!uploadConfig.enabled||(!counts.records&&!counts.diagnostics);
  $('phaseMaximum').disabled=collecting||uploading;$('scanInterval').disabled=uploading;$('saveOriginal').disabled=uploading;
  $('phaseCount').textContent=`${phaseCount} / ${phaseMaximum()}`;
  const activity=phaseActivity.get(activeKey);
  $('seenCount').textContent=String(activity?.seen??0);
  $('replacementCount').textContent=String(activity?.replaced??0);
  const phaseCounts=activeInfo?Array.from({length:activeInfo.phases},(_,p)=>buffer.records.filter(row=>row.info.id===activeInfo.id&&row.info.phase===p).length):[];
  $('activePhase').textContent=activeInfo?`${activeInfo.size}×${activeInfo.size} · phase ${activeInfo.phase+1}/${activeInfo.phases} · retained [${phaseCounts.join(', ')}] · ${activeInfo.id}`:'No matrix detected yet';
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
  const frame=sender.frames[index],image=await client.call('render',{frame,pixels:frame.size>=128?2048:1024});
  if(token!==epoch)return;
  phase=index;
  const canvas=$('senderCanvas');canvas.width=image.width;canvas.height=image.height;
  canvas.getContext('2d').putImageData(new ImageData(image.data,image.width,image.height),0,0);
  $('senderCanvasWrap').hidden=false;
  $('senderStatus').textContent=`${frame.size}×${frame.size}, ${frame.colors} colors · phase ${phase+1}/${sender.frames.length} · collection ${sender.manifest.collectionCode}. Keep this image unchanged while the receiver collects.`;
}
async function createSender(){
  if(busy||uploading||!canDiscardSender())return;showError();busy=true;const token=++epoch;$('senderCanvasWrap').hidden=true;refresh();
  try{
    const created=await client.call('create',{...captureProfile(Number($('senderSize').value),16,4),sessionId:senderSession});
    if(token!==epoch)return;
    senderSplit??=$('datasetSplit').value;
    sender={...created,manifest:{...created.manifest,datasetSplit:senderSplit,environment:environment()}};
    senderSaved=false;$('senderUploadStatus').textContent='Save this manifest once; it describes every palette phase.';
    phase=0;await displayPhase(0,token);
  }catch(error){if(token===epoch)showError(error.message);}
  finally{if(token===epoch){busy=false;refresh();}}
}
async function nextPhase(){
  if(busy||uploading||!sender)return;showError();busy=true;const token=++epoch;refresh();
  try{await displayPhase((phase+1)%sender.frames.length,token);}
  catch(error){if(token===epoch)showError(error.message);}finally{if(token===epoch){busy=false;refresh();}}
}
function overlay(record,width,height){
  const canvas=$('alignmentOverlay');canvas.width=width;canvas.height=height;
  const context=canvas.getContext('2d');context.clearRect(0,0,width,height);
  const points=(record.ok?record.corners:record.tentativeCorners)??(record.transform?[[0,0],[256,0],[256,256],[0,256]].map(([x,y])=>project(record.transform,x,y)):null);
  if(!Array.isArray(points)||points.length!==4||points.some(point=>!Array.isArray(point)||point.length!==2||point.some(value=>!Number.isFinite(value))))return;
  context.strokeStyle=record.ok?'#8ff3ad':'#ffbe32';context.lineWidth=Math.max(2,width/400);context.beginPath();
  points.forEach(([x,y],i)=>i?context.lineTo(x,y):context.moveTo(x,y));context.closePath();context.fillStyle=record.ok?'rgba(50,235,135,0.26)':'rgba(255,190,50,0.25)';context.fill();context.stroke();
}
function showCapture(record){
  $('latestCapture').textContent=record.ok
    ? `Orientation and phase cues accepted · ${record.info.size}×${record.info.size} / ${record.info.colors} colors · phase ${record.info.phase+1}/${record.info.phases}. Collection ${record.info.id}; payload labels are joined offline.`
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
  const selected=assignment();if(!selected)throw Error('Enter the sender’s eight-character collection code and select its grid.');
  const options={...currentImage(),...selected};
  const full=collecting;
  options.original=full&&$('saveOriginal').checked&&activeKey&&!originalAttempts.has(activeKey)&&buffer.originals<2;
  if(options.original)originalAttempts.add(activeKey);
  const record=await client.call(full?'capture':'probe',options,[options.image.data.buffer]);
  return {record,width:options.image.width,height:options.image.height};
}
async function receiveScan({record,width,height}){
  const token=receiverEpoch;
  if(uploading)return;
  overlay(record,width,height);showCapture(record);
  const selected=assignment();
  if(record.ok&&(!selected||record.info.id!==selected.collectionCode||record.info.size!==selected.profile.size||record.info.colors!==16||record.info.phases!==4)){
    collecting=false;buffer.resetStability();status('Received a different profile or collection code. Save and clear the batch before changing its assignment.');refresh();return;
  }
  if(record.ok){activeKey=record.key;activeInfo=record.info;}else buffer.resetStability();
  if(collecting&&record.ok){
    const first=buffer.records[0];
    if(first&&(first.info.size!==record.info.size||first.info.colors!==record.info.colors||first.info.phases!==record.info.phases)){
      collecting=false;buffer.resetStability();
      status(`Detected a different profile (G${record.info.size}, ${record.info.colors} colors, ${record.info.phases} phases). Save and clear this G${first.info.size} batch before changing profiles.`);
      refresh();return;
    }
    if(first&&first.camera?.deviceId&&record.camera?.deviceId&&first.camera.deviceId!==record.camera.deviceId){
      collecting=false;buffer.resetStability();status('Camera changed. Save this batch and start a new physical session before collecting.');refresh();return;
    }
    const maximum=recommendedPhaseMaximum(record.info.size,record.info.phases);
    if(!first&&phaseMaximum()>maximum){$('phaseMaximum').value=String(maximum);previousMaximum=maximum;invalidateReceiver();}
  }
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
        const selected=await client.call('select',{key:record.key,reference:phaseReferences.get(record.key),signatures:buffer.phaseSignatures(record.key),incomingSignature:record.signature});
        if(token!==receiverEpoch)return;
        if(selected.reference?.ready)phaseReferences.set(record.key,selected.reference);
        if(selected.reason==='no-majority-cluster'){
          collecting=false;buffer.resetStability();
          showError('This phase’s initial pool has no consistent majority. Collection paused: save or clear this batch, check the code, grid and camera alignment, then record a steadier pool.');
          status('Collection paused: inconsistent initial sample pool.');refresh();return;
        }
        replaceIndex=selected.replaceIndex;
      }
    }
    const outcome=buffer.add(record,{alreadyValidated:true,maxPerPhase:phaseMaximum(),replaceIndex});
    if(outcome.stage!=='kept')invalidateReceiver();
    if(outcome.stage==='replaced')phaseActivity.get(record.key).replaced++;
    if(outcome.stage==='full'){
      collecting=false;budgetFull=true;
      status('Export budget reached. Save or upload receiver captures, clear the saved batch, then resume.');
    }else if(!record.ok)status('Looking for clear orientation and matching phase cues. Payload errors remain useful training evidence.');
    else if(outcome.stage==='settling')status('Matrix detected. Waiting for stable geometry and phase cues before saving samples…');
    else status(buffer.phaseRecords(activeKey).length>=phaseMaximum()
      ? 'Pool full: balancing core, shoulder and tail samples as new captures arrive. Pause when ready to save.'
      : 'Collecting this phase. Keep the matrix static; small changes in angle or focus can add variety.');
  }else if(collecting){
    status(record.ok&&buffer.phaseRecords(activeKey).length>=phaseMaximum()
      ? 'Pool full: recording continues with balanced sample selection.'
      : 'Collection armed. Waiting for stable captures of the detected phase…');
  }else if(budgetFull)status('Export budget reached. Save or upload receiver captures, clear the saved batch, then resume.');
  else if(record.ok)status('Matrix detected. Start / resume collection when ready.');
  else status('Looking for a Neural-QR. Fit all four markers, phase cues and palette strips in view.');
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
  if(!stream||collecting||budgetFull||uploading)return;
  const selected=assignment();if(!selected){showError('Enter the sender’s collection code and grid before recording.');return;}
  receiverAssignment=selected;
  receiverSplit??=$('datasetSplit').value;receiverEnvironment??=environment();
  sessionCameraId??=cameraInfo().deviceId??cameraInfo().label??null;
  showError();collecting=true;restartScanner();
  status('Collection armed. Keep this matrix static; samples save after orientation and phase cues settle.');refresh();
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
  const first=buffer.records[0];
  if(first&&maximum>recommendedPhaseMaximum(first.info.size,first.info.phases)){
    $('phaseMaximum').value=String(previousMaximum);showError('That pool size would leave insufficient room for the other phases of this matrix.');return;
  }
  previousMaximum=maximum;invalidateReceiver();showError();
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
  cameraInfo();
  stream?.getTracks().forEach(track=>track.stop());stream=null;cameraStarting=false;lastMediaTime=-1;buffer.resetStability();
  $('video').srcObject=null;$('cameraPlaceholder').hidden=false;$('cameraStatus').textContent='Camera is off. Its label and settings remain attached to collected samples.';
  const canvas=$('alignmentOverlay');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);refresh();
}
async function startCamera(){
  if(cameraStarting||stream||busy||uploading)return;if(!assignment()){showError('Enter the sender’s collection code and grid first.');return;}showError();cameraStarting=true;const token=++cameraEpoch;refresh();
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw Error('Camera access requires trusted HTTPS or localhost.');
    const height=Number($('resolution').value),width=height===480?640:Math.round(height*16/9);
    const acquired=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:width},height:{ideal:height},
      ...($('camera').value?{deviceId:{exact:$('camera').value}}:{facingMode:{ideal:'environment'}})}});
    if(token!==cameraEpoch){acquired.getTracks().forEach(track=>track.stop());return;}
    const newCamera=cameraMetadata(acquired.getVideoTracks()[0],{requestedHeight:height}),newId=newCamera.deviceId??newCamera.label??null;
    if(sessionCameraId&&newId&&sessionCameraId!==newId){
      acquired.getTracks().forEach(track=>track.stop());throw Error('A different camera needs a new physical session. Save the current batch, then select New physical session before starting this camera.');
    }
    stream=acquired;$('video').srcObject=stream;await $('video').play();if(token!==cameraEpoch)return;
    lastMediaTime=-1;$('cameraPlaceholder').hidden=true;const actual=cameraInfo();
    $('cameraStatus').textContent=`Actual capture ${actual.width}×${actual.height}, ${actual.frameRate?.toFixed(0)??'unknown'} fps. Fit all four markers, phase cues and palette strips in view.`;
    await enumerateCameras();
    if(token===cameraEpoch)scanner.start();
  }catch(error){if(token===cameraEpoch){stopCamera();showError(error.message);}}
  finally{if(token===cameraEpoch){cameraStarting=false;refresh();}}
}
function chooseRole(value){if(busy||uploading)return;if(value==='sender')stopCamera();role=value;showError();refresh();}
function shutdown(){
  closeFocus(false);epoch++;busy=false;
  if(uploading){$(role==='sender'?'senderUploadStatus':'receiverUploadStatus').textContent='Upload interrupted. The recording remains available; retrying is safe.';}
  uploadEpoch++;uploadController?.abort();uploadController=null;uploading=false;$('uploadProgress').hidden=true;
  stopCamera();client.cancel();
}
function clearReceiver(newSession=false){
  if(collecting||uploading||!canDiscardReceiver())return;
  receiverEpoch++;scanner.stop();buffer.reset();budgetFull=false;distances.clear();originalAttempts.clear();phaseActivity.clear();phaseReferences.clear();
  activeKey=null;activeInfo=null;receiverAssignment=null;invalidateReceiver();receiverSavedRevision=receiverRevision;
  if(newSession){receiverSession=session('receiver');receiverSplit=null;receiverEnvironment=null;sessionCameraId=null;}
  $('receiverUploadStatus').textContent='';if(stream)scanner.start();
  status(newSession?'A new physical receiver session is ready. Select its split before recording.':'Saved batch cleared. The physical session and split are retained for the next recording.');refresh();
}
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
$('downloadSender').addEventListener('click',()=>{if(sender&&!uploading){saveJSON(sender.manifest,recordingFilename(sender.manifest));senderSaved=true;}});
$('newSenderSession').addEventListener('click',()=>{if(uploading||busy||!canDiscardSender())return;senderSession=session('sender');sender=null;senderSplit=null;senderSaved=false;phase=0;$('senderCanvasWrap').hidden=true;$('senderUploadStatus').textContent='';$('senderStatus').textContent='New physical session. Select its split, then create and save a fresh sequence.';refresh();});
$('uploadSender').addEventListener('click',()=>upload('sender'));
$('uploadCaptures').addEventListener('click',()=>upload('receiver'));
$('refreshUploads').addEventListener('click',checkUploads);
$('deviceLabel').value=identity.deviceLabel;
$('deviceLabel').addEventListener('input',()=>{saveDeviceLabel($('deviceLabel').value);refresh();});
$('datasetSplit').addEventListener('change',()=>{invalidateReceiver();refresh();});
$('captureNotes').addEventListener('input',invalidateReceiver);
$('startCamera').addEventListener('click',startCamera);$('stopCamera').addEventListener('click',stopCamera);
// CSS mirrors the displayed video and alignment overlay together. The camera
// source, currentImage() canvas, worker inputs, and exported pixels stay intact.
$('mirrorPreview').addEventListener('click',()=>{previewMirrored=!previewMirrored;refresh();});
$('collectBurst').addEventListener('click',collectBurst);$('cancelBurst').addEventListener('click',cancel);
$('phaseMaximum').addEventListener('change',changePoolSize);
$('receiverCode').addEventListener('input',changeAssignment);$('receiverSize').addEventListener('change',changeAssignment);
$('scanInterval').addEventListener('change',()=>{invalidateReceiver();if(stream)restartScanner();});
$('clearCaptures').addEventListener('click',()=>clearReceiver());
$('newReceiverSession').addEventListener('click',()=>clearReceiver(true));
$('downloadCaptures').addEventListener('click',()=>{if(collecting||uploading)return;const recording=snapshotReceiver();saveJSON(recording,recordingFilename(recording));receiverSavedRevision=receiverRevision;});
window.addEventListener('beforeunload',event=>{if(receiverDirty()||sender&&!senderSaved||uploading){event.preventDefault();event.returnValue='';}});
window.addEventListener('pagehide',shutdown);document.addEventListener('visibilitychange',()=>{if(document.hidden)shutdown();});
if(!supported)showError('Use a current browser with Web Workers and Web Crypto on trusted HTTPS or localhost.');
refresh();enumerateCameras();checkUploads();
