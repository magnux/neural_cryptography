import {PALETTE,capacity,encodeSequence,randomBytes,renderFrame,hex,crc32,phaseFrames,transportProfile} from './format.js';
import {CONTENT_TYPES,contentDescriptor,imageType,sampleWav,samplePng} from './content.js';
import {ContentView,downloadBlob} from './content-view.js';
import {ReceiveSession} from './receive-session.js';
import {DecoderClient} from './decoder-client.js';
import {probePayload,ExperimentReport} from './diagnostics.js';
import {TransportView} from './transport-view.js';

const $=id=>document.getElementById(id),fmt=n=>new Intl.NumberFormat().format(n);
const state={frames:[],timeline:[],payload:null,index:0,source:null,playing:false,timer:null,files:{},fileToken:0,
  stream:null,cameraTimer:null,cameraOpening:false,workspace:'live',view:'both',history:[],labBusy:false};
const live={session:new ReceiveSession(),viewer:new ContentView($('liveContent')),worker:null,source:'camera'};
const lab={session:new ReceiveSession(),viewer:new ContentView($('labContent')),worker:null,source:'none',expected:null};
const channels={live,lab};
for(const [name,channel] of Object.entries(channels)){channel.report=new ExperimentReport();channel.transportView=new TransportView($(`${name}Calibration`));}
const HINTS={diagnostic:'Known test content fills every data frame. Both decoders classify independently; diagnostic truth is used afterward to count errors.',random:'Random bytes fill the color payload. The content descriptor travels in the monochrome header.',raw:'Choose any binary file. Its original bytes are recovered and downloadable.',text:'The complete UTF-8 message is displayed after reception, including accents, emoji, and line breaks.',image:'PNG for graphics; JPEG for photos. Small files are easiest to transmit. File bytes are preserved.',audio:'Use a short PCM WAV: mono/stereo, 8- or 16-bit samples. Playback is available after reception.'};
function log(message){state.history.push({at:new Date().toISOString(),message});if(state.history.length>60)state.history.shift();}
function notice(message){$('environmentNotice').textContent=message;$('environmentNotice').hidden=!message;}
function paint(canvas,image){canvas.width=image.width;canvas.height=image.height;canvas.getContext('2d').putImageData(new ImageData(image.data,image.width,image.height),0,0);}
function clearCanvas(canvas){canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);}
function status(channel,title,detail,kind=''){
  const prefix=channel===live?'':'lab';
  $(prefix?'labStatusTitle':'statusTitle').textContent=title;
  $(prefix?'labStatusDetail':'statusDetail').textContent=detail;
  $(prefix?'labStatus':'receiveStatus').className=`receive-status ${kind}`;
}
function swatches(container,colors){container.replaceChildren(...colors.map(rgb=>{const span=document.createElement('span');span.style.background=rgb?`rgb(${rgb.map(v=>Math.round(v)).join(',')})`:'#edf0e7';return span;}));}
function config(){return {size:Number($('matrixSize').value),colors:Number($('colorCount').value),maxFrames:Number($('frameLimit').value),gridBlock:Number($('gridBlock').value),parity:Number($('fecParity').value),phases:Number($('phaseCount').value),permute:Number($('phaseCount').value)>1&&$('phaseMode').value==='permute',holdMs:Math.round(1000/Number($('fps').value))};}
function receiverConfig(name){
  return {filter:$(`${name}Filter`).value,window:Number($(name==='live'?'liveWindow':'labSamples').value),overhead:Number($(`${name}Overhead`).value),minPhases:Number($(`${name}MinPhases`).value),recovery:$(`${name}Recovery`).value,spatial:$(`${name}Spatial`).value,settleMs:name==='live'?300:0};
}
function refreshSamplingControls(){
  for(const name of ['live','lab']){
    const filter=$(`${name}Filter`).value,busy=name==='lab'&&state.labBusy;
    $(name==='live'?'liveWindow':'labSamples').disabled=busy||filter==='none';
    $(`${name}Overhead`).disabled=busy||filter!=='robust';
  }
}
function temporalSummary(t){
  const pool=t.filter==='robust'?`select ${t.target} of ${t.poolCapacity}`:`target ${t.target}`;
  const limited=t.memoryLimited?` · memory-limited: requested ${t.requested} samples, ${t.requestedPool} pool slots; retained phase limit ${t.retainedPhaseLimit}/${t.phases}`:'';
  return `${t.filter} · phase samples ${t.counts.join(' / ')} · ${pool} each · ${t.eligiblePhases.length}/${t.minPhases} sampled phases needed${t.ready?' · refining':' · collecting'}${limited} · latest single capture ${t.singleCRC?'verified':'not verified'}${t.evicted?` · ${t.evicted} old windows released`:''}`;
}
function displayedFrames(){return state.timeline.length?state.timeline:state.frames;}
function advanceFrame(){
  const frames=displayedFrames();if(!frames.length)return;
  state.index++;
  if(state.index>=frames.length)state.index=0;
  showFrame();
}
function setPlaying(playing){
  state.playing=playing&&displayedFrames().length>1;clearTimeout(state.timer);state.timer=null;
  $('playButton').textContent=state.playing?'Ⅱ Pause':'▶ Play';
  if(state.playing){
    const hold=displayedFrames()[state.index]?.holdMs??Math.round(1000/Number($('fps').value));
    state.timer=setTimeout(()=>{advanceFrame();setPlaying(true);},hold);
  }
}
function refreshTransmitButtons(){
  $('nextButton').disabled=displayedFrames().length<2;$('playButton').disabled=displayedFrames().length<2;
  $('savePNG').disabled=!state.frames.length;
  $('loopbackButton').disabled=state.labBusy?(lab.source!=='simulated'||state.cancelLab):!state.frames.length;
  $('loopbackButton').textContent=state.labBusy&&lab.source==='simulated'?'Stop test':'Run transport test ↗';
}
function renderSize(frame){return Math.min(4096,Math.max(1024,Math.ceil(frame.size*6.4/256)*256));}
function showFrame(){
  if(!state.frames.length)return;
  const frame=displayedFrames()[state.index];state.source=renderFrame(frame,renderSize(frame));paint($('matrixCanvas'),state.source);
  $('matrixCanvas').hidden=false;$('matrixEmpty').hidden=true;
  $('sequenceId').textContent=state.view==='receive'?'LIVE CAMERA · CONTENT RECEIVER':`SEQ ${frame.id.slice(0,8)} · OBM5`;
  $('frameLabel').textContent=`DATA ${frame.index+1} / ${frame.count} · PHASE ${frame.phase+1} / ${frame.phases}`;
  $('densityLabel').textContent=`${frame.size} × ${frame.size} · ${Math.log2(frame.colors)} bits/cell`;refreshTransmitButtons();
}
function clearTransmission(message){
  setPlaying(false);state.frames=[];state.timeline=[];state.payload=null;state.source=null;
  $('matrixCanvas').hidden=true;$('matrixEmpty').hidden=false;$('matrixEmpty').textContent=message;
  $('sequenceId').textContent='TRANSMITTER NOT READY';$('frameLabel').textContent='NO FRAMES';$('densityLabel').textContent='';
  $('setupSummary').textContent='Content needs attention';refreshTransmitButtons();
}
function generate({expandSample=false}={}){
  try{
    const options=config(),cap=capacity(options.size,options.colors,options),mode=$('payloadMode').value;
    let bytes,type;
    if(mode==='diagnostic'){
      type='raw';options.probePattern=Number($('probePattern').value);options.probeSeed=Number($('probeSeed').value);
      if(!Number.isInteger(options.probeSeed)||options.probeSeed<1||options.probeSeed>4294967295)throw Error('Use a payload seed from 1 to 4294967295.');
      bytes=probePayload(options);
    }
    else if(mode==='random'){type='raw';bytes=randomBytes(cap*options.maxFrames);}
    else if(mode==='text'){type='text';bytes=new TextEncoder().encode($('payloadText').value);}
    else{
      const file=state.files[mode];if(!file)throw Error(`Choose ${mode==='image'?'a PNG/JPEG image':mode==='audio'?'a PCM WAV audio file':'a binary file'} in transmission settings.`);
      ({bytes,type}=file);
    }
    const payload=bytes,needed=Math.max(1,Math.ceil(payload.length/cap));
    if(expandSample&&needed>options.maxFrames&&needed<=8){options.maxFrames=needed;$('frameLimit').value=String(needed);$('contentHint').textContent=`Sample loaded. Frame limit increased to ${needed} to fit it. ${HINTS[mode]}`;}
    if(needed>options.maxFrames)throw Error(`${fmt(bytes.length)} content bytes need ${needed} frames at this profile; the limit is ${options.maxFrames}. Increase the frame limit, grid, or palette, or choose smaller content.`);
    const frames=encodeSequence(payload,{...options,contentType:type});
    setPlaying(false);state.frames=frames;state.timeline=phaseFrames(frames);state.payload=payload;state.index=0;showFrame();
    const profile=transportProfile(options.size,options.colors,options);
    $('phaseMode').disabled=options.phases===1;
    $('capacityLabel').textContent=`${fmt(bytes.length)} content B · ${frames.length} frame${frames.length===1?'':'s'} · ${fmt(cap)} content B/frame · ${profile.pilotCells} reference cells · ${profile.blocks*options.parity} parity B/frame · ${options.phases} phases/frame · ${(options.holdMs/1000).toFixed(2)} s/phase`;
    $('setupSummary').textContent=`${options.size} × ${options.size} · ${options.colors} colors · ${CONTENT_TYPES[type].label} · ${frames.length} frame${frames.length===1?'':'s'}`;
    $('transmitError').hidden=true;log(`Generated ${type}: ${bytes.length} content bytes, ${frames.length} frame(s).`);
  }catch(error){
    clearTransmission(error.message);$('transmitError').textContent=error.message;$('transmitError').hidden=false;
    try{const c=config();$('capacityLabel').textContent=`${fmt(capacity(c.size,c.colors,c)*c.maxFrames)} content bytes available; descriptor is outside the color payload`;}catch{$('capacityLabel').textContent='Choose a valid optical profile.';}
  }
}
function updatePayloadControls(){
  state.fileToken++;const mode=$('payloadMode').value;
  $('payloadText').hidden=mode!=='text';$('probeControls').hidden=mode!=='diagnostic';$('fileControls').hidden=!['raw','image','audio'].includes(mode);
  $('samplePayload').hidden=mode==='raw';$('payloadFile').accept=mode==='image'?'image/png,image/jpeg':mode==='audio'?'.wav,audio/wav,audio/x-wav':'';
  $('payloadFileLabel').textContent=state.files[mode]?.name??'No file selected.';$('contentHint').textContent=HINTS[mode];generate();
}
async function loadPayload(file){
  if(!file)return;
  const mode=$('payloadMode').value,token=++state.fileToken;
  try{
    if(file.size>capacity(512,16)*8)throw Error('This file exceeds the eight-frame maximum (1 MiB). Choose a smaller file.');
    const bytes=new Uint8Array(await file.arrayBuffer());if(token!==state.fileToken||mode!==$('payloadMode').value)return;
    const type=mode==='image'?imageType(bytes):mode==='audio'?'wav':'raw';contentDescriptor(bytes,type);
    state.files[mode]={bytes,type,name:file.name};$('payloadFileLabel').textContent=`${file.name} · ${fmt(bytes.length)} B`;
    $('contentHint').textContent=HINTS[mode];generate();
  }catch(error){
    if(token!==state.fileToken)return;
    delete state.files[mode];$('payloadFileLabel').textContent='File not accepted.';clearTransmission(error.message);
    $('transmitError').textContent=error.message;$('transmitError').hidden=false;
  }finally{$('payloadFile').value='';}
}
function loadSample(){
  const mode=$('payloadMode').value;if(!['image','audio'].includes(mode))return;
  state.fileToken++;const image=mode==='image',bytes=image?samplePng():sampleWav();
  state.files[mode]={bytes,type:image?'png':'wav',name:image?'sample-tile.png':'sample-tone.wav'};
  $('payloadFileLabel').textContent=`${state.files[mode].name} · ${fmt(bytes.length)} B`;generate({expandSample:true});
}
function drawOverlay(canvas,result,dimensions){
  canvas.width=dimensions.width;canvas.height=dimensions.height;const ctx=canvas.getContext('2d');ctx.lineWidth=Math.max(2,dimensions.width/400);ctx.strokeStyle='#c9eb9b';
  if(result.corners){ctx.beginPath();result.corners.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();}
  else for(const marker of result.markers??[]){ctx.beginPath();ctx.arc(marker.x,marker.y,marker.module*3.6,0,Math.PI*2);ctx.stroke();}
}
function pipeline(channel,result){
  const steps=['detect','header','settle','calibration','accumulate','payload','complete'],index=steps.indexOf(result.stage),name=channel===live?'live':'lab';
  for(const item of document.querySelectorAll(`[data-pipeline="${name}"] [data-stage]`)){
    const i=steps.indexOf(item.dataset.stage);item.className=result.ok||i<index?'done':i===index?'current':'';
  }
}
function resetReceiver(channel){
  channel.session.reset();channel.worker?.reset();channel.viewer.clear();channel.report.reset();channel.transportView.clear();
  const liveMode=channel===live;
  clearCanvas($(liveMode?'overlayCanvas':'labOverlay'));clearCanvas($(liveMode?'rectifiedCanvas':'labRectified'));
  for(const span of document.querySelectorAll(`[data-pipeline="${liveMode?'live':'lab'}"] [data-stage]`))span.className='';
  if(liveMode){
    $('metricFrames').innerHTML='0 <small>/ —</small>';$('metricCell').innerHTML='— <small>px</small>';$('metricColor').innerHTML='— <small>/ 255</small>';$('metricTime').innerHTML='— <small>ms</small>';
    $('metricIntegrity').textContent='Waiting';$('metricIntegrity').className='neutral';$('attemptCount').textContent='0 captures';
    $('rectifiedLabel').textContent='Waiting for detection';$('calibrationCaption').textContent='Known colors estimate RGB affine correction.';
    $('temporalStatus').textContent='Evidence cleared. Each cell selects from recent observations; phase thresholds are capped at the transmitted phase count.';
    $('matchLabel').textContent='LIVE CAMERA ONLY';swatches($('observedPalette'),Array(16).fill(null));
  }else{$('labMetrics').textContent='Geometry, color fit, and decode time will appear here.';$('labMatchLabel').textContent='IMAGE LAB ONLY';lab.experiment=null;$('burstComparison').hidden=true;}
  status(channel,liveMode?'Live receiver reset':'Image lab reset','Ready for a new sequence.');log(`${liveMode?'Live':'Lab'} receiver reset.`);
}
function renderDiagnostics(channel,result,dimensions){
  const s=result.stats??{},isLive=channel===live;
  pipeline(channel,result);drawOverlay($(isLive?'overlayCanvas':'labOverlay'),result,dimensions);
  if(result.rectified)paint($(isLive?'rectifiedCanvas':'labRectified'),result.rectified);
  if(!isLive){
    $('labMetrics').textContent=`${s.markerCount??0} finder candidates · ${Number.isFinite(s.ms)?Math.round(s.ms):'—'} ms\n${s.pixelsPerCell?.toFixed(1)??'—'} px/cell · RGB fit error ${s.colorRMSE?.toFixed(1)??'—'} / 255\n${s.temporal?temporalSummary(s.temporal):''}\nPreview: latest rectified capture.`;
    return;
  }
  $('metricTime').innerHTML=`${Number.isFinite(s.ms)?Math.round(s.ms):'—'} <small>ms</small>`;
  $('metricCell').innerHTML=`${s.pixelsPerCell?.toFixed(1)??'—'} <small>px</small>`;
  $('metricColor').innerHTML=`${(result.colorComparison?.global?.stats?.colorRMSE??s.colorRMSE)?.toFixed(1)??'—'} <small>/ 255</small>`;
  if(result.rectified)$('rectifiedLabel').textContent=`${result.info.size} × ${result.info.size} · ${result.info.colors} colors`;
  if(result.observedPalette)swatches($('observedPalette'),result.observedPalette.slice(0,16));
  if(Number.isFinite(s.confidence))$('calibrationCaption').textContent=`${result.decoder==='local'?'Current-frame grid':'Border fit'} · ${s.temporal?.filter??'single'} · color separation ${(s.confidence*100).toFixed(1)}% · ${fmt(s.lowConfidence)} uncertain cells.${s.pixelsPerCell<4?' Aim for ≥4 px/cell.':''} Preview: latest capture.`;
  if(s.temporal)$('temporalStatus').textContent=temporalSummary(s.temporal);
}
function handleResult(channel,result,dimensions,expected){
  channel.report.accept(result);channel.transportView.render(channel.report.snapshot());
  renderDiagnostics(channel,result,dimensions);
  if(result.stage==='settle'){
    channel.session.lastResult=result;
    status(channel,'Display settling',result.error,'working');return;
  }
  try{
    const progress=channel.session.accept(result,expected),session=channel.session;
    if(channel===live)$('attemptCount').textContent=`${session.captures} captures`;
    if(!progress.valid){
      const accumulating=result.stage==='accumulate'&&result.stats?.temporal&&!result.stats.temporal.ready;
      status(channel,result.stage==='detect'?'Looking for a matrix':accumulating?'Stabilizing the matrix':'Capture needs another look',result.error,'working');
      if(channel===live&&!session.received){$('metricIntegrity').textContent=accumulating?'Accumulating':'Not accepted';$('metricIntegrity').className=accumulating?'neutral':'failed';}return;
    }
    if(channel===live)$('metricFrames').innerHTML=`${progress.received} <small>/ ${progress.count}</small>`;
    if(!progress.complete){
      status(channel,`Received ${progress.received} of ${progress.count} frames`,'Content will appear when every frame has arrived.','working');
      if(channel===live){$('metricIntegrity').textContent='Frame valid';$('metricIntegrity').className='passed';}return;
    }
    if(channel===live){$('metricIntegrity').textContent='CRC passed';$('metricIntegrity').className='passed';}
    if(progress.fresh){
      if(progress.contentError)channel.viewer.error(progress.contentError,session.received,result.info.id);
      else channel.viewer.show(progress.content,result.info.id);
      const match=progress.exact?'EXACT BYTE MATCH':`CRC PASSED · ${result.info.id.slice(0,8)}`;
      $(channel===live?'matchLabel':'labMatchLabel').textContent=match;
      log(`${channel===live?'Live':'Lab'}: received ${session.received.length} wire bytes; ${progress.content?.label??progress.contentError}.`);
    }
    if(progress.contentError){status(channel,'Bytes received; content not supported',progress.contentError,'error');return;}
    const content=progress.content,detail=`${content.label}: ${fmt(content.bytes.length)} content bytes.${progress.exact?' Exact match with the generated payload.':' Frame and sequence checksums passed.'}`;
    status(channel,channel===lab&&channel.source==='simulated'?'Loopback passed':'Content received',detail,'success');
  }catch(error){status(channel,'Sequence not accepted',error.message,'error');log(error.message);}
}
async function decode(channel,image,expected=null,options=null){
  if(!channel.worker)throw Error('Decoder unavailable. Reload this app over HTTP(S).');
  const epoch=channel.session.epoch,dimensions={width:image.width,height:image.height};
  const settings=options??receiverConfig(channel===live?'live':'lab');
  const result=await channel.worker.decode(image,settings);
  if(epoch===channel.session.epoch)handleResult(channel,result,dimensions,expected);
  return result;
}
function stopCamera(){
  live.session.epoch++;live.worker?.reset();clearInterval(state.cameraTimer);state.cameraTimer=null;
  state.stream?.getTracks().forEach(track=>track.stop());state.stream=null;
  $('cameraVideo').srcObject=null;$('cameraVideo').hidden=true;$('capturePlaceholder').hidden=false;clearCanvas($('overlayCanvas'));
  $('cameraButton').textContent='Start camera';$('sourceBadge').textContent='CAMERA OFF';$('captureSize').textContent='';$('cameraIndicator').className='';
  $('cameraPlaceholderText').textContent='Start camera to align';
}
function cameraError(error){
  return {NotAllowedError:'Camera permission was denied. Allow this site to use the camera, then try again.',NotFoundError:'No camera was found. Image lab tests are still available.',NotReadableError:'The camera is busy or unavailable. Close other camera apps and try again.',OverconstrainedError:'The selected camera is unavailable. Choose another camera.'}[error.name]??error.message;
}
async function startCamera(){
  if(state.cameraOpening)return;
  if(state.stream){stopCamera();status(live,'Camera stopped','The received content is preserved.');return;}
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){status(live,'Camera needs a secure page','Use HTTPS or localhost.','error');return;}
  if(!live.worker){status(live,'Decoder unavailable','Reload the app over HTTP(S).','error');return;}
  const video=$('cameraVideo'),selection=$('cameraSelect').value,[requestedWidth,requestedHeight]=$('cameraResolution').value.split('x').map(Number);
  state.cameraOpening=true;for(const id of ['cameraButton','cameraSelect','cameraResolution'])$(id).disabled=true;
  status(live,'Opening camera','Allow camera access in your browser.','working');
  try{
    live.session.epoch++;live.worker.reset();const constraints={width:{ideal:requestedWidth,max:4096},height:{ideal:requestedHeight,max:4096},frameRate:{ideal:15,max:30}};
    if(['user','environment'].includes(selection))constraints.facingMode={ideal:selection};else constraints.deviceId={exact:selection};
    const stream=await navigator.mediaDevices.getUserMedia({video:constraints,audio:false});state.stream=stream;video.srcObject=stream;await video.play();
    // A view can change while the permission prompt is open.
    if(state.view==='display'){stopCamera();return;}
    video.hidden=false;$('capturePlaceholder').hidden=true;$('cameraButton').textContent='Stop camera';
    $('sourceBadge').textContent='LIVE';$('captureSize').textContent=`${video.videoWidth} × ${video.videoHeight}`;$('cameraIndicator').className='active';
    const settings=stream.getVideoTracks()[0].getSettings();
    state.cameraSettings={requested:{width:requestedWidth,height:requestedHeight},actual:{width:settings.width??video.videoWidth,height:settings.height??video.videoHeight,frameRate:settings.frameRate,exposureMode:settings.exposureMode,whiteBalanceMode:settings.whiteBalanceMode,focusMode:settings.focusMode},cameraLabel:stream.getVideoTracks()[0].label};
    $('cameraResolutionNote').textContent=`Requested ${requestedWidth} × ${requestedHeight} · actual ${video.videoWidth} × ${video.videoHeight}. Full-resolution cell sampling.`;
    status(live,'Camera running','Align all four markers with the small preview. Content appears below.','working');
    try{
      const selected=stream.getVideoTracks()[0].getSettings().deviceId,cameras=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
      if(cameras.length)$('cameraSelect').replaceChildren(...cameras.map((c,i)=>{const o=new Option(c.label||`Camera ${i+1}`,c.deviceId);o.selected=c.deviceId===selected;return o;}));
    }catch{/* Enumeration is optional. */}
    const scratch=document.createElement('canvas'),ctx=scratch.getContext('2d',{willReadFrequently:true});let lastVideoTime=-1;
    state.cameraTimer=setInterval(async()=>{
      if(!state.stream||live.worker.busy||video.readyState<2||document.hidden||state.workspace!=='live'||video.currentTime===lastVideoTime)return;
      lastVideoTime=video.currentTime;scratch.width=video.videoWidth;scratch.height=video.videoHeight;
      ctx.drawImage(video,0,0,scratch.width,scratch.height);const raw=ctx.getImageData(0,0,scratch.width,scratch.height);
      const expected=state.frames[0]?{id:state.frames[0].id,payload:state.payload}:null;
      try{await decode(live,{width:raw.width,height:raw.height,data:raw.data},expected);}catch(error){status(live,'Capture interrupted',error.message,'error');}
    },250);
  }catch(error){stopCamera();status(live,'Camera unavailable',cameraError(error),'error');}
  finally{state.cameraOpening=false;for(const id of ['cameraButton','cameraSelect','cameraResolution'])$(id).disabled=false;}
}
function setLabBusy(busy){state.labBusy=busy;for(const id of ['uploadButton','resetLab','simulationPreset'])$(id).disabled=busy;for(const el of $('labSettings').querySelectorAll('input,select'))el.disabled=busy;$('labComparePhases').disabled=busy;refreshSamplingControls();refreshTransmitButtons();}
function showLabImage(image,label,dimensions=image){paint($('captureCanvas'),image);$('captureCanvas').hidden=false;$('labPlaceholder').hidden=true;$('labSourceBadge').textContent=label;$('labCaptureSize').textContent=`${dimensions.width} × ${dimensions.height}`;}
function labConfig(){
  const settings={...receiverConfig('lab'),preset:$('simulationPreset').value,extraCaptures:Number($('labExtraCaptures').value),repeatCycles:Number($('labRepeatCycles').value),noise:Number($('labNoise').value),moire:Number($('labMoire').value),moirePeriod:Number($('labMoirePeriod').value),phaseMode:$('labPhase').value,resolution:$('labResolution').value,seed:Number($('labSeed').value),curvature:Number($('labCurvature').value),colorDependence:Number($('labColorDependence').value),gainSwing:Number($('labGain').value),blur:Number($('labBlur').value),comparePhases:$('labComparePhases').checked};
  settings.exposuresPerPhase=(settings.filter==='none'?1:settings.window)+settings.extraCaptures;
  if(!Number.isFinite(settings.moirePeriod)||settings.moirePeriod<1||settings.moirePeriod>32||!Number.isInteger(settings.seed)||settings.seed<1||settings.seed>4294967295)throw Error('Use a band spacing from 1–32 and an integer seed from 1–4294967295.');
  return settings;
}
function updateComparison(){
  const e=lab.experiment;$('burstProgress').textContent=`${e.captures} / ${e.target} exposures`;
  $('burstResults').replaceChildren(...e.runs.map(run=>{
    const tr=document.createElement('tr'),row=run.report?.methods?.accepted;
    for(const value of [run.name,`${run.captures} · ${run.headers} headers`,row?`${row.rawPasses}/${row.attempts}`:'—',row?`${row.hardPasses??0}/${row.attempts}`:'—',row?.erasurePasses??'—',row?.cells?`${(100*row.symbolErrorRate).toFixed(3)}%`:'Diagnostic needed',row?.firstVerifiedMs!=null?`${(row.firstVerifiedMs/1000).toFixed(1)} s · capture ${row.firstVerifiedCapture}`:'—',`${row?.frames?.length??0}/${e.frames}`]){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;
  }));
  $('burstNote').textContent='Equal requested exposures per variant, including refinement after success. Ordinary RS includes raw-valid captures; extra erasure passes recover candidates that ordinary RS could not verify. Each variant retains all decoder comparisons in the report. First verified time measures lab processing, not optical throughput.';
}
async function loopback(){
  if(state.labBusy||lab.worker?.busy||!state.frames.length)return;
  if(!lab.worker){status(lab,'Decoder unavailable','Reload the app over HTTP(S).','error');return;}
  lab.source='simulated';state.cancelLab=false;setLabBusy(true);resetReceiver(lab);
  status(lab,'Testing optical transport','Comparing current-frame color estimates and phase evidence.','working');
  try{
    const settings=labConfig(),first=state.frames[0],payload=state.payload.slice();
    const variants=settings.comparePhases&&first.phases>1?[false,true]:[first.permute];
    const experiment={settings,frames:first.count,captures:0,target:variants.length*first.count*first.phases*settings.exposuresPerPhase*settings.repeatCycles,runs:[]};
    lab.experiment=experiment;$('burstComparison').hidden=false;
    outer:for(const permute of variants){
      resetReceiver(lab);lab.experiment=experiment;$('burstComparison').hidden=false;
      const frames=encodeSequence(payload,{...first,maxFrames:first.count,id:randomBytes(8),contentType:first.content.type,permute});
      const expected={id:frames[0].id,payload};lab.expected=expected;
      const run={name:permute?'Palette permutations':'Identical repetitions',permute,captures:0,headers:0,report:null,complete:false};experiment.runs.push(run);let ordinal=0;
      for(let cycle=0;cycle<settings.repeatCycles;cycle++)for(const frame of phaseFrames(frames)){
        const width=settings.resolution==='auto'?(frame.size>=256?Math.min(4096,frame.size*12):960):Number(settings.resolution);
        for(let exposure=0;exposure<settings.exposuresPerPhase;exposure++,ordinal++){
          if(state.cancelLab)break outer;
          const drifting=settings.phaseMode==='shifted',phase=drifting?ordinal*1.1:0;
          const simulation={...settings,width,height:Math.round(width*760/960),seed:(settings.seed+ordinal*104729)>>>0,phase,
            gain:1+settings.gainSwing*Math.sin(ordinal*1.7),blur:settings.blur*(drifting?.65+.35*Math.sin(ordinal*1.3):1)};
          const epoch=lab.session.epoch,result=await lab.worker.simulate(frame,renderSize(frame),simulation,{filter:settings.filter,window:settings.window,overhead:settings.overhead,minPhases:settings.minPhases,recovery:settings.recovery,spatial:settings.spatial,settleMs:0});
          const {inputPreview,inputDimensions}=result;delete result.inputPreview;delete result.inputDimensions;
          if(epoch!==lab.session.epoch)break outer;
          if(inputPreview)showLabImage(inputPreview,`${run.name.toUpperCase()} · CYCLE ${cycle+1} · DATA ${frame.index+1} · PHASE ${frame.phase+1} · SAMPLE ${exposure+1}/${settings.exposuresPerPhase}`,inputDimensions);
          handleResult(lab,result,inputDimensions??{width,height:simulation.height},expected);
          run.captures++;experiment.captures++;if(result.info)run.headers++;run.report=lab.report.snapshot();updateComparison();
          await new Promise(resolve=>requestAnimationFrame(resolve));
        }
      }
      run.complete=true;run.contentRecovered=Boolean(lab.session.received);run.report=lab.report.snapshot();updateComparison();
    }
    if(state.cancelLab)status(lab,'Test stopped','Partial results are preserved in the report.');
    else status(lab,'Transport comparison complete','Inspect raw errors and post-FEC acceptance for both variants. The content viewer shows the last variant.',lab.session.received?'success':'working');
  }catch(error){status(lab,'Test failed',error.message,'error');}
  finally{setLabBusy(false);}
}
async function openImage(file){
  if(!file||state.labBusy||lab.worker?.busy)return;
  if(file.size>40*1024*1024){status(lab,'Image too large','Choose a matrix photograph under 40 MB.','error');return;}
  lab.source='uploaded matrix photo';setLabBusy(true);resetReceiver(lab);lab.experiment=null;$('burstComparison').hidden=true;
  try{
    const url=URL.createObjectURL(file),img=new Image();
    try{await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(Error('This photo could not be opened. Try PNG or JPEG.'));img.src=url;});}finally{URL.revokeObjectURL(url);}
    const scale=Math.min(1,4096/Math.max(img.naturalWidth,img.naturalHeight)),canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(img.naturalWidth*scale));canvas.height=Math.max(1,Math.round(img.naturalHeight*scale));
    const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(img,0,0,canvas.width,canvas.height);const raw=ctx.getImageData(0,0,canvas.width,canvas.height),image={width:raw.width,height:raw.height,data:raw.data};
    showLabImage(image,'UPLOADED MATRIX PHOTO');status(lab,'Reading matrix photo',file.name,'working');
    await decode(lab,image,state.frames[0]?{id:state.frames[0].id,payload:state.payload}:null,{...receiverConfig('lab'),filter:'none',singlePhase:true});
  }catch(error){status(lab,'Photo could not be decoded',error.message,'error');}
  finally{setLabBusy(false);$('imageFile').value='';}
}
function exportReport(channel){
  const s=channel.session,r=s.lastResult;
  const report={application:'Optical Bench',version:'0.6.0',format:'OBM5',at:new Date().toISOString(),source:channel.source,notes:$('testNotes').value,userAgent:navigator.userAgent,
    transportExperiment:channel.report.snapshot(),
    camera:channel===live?state.cameraSettings??null:null,receiverSettings:receiverConfig(channel===live?'live':'lab'),burst:channel===lab?lab.experiment??null:null,
    currentTransmitter:state.frames[0]?{size:state.frames[0].size,colors:state.frames[0].colors,frames:state.frames.length,wireBytes:state.payload.length,id:state.frames[0].id,gridBlock:state.frames[0].gridBlock,parity:state.frames[0].parity,phases:state.frames[0].phases,permute:state.frames[0].permute,holdMs:state.frames[0].holdMs}:null,
    receiver:{captures:s.captures,validCaptures:s.validCaptures,frames:s.assembler.parts.size,complete:s.received!==null,wireBytes:s.received?.length??null,contentType:s.content?.type??null,contentBytes:s.content?.bytes.length??null,contentError:s.contentError,crc32:s.received?crc32(s.received).toString(16):null},
    lastCapture:r?{ok:r.ok,stage:r.stage,error:r.error,info:r.info,stats:r.stats,corners:r.corners,colorTransform:r.colorTransform}:null,events:state.history};
  downloadBlob(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}),`optical-bench-${channel===live?'live':'lab'}-report.json`);
}
function setView(view){
  state.view=view;$('displaySurface').dataset.view=view;$('cameraControls').hidden=view==='display';$('receiverControls').hidden=view==='display';$('playbackControls').hidden=view==='receive';
  $('transmitterSetup').hidden=state.workspace==='live'&&view==='receive';
  $('receiveStatus').hidden=view==='display';$('liveReceivedPanel').hidden=view==='display';$('liveDiagnostics').hidden=view==='display';$('liveCalibration').hidden=view==='display';
  if(view==='display'&&state.stream)stopCamera();
  for(const button of document.querySelectorAll('button[data-view]')){const active=button.dataset.view===view;button.classList.toggle('selected',active);button.setAttribute('aria-pressed',String(active));}
  if(state.frames.length)showFrame();
}
function setWorkspace(workspace){
  state.workspace=workspace;document.body.dataset.workspace=workspace;
  $('liveWorkspace').hidden=workspace!=='live';$('labWorkspace').hidden=workspace!=='lab';
  $('transmitterSetup').hidden=workspace==='live'&&state.view==='receive';
  for(const button of document.querySelectorAll('button[data-workspace]')){const active=button.dataset.workspace===workspace;button.classList.toggle('selected',active);button.setAttribute('aria-pressed',String(active));}
}

$('generateButton').onclick=()=>generate();
for(const id of ['probePattern','probeSeed'])$(id).onchange=()=>generate();
for(const id of ['matrixSize','colorCount','frameLimit','gridBlock','fecParity','phaseCount','phaseMode'])$(id).onchange=()=>generate();
$('payloadMode').onchange=updatePayloadControls;$('payloadText').onchange=()=>generate();
$('choosePayload').onclick=()=>$('payloadFile').click();$('payloadFile').onchange=event=>loadPayload(event.target.files[0]);$('samplePayload').onclick=loadSample;
$('playButton').onclick=()=>setPlaying(!state.playing);$('nextButton').onclick=()=>{advanceFrame();if(state.playing)setPlaying(true);};
$('fps').onchange=()=>{const playing=state.playing;generate();setPlaying(playing);};
$('savePNG').onclick=()=>{if(state.frames.length)$('matrixCanvas').toBlob(blob=>{if(blob)downloadBlob(blob,`obm5-${displayedFrames()[state.index].id}-${state.index+1}.png`);},'image/png');};
$('cameraButton').onclick=startCamera;$('cameraSelect').onchange=async()=>{if(state.stream){stopCamera();await startCamera();}};
$('cameraResolution').onchange=async()=>{if(state.stream){stopCamera();await startCamera();}};
for(const id of ['liveFilter','liveWindow','liveOverhead','liveMinPhases','liveRecovery','liveSpatial'])$(id).onchange=()=>{live.session.epoch++;live.worker?.reset();live.report.reset();live.transportView.clear();refreshSamplingControls();$('temporalStatus').textContent='Sampling changed. Evidence pools reset; received content is preserved. Reset receiver for a fresh time-to-recovery measurement.';};
$('labFilter').onchange=refreshSamplingControls;
$('labNoise').oninput=()=>{$('labNoiseValue').textContent=`${$('labNoise').value} / 255 σ`;};
$('labMoire').oninput=()=>{$('labMoireValue').textContent=`${$('labMoire').value} / 255`;};
$('loopbackButton').onclick=()=>{if(state.labBusy){state.cancelLab=true;$('loopbackButton').disabled=true;status(lab,'Stopping the burst','Finishing the current exposure…','working');}else loopback();};
$('uploadButton').onclick=()=>$('imageFile').click();$('imageFile').onchange=event=>openImage(event.target.files[0]);
$('resetReceiver').onclick=()=>resetReceiver(live);$('resetLab').onclick=()=>resetReceiver(lab);
$('exportReport').onclick=()=>exportReport(live);$('exportLabReport').onclick=()=>exportReport(lab);
for(const button of document.querySelectorAll('button[data-view]'))button.onclick=()=>setView(button.dataset.view);
for(const button of document.querySelectorAll('button[data-workspace]'))button.onclick=()=>setWorkspace(button.dataset.workspace);
$('fullscreenButton').onclick=async()=>{
  const surface=$('displaySurface');if(document.fullscreenElement){await document.exitFullscreen();return;}
  if(surface.classList.contains('expanded')){surface.classList.remove('expanded');return;}
  try{if(surface.requestFullscreen)await surface.requestFullscreen();else surface.classList.add('expanded');}catch{surface.classList.add('expanded');}
};
document.addEventListener('keydown',event=>{if(event.key==='Escape')$('displaySurface').classList.remove('expanded');});
$('helpButton').onclick=()=>$('guideDialog').showModal();$('closeGuide').onclick=()=>$('guideDialog').close();
$('guideDialog').onclick=event=>{if(event.target===$('guideDialog')){const r=$('guideDialog').getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)$('guideDialog').close();}};
window.addEventListener('pagehide',()=>{stopCamera();setPlaying(false);});
document.addEventListener('visibilitychange',()=>{if(document.hidden)setPlaying(false);});
try{
  let id=localStorage.getItem('optical-bench-device-id');if(!id){id=crypto.randomUUID?crypto.randomUUID():hex(randomBytes(16));localStorage.setItem('optical-bench-device-id',id);}
  $('deviceId').textContent=`DEVICE ${id.slice(0,8).toUpperCase()}`;
}catch{$('deviceId').textContent='SESSION ONLY';}
for(const channel of Object.values(channels)){try{channel.worker=new DecoderClient();}catch(error){notice(`Decoder unavailable: ${error.message}. Open this app through its server, not file://.`);}}
swatches($('referencePalette'),PALETTE);swatches($('observedPalette'),Array(16).fill(null));
if(location.protocol==='file:')notice('Open this app through the supplied server. JavaScript modules and camera access need HTTP(S).');
else if(!window.isSecureContext)notice('Live camera access needs HTTPS or localhost. The Image lab can still decode local files.');
else if(!navigator.mediaDevices?.getUserMedia)notice('Camera access is unavailable here. Try opening the app as a full page, or use the Image lab.');
refreshSamplingControls();generate();
