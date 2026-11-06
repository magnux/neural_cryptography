/** Same-origin recording/session transport. Receiver discovery exposes no sender labels. */
import {recordingBlob} from './recording-json.js';
import {NEURAL_QR_VERSION,validateProfile,validateCollectionCode,validateSessionId,canonicalJSON} from './contract.js';
export const DATASET_SPLITS=Object.freeze(['train','validation','test']);
export async function recordingUploadConfig(fetcher=globalThis.fetch){
  try{
    const response=await fetcher('/api/recordings/config',{cache:'no-store',credentials:'same-origin'});
    if(!response.ok)return {enabled:false,reason:'Uploads unavailable on this server. Download files or start the demo server with --recordings-dir.'};
    const config=await response.json();
    if(!config.enabled)return {enabled:false,reason:'Uploads are off. Start the demo server with --recordings-dir.'};
    if(!Number.isSafeInteger(config.maxUploadBytes)||config.maxUploadBytes<1||typeof config.uploadToken!=='string'||!config.uploadToken)
      throw Error('Invalid upload configuration.');
    return {enabled:true,maxUploadBytes:config.maxUploadBytes,uploadToken:config.uploadToken};
  }catch{return {enabled:false,reason:'Cannot reach the recording server. Downloads remain available.'};}
}
export function validateCollectionSession(value){
  if(!value||value.format!=='neural-qr-collection-session'||value.version!==NEURAL_QR_VERSION||!DATASET_SPLITS.includes(value.datasetSplit)||
    typeof value.created!=='string'||!Number.isFinite(Date.parse(value.created)))throw Error('Invalid active collection session.');
  validateSessionId(value.id);validateSessionId(value.senderSessionId);validateCollectionCode(value.collectionCode);validateProfile(value.profile);
  // Explicit allowlist: response metadata cannot carry sender pixels, labels, hashes or seeds into acquisition.
  const {size,colors,phases,gridBlock,parity,holdMs,permute,pixelsPerCell}=value.profile;
  return {format:value.format,version:NEURAL_QR_VERSION,id:value.id,collectionCode:value.collectionCode,senderSessionId:value.senderSessionId,
    profile:{size,colors,phases,gridBlock,parity,holdMs,permute,pixelsPerCell},datasetSplit:value.datasetSplit,created:value.created};
}
export async function fetchCollectionSession({config,signal,fetcher=globalThis.fetch}={}){
  if(!config?.enabled)return null;
  const response=await fetcher('/api/collection-session',{cache:'no-store',credentials:'same-origin',signal});
  const body=await response.json();
  if(!response.ok)throw Error(body.error??'Could not read the active collection session.');
  if(body.enabled!==true)throw Error('Automatic collection sessions are disabled on this server.');
  return body.session===null?null:validateCollectionSession(body.session);
}
export async function abandonCollectionSession(id,{config,signal,fetcher=globalThis.fetch}={}){
  validateSessionId(id);if(!config?.enabled)throw Error('Recording uploads are not enabled on this server.');
  const response=await fetcher('/api/collection-session/abandon',{method:'POST',credentials:'same-origin',signal,
    headers:{'Content-Type':'application/json','X-Upload-Token':config.uploadToken},body:JSON.stringify({id})});
  const body=await response.json();if(!response.ok||body.ok!==true)throw Error(body.error??'Could not abandon the active collection session.');
  return body;
}
export function uploadRecording(recording,options={}){return sendRecording(recording,options,'/api/recordings');}
export async function stageCollectionSession(recording,options={}){
  if(recording?.format!=='neural-qr-sender')throw Error('Only a sender manifest starts a collection session.');
  const result=await sendRecording(recording,options,'/api/collection-session');
  if(result.session===null&&result.sessionCompleted===true)return result;
  const session=validateCollectionSession(result.session);
  if(session.collectionCode!==recording.collectionCode||session.senderSessionId!==recording.sessionId||session.datasetSplit!==recording.datasetSplit||canonicalJSON(session.profile)!==canonicalJSON(recording.profile))throw Error('The staged collection does not match this sender manifest.');
  return {...result,session};
}
function sendRecording(recording,{split,config,onProgress=()=>{},signal,xhrFactory=()=>new XMLHttpRequest()}={},endpoint){
  if(!DATASET_SPLITS.includes(split)||recording.datasetSplit!==split)return Promise.reject(Error('Recording split does not match the selected destination.'));
  if(!config?.enabled)return Promise.reject(Error('Recording uploads are not enabled on this server.'));
  const body=recordingBlob(recording);
  if(body.size>config.maxUploadBytes)return Promise.reject(Error('Recording exceeds the server upload limit. Download this batch instead.'));
  return new Promise((resolve,reject)=>{
    const xhr=xhrFactory();let completed=false;
    const finish=(error,value)=>{if(completed)return;completed=true;signal?.removeEventListener('abort',abort);error?reject(error):resolve(value);};
    const abort=()=>{xhr.abort();finish(Error('Upload cancelled. The recording remains on this device.'));};
    if(signal?.aborted){finish(Error('Upload cancelled. The recording remains on this device.'));return;}
    xhr.open('POST',`${endpoint}?split=${encodeURIComponent(split)}`);
    xhr.setRequestHeader('Content-Type','application/json');xhr.setRequestHeader('X-Upload-Token',config.uploadToken);
    xhr.timeout=endpoint==='/api/recordings'?900000:300000;
    xhr.upload.onprogress=event=>onProgress({loaded:event.loaded,total:event.lengthComputable?event.total:body.size});
    xhr.onerror=()=>finish(Error('Upload failed. The recording remains on this device; retry or download it.'));
    xhr.ontimeout=()=>finish(Error('Upload timed out. Retry is safe; the server checks duplicate files.'));
    xhr.onabort=()=>finish(Error('Upload cancelled. The recording remains on this device.'));
    xhr.onload=()=>{
      let result;try{result=JSON.parse(xhr.responseText);}catch{return finish(Error('The server returned an invalid upload response.'));}
      if(xhr.status<200||xhr.status>=300||result.ok!==true)return finish(Error(result.error??`Upload failed (${xhr.status}).`));
      if(typeof result.storedPath!=='string')return finish(Error('The server did not confirm a stored recording.'));
      finish(null,result);
    };
    signal?.addEventListener('abort',abort,{once:true});onProgress({loaded:0,total:body.size});xhr.send(body);
  });
}
