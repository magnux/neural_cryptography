/** Explicit same-origin recording upload. Download/export remains independent. */
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
export function uploadRecording(recording,{split,config,onProgress=()=>{},signal,xhrFactory=()=>new XMLHttpRequest()}={}){
  if(!DATASET_SPLITS.includes(split)||recording.datasetSplit!==split)return Promise.reject(Error('Recording split does not match the selected destination.'));
  if(!config?.enabled)return Promise.reject(Error('Recording uploads are not enabled on this server.'));
  const body=new Blob([JSON.stringify(recording)],{type:'application/json'});
  if(body.size>config.maxUploadBytes)return Promise.reject(Error('Recording exceeds the server upload limit. Download this batch instead.'));
  return new Promise((resolve,reject)=>{
    const xhr=xhrFactory();let completed=false;
    const finish=(error,value)=>{if(completed)return;completed=true;signal?.removeEventListener('abort',abort);error?reject(error):resolve(value);};
    const abort=()=>{xhr.abort();finish(Error('Upload cancelled. The recording remains on this device.'));};
    if(signal?.aborted){finish(Error('Upload cancelled. The recording remains on this device.'));return;}
    xhr.open('POST',`/api/recordings?split=${encodeURIComponent(split)}`);
    xhr.setRequestHeader('Content-Type','application/json');xhr.setRequestHeader('X-Upload-Token',config.uploadToken);
    xhr.timeout=300000;
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
