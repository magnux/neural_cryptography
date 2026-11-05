/** Same recording JSON schema, without a single full-file JavaScript string.
 * A G256 four-phase/32-sample recording exceeds common engines' string limit.
 * Materialize one record at a time into Blob storage, then compose those blobs.
 * Existing input strings stay in the recording; only the serialization is chunked.
 */
export function recordingBlob(recording){
  if(!recording||typeof recording!=='object'||Array.isArray(recording))throw Error('Expected a recording object.');
  const parts=['{'];let first=true;
  for(const key of Object.keys(recording)){
    const value=recording[key];
    if((key==='records'||key==='diagnostics')&&Array.isArray(value)){
      if(!first)parts.push(',');first=false;
      parts.push(JSON.stringify(key),':[');
      for(let i=0;i<value.length;i++){
        if(i)parts.push(',');
        parts.push(new Blob([JSON.stringify(value[i])??'null']));
      }
      parts.push(']');
    }else{
      const encoded=JSON.stringify(value);
      if(encoded===undefined)continue;
      if(!first)parts.push(',');first=false;
      parts.push(JSON.stringify(key),':',new Blob([encoded]));
    }
  }
  parts.push('}');
  return new Blob(parts,{type:'application/json'});
}
