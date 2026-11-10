/** Camera negotiation and acquisition bounds shared by the page and worker. */
export const MAX_CAPTURE_PIXELS=20_000_000;
const PROFILES={480:[640,480],720:[1280,720],1080:[1920,1080],1440:[2560,1440],2160:[3840,2160],max:[8192,8192]};

export function cameraRequest(resolution){
  const key=String(resolution),dimensions=PROFILES[key];
  if(!dimensions)throw Error('Choose a supported camera resolution.');
  return {requestedResolution:key,requestedWidth:dimensions[0],requestedHeight:dimensions[1]};
}

export function captureDimensions(width,height){
  if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<32||height<32)throw Error('Camera has no valid image dimensions yet.');
  const scale=Math.min(1,Math.sqrt(MAX_CAPTURE_PIXELS/(width*height)));
  return {width:Math.max(1,Math.floor(width*scale)),height:Math.max(1,Math.floor(height*scale)),scale};
}

function constraints(request,deviceId,dimensions=true){
  return {audio:false,video:{...(dimensions?{width:{ideal:request.requestedWidth},height:{ideal:request.requestedHeight},resizeMode:{ideal:'none'}}:{}),
    ...(deviceId?{deviceId:{exact:deviceId}}:{facingMode:{ideal:'environment'}})}};
}

export async function openCaptureCamera(mediaDevices,{resolution,deviceId=''}={}){
  if(!mediaDevices?.getUserMedia)throw Error('Camera access requires trusted HTTPS or localhost.');
  const request=cameraRequest(resolution);let stream;
  try{stream=await mediaDevices.getUserMedia(constraints(request,deviceId));}
  catch(error){
    // Permissions and a missing selected device must never switch cameras.
    if(error.name!=='OverconstrainedError'||error.constraint==='deviceId'||error.constraint==='facingMode')throw error;
    stream=await mediaDevices.getUserMedia(constraints(request,deviceId,false));
    request.resolutionFallback='Resolution constraints rejected; using this camera’s negotiated default.';
  }
  const track=stream.getVideoTracks()[0];
  if(request.requestedResolution==='max'&&track?.applyConstraints){
    // Capability maxima are ideals, not a promise that their combination exists.
    // Keep the opened stream if this additional request cannot be satisfied.
    const caps=track.getCapabilities?.()??{},width=caps.width?.max,height=caps.height?.max;
    if(Number.isSafeInteger(width)&&Number.isSafeInteger(height)&&width>=32&&height>=32){
      const target=captureDimensions(width,height);
      request.requestedWidth=target.width;request.requestedHeight=target.height;
      try{await track.applyConstraints({width:{ideal:target.width},height:{ideal:target.height},resizeMode:{ideal:'none'}});}
      catch{request.resolutionFallback='Maximum-resolution request was unavailable; keeping the camera’s negotiated stream.';}
    }
  }
  return {stream,request};
}

export function cameraStatus(camera,dimensions){
  const requested=camera.requestedResolution==='max'?'maximum available':`${camera.requestedWidth}×${camera.requestedHeight}`;
  const processed=dimensions??captureDimensions(camera.width,camera.height);
  const scaleNote=processed.scale<1?` The 20-megapixel processing bound reduces this to ${processed.width}×${processed.height}.`:' Full camera resolution reaches acquisition.';
  return `Requested ${requested}; actual camera ${camera.width}×${camera.height}, ${camera.frameRate?.toFixed(0)??'unknown'} fps.${scaleNote}${camera.resolutionFallback?` ${camera.resolutionFallback}`:''}`;
}
