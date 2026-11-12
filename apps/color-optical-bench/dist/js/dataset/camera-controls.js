/** Optional camera controls. These change camera acquisition, never captured RGB. */
const MODES=['continuous','single-shot','manual'];
const MODE_KEYS=['exposureMode','focusMode'];
const RANGE_KEYS=['exposureCompensation','exposureTime','focusDistance'];
const KEYS=[...MODE_KEYS,...RANGE_KEYS];
const closedError=()=>new DOMException('Camera controls are no longer active.','AbortError');
const clone=value=>JSON.parse(JSON.stringify(value));

function safeRead(track,method){try{return track?.[method]?.()??{};}catch{return {};}}
export function readCameraControls(track){
  const settings=safeRead(track,'getSettings'),result={};
  for(const key of KEYS){const value=settings[key];if(MODE_KEYS.includes(key)?MODES.includes(value):Number.isFinite(value))result[key]=value;}
  return result;
}
export function cameraControlCapabilities(track){
  if(typeof track?.applyConstraints!=='function')return {};
  const raw=safeRead(track,'getCapabilities'),result={};
  for(const key of MODE_KEYS)if(Array.isArray(raw[key])){
    const values=MODES.filter(mode=>raw[key].includes(mode));if(values.length>1)result[key]=values;
  }
  for(const key of RANGE_KEYS){
    const range=raw[key];
    if(!range||!Number.isFinite(range.min)||!Number.isFinite(range.max)||range.max<=range.min)continue;
    if(range.step!==undefined&&(!Number.isFinite(range.step)||range.step<=0))continue;
    result[key]={min:range.min,max:range.max,...(range.step===undefined?{}:{step:range.step})};
  }
  return result;
}

function constrainedValue(capability,value){
  if(Array.isArray(capability)){
    if(!capability.includes(value))throw Error('Choose an available camera mode.');
    return value;
  }
  if(typeof value!=='number'||!Number.isFinite(value))throw Error('Camera control value must be finite.');
  const {min,max,step}=capability;
  let bounded=Math.min(max,Math.max(min,value));
  if(step)bounded=min+Math.min(Math.floor((max-min)/step+1e-9),Math.max(0,Math.round((bounded-min)/step)))*step;
  return Number(Math.min(max,Math.max(min,bounded)).toPrecision(12));
}

/** Serializes driver requests. Closing invalidates queued work and late results. */
export class CameraControls {
  constructor(track){
    this.track=track;this.capabilities=cameraControlCapabilities(track);this.requested={};
    this.actual=readCameraControls(track);this.closed=false;this.pending=0;this.tail=Promise.resolve();this.lastError=null;
  }
  snapshot(){
    if(!this.closed)this.actual=readCameraControls(this.track);
    return clone({capabilities:this.capabilities,requested:this.requested,actual:this.actual,pending:this.pending,closed:this.closed,lastError:this.lastError});
  }
  apply(changes){
    if(this.closed||this.track?.readyState==='ended')return Promise.reject(closedError());
    if(!changes||typeof changes!=='object'||Array.isArray(changes)||!Object.keys(changes).length)
      return Promise.reject(Error('Choose a camera control to change.'));
    if(this.pending>=8)return Promise.reject(Error('Wait for the camera control update to finish.'));
    let request={};
    try{
      for(const [key,value] of Object.entries(changes)){
        if(!Object.hasOwn(this.capabilities,key))throw Error(`This camera does not expose ${key}.`);
        request[key]=constrainedValue(this.capabilities[key],value);
      }
    }catch(error){return Promise.reject(error);}
    this.pending++;
    const operation=this.tail.then(async()=>{
      if(this.closed||this.track?.readyState==='ended')throw closedError();
      // A previous rejected request can remove an unsupported capability.
      for(const key of Object.keys(request))if(!Object.hasOwn(this.capabilities,key))throw Error(`This camera does not expose ${key}.`);
      const actual=readCameraControls(this.track);
      for(const [valueKey,modeKey] of [['exposureTime','exposureMode'],['focusDistance','focusMode']]){
        if(Object.hasOwn(request,valueKey)&&this.capabilities[modeKey]?.includes('manual')&&
          (request[modeKey]??actual[modeKey])!=='manual')throw Error(`Choose Manual ${modeKey==='focusMode'?'focus':'exposure'} before changing its value.`);
      }
      this.requested={...this.requested,...request};this.lastError=null;
      try{
        // Image Capture controls use the advanced constraint set. Drivers may
        // ignore a request, so report getSettings() separately from requested.
        await this.track.applyConstraints({advanced:[request]});
      }catch(error){
        if(this.closed)throw closedError();
        this.lastError=String(error.message||error).slice(0,300);
        if(error.name==='NotSupportedError'){
          const keys=Object.hasOwn(request,error.constraint)?[error.constraint]:Object.keys(request);
          for(const key of keys)delete this.capabilities[key];
        }
        throw error;
      }
      if(this.closed||this.track?.readyState==='ended')throw closedError();
      this.actual=readCameraControls(this.track);return this.snapshot();
    });
    this.tail=operation.catch(()=>{});
    return operation.finally(()=>{this.pending--;}).then(()=>this.snapshot());
  }
  close(){this.closed=true;}
}

/** Compact, image-free diagnostics: latest result only, with bounded arrays. */
export function acquisitionDiagnostics(result){
  const fields=['geometrySource','previousGeometryTried','geometryMaxAgeMs','captureIntervalMs','markerCount',
    'detectionMode','detectionMaxSide','detectionWidth','detectionHeight','detectionThreshold','quadCount',
    'orientationError','orientationMargin','markerClippedFraction','finderContrasts','finderBlackLevels','finderWhiteLevels',
    'sizeContrasts','sizeMargin','detectedSize','phaseError','phaseMargin','phaseSeparation','colorVolume','pixelsPerCell','colorRMSE','ms'];
  const stats={};
  for(const key of fields){
    const value=result.stats?.[key];
    if(typeof value==='boolean'||Number.isFinite(value))stats[key]=value;
    else if(typeof value==='string')stats[key]=value.slice(0,120);
    else if(Array.isArray(value))stats[key]=value.slice(0,8).filter(Number.isFinite);
  }
  return {stage:String(result.stage??'unknown').slice(0,40),error:result.error?String(result.error).slice(0,300):null,stats};
}
