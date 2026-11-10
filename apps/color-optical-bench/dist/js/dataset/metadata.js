/** Browser-reported collection context; no high-entropy hints or extra permissions. */
const DEVICE_KEY='neural-corrector-collection-device';
const LABEL_KEY='neural-corrector-collection-label';
const short=(value,max=300)=>typeof value==='string'?value.slice(0,max):undefined;
const finite=value=>Number.isFinite(value)?value:undefined;
function localStorageOrNull(){try{return globalThis.localStorage??null;}catch{return null;}}

export function collectionIdentity({storage=localStorageOrNull(),randomUUID=()=>globalThis.crypto?.randomUUID?.()??'unavailable'}={}){
  let deviceId,deviceLabel='',persisted=false;
  try{
    deviceId=storage?.getItem(DEVICE_KEY);
    if(!/^[a-f0-9-]{36}$/i.test(deviceId??''))deviceId=randomUUID();
    storage?.setItem(DEVICE_KEY,deviceId);persisted=Boolean(storage);
    deviceLabel=storage?.getItem(LABEL_KEY)??'';
  }catch{deviceId??=randomUUID();}
  return {deviceId,deviceLabel:short(deviceLabel,80)??'',persisted};
}
export function saveDeviceLabel(label,storage=localStorageOrNull()){
  try{storage?.setItem(LABEL_KEY,String(label).slice(0,80));}catch{/* Session-only label still works. */}
}
export function environmentMetadata(identity,{navigator:nav=globalThis.navigator,screen=globalThis.screen,
  innerWidth=globalThis.innerWidth,innerHeight=globalThis.innerHeight,devicePixelRatio=globalThis.devicePixelRatio}={}){
  const ua=nav?.userAgent??'',hints=nav?.userAgentData;
  const deviceClass=/iPad|Tablet/i.test(ua)?'tablet':hints?.mobile===true||/iPhone|Android.*Mobile|Mobile/i.test(ua)?'mobile':
    /Windows NT|Macintosh|X11|CrOS/i.test(ua)?'desktop':'unknown';
  return {deviceId:identity.deviceId,deviceIdScope:identity.persisted?'browser-origin storage':'this page session',
    deviceLabel:short(identity.deviceLabel,80)??'',deviceClass,deviceClassSource:'browser hint / user-agent heuristic',
    browser:{userAgent:short(ua,1000),platform:short(nav?.platform),language:short(nav?.language,40),
      ...(hints?{clientHints:{mobile:hints.mobile,platform:short(hints.platform),
        brands:Array.isArray(hints.brands)?hints.brands.slice(0,8).map(item=>({brand:short(item.brand,80),version:short(item.version,40)})):[]}}:{})},
    display:{screenWidth:finite(screen?.width),screenHeight:finite(screen?.height),
      availableWidth:finite(screen?.availWidth),availableHeight:finite(screen?.availHeight),
      viewportWidth:finite(innerWidth),viewportHeight:finite(innerHeight),devicePixelRatio:finite(devicePixelRatio)}};
}
export function cameraMetadata(track,{requestedHeight,requestedWidth,requestedResolution,resolutionFallback,previous={}}={}){
  if(!track)return {...previous,active:false};
  const settings=track.getSettings?.()??{};
  return {active:true,label:short(track.label,300),width:finite(settings.width),height:finite(settings.height),
    frameRate:finite(settings.frameRate),facingMode:short(settings.facingMode,40),
    deviceId:short(settings.deviceId,300),groupId:short(settings.groupId,300),
    cameraIdScope:'browser-origin permission-scoped identifiers; not a global hardware identity',
    resizeMode:short(settings.resizeMode,40),requestedHeight:finite(requestedHeight),
    requestedWidth:finite(requestedWidth),requestedResolution:short(requestedResolution,20),
    resolutionFallback:short(resolutionFallback,300)};
}
