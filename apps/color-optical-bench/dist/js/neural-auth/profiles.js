/** Physical device directions are independent of authentication roles. */
export const DEVICE_PROFILES=Object.freeze({
  pc:Object.freeze({name:'Computer',outgoingSizes:Object.freeze([32,64,128]),defaultOutgoingSize:128,incomingMaxSize:64}),
  phone:Object.freeze({name:'Phone',outgoingSizes:Object.freeze([32,64]),defaultOutgoingSize:64,incomingMaxSize:128}),
});

export function deviceProfile(device,g128Test=false){
  if(!Object.hasOwn(DEVICE_PROFILES,device))throw Error('Choose Computer or Phone for this screen.');
  if(typeof g128Test!=='boolean')throw Error('Invalid G128 test setting.');
  if(g128Test)return Object.freeze({...DEVICE_PROFILES[device],name:`${DEVICE_PROFILES[device].name} · G128 test`,
    outgoingSizes:DEVICE_PROFILES.pc.outgoingSizes,defaultOutgoingSize:128,incomingMaxSize:128});
  return DEVICE_PROFILES[device];
}

export function transportProfile(device,outgoingSize,g128Test=false){
  const profile=deviceProfile(device,g128Test),size=Number(outgoingSize);
  if(!Number.isSafeInteger(size)||!profile.outgoingSizes.includes(size))throw Error(`Choose a supported outgoing grid for ${profile.name}.`);
  return {device,outgoingSize:size,incomingMaxSize:profile.incomingMaxSize,controlSize:32,...(g128Test?{g128Test:true}:{})};
}

export function transferMetrics(bytes,elapsedMs){
  if(!Number.isSafeInteger(bytes)||bytes<0||!Number.isFinite(elapsedMs)||elapsedMs<0)throw Error('Invalid transfer measurements.');
  return {bytes,elapsedMs,goodputBytesPerSecond:elapsedMs>0?1000*bytes/elapsedMs:0};
}
