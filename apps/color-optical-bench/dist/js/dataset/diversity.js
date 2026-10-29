/** Diagnostic only: fixed stratified pixels in the calibrated CHW crop. */
export function captureSignature(input, size) {
  const side=4*size, pixels=side*side;
  if(!(input instanceof Float32Array)||![64,128].includes(size)||input.length!==3*pixels)throw Error('Invalid diversity input.');
  const values=new Float32Array(32*32*3), bin=side/32;
  for(let i=0;i<1024;i++){
    // Deterministic jitter avoids sampling the same subcell offset everywhere.
    const h=Math.imul(i+1,2654435761)>>>0;
    const x=(i%32)*bin+h%bin,y=Math.floor(i/32)*bin+(h>>>16)%bin;
    for(let c=0;c<3;c++)values[3*i+c]=input[c*pixels+y*side+x];
  }
  return values;
}
export function meanCaptureDistance(signatures) {
  if(!Array.isArray(signatures)||signatures.length>64)throw Error('Invalid comparison pool.');
  for(const a of signatures){
    if(!(a instanceof Float32Array)||a.length!==3072)throw Error('Invalid capture signature.');
    for(const v of a)if(!Number.isFinite(v)||Math.abs(v)>1.5)throw Error('Invalid signature value.');
  }
  let sum=0,pairs=0;
  for(let i=0;i<signatures.length;i++)for(let j=0;j<i;j++){
    let squared=0;
    for(let k=0;k<3072;k++)squared+=(signatures[i][k]-signatures[j][k])**2;
    sum+=50*Math.sqrt(squared/3072);pairs++;
  }
  return {mean:pairs?sum/pairs:null,pairs,samples:signatures.length,metric:'mean-pairwise-RGB-RMS-percent'};
}
