/** Diagnostic only: fixed stratified pixels in the calibrated CHW crop. */
export function captureSignature(input, size, pixelsPerCell=4) {
  const side=pixelsPerCell*size, pixels=side*side;
  if(!(input instanceof Float32Array)||![8,16,32,64,128,256].includes(size)||![2,4].includes(pixelsPerCell)||side>512||input.length!==3*pixels)throw Error('Invalid diversity input.');
  const values=new Float32Array(32*32*3), bin=side/32;
  for(let i=0;i<1024;i++){
    // Deterministic jitter avoids sampling the same subcell offset everywhere.
    const h=Math.imul(i+1,2654435761)>>>0;
    // Fractional bins at G8 intentionally repeat source pixels while preserving
    // the same bounded signature shape used by larger capture pools.
    const x=Math.floor(((i%32)+(h&65535)/65536)*bin),y=Math.floor((Math.floor(i/32)+(h>>>16)/65536)*bin);
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

function validateSignature(signature) {
  if(!(signature instanceof Float32Array)||signature.length!==3072)throw Error('Invalid capture signature.');
  for(const value of signature)if(!Number.isFinite(value)||Math.abs(value)>1.5)throw Error('Invalid signature value.');
}
function sampledIndices(length,count,random){
  const indices=Array.from({length},(_,i)=>i);
  for(let i=0;i<count;i++){
    const draw=random();if(!Number.isFinite(draw)||draw<0||draw>=1)throw Error('Invalid selection randomness.');
    const j=i+Math.floor(draw*(length-i));[indices[i],indices[j]]=[indices[j],indices[i]];
  }
  return indices.slice(0,count);
}
// Percentage of nominal RGB range, capped per component at 25 percentage points.
// This is a coverage score, not a sharpness/quality or correct-label score.
function clippedDistance(a,b){
  let squared=0;
  for(let i=0;i<a.length;i++)squared+=Math.min(.5,Math.abs(a[i]-b[i]))**2;
  return 50*Math.sqrt(squared/a.length);
}

/** Cheap randomized crowding replacement. Never receives labels or CRC status.
 * At most 8 reference + 8 candidate signatures are compared. A novelty margin
 * avoids churn on near-duplicates; the isolation guard avoids rewarding a lone
 * extreme capture merely for being unlike a currently consistent pool.
 */
export function selectCaptureReplacement(signatures,incomingSignature,{random=Math.random}={}){
  if(!Array.isArray(signatures)||signatures.length>64)throw Error('Invalid comparison pool.');
  signatures.forEach(validateSignature);validateSignature(incomingSignature);
  const result=(reason,replaceIndex=null,candidateDistance=null,replacedDistance=null)=>({replaceIndex,reason,candidateDistance,replacedDistance});
  if(signatures.length<2)return result('insufficient-pool');
  const references=sampledIndices(signatures.length,Math.min(8,signatures.length),random);
  const candidates=sampledIndices(signatures.length,Math.min(8,signatures.length),random);
  const nearest=(signature,exclude=-1)=>Math.min(...references.filter(i=>i!==exclude).map(i=>clippedDistance(signature,signatures[i])));
  const novelty=nearest(incomingSignature);
  const pairDistances=[];
  for(let i=0;i<references.length;i++)for(let j=0;j<i;j++)pairDistances.push(clippedDistance(signatures[references[i]],signatures[references[j]]));
  pairDistances.sort((a,b)=>a-b);
  const middle=Math.floor(pairDistances.length/2);
  const typical=pairDistances.length%2?pairDistances[middle]:(pairDistances[middle-1]+pairDistances[middle])/2;
  if(novelty>Math.max(12,3*typical))return result('isolated-candidate',null,novelty);
  let victim=null,redundancy=Infinity;
  // Candidate traversal is randomized, so equally crowded records do not always
  // evict the oldest or first retained frame.
  for(const index of candidates){const distance=nearest(signatures[index],index);if(distance<redundancy){victim=index;redundancy=distance;}}
  if(victim===null||novelty<=redundancy+Math.max(.05,.05*redundancy))return result('no-diversity-gain',null,novelty,redundancy);
  return result('greater-local-diversity',victim,novelty,redundancy);
}
