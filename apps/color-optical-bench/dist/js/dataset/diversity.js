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
function median(values){
  const ordered=values.slice().sort((a,b)=>a-b),middle=Math.floor(ordered.length/2);
  return ordered.length%2?ordered[middle]:(ordered[middle-1]+ordered[middle])/2;
}
function distance(a,b){
  let squared=0;
  for(let i=0;i<a.length;i++)squared+=(a[i]-b[i])**2;
  return 50*Math.sqrt(squared/a.length);
}
function draw(random){
  const value=random();
  if(!Number.isFinite(value)||value<0||value>=1)throw Error('Invalid selection randomness.');
  return value;
}
function validatePool(signatures){
  if(!Array.isArray(signatures)||signatures.length>64)throw Error('Invalid comparison pool.');
  signatures.forEach(validateSignature);
}

/** Freeze once per phase when its initial pool fills; discard on a new batch.
 * Keeping this independent of subsequent retention prevents the acceptance
 * radius from shrinking each time a more central frame replaces another one.
 * A trimmed medoid and MAD describe agreement, not image correctness.
 */
export function createCaptureReference(signatures){
  validatePool(signatures);
  if(signatures.length<2)return {ready:false,reason:'insufficient-pool',samples:signatures.length};
  const count=signatures.length,matrix=Array.from({length:count},()=>new Float64Array(count));
  for(let i=0;i<count;i++)for(let j=0;j<i;j++)matrix[i][j]=matrix[j][i]=distance(signatures[i],signatures[j]);
  const neighbors=Math.max(1,Math.ceil((count-1)*.75));
  let medoid=0,best=Infinity;
  for(let i=0;i<count;i++){
    const nearest=Array.from(matrix[i]).filter((_,j)=>j!==i).sort((a,b)=>a-b).slice(0,neighbors);
    const cost=nearest.reduce((sum,value)=>sum+value,0)/neighbors;
    if(cost<best){best=cost;medoid=i;}
  }
  // Estimate from the central 75%, so a few terrible initial frames cannot
  // stretch the acceptance range. The 1 percentage-point scale floor allows
  // modest variation even after an almost identical initial warm-up.
  const radii=Array.from(matrix[medoid]),inner=radii.filter((_,i)=>i!==medoid).sort((a,b)=>a-b).slice(0,neighbors);
  const radius=median(inner),scale=Math.max(1,1.4826*median(inner.map(value=>Math.abs(value-radius))));
  const limits=[radius+scale,radius+2*scale,radius+3*scale];
  const coherentSamples=radii.filter(value=>value<=limits[2]).length;
  if(coherentSamples<count*.6)return {ready:false,reason:'no-majority-cluster',samples:count,coherentSamples};
  return {ready:true,method:'central-bands',center:signatures[medoid].slice(),radius,scale,limits,
    samples:count,coherentSamples};
}
function validateReference(reference){
  if(!reference||typeof reference.ready!=='boolean')throw Error('Invalid capture reference.');
  if(!reference.ready)return;
  validateSignature(reference.center);
  if(!Array.isArray(reference.limits)||reference.limits.length!==3||
    reference.limits.some((value,i)=>!Number.isFinite(value)||value<=0||i&&value<=reference.limits[i-1]))throw Error('Invalid capture reference.');
}
function bandOf(radius,reference){
  const index=reference.limits.findIndex(limit=>radius<=limit);
  return index<0?3:index;
}
function quotas(count){
  const exact=[.7,.25,.05].map(weight=>weight*count),result=exact.map(Math.floor);
  const order=[0,1,2].sort((a,b)=>(exact[b]-result[b])-(exact[a]-result[a])||a-b);
  let remaining=count-result.reduce((sum,value)=>sum+value,0);
  for(let i=0;i<remaining;i++)result[order[i]]++;
  return result;
}

/** Centrally concentrated, randomized retention; no labels or CRC status.
 * Nominal core/shoulder/tail budgets are 70/25/5%, rounded to the pool size.
 * These are quotas around robust RMS-distance bands, not a claim that radial
 * RGB distances themselves follow a Gaussian distribution. Absent bands are
 * never fabricated. Isolated observations outside three scale units are skipped.
 * Once a band is full, a 25% stochastic refresh avoids maximizing distance or
 * always evicting the oldest sample. Close duplicates do not cause churn.
 */
export function selectCaptureReplacement(signatures,incomingSignature,{reference=null,random=Math.random}={}){
  validatePool(signatures);validateSignature(incomingSignature);
  const result=(reason,replaceIndex=null,extra={})=>({replaceIndex,reason,candidateDistance:null,replacedDistance:null,...extra});
  if(signatures.length<2)return result('insufficient-pool');
  reference??=createCaptureReference(signatures);validateReference(reference);
  if(!reference.ready)return result(reference.reason??'no-majority-cluster');
  const radii=signatures.map(signature=>distance(signature,reference.center));
  const bands=radii.map(radius=>bandOf(radius,reference));
  const counts=[0,0,0,0];bands.forEach(band=>counts[band]++);
  const target=quotas(signatures.length),candidateDistance=distance(incomingSignature,reference.center),band=bandOf(candidateDistance,reference);
  const extra={candidateDistance,band,counts,target,limits:reference.limits.slice()};
  if(band===3)return result('isolated-candidate',null,extra);
  // Purge bad warm-up frames first. The nearest valid replacement may coincide
  // with another retained frame: removing the outlier takes priority here.
  let victims=bands.map((value,index)=>value===3?index:-1).filter(index=>index>=0),reason='replace-outlier';
  if(!victims.length){
    if(signatures.some(signature=>distance(signature,incomingSignature)<.05))return result('near-duplicate',null,extra);
    if(!target[band])return result('band-budget-zero',null,extra);
    if(counts[band]<target[band]){
      const excess=counts.slice(0,3).map((count,index)=>count-target[index]);
      const most=Math.max(...excess);
      victims=bands.map((value,index)=>excess[value]===most?index:-1).filter(index=>index>=0);
      reason='balance-bands';
    }else{
      if(draw(random)>=.25)return result('kept-band',null,extra);
      victims=bands.map((value,index)=>value===band?index:-1).filter(index=>index>=0);
      reason='refresh-band';
    }
  }
  const replaceIndex=victims[Math.floor(draw(random)*victims.length)];
  return result(reason,replaceIndex,{...extra,replacedDistance:radii[replaceIndex]});
}
