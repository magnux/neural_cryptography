/** Componentwise filtering of bounded byte evidence or floating-point measurements. */
export function filterSamples(samples,method){
  if(!samples.length||!['mean','median'].includes(method))throw Error('Invalid temporal filter.');
  const length=samples[0].length;
  if(samples.some(s=>(!(s instanceof Float32Array)&&!(s instanceof Uint8Array))||s.length!==length))throw Error('Incompatible sample grids.');
  const output=new Float32Array(length),n=samples.length;
  if(method==='mean'){
    for(const sample of samples)for(let i=0;i<length;i++)output[i]+=sample[i]/n;
  }else{
    const sorted=new Float32Array(n);
    for(let i=0;i<length;i++){
      // Tiny windows (at most nine): insertion sort avoids per-cell allocations.
      for(let j=0;j<n;j++){const value=samples[j][i];let k=j;while(k>0&&sorted[k-1]>value){sorted[k]=sorted[k-1];k--;}sorted[k]=value;}
      output[i]=n%2?sorted[n>>1]:(sorted[n/2-1]+sorted[n/2])/2;
    }
  }
  return output;
}

/**
 * Select a consistent subset independently for each cell's logical-symbol vector.
 * The caller owns the rolling candidate pool: rejected observations are not kept
 * or replaced here, and this estimator never sees payload truth or CRC results.
 */
export function robustSamples(samples,colors,keepCount){
  if(!Array.isArray(samples)||!samples.length||samples.length>11||
    ![2,4,8,16].includes(colors)||!Number.isInteger(keepCount)||
    keepCount<1||keepCount>samples.length)throw Error('Invalid robust sampling settings.');
  const length=samples[0]?.length,n=samples.length;
  if(!Number.isInteger(length)||!length||length%colors||
    samples.some(s=>!(s instanceof Uint8Array)||s.length!==length))throw Error('Incompatible symbol evidence grids.');

  const scores=new Float32Array(length),cells=length/colors;
  // Reuse these tiny workspaces for all cells; storage does not grow with frames.
  const normalized=new Float64Array(n*colors),distances=new Float64Array(n*n);
  const totals=new Float64Array(n),order=new Uint8Array(n);
  let trimmedCells=0,ambiguousCells=0,selectionDistance=0,discardedDistance=0;
  for(let offset=0;offset<length;offset+=colors){
    for(let j=0;j<n;j++){
      let sum=0;for(let k=0;k<colors;k++)sum+=samples[j][offset+k];
      for(let k=0;k<colors;k++)normalized[j*colors+k]=sum?samples[j][offset+k]/sum:1/colors;
    }
    totals.fill(0);
    for(let a=0;a<n;a++)for(let b=a+1;b<n;b++){
      let distance=0;
      for(let k=0;k<colors;k++)distance+=Math.abs(normalized[a*colors+k]-normalized[b*colors+k]);
      distances[a*n+b]=distances[b*n+a]=distance;
      totals[a]+=distance;totals[b]+=distance;
    }
    // Stable ties select the earliest candidate; the caller expires it by age.
    let medoid=0;
    for(let j=1;j<n;j++)if(totals[j]<totals[medoid]-1e-12)medoid=j;
    for(let j=0;j<n;j++){
      let position=j;
      while(position>0&&distances[medoid*n+order[position-1]]>distances[medoid*n+j]+1e-12){
        order[position]=order[position-1];position--;
      }
      order[position]=j;
    }
    let discardedDifferent=false;
    for(let j=0;j<n;j++){
      const distance=distances[medoid*n+order[j]];
      if(j<keepCount)selectionDistance+=distance;
      else{discardedDistance+=distance;if(distance>1e-12)discardedDifferent=true;}
    }
    if(discardedDifferent)trimmedCells++;
    let best=-1,second=-1;
    for(let k=0;k<colors;k++){
      let sum=0;for(let j=0;j<keepCount;j++)sum+=normalized[order[j]*colors+k];
      const value=sum/keepCount;
      scores[offset+k]=255*value;
      if(value>best){second=best;best=value;}else if(value>second)second=value;
    }
    if(best-second<1e-7)ambiguousCells++;
  }
  return {scores,stats:{cells,candidates:n,kept:keepCount,trimmedCells,ambiguousCells,
    meanSelectionDistance:selectionDistance/(cells*keepCount),
    meanDiscardedDistance:n>keepCount?discardedDistance/(cells*(n-keepCount)):0}};
}

