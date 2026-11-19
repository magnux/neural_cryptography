// Refinement of a globally calibrated Neural-QR crop, prior to normalization.
// Only fixed, unpermuted pilot RGBs fit the correction; no payload decisions.
import {gridLayout} from '../grid.js';
import {PALETTE} from '../format.js';

const PPC=4, STRIDE=9, MID=127.5;
const clamp=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));
const geometryBySize=new Map(); // Only four supported layouts; never cache measurements.
function geometry(size){
  if(![16,32,64,128].includes(size))throw Error('Unsupported grid color correction geometry.');
  if(geometryBySize.has(size))return geometryBySize.get(size);
  const {pilots,reference}=gridLayout(size,16,8),side=PPC*size;
  const nodes=1+Math.round((size-1)/STRIDE),spacing=(size-1)/(nodes-1);
  const radius=1.35*spacing,patches=[],center=new Int32Array(pilots.length),pilotId=new Int32Array(size*size);
  for(let i=0;i<pilots.length;i++){
    const c=pilots[i],y=Math.floor(c/size),x=c%size;
    center[i]=3*((PPC*y+1)*side+PPC*x+1);pilotId[c]=i;
  }
  // Overlapping neighborhoods, including terminal pilot lines at each edge.
  for(let py=0;py<nodes;py++)for(let px=0;px<nodes;px++){
    const cx=px*spacing,cy=py*spacing,ids=[],weights=[];
    for(let y=Math.max(0,Math.ceil(cy-radius));y<=Math.min(size-1,Math.floor(cy+radius));y++)
      for(let x=Math.max(0,Math.ceil(cx-radius));x<=Math.min(size-1,Math.floor(cx+radius));x++){
        const c=y*size+x;if(reference[c]<0)continue;
        const w=(1-Math.abs(x-cx)/radius)*(1-Math.abs(y-cy)/radius);
        if(w>0){ids.push(pilotId[c]);weights.push(w);}
      }
    patches.push({ids:Uint16Array.from(ids),weights:Float32Array.from(weights)});
  }
  const base=new Uint8Array(side),frac=new Float32Array(side);
  for(let p=0;p<side;p++){
    const v=clamp(((p+.5)/PPC-.5)/spacing,0,nodes-1);
    base[p]=Math.min(nodes-2,Math.floor(v));frac[p]=v-base[p];
  }
  const target=Uint8Array.from(pilots,c=>reference[c]); // physical palette index, invariant across phases
  const g={side,nodes,patches,center,target,base,frac};geometryBySize.set(size,g);return g;
}

// Fit measured-minus-known response against known mid-gray-centered intensity.
// Two passes downweight large residuals and isolated glare/blurred pilot samples.
// A collapsed color channel or too little support leaves the border fit in place.
function fitChannel(patch,target,observed,k,coeff,at){
  let slope=0,bias=0;
  for(let pass=0;pass<2;pass++){
    let sw=0,sx=0,sd=0,sxx=0,sxd=0;
    for(let j=0;j<patch.ids.length;j++){
      const id=patch.ids[j],known=PALETTE[target[id]][k],x=known-MID,d=observed[3*id+k]-known;
      const error=d-(slope*x+bias),width=pass===0?50:24;
      const w=patch.weights[j]/(1+(error/width)**2);
      sw+=w;sx+=w*x;sd+=w*d;sxx+=w*x*x;sxd+=w*x*d;
    }
    if(sw<3) return false;
    const mx=sx/sw,md=sd/sw,variance=sxx/sw-mx*mx;
    if(variance<700)return false;
    const cov=sxd/sw-mx*md, response=1+cov/variance;
    if(response<.45||response>1.8)return false;
    // Ridge and support shrinkage towards the unmodified border calibration.
    slope=cov/(variance+400);bias=md-slope*mx;
    if(pass===1){
      const strength=sw/(sw+2),gain=1+strength*(clamp(1/(1+slope),.75,1.35)-1);
      const offset=strength*clamp(-bias/(1+slope),-36,36);
      coeff[at]=gain;coeff[at+1]=MID*(1-gain)+offset;
    }
  }
  return true;
}

/** In-place floating-point RGB (HWC, globally calibrated, 4x4 samples/cell). */
export function applyGridColorCorrection(rgb,size){
  const {side,nodes,patches,center,target,base,frac}=geometry(size);
  if(!(rgb instanceof Float32Array)||rgb.length!==3*side*side)throw Error('Expected a dense 4x4 RGB crop.');
  // Average only the central 2x2 samples per pilot, staying away from its edges.
  const observed=new Float32Array(3*center.length),pitch=3*side;
  for(let i=0;i<center.length;i++)for(let k=0;k<3;k++){
    const t=center[i]+k;observed[3*i+k]=.25*(rgb[t]+rgb[t+3]+rgb[t+pitch]+rgb[t+pitch+3]);
  }
  const coeff=new Float32Array(nodes*nodes*6);let fallbackChannels=0;
  for(let n=0;n<patches.length;n++)for(let k=0;k<3;k++){
    const at=6*n+2*k;coeff[at]=1;
    if(!fitChannel(patches[n],target,observed,k,coeff,at))fallbackChannels++;
  }
  // Bilinear interpolation, no tile edges and no per-pixel temporary objects.
  const row=new Float32Array(nodes*6);
  for(let y=0;y<side;y++){
    const off=base[y]*row.length,fy=frac[y];
    for(let j=0;j<row.length;j++)row[j]=coeff[off+j]+fy*(coeff[off+row.length+j]-coeff[off+j]);
    for(let x=0;x<side;x++){
      const p=3*(y*side+x),b=base[x]*6,fx=frac[x];
      for(let k=0;k<3;k++){
        const j=b+2*k,gain=row[j]+fx*(row[j+6]-row[j]),bias=row[j+1]+fx*(row[j+7]-row[j+1]);
        rgb[p+k]=gain*rgb[p+k]+bias;
      }
    }
  }
  return {pilots:center.length,nodes:nodes*nodes,fallbackChannels};
}
