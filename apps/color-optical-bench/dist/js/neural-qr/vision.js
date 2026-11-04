/** Header-free acquisition. The known profile contains no sender target data. */
import {findMarkers,quadProposals,homography,project,sample,patch,calibrate} from '../vision.js';
import {normalizeCalibratedRGB} from '../learned-input.js';
import {NQR_LAYOUT as L,validateNeuralQRProfile} from './layout.js';

const luma=c=>.2126*c[0]+.7152*c[1]+.0722*c[2];
const corners=h=>[[0,0],[256,0],[256,256],[0,256]].map(p=>project(h,...p));
export const GEOMETRY_MAX_AGE_MS=400;

function readOrientation(image,h){
  // Every reuse is checked against this exposure's localization cores, rings,
  // exterior and chiral satellites. Old phase or color samples are never reused.
  const dark=[],light=[];
  for(const [x,y] of L.centers){
    dark.push(luma(patch(image,h,x,y,1)));
    for(const [dx,dy] of [[8,0],[-8,0],[0,8],[0,-8]])light.push(luma(patch(image,h,x+dx,y+dy,.35)));
    for(const [dx,dy] of [[12,0],[-12,0],[0,12],[0,-12]])dark.push(luma(patch(image,h,x+dx,y+dy,.35)));
  }
  const mean=a=>a.reduce((s,v)=>s+v,0)/a.length,d=mean(dark),w=mean(light),contrast=w-d;
  if(contrast<45||Math.max(...dark)>d+.42*contrast||Math.min(...light)<w-.42*contrast)
    throw Error('Finder contrast or geometry is uncertain.');
  const threshold=(d+w)/2;
  for(const [x,y,bit] of L.orientation){
    const value=luma(patch(image,h,x,y,1.1));
    if(bit?value>threshold-.12*contrast:value<threshold+.12*contrast)
      throw Error('Asymmetric orientation marker is unclear.');
  }
  return contrast;
}

function readReferences(image,h){
  const values=new Float32Array(96);let at=0;
  for(const x of L.referenceX)for(let i=0;i<16;i++){
    values.set(patch(image,h,x,L.referenceY+L.referenceStep*i,1.5),at);at+=3;
  }
  return values;
}

function readPhase(image,h,references){
  // Compare in observed RGB space. This avoids trusting an ill-conditioned
  // color transform to separate phase cues after clipping or lost chroma.
  const observed=Array.from({length:4},(_,i)=>[0,1,2].map(k=>(references[3*i+k]+references[48+3*i+k])/2));
  const distance=(a,b)=>Math.hypot(...a.map((v,k)=>v-b[k]));
  const vectors=observed.slice(1).map(c=>c.map((v,k)=>v-observed[0][k]));
  const [a,b,c]=vectors,volume=Math.abs(a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]));
  const colorVolume=volume/Math.max(1,vectors.reduce((v,row)=>v*Math.hypot(...row),1));
  if(colorVolume<.04)throw Error('Color references have lost independent RGB information.');
  let separation=Infinity;
  for(let a=0;a<4;a++)for(let b=a+1;b<4;b++)separation=Math.min(separation,distance(observed[a],observed[b]));
  if(separation<55)throw Error('Phase reference colors are not sufficiently separated.');
  const decisions=L.phasePairs.map(pair=>{
    const measured=pair.map(([x,y])=>patch(image,h,x,y,3));
    const scores=Array.from({length:4},(_,phase)=>({phase,error:Math.sqrt(measured.reduce((s,c,i)=>s+distance(c,observed[(phase+i)%4])**2,0)/2)})).sort((a,b)=>a.error-b.error);
    if(scores[0].error>Math.min(70,.42*separation)||scores[1].error-scores[0].error<.25*separation)
      throw Error('Phase cues are uncertain; avoid glare or a display transition.');
    return {...scores[0],margin:scores[1].error-scores[0].error};
  });
  if(decisions[0].phase!==decisions[1].phase)throw Error('Opposite phase cues disagree; skipping a transition exposure.');
  return {phase:decisions[0].phase,phaseError:Math.max(...decisions.map(d=>d.error)),phaseMargin:Math.min(...decisions.map(d=>d.margin)),phaseSeparation:separation,colorVolume};
}

/** A successful result is optical evidence, not an accepted packet. */
export function extractNeuralQR(image,{profile,previousGeometry=null,now=Date.now(),dense=true}={}){
  const start=performance.now(),result={ok:false,stage:'detect',markers:[],tentativeCorners:[],stats:{geometrySource:'none',previousGeometryTried:false},error:'Find all four Neural-QR markers.'};
  let known;
  try{known=validateNeuralQRProfile(profile);}catch(error){return {...result,stage:'profile',error:error.message};}
  if(!Number.isFinite(now)||!image||!Number.isInteger(image.width)||!Number.isInteger(image.height)||image.width<32||image.height<32||image.width*image.height>20_000_000||
    !(image.data instanceof Uint8Array||image.data instanceof Uint8ClampedArray)||image.data.length!==4*image.width*image.height)
    return {...result,error:'Unsupported camera image or timestamp.'};
  let selected=null,oriented=false,lastError=null;
  function inspect(h,source,acquiredAt){
    readOrientation(image,h);oriented=true;result.tentativeCorners=corners(h);
    try{
      const references=readReferences(image,h),phase=readPhase(image,h,references);
      return {h,references,phase,source,acquiredAt};
    }catch(error){error.neuralQRStage='phase';throw error;}
  }
  const previous=previousGeometry;
  if(previous?.width===image.width&&previous?.height===image.height&&Number.isFinite(previous.acquiredAt)&&now>=previous.acquiredAt&&now-previous.acquiredAt<GEOMETRY_MAX_AGE_MS&&
    Array.isArray(previous.transform)&&previous.transform.length===9&&previous.transform.every(Number.isFinite)){
    result.stats.previousGeometryTried=true;
    try{selected=inspect(previous.transform.slice(),'previous',previous.acquiredAt);}catch(error){lastError=error.message;}
  }
  if(!selected){
    const found=findMarkers(image);result.markers=found.markers;
    const proposals=quadProposals(found.markers,image.width,image.height);
    if(!result.tentativeCorners.length&&proposals.length){try{result.tentativeCorners=corners(homography(L.centers,proposals[0].q));}catch{}}
    outer:for(const {q} of proposals)for(const direction of [1,-1])for(let rotation=0;rotation<4;rotation++){
      try{
        const ordered=Array.from({length:4},(_,i)=>q[(rotation+direction*i+8)%4]);
        selected=inspect(homography(L.centers,ordered),'detected',now);break outer;
      }catch(error){if(error.neuralQRStage==='phase')lastError=error.message;}
    }
  }
  result.stats.markerCount=result.markers.length;
  if(!selected){result.stage=oriented?'phase':result.markers.length>=4?'orientation':'detect';result.error=lastError??(result.stage==='orientation'?'Markers found; asymmetric orientation is unclear.':result.error);result.stats.ms=performance.now()-start;return result;}
  const {h,references,phase,source,acquiredAt}=selected,info={...known,phase:phase.phase};
  Object.assign(result,{info,transform:h,corners:corners(h),tentativeCorners:[],geometry:{width:image.width,height:image.height,transform:h.slice(),acquiredAt}});
  Object.assign(result.stats,{geometrySource:source,...phase});
  try{
    const calibration=calibrate(references,0),p=L.payload,g=info.size,cell=p.side/g,side=2*g;
    const horizontal=Math.hypot(...project(h,p.x+p.side,p.y).map((v,k)=>v-project(h,p.x,p.y)[k]))/g;
    const vertical=Math.hypot(...project(h,p.x,p.y+p.side).map((v,k)=>v-project(h,p.x,p.y)[k]))/g;
    Object.assign(result.stats,{pixelsPerCell:Math.min(horizontal,vertical),colorRMSE:calibration.rmse});
    if(!dense){result.ok=true;result.stage='acquired';result.error=null;result.stats.ms=performance.now()-start;return result;}
    result.stage='raster';const samples=new Float32Array(g*g*3+96),rgb=new Float32Array(side*side*3);
    for(let y=0;y<g;y++)for(let x=0;x<g;x++)samples.set(patch(image,h,p.x+(x+.5)*cell,p.y+(y+.5)*cell,.16*cell),3*(y*g+x));
    samples.set(references,g*g*3);
    for(let y=0;y<side;y++)for(let x=0;x<side;x++){
      const observed=sample(image,...project(h,p.x+(x+.5)*p.side/side,p.y+(y+.5)*p.side/side));
      if(!observed)throw Error('Payload extends outside the camera image.');
      rgb.set(calibration.apply(observed),3*(y*side+x));
    }
    Object.assign(result,{ok:true,stage:'acquired',error:null,samples,input:normalizeCalibratedRGB(rgb,side,side),dims:[1,3,side,side],pixelsPerCell:2,colorTransform:calibration.coefficients,observedPalette:calibration.observed});
  }catch(error){result.error=error.message;}
  result.stats.ms=performance.now()-start;return result;
}
