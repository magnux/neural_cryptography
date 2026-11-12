/** Header-free acquisition. Size and phase are measured before raster sampling. */
import {homography,project,sample,patch,calibrate} from '../vision.js';
import {markerPasses} from './markers.js';
import {normalizeCalibratedRGB} from '../learned-input.js';
import {NQR_LAYOUT as L,validateNeuralQRProfile} from './layout.js';

const luma=c=>.2126*c[0]+.7152*c[1]+.0722*c[2];
const corners=h=>[[0,0],[256,0],[256,256],[0,256]].map(p=>project(h,...p));
export const DEFAULT_CAPTURE_INTERVAL_MS=100;
export function geometryHintMaxAgeMs(captureIntervalMs=DEFAULT_CAPTURE_INTERVAL_MS){
  if(!Number.isFinite(captureIntervalMs)||captureIntervalMs<50||captureIntervalMs>5000)
    throw Error('Capture interval must be 50–5000 ms.');
  return 2*captureIntervalMs;
}
const median=values=>{const sorted=values.slice().sort((a,b)=>a-b),m=sorted.length>>>1;return sorted.length%2?sorted[m]:(sorted[m-1]+sorted[m])/2;};
const mean=values=>values.reduce((sum,value)=>sum+value,0)/values.length;
const clamp=value=>Math.max(0,Math.min(1,value));
function fail(stage,message,stats={}){const error=Error(message);error.neuralQRStage=stage;error.stats=stats;return error;}

function readOrientation(image,h){
  // Normalize each finder locally: a reflection over one corner must not set
  // the black/white threshold for the other three. Robust medians tolerate an
  // isolated contaminated ring sample, but every finder still needs a dark
  // core, contrasting white ring and a majority of dark outer-ring samples.
  let clipped=0,measurements=0;
  const measure=(x,y,radius)=>{
    const rgb=patch(image,h,x,y,radius);clipped+=rgb.filter(v=>v>=254).length;measurements+=3;return luma(rgb);
  };
  const refs=L.centers.map(([x,y])=>{
    const core=measure(x,y,1),dark=[core],light=[];
    for(const [dx,dy] of [[8,0],[-8,0],[0,8],[0,-8]])light.push(measure(x+dx,y+dy,.35));
    for(const [dx,dy] of [[12,0],[-12,0],[0,12],[0,-12]])dark.push(measure(x+dx,y+dy,.35));
    const d=median(dark),w=median(light),contrast=w-d,threshold=(d+w)/2;
    return {dark:d,light:w,contrast,valid:contrast>=30&&core<threshold-.08*contrast&&
      dark.filter(v=>v<threshold-.08*contrast).length>=4&&light.filter(v=>v>threshold+.08*contrast).length>=3};
  });
  const stats={finderContrasts:refs.map(r=>r.contrast),finderBlackLevels:refs.map(r=>r.dark),
    finderWhiteLevels:refs.map(r=>r.light),markerClippedFraction:clipped/measurements};
  if(refs.some(r=>!r.valid))throw fail('contrast','Finder contrast is uncertain; reduce reflections or adjust camera exposure.',stats);
  const local=refs[2];
  const values=L.orientation.map(([x,y])=>clamp((measure(x,y,1.1)-local.dark)/local.contrast));
  const target=L.orientation.map(([, ,bit])=>1-bit),mirrored=[...target.slice(3),...target.slice(0,3)];
  const error=pattern=>mean(values.map((value,i)=>(value-pattern[i])**2));
  const score=error(target),mirrorScore=error(mirrored),blankScore=error(values.map(()=>1));
  const margin=Math.min(mirrorScore,blankScore)-score;
  Object.assign(stats,{orientationError:score,orientationMargin:margin,markerClippedFraction:clipped/measurements});
  // Soft whole-pattern matching tolerates an uncertain patch. A contradicted
  // chirality bit cannot silently choose the mirrored order: it removes the
  // runner-up margin. Entirely missing/flat satellites also remain ambiguous.
  if(score>.12||margin<.08)throw fail('orientation','Orientation patches are ambiguous; keep the asymmetric corner clear.',stats);
  return {...stats,refs};
}

function readSize(image,h){
  // Each balanced copy has four dark and four light patches. Estimate its own
  // levels without using an expected size or repairing any bit. This keeps a
  // glare/shade gradient at a finder from imposing the wrong level on a remote
  // size code. Every measured bit and both complete codewords must still agree.
  const decisions=L.sizeCopies.map(copy=>{
    const values=copy.map(([x,y])=>luma(patch(image,h,x,y,1.5))),sorted=values.slice().sort((a,b)=>a-b);
    const dark=median(sorted.slice(0,4)),light=median(sorted.slice(4)),contrast=light-dark,threshold=(dark+light)/2;
    if(contrast<30)throw fail('size','Grid-size patches have insufficient local contrast.');
    let bits='',margin=Infinity;
    for(const value of values){
      const confidence=Math.abs(value-threshold)/contrast;
      if(confidence<.18)throw fail('size','Grid-size patches are uncertain; avoid glare or a display transition.');
      margin=Math.min(margin,confidence);bits+=value<threshold?'1':'0';
    }
    const index=L.sizeCodes.indexOf(bits);
    if(index<0)throw fail('size','Grid-size pattern is invalid; keep both binary codes visible.');
    return {size:L.sizes[index],margin,contrast};
  });
  if(decisions[0].size!==decisions[1].size)throw fail('size','Opposite grid-size codes disagree; skipping a transition exposure.');
  return {detectedSize:decisions[0].size,sizeMargin:Math.min(...decisions.map(d=>d.margin)),sizeContrasts:decisions.map(d=>d.contrast)};
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
export function extractNeuralQR(image,{profile,previousGeometry=null,now=Date.now(),dense=true,captureIntervalMs=DEFAULT_CAPTURE_INTERVAL_MS}={}){
  const start=performance.now(),result={ok:false,stage:'detect',markers:[],tentativeCorners:[],stats:{geometrySource:'none',previousGeometryTried:false},error:'Find all four Neural-QR markers.'};
  let expected=null,maxAge;
  try{maxAge=geometryHintMaxAgeMs(captureIntervalMs);if(profile!=null)expected=validateNeuralQRProfile(profile);}
  catch(error){return {...result,stage:'profile',error:error.message};}
  Object.assign(result.stats,{captureIntervalMs,geometryMaxAgeMs:maxAge});
  if(expected)result.stats.expectedSize=expected.size;
  if(!Number.isFinite(now)||!image||!Number.isInteger(image.width)||!Number.isInteger(image.height)||image.width<32||image.height<32||image.width*image.height>20_000_000||
    !(image.data instanceof Uint8Array||image.data instanceof Uint8ClampedArray)||image.data.length!==4*image.width*image.height)
    return {...result,error:'Unsupported camera image or timestamp.'};
  let selected=null,partial=null,bestFailure=null;
  const stageRank={detect:0,quad:1,contrast:2,orientation:3,size:4,profile:5,phase:6};
  function recordFailure(error,h){
    const stage=error.neuralQRStage??'quad',score=error.stats?.orientationError??Infinity;
    if(!bestFailure||stageRank[stage]>stageRank[bestFailure.stage]||
      stageRank[stage]===stageRank[bestFailure.stage]&&score<bestFailure.score){
      bestFailure={stage,error:error.message,stats:error.stats??{},score,h};
    }
  }
  function inspect(h,levels,source,acquiredAt){
    const size=readSize(image,h);
    if(expected&&size.detectedSize!==expected.size)
      throw fail('profile',`Optical G${size.detectedSize} does not match collection G${expected.size}.`,{...levels,...size});
    const geometry={width:image.width,height:image.height,transform:h.slice(),acquiredAt,size:size.detectedSize};
    const stats={...levels,...size,geometrySource:source};delete stats.refs;
    // A transient phase failure may preserve geometry, never RGB, phase or a
    // successful observation. Its age remains measured from full detection.
    partial={geometry,stats,h};
    try{
      const references=readReferences(image,h),phase=readPhase(image,h,references);
      return {h,references,phase,size,source,geometry,stats};
    }catch(error){throw fail('phase',error.message,stats);}
  }
  const previous=previousGeometry;
  if(previous?.width===image.width&&previous?.height===image.height&&Number.isFinite(previous.acquiredAt)&&now>=previous.acquiredAt&&now-previous.acquiredAt<maxAge&&
    Array.isArray(previous.transform)&&previous.transform.length===9&&previous.transform.every(Number.isFinite)){
    result.stats.previousGeometryTried=true;
    try{const h=previous.transform.slice();selected=inspect(h,readOrientation(image,h),'previous',previous.acquiredAt);}
    catch(error){recordFailure(error,previous.transform);}
  }
  if(!selected&&!partial){
    outer:for(const found of markerPasses(image)){
      result.markers=found.markers;Object.assign(result.stats,found.stats);
      if(!found.proposals.length){
        if(found.markers.length>=4)recordFailure(fail('quad','Markers found, but their geometry does not form a usable matrix.'));
        continue;
      }
      for(const {q} of found.proposals){
        const candidates=[];
        for(const direction of [1,-1])for(let rotation=0;rotation<4;rotation++){
          let h;
          try{
            const ordered=Array.from({length:4},(_,i)=>q[(rotation+direction*i+8)%4]);
            h=homography(L.centers,ordered);
            const levels=readOrientation(image,h);candidates.push({h,levels});
          }catch(error){recordFailure(error,h);}
        }
        candidates.sort((a,b)=>a.levels.orientationError-b.levels.orientationError);
        if(!candidates.length)continue;
        if(candidates.length>1&&candidates[1].levels.orientationError-candidates[0].levels.orientationError<.08){
          recordFailure(fail('orientation','More than one corner order matches; orientation is ambiguous.',candidates[0].levels),candidates[0].h);continue;
        }
        const {h,levels}=candidates[0];
        try{selected=inspect(h,levels,'detected',now);break outer;}
        catch(error){recordFailure(error,h);if(partial)break outer;}
      }
    }
  }
  result.stats.markerCount=result.markers.length;
  if(!selected){
    if(bestFailure){result.stage=bestFailure.stage;result.error=bestFailure.error;Object.assign(result.stats,bestFailure.stats);delete result.stats.refs;
      if(bestFailure.h)try{result.tentativeCorners=corners(bestFailure.h);}catch{}}
    if(partial){result.geometry=partial.geometry;Object.assign(result.stats,partial.stats);result.tentativeCorners=corners(partial.h);}
    result.stats.ms=performance.now()-start;return result;
  }
  const {h,references,phase,size,geometry,stats}=selected;
  const info={...validateNeuralQRProfile({size:size.detectedSize,colors:16,phases:4,gridBlock:8,pixelsPerCell:4}),phase:phase.phase};
  Object.assign(result,{info,transform:h,corners:corners(h),tentativeCorners:[],geometry});
  Object.assign(result.stats,stats,phase);
  try{
    const calibration=calibrate(references,0),p=L.payload,g=info.size,cell=p.side/g,side=4*g;
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
    Object.assign(result,{ok:true,stage:'acquired',error:null,samples,input:normalizeCalibratedRGB(rgb,side,side),dims:[1,3,side,side],pixelsPerCell:4,colorTransform:calibration.coefficients,observedPalette:calibration.observed});
  }catch(error){result.error=error.message;}
  result.stats.ms=performance.now()-start;return result;
}
