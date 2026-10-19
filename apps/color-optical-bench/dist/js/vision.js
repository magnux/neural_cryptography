import {FORMAT,PALETTE,parseHeader,decodeSymbols} from './format.js';
import {gridLayout,paletteMap} from './grid.js';
import {currentReferences} from './references.js';

export function solve(matrix,rhs){
  const n=rhs.length,a=matrix.map((row,i)=>[...row,rhs[i]]);
  for(let col=0;col<n;col++){
    let pivot=col;for(let r=col+1;r<n;r++)if(Math.abs(a[r][col])>Math.abs(a[pivot][col]))pivot=r;
    if(Math.abs(a[pivot][col])<1e-10)throw Error('Degenerate transform.');
    [a[col],a[pivot]]=[a[pivot],a[col]];const div=a[col][col];for(let j=col;j<=n;j++)a[col][j]/=div;
    for(let r=0;r<n;r++)if(r!==col){const mul=a[r][col];for(let j=col;j<=n;j++)a[r][j]-=mul*a[col][j];}
  }
  return a.map(row=>row[n]);
}
export function homography(source,dest){
  const a=[],b=[];
  for(let i=0;i<4;i++){const [x,y]=source[i],[u,v]=dest[i];
    a.push([x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]);b.push(u,v);
  }
  return [...solve(a,b),1];
}
export function project(h,x,y){const z=h[6]*x+h[7]*y+h[8];return [(h[0]*x+h[1]*y+h[2])/z,(h[3]*x+h[4]*y+h[5])/z];}
const luma = c => .2126*c[0]+.7152*c[1]+.0722*c[2];
export function sample(image,x,y){
  x-=.5;y-=.5;
  if(x<0||y<0||x>image.width-1||y>image.height-1)return null;
  const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(x0+1,image.width-1),y1=Math.min(y0+1,image.height-1),fx=x-x0,fy=y-y0;
  const rgb=[0,0,0];
  for(let k=0;k<3;k++)rgb[k]=(image.data[4*(y0*image.width+x0)+k]*(1-fx)+image.data[4*(y0*image.width+x1)+k]*fx)*(1-fy)+(image.data[4*(y1*image.width+x0)+k]*(1-fx)+image.data[4*(y1*image.width+x1)+k]*fx)*fy;
  return rgb;
}
function patch(image,h,x,y,radius=0){
  const offsets=radius?[[0,0],[-radius,0],[radius,0],[0,-radius],[0,radius]]:[[0,0]],rgb=[0,0,0];
  for(const [dx,dy] of offsets){const p=project(h,x+dx,y+dy),c=sample(image,...p);if(!c)throw Error('Matrix extends outside the image.');for(let k=0;k<3;k++)rgb[k]+=c[k]/offsets.length;}
  return rgb;
}
function cellPatch(image,h,x,y,cell,spatial){
  if(spatial==='cross')return patch(image,h,x,y,cell*.16);
  if(spatial!=='gaussian')throw Error('Unknown cell sampling method.');
  const rgb=[0,0,0],weights=[1,2,1];
  for(let iy=0;iy<3;iy++)for(let ix=0;ix<3;ix++){
    const value=patch(image,h,x+(ix-1)*cell*.22,y+(iy-1)*cell*.22);
    const weight=weights[ix]*weights[iy]/16;for(let k=0;k<3;k++)rgb[k]+=weight*value[k];
  }
  return rgb;
}
function otsu(hist,total){
  let sum=0;for(let i=0;i<256;i++)sum+=i*hist[i];
  let w=0,s=0,best=-1,low=80,high=180;
  for(let i=0;i<255;i++){w+=hist[i];s+=i*hist[i];if(!w||w===total)continue;
    const delta=s/w-(sum-s)/(total-w),v=w*(total-w)*delta*delta;
    if(v>best+1e-5){best=v;low=high=i;}else if(Math.abs(v-best)<1e-5)high=i;
  }
  return Math.max(35,Math.min(215,(low+high)/2));
}
function ratio(r){const t=r.reduce((s,v)=>s+v,0),m=t/7;if(t<9||r.some(v=>!v))return false;return r.every((v,i)=>Math.abs(v-(i===2?3:1)*m)<m*(i===2?1.15:.8));}
function cross(binary,w,h,x,y,vertical,limit){
  const at=(u)=>vertical?binary[u*w+x]:binary[y*w+u],p=vertical?y:x,max=vertical?h:w;
  if(!at(p))return null;const runs=[0,0,0,0,0];let i=p;
  while(i>=0&&at(i)&&runs[2]<=limit){runs[2]++;i--;}
  const centralStart=i+1;
  while(i>=0&&!at(i)&&runs[1]<=limit){runs[1]++;i--;}
  while(i>=0&&at(i)&&runs[0]<=limit){runs[0]++;i--;}
  i=p+1;
  while(i<max&&at(i)&&runs[2]<=limit){runs[2]++;i++;}
  const centralEnd=i;
  while(i<max&&!at(i)&&runs[3]<=limit){runs[3]++;i++;}
  while(i<max&&at(i)&&runs[4]<=limit){runs[4]++;i++;}
  if(!ratio(runs))return null;
  return {center:(centralStart+centralEnd)/2,module:runs.reduce((s,v)=>s+v,0)/7};
}
function hull(points){
  points.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const lo=[],hi=[];
  for(const p of points){while(lo.length>1&&cross(lo.at(-2),lo.at(-1),p)<=0)lo.pop();lo.push(p);}
  for(const p of points.slice().reverse()){while(hi.length>1&&cross(hi.at(-2),hi.at(-1),p)<=0)hi.pop();hi.push(p);}
  return lo.slice(0,-1).concat(hi.slice(0,-1));
}
function refineCenter(marker,binary,w,h){
  const sx=Math.floor(marker.x),sy=Math.floor(marker.y),radius=Math.ceil(marker.module*3),left=Math.max(0,sx-radius),top=Math.max(0,sy-radius),right=Math.min(w,sx+radius+1),bottom=Math.min(h,sy+radius+1),bw=right-left;
  if(!binary[sy*w+sx])return marker;
  const seen=new Uint8Array(bw*(bottom-top)),queue=[[sx,sy]],boundary=[];seen[(sy-top)*bw+sx-left]=1;
  for(let at=0;at<queue.length;at++){
    const [x,y]=queue[at];let edge=false;
    for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){
      const xx=x+dx,yy=y+dy;
      if(xx<left||xx>=right||yy<top||yy>=bottom)return marker;
      if(!binary[yy*w+xx]){edge=true;continue;}
      const j=(yy-top)*bw+xx-left;if(!seen[j]){seen[j]=1;queue.push([xx,yy]);}
    }
    if(edge)boundary.push([x+.5,y+.5]);
  }
  if(boundary.length<8)return marker;
  const poly=hull(boundary);
  while(poly.length>4){let least=Infinity,index=-1;for(let i=0;i<poly.length;i++){
    const a=poly[(i+poly.length-1)%poly.length],b=poly[i],c=poly[(i+1)%poly.length];
    const area=Math.abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]));if(area<least){least=area;index=i;}
  }poly.splice(index,1);}
  if(poly.length!==4)return marker;
  try{const H=homography([[0,0],[1,0],[1,1],[0,1]],poly),[x,y]=project(H,.5,.5);return {...marker,x,y};}catch{return marker;}
}
export function findMarkers(image){
  // Downsample detection only; payload/color sampling always uses the original image.
  const scale=Math.min(1,1000/Math.max(image.width,image.height)),w=Math.max(1,Math.round(image.width*scale)),h=Math.max(1,Math.round(image.height*scale));
  const gray=new Uint8Array(w*h),hist=new Uint32Array(256);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const xx=Math.min(image.width-1,Math.floor((x+.5)/scale)),yy=Math.min(image.height-1,Math.floor((y+.5)/scale)),p=4*(yy*image.width+xx);
    const v=Math.round(.2126*image.data[p]+.7152*image.data[p+1]+.0722*image.data[p+2]);gray[y*w+x]=v;hist[v]++;
  }
  const threshold=otsu(hist,w*h),binary=gray.map(v=>v<=threshold?1:0),found=[];
  for(let y=0;y<h;y+=Math.max(1,Math.floor(h/650))){
    const runs=[];let begin=0,color=binary[y*w];
    for(let x=1;x<=w;x++)if(x===w||binary[y*w+x]!==color){
      runs.push({start:begin,n:x-begin,color});begin=x;color=binary[y*w+x];
      if(runs.length<5)continue;const r=runs.slice(-5);
      if(!r[0].color||!ratio(r.map(v=>v.n)))continue;
      const cx=r[2].start+r[2].n/2,m=r.reduce((s,v)=>s+v.n,0)/7;
      const v=cross(binary,w,h,Math.floor(cx),y,true,m*6);if(!v||v.module/m<.4||v.module/m>2.5)continue;
      const horizontal=cross(binary,w,h,Math.floor(cx),Math.floor(v.center),false,m*6);if(!horizontal)continue;
      const xx=horizontal.center,yy=v.center,module=(v.module+horizontal.module)/2;
      const existing=found.find(f=>Math.hypot(f.x-xx,f.y-yy)<Math.max(module,f.module)*1.5);
      if(existing){const n=existing.hits;existing.x=(existing.x*n+xx)/(n+1);existing.y=(existing.y*n+yy)/(n+1);existing.module=(existing.module*n+module)/(n+1);existing.hits++;}
      else found.push({x:xx,y:yy,module,hits:1});
    }
  }
  const markers=found.filter(f=>f.hits>=2).sort((a,b)=>b.module*Math.sqrt(b.hits)-a.module*Math.sqrt(a.hits)).slice(0,16).map(f=>refineCenter(f,binary,w,h)).map(f=>{
    const full={...f,x:f.x/scale,y:f.y/scale,module:f.module/scale};
    if(scale===1)return full;
    // Refine the tiny inner square at original resolution. A subpixel error in
    // the coarse detector becomes several payload cells in a dense matrix.
    const r=Math.ceil(full.module*4),left=Math.max(0,Math.floor(full.x)-r),top=Math.max(0,Math.floor(full.y)-r);
    const cw=Math.min(image.width-left,2*r+1),ch=Math.min(image.height-top,2*r+1),crop=new Uint8Array(cw*ch);
    for(let yy=0;yy<ch;yy++)for(let xx=0;xx<cw;xx++){
      const j=4*((yy+top)*image.width+xx+left);crop[yy*cw+xx]=.2126*image.data[j]+.7152*image.data[j+1]+.0722*image.data[j+2]<=threshold?1:0;
    }
    const refined=refineCenter({...full,x:full.x-left,y:full.y-top},crop,cw,ch);
    return {...refined,x:refined.x+left,y:refined.y+top};
  });
  return {markers,threshold};
}
function quadProposals(markers,width,height){
  const proposals=[];
  for(let a=0;a<markers.length-3;a++)for(let b=a+1;b<markers.length-2;b++)for(let c=b+1;c<markers.length-1;c++)for(let d=c+1;d<markers.length;d++){
    const ms=[markers[a],markers[b],markers[c],markers[d]];
    if(Math.max(...ms.map(m=>m.module))>3*Math.min(...ms.map(m=>m.module)))continue;
    const cx=ms.reduce((s,m)=>s+m.x,0)/4,cy=ms.reduce((s,m)=>s+m.y,0)/4;
    const q=ms.map(m=>[m.x,m.y]).sort((a,b)=>Math.atan2(a[1]-cy,a[0]-cx)-Math.atan2(b[1]-cy,b[0]-cx));
    let area=0,convex=true;const lengths=[];
    for(let i=0;i<4;i++){const p=q[i],r=q[(i+1)%4],s=q[(i+2)%4];area+=p[0]*r[1]-p[1]*r[0];lengths.push(Math.hypot(p[0]-r[0],p[1]-r[1]));if((r[0]-p[0])*(s[1]-r[1])-(r[1]-p[1])*(s[0]-r[0])<=0)convex=false;}
    area=Math.abs(area)/2;if(!convex||area<width*height*.012||Math.min(...lengths)<30||Math.max(...lengths)>4*Math.min(...lengths))continue;
    const expectedModule=Math.sqrt(area)/53,relative=ms.map(m=>m.module/expectedModule);
    if(relative.some(r=>r<.45||r>2.1))continue;
    proposals.push({q,score:area/(1+relative.reduce((s,v)=>s+Math.abs(Math.log(v)),0))});
  }
  return proposals.sort((a,b)=>b.score-a.score).slice(0,24);
}
function readHeader(image,h){
  const dark=FORMAT.centers.reduce((s,[x,y])=>s+luma(patch(image,h,x,y,1)),0)/4;
  const light=FORMAT.centers.reduce((s,[x,y])=>s+luma(patch(image,h,x+8,y,.4)),0)/4;
  if(light-dark<45)throw Error('Not enough marker contrast.');
  const threshold=(dark+light)/2;
  const head=FORMAT.header,bytes=new Uint8Array(FORMAT.headerBytes);
  for(let bit=0;bit<bytes.length*8;bit++){
    const x=head.x+(bit%32+.5)*head.cell,y=head.y+(Math.floor(bit/32)+.5)*head.cellHeight;
    if(luma(patch(image,h,x,y,.45))<threshold)bytes[bit>>>3]|=1<<(7-(bit&7));
  }
  return parseHeader(bytes);
}
function calibrate(values,offset){
  const rows=[],targets=[],observed=[];
  for(let i=0;i<32;i++){
    const rgb=Array.from(values.subarray(offset+i*3,offset+i*3+3));rows.push([...rgb.map(v=>v/255),1]);targets.push(PALETTE[i%16]);observed.push(rgb);
  }
  const normal=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>rows.reduce((s,r)=>s+r[i]*r[j],0)+(i===j?1e-8:0)));
  const coefficients=[0,1,2].map(k=>solve(normal,[0,1,2,3].map(i=>rows.reduce((s,r,j)=>s+r[i]*targets[j][k],0))));
  const apply=rgb=>coefficients.map(c=>c[0]*rgb[0]/255+c[1]*rgb[1]/255+c[2]*rgb[2]/255+c[3]);
  let squared=0;for(let i=0;i<rows.length;i++){const fitted=apply(observed[i]);for(let k=0;k<3;k++)squared+=(fitted[k]-targets[i][k])**2;}
  const rmse=Math.sqrt(squared/96);
  if(!Number.isFinite(rmse))throw Error('Color calibration is unreliable.');
  return {apply,rmse,observed,coefficients};
}
export function rectify(image,h,pixels=320){
  const data=new Uint8ClampedArray(pixels*pixels*4);
  for(let y=0;y<pixels;y++)for(let x=0;x<pixels;x++){
    const c=sample(image,...project(h,(x+.5)*256/pixels,(y+.5)*256/pixels))??[80,80,80],i=4*(y*pixels+x);
    data[i]=c[0];data[i+1]=c[1];data[i+2]=c[2];data[i+3]=255;
  }
  return {width:pixels,height:pixels,data};
}
/** Rectify cell-center RGB samples and palette pilots, even if payload CRC fails. */
export function extractCapture(image,{preview=true,spatial='cross'}={}){
  const start=performance.now(),result={ok:false,stage:'detect',markers:[],stats:{},error:'Find all four corner markers. Move closer and keep the full frame visible.'};
  if(!image||!Number.isInteger(image.width)||!Number.isInteger(image.height)||image.width<32||image.height<32||image.width*image.height>20_000_000||image.data.length!==image.width*image.height*4){result.error='Unsupported image dimensions.';return result;}
  const detection=findMarkers(image);result.markers=detection.markers;
  const proposals=quadProposals(detection.markers,image.width,image.height);
  let h=null,info=null;
  outer:for(const {q} of proposals)for(const direction of [1,-1])for(let rotation=0;rotation<4;rotation++){
    const ordered=Array.from({length:4},(_,i)=>q[(rotation+direction*i+8)%4]);
    try{const candidate=homography(FORMAT.centers,ordered),header=readHeader(image,candidate);h=candidate;info=header;break outer;}catch{/* A checksum-valid header chooses the correct corner order. */}
  }
  result.stats.markerCount=detection.markers.length;
  if(!h){result.stage=detection.markers.length>=4?'header':'detect';if(result.stage==='header')result.error='Markers found, but the header could not be read. Hold steady or move closer.';result.stats.ms=performance.now()-start;return result;}
  result.info=info;result.corners=[[0,0],[256,0],[256,256],[0,256]].map(p=>project(h,...p));result.transform=h;result.stage='calibration';
  try{
    const p=FORMAT.payload,cell=p.side/info.size,values=new Float32Array(info.size*info.size*3+96);
    const horizontal=Math.hypot(...project(h,p.x+p.side,p.y).map((v,k)=>v-project(h,p.x,p.y)[k]))/info.size;
    const vertical=Math.hypot(...project(h,p.x,p.y+p.side).map((v,k)=>v-project(h,p.x,p.y)[k]))/info.size;
    result.stats.pixelsPerCell=Math.min(horizontal,vertical);
    for(let y=0;y<info.size;y++)for(let x=0;x<info.size;x++){
      values.set(cellPatch(image,h,p.x+(x+.5)*cell,p.y+(y+.5)*cell,cell,spatial),3*(y*info.size+x));
    }
    let offset=info.size*info.size*3;
    for(const x of [22,234])for(let i=0;i<16;i++){values.set(patch(image,h,x,52+i*10,1.5),offset);offset+=3;}
    result.samples=values;result.error=null;
  }catch(error){result.error=error.message;}
  if(preview)result.rectified=rectify(image,h);
  result.stats.ms=performance.now()-start;return result;
}

/** Per-exposure calibration and bounded soft evidence in logical symbol space. */
export function classifySamples(info,values,{local=Boolean(info.gridBlock)}={}){
  if(!(values instanceof Float32Array)||values.length!==info.size**2*3+96)throw Error('Invalid rectified RGB samples.');
  const cal=calibrate(values,info.size**2*3),layout=gridLayout(info.size,info.colors,info.gridBlock);
  const references=local&&info.gridBlock?currentReferences(info,values,cal.apply):null;
  const scores=new Uint8Array(layout.cells.length*info.colors),weights=new Float64Array(info.colors);
  for(let i=0;i<layout.cells.length;i++){
    const cell=layout.cells[i],rgb=cal.apply(values.subarray(3*cell,3*cell+3)),model=references?.at(cell);let minimum=Infinity;
    for(let logical=0;logical<info.colors;logical++){
      const color=paletteMap(logical,info),target=model?model.subarray(3*color,3*color+3):PALETTE[color];
      const distance=(rgb[0]-target[0])**2+(rgb[1]-target[1])**2+(rgb[2]-target[2])**2;weights[logical]=distance;minimum=Math.min(minimum,distance);
    }
    let sum=0;for(let k=0;k<info.colors;k++){weights[k]=Math.exp(-(weights[k]-minimum)/3600);sum+=weights[k];}
    for(let k=0;k<info.colors;k++)scores[i*info.colors+k]=Math.round(255*weights[k]/sum);
  }
  return {scores,decoder:references?'local':'global',observedPalette:cal.observed,colorTransform:cal.coefficients,
    stats:{colorRMSE:cal.rmse,references:references?.stats??{pilotCells:0,gridBlock:0}}};
}
export function decodeScores(info,scores,base={}, {recovery='hard'}={}){
  const layout=gridLayout(info.size,info.colors,info.gridBlock),result={...base,ok:false,stage:'payload',error:null,stats:{...base.stats,rawCRC:false,hardCRC:false,correctedBytes:0,erasedBytes:0,erasureTrials:0,recovery:'raw'}},symbols=new Uint8Array(info.size**2),reliability=new Uint8Array(layout.cells.length);
  if(scores.length!==layout.cells.length*info.colors)throw Error('Invalid symbol evidence.');
  let confidence=0,uncertain=0;
  for(let i=0;i<layout.cells.length;i++){
    let best=-1,second=-1,symbol=0;
    for(let k=0;k<info.colors;k++){const value=scores[i*info.colors+k];if(value>best){second=best;best=value;symbol=k;}else second=Math.max(second,value);}
    symbols[layout.cells[i]]=symbol;const margin=(best-second)/(best+1e-9);confidence+=margin;if(margin<.35)uncertain++;reliability[i]=Math.round(255*Math.max(0,Math.min(1,margin)));
  }
  result.symbols=symbols;result.stats.confidence=confidence/layout.cells.length;result.stats.lowConfidence=uncertain;
  const fields=['rawCRC','hardCRC','correctedBytes','erasedBytes','erasureTrials','recovery'];
  try{const decoded=decodeSymbols(symbols,info,{reliability,recovery});result.chunk=decoded.chunk;for(const field of fields)result.stats[field]=decoded[field];result.stats.rsBlocks=decoded.blocks;result.ok=true;result.stage='complete';}
  catch(error){result.error=error.message;for(const field of fields)if(error[field]!==undefined)result.stats[field]=error[field];}
  return result;
}
export function decodeSamples(info,values,options={}){
  try{const classified=classifySamples(info,values,options);return decodeScores(info,classified.scores,classified,options);}
  catch(error){return {ok:false,stage:'calibration',error:error.message,stats:{}};}
}
export function decodeImage(image,options={}){
  const start=performance.now(),capture=extractCapture(image,options),samples=capture.samples;delete capture.samples;
  if(samples){const decoded=decodeSamples(capture.info,samples,options);delete decoded.scores;return {...capture,...decoded,stats:{...capture.stats,...decoded.stats,ms:performance.now()-start}};}
  return capture;
}

/** Deterministic synthetic capture. This is an image test, never camera evidence. */
export function simulateCapture(source,{preset='perspective',width=960,height=760,seed=1234,noise=0,moire=0,moirePeriod=6,phase=0,curvature=0,colorDependence=0,gain=1,blur=0}={}){
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<32||height<32||width*height>20_000_000)throw Error('Invalid simulation dimensions.');
  if(![noise,moire,moirePeriod,phase,curvature,colorDependence,gain,blur].every(Number.isFinite)||noise<0||noise>100||moire<0||moire>100||moirePeriod<1||curvature<0||curvature>4||colorDependence<0||colorDependence>1||gain<.2||gain>2||blur<0||blur>3)throw Error('Invalid simulation settings.');
  const data=new Uint8ClampedArray(width*height*4);let state=(seed>>>0)||1;
  const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};
  let corners=[[.13,.07],[.87,.07],[.87,.93],[.13,.93]].map(([x,y])=>[x*width,y*height]);
  if(preset!=='clean')corners=[[.14,.06],[.89,.13],[.84,.91],[.07,.96]].map(([x,y])=>[x*width,y*height]);
  if(preset==='rotated'){corners=[corners[1],corners[2],corners[3],corners[0]];}
  if(preset==='mirrored'){corners=[corners[1],corners[0],corners[3],corners[2]];}
  const H=homography(corners,[[0,0],[256,0],[256,256],[0,256]]),tint=preset!=='clean';
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const [u,v]=project(H,x+.5,y+.5),i=4*(y*width+x);let rgb=[37,47,58];
    const inside=u>=0&&u<256&&v>=0&&v<256;
    if(inside)rgb=sample(source,u*source.width/256,v*source.height/256)??[255,255,255];
    if(tint){const [r,g,b]=rgb;rgb=[.82*r+.04*g+12,.03*r+.91*g+5,.06*g+.74*b+16];}
    for(let k=0;k<3;k++){
      // An interference-pattern stressor, not a physical LCD/sensor model.
      const bands=moire&&inside?moire*(Math.sin(2*Math.PI*(u+.18*v+curvature*(v-128)**2/256)/moirePeriod+phase+k*2.0944)+.35*Math.sin(2*Math.PI*(v-.11*u+curvature*(u-128)*(v-128)/512)/(moirePeriod*1.13)-phase+k*2.0944))/1.35:0;
      const additive=noise?(random()-.5)*Math.sqrt(12)*noise:0; // Uniform, requested std. dev. before clipping.
      const amplitude=1-colorDependence+colorDependence*(.25+.75*rgb[k]/255);
      data[i+k]=gain*(rgb[k]+bands*amplitude)+additive;
    }
    data[i+3]=255;
  }
  if(blur){
    const radius=Math.ceil(blur*2),kernel=Array.from({length:2*radius+1},(_,i)=>Math.exp(-((i-radius)**2)/(2*blur*blur))),sum=kernel.reduce((a,b)=>a+b,0),tmp=new Uint8ClampedArray(data.length);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)for(let k=0;k<3;k++){
      let v=0;for(let j=-radius;j<=radius;j++)v+=kernel[j+radius]*data[4*(y*width+Math.max(0,Math.min(width-1,x+j)))+k];tmp[4*(y*width+x)+k]=v/sum;
    }
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)for(let k=0;k<3;k++){
      let v=0;for(let j=-radius;j<=radius;j++)v+=kernel[j+radius]*tmp[4*(Math.max(0,Math.min(height-1,y+j))*width+x)+k];data[4*(y*width+x)+k]=v/sum;
    }
  }
  return {width,height,data};
}
