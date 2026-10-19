import {gridLayout} from './grid.js';
import {PALETTE} from './format.js';

// Nearest four same-color references around each block. Geometry is cached;
// RGB measurements are rebuilt for every exposure. No calibration is retained.
const cache=new Map();
function neighborhoods(info){
  const key=`${info.size}/${info.colors}/${info.gridBlock}`;if(cache.has(key))return cache.get(key);
  const {size,colors,gridBlock}=info,step=gridBlock+1,side=Math.ceil(size/step),layout=gridLayout(size,colors,gridBlock),indices=new Int32Array(side*side*colors*4),weights=new Float32Array(side*side*colors*4);indices.fill(-1);
  for(let by=0;by<side;by++)for(let bx=0;bx<side;bx++){
    const cx=Math.min(size-1,bx*step+step/2),cy=Math.min(size-1,by*step+step/2),nearest=Array.from({length:colors},()=>[]);
    for(let y=Math.max(0,by*step-step);y<Math.min(size,(by+3)*step);y++)for(let x=Math.max(0,bx*step-step);x<Math.min(size,(bx+3)*step);x++){
      const cell=y*size+x,color=layout.reference[cell];if(color<0)continue;
      const d=(x-cx)**2+(y-cy)**2,list=nearest[color];list.push({cell,d});list.sort((a,b)=>a.d-b.d);if(list.length>4)list.pop();
    }
    nearest.forEach((list,color)=>{const sum=list.reduce((s,p)=>s+1/(1+p.d),0);list.forEach((p,i)=>{const at=((by*side+bx)*colors+color)*4+i;indices[at]=p.cell;weights[at]=1/(1+p.d)/sum;});});
  }
  const result={indices,weights,side,step};cache.set(key,result);if(cache.size>2)cache.delete(cache.keys().next().value);return result;
}
export function currentReferences(info,values,apply){
  const geometry=neighborhoods(info),layout=gridLayout(info.size,info.colors,info.gridBlock),observed=new Float32Array(info.size**2*3);
  for(const cell of layout.pilots)observed.set(apply(values.subarray(3*cell,3*cell+3)),3*cell);
  let missing=0,minimumSeparation=Infinity;
  const models=Array.from({length:geometry.side**2},(_,group)=>{
    const model=new Float32Array(info.colors*3);
    for(let color=0;color<info.colors;color++){
      const offset=(group*info.colors+color)*4,hasRefs=geometry.indices[offset]>=0;if(!hasRefs)missing++;
      for(let k=0;k<3;k++){
        // Shrink noisy estimates toward the same-exposure border calibration.
        let measured=hasRefs?0:PALETTE[color][k];for(let i=0;i<4;i++){const cell=geometry.indices[offset+i];if(cell>=0)measured+=geometry.weights[offset+i]*observed[3*cell+k];}
        model[3*color+k]=.8*measured+.2*PALETTE[color][k];
      }
    }
    for(let a=0;a<info.colors;a++)for(let b=a+1;b<info.colors;b++)minimumSeparation=Math.min(minimumSeparation,Math.hypot(model[3*a]-model[3*b],model[3*a+1]-model[3*b+1],model[3*a+2]-model[3*b+2]));
    return model;
  });
  return {at(cell){return models[Math.floor(Math.floor(cell/info.size)/geometry.step)*geometry.side+Math.floor((cell%info.size)/geometry.step)];},
    stats:{pilotCells:layout.pilots.length,gridBlock:info.gridBlock,missingPrototypes:missing,minimumSeparation}};
}
