import {TemporalDecoder} from './temporal.js';
import {renderFrame} from './format.js';
import {simulateCapture} from './vision.js';
const decoder=new TemporalDecoder();
function thumbnail(image){
  const scale=Math.min(1,1000/Math.max(image.width,image.height)),width=Math.round(image.width*scale),height=Math.round(image.height*scale),data=new Uint8ClampedArray(width*height*4);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const from=4*(Math.min(image.height-1,Math.floor((y+.5)/scale))*image.width+Math.min(image.width-1,Math.floor((x+.5)/scale)));
    const to=4*(y*width+x);for(let channel=0;channel<4;channel++)data[to+channel]=image.data[from+channel];
  }
  return {width,height,data};
}
self.onmessage = ({data}) => {
  const {id,epoch,options}=data;let image=data.image;
  if(data.type==='reset'){decoder.reset();return;}
  try{
    const start=performance.now();
    if(data.type==='simulate')image=simulateCapture(renderFrame(data.frame,data.renderPixels),data.simulation);
    const simulationMs=performance.now()-start;
    const result=decoder.decode(image,options),transfers=[];
    if(data.type==='simulate'){
      result.inputPreview=thumbnail(image);result.inputDimensions={width:image.width,height:image.height};
      result.stats.simulationMs=simulationMs;transfers.push(result.inputPreview.data.buffer);
    }
    if(result.rectified)transfers.push(result.rectified.data.buffer);
    if(result.chunk)transfers.push(result.chunk.buffer);
    self.postMessage({id,epoch,result},transfers);
  }catch(error){self.postMessage({id,epoch,result:{ok:false,stage:'error',error:error.message,markers:[],stats:{}}});}
};
