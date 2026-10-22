import {createSender} from './contract.js';
import {captureFrame,validateCapture} from './capture.js';
import {renderFrame} from '../format.js';

let busy=false;
function toBase64(bytes){let text='';for(let i=0;i<bytes.length;i+=8192)text+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(text);}
async function originalPNG(image){
  if(typeof OffscreenCanvas!=='function')return {omitted:'This browser does not support worker PNG encoding.'};
  const canvas=new OffscreenCanvas(image.width,image.height),context=canvas.getContext('2d');
  context.putImageData(new ImageData(image.data,image.width,image.height),0,0);
  const blob=await canvas.convertToBlob({type:'image/png'});
  if(blob.size>1024*1024)return {omitted:'Original PNG exceeded the 1 MiB snapshot limit.'};
  return {mime:'image/png',base64:toBase64(new Uint8Array(await blob.arrayBuffer())),width:image.width,height:image.height};
}
self.onmessage=async event=>{
  const {id,operation,options={}}=event.data??{};
  if(!Number.isSafeInteger(id))return;
  if(busy){self.postMessage({id,type:'error',error:'A capture operation is already running.'});return;}
  busy=true;
  try{
    let result,transfer=[];
    if(operation==='create'){
      // Public random training data only. This operation is never used by capture.
      const {size,colors,phases,sessionId}=options;
      result=await createSender({size,colors,phases,sessionId,gridBlock:8,parity:32,holdMs:4000});
    }else if(operation==='render'){
      const {frame,pixels}=options;
      if(![1024,2048].includes(pixels)||![64,128].includes(frame?.size)||![4,16].includes(frame?.colors))throw Error('Unsupported capture display profile.');
      result=renderFrame(frame,pixels);transfer=[result.data.buffer];
    }else if(operation==='capture'){
      const {image,timestamp,camera,original=false}=options;
      if(!image||!Number.isSafeInteger(image.width)||!Number.isSafeInteger(image.height)||
         image.width<32||image.height<32||image.width*image.height>2073600||
         !(image.data instanceof Uint8ClampedArray)||image.data.length!==image.width*image.height*4)throw Error('Capture exceeds the 1080p pixel limit or has invalid pixels.');
      // No sender manifest, labels, expected bytes, or sequence seed enters here.
      result=await captureFrame(image,{timestamp,camera});
      if(original){
        try{const snapshot=await originalPNG(image);if(snapshot.omitted)result.originalNote=snapshot.omitted;else result.original=snapshot;}
        catch(error){result.originalNote=`Original snapshot unavailable: ${error.message}`;}
      }
      if(result.ok)validateCapture(result);
    }else throw Error('Unknown dataset operation.');
    self.postMessage({id,type:'result',result},transfer);
  }catch(error){self.postMessage({id,type:'error',error:error.message||String(error)});}
  finally{busy=false;}
};
