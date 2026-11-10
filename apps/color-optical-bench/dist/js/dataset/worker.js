import {MAX_CAPTURE_PIXELS} from './camera.js';
import {captureProfile,CAPTURE_SIZES,createSender} from './contract.js';
import {captureFrame,validateCapture} from './capture.js';
import {meanCaptureDistance,selectCaptureReplacement,createCaptureReference} from './diversity.js';
import {renderNeuralQR} from '../neural-qr/layout.js';

let busy=false,previousGeometry=null,geometryAssignment=null;
const references=new Map();
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
      result=await createSender({...captureProfile(size,colors,phases),sessionId});
    }else if(operation==='render'){
      const {frame,pixels}=options;
      if(![1024,2048].includes(pixels)||!CAPTURE_SIZES.includes(frame?.size)||frame?.colors!==16)throw Error('Unsupported capture display profile.');
      result=renderNeuralQR(frame,pixels);transfer=[result.data.buffer];
    }else if(operation==='compare'){
      result=meanCaptureDistance(options.signatures);
    }else if(operation==='select'){
      const key=options.key;
      if(typeof key!=='string'||key.length>200)throw Error('Selection requires a phase group key.');
      if(options.reference)references.set(key,options.reference);
      if(!references.has(key)){references.set(key,createCaptureReference(options.signatures));}
      if(references.size>128)references.delete(references.keys().next().value);
      result={...selectCaptureReplacement(options.signatures,options.incomingSignature,{reference:references.get(key)}),
        ...(!options.reference?{reference:references.get(key)}:{})};
    }else if(operation==='capture'||operation==='probe'){
      const {image,timestamp,camera,original=false}=options;
      if(!image||!Number.isSafeInteger(image.width)||!Number.isSafeInteger(image.height)||
         image.width<32||image.height<32||image.width*image.height>MAX_CAPTURE_PIXELS||
         !(image.data instanceof Uint8ClampedArray)||image.data.length!==image.width*image.height*4)throw Error('Capture exceeds the 20-megapixel processing limit or has invalid pixels.');
      // Public assignment only: no sender manifest, labels, expected bytes, or seed.
      const {collectionCode,profile}=options;
      const assignment=`${collectionCode}/${profile?.size}`;
      if(assignment!==geometryAssignment){previousGeometry=null;geometryAssignment=assignment;references.clear();}
      result=captureFrame(image,{timestamp,camera,signature:operation==='capture',profile,collectionCode,
        previousGeometry,probe:operation==='probe'});
      if(result.ok)previousGeometry=result.geometry;
      else if(['detect','orientation','size','profile'].includes(result.stage))previousGeometry=null;
      if(original){
        try{const snapshot=await originalPNG(image);if(snapshot.omitted)result.originalNote=snapshot.omitted;else result.original=snapshot;}
        catch(error){result.originalNote=`Original snapshot unavailable: ${error.message}`;}
      }
      if(result.ok&&!result.probe){validateCapture(result);transfer=[result.signature.buffer];}
    }else throw Error('Unknown dataset operation.');
    self.postMessage({id,type:'result',result},transfer);
  }catch(error){self.postMessage({id,type:'error',error:error.message||String(error)});}
  finally{busy=false;}
};

