/** Eight-byte content descriptor in the current monochrome header. */
export const CONTENT_HEADER_BYTES = 8;
export const CONTENT_TYPES = Object.freeze({
  raw:  {id:0, label:'Raw bytes', mime:'application/octet-stream', extension:'bin'},
  text: {id:1, label:'UTF-8 text', mime:'text/plain;charset=utf-8', extension:'txt'},
  png:  {id:2, label:'PNG image', mime:'image/png', extension:'png'},
  jpeg: {id:3, label:'JPEG image', mime:'image/jpeg', extension:'jpg'},
  wav:  {id:4, label:'WAV audio', mime:'audio/wav', extension:'wav'},
});
const ascii=(bytes,offset,length)=>String.fromCharCode(...bytes.subarray(offset,offset+length));
export const decodeText = bytes => new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(bytes);

export function inspectWav(bytes){
  if(bytes.length<44||ascii(bytes,0,4)!=='RIFF'||ascii(bytes,8,4)!=='WAVE')throw Error('Choose a WAV file containing PCM audio.');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  if(view.getUint32(4,true)+8!==bytes.length)throw Error('WAV file length is inconsistent.');
  let offset=12,format=null,dataLength=null;
  while(offset+8<=bytes.length){
    const tag=ascii(bytes,offset,4),length=view.getUint32(offset+4,true),start=offset+8,end=start+length;
    if(end>bytes.length)throw Error('Truncated WAV chunk.');
    if(tag==='fmt '){
      if(format||length<16)throw Error('Invalid WAV format chunk.');
      const code=view.getUint16(start,true),channels=view.getUint16(start+2,true),sampleRate=view.getUint32(start+4,true),byteRate=view.getUint32(start+8,true),align=view.getUint16(start+12,true),bits=view.getUint16(start+14,true);
      if(code!==1||![1,2].includes(channels)||![8,16].includes(bits)||sampleRate<8000||sampleRate>192000)throw Error('Use uncompressed PCM WAV: mono/stereo, 8-bit or 16-bit, 8–192 kHz.');
      if(align!==channels*bits/8||byteRate!==sampleRate*align)throw Error('Invalid WAV sample layout.');
      format={channels,sampleRate,bits,align};
    }
    if(tag==='data'){if(dataLength!==null)throw Error('Use a WAV file with one audio data chunk.');dataLength=length;}
    offset=end+(length&1);
  }
  if(offset!==bytes.length||!format||dataLength===null||dataLength%format.align)throw Error('Incomplete WAV audio data.');
  return {...format,duration:dataLength/(format.sampleRate*format.align)};
}
function validate(bytes,type){
  if(!(bytes instanceof Uint8Array))throw Error('Content must be a Uint8Array.');
  if(!Object.hasOwn(CONTENT_TYPES,type))throw Error('Unsupported content type.');
  if(type==='text'){try{return {text:decodeText(bytes)};}catch{throw Error('The received text is not valid UTF-8.');}}
  if(type==='png'){
    const signature=[137,80,78,71,13,10,26,10];
    if(bytes.length<33||signature.some((v,i)=>bytes[i]!==v)||ascii(bytes,12,4)!=='IHDR')throw Error('The content does not have a PNG image header.');
  }
  if(type==='jpeg'&&(bytes.length<4||bytes[0]!==255||bytes[1]!==216||bytes[2]!==255))throw Error('The content does not have a JPEG image header.');
  if(type==='wav')return {audio:inspectWav(bytes)};
  return {};
}
export function imageType(bytes){
  if(bytes[0]===137){validate(bytes,'png');return 'png';}
  if(bytes[0]===255){validate(bytes,'jpeg');return 'jpeg';}
  throw Error('Choose a PNG or JPEG file. The original file bytes are transmitted unchanged.');
}
export function contentDescriptor(bytes,type='raw'){
  validate(bytes,type);
  if(bytes.length>0xffffffff)throw Error('Content is too large.');
  const header=new Uint8Array(CONTENT_HEADER_BYTES);
  header.set([79,67,49,CONTENT_TYPES[type].id]);
  new DataView(header.buffer).setUint32(4,bytes.length);
  return header;
}
export function parseContentDescriptor(header){
  if(!(header instanceof Uint8Array)||header.length!==8)throw Error('Truncated content header.');
  if(header[0]!==79||header[1]!==67||header[2]!==49)throw Error('Unsupported content-envelope version.');
  const type=Object.keys(CONTENT_TYPES).find(key=>CONTENT_TYPES[key].id===header[3]);
  if(!type)throw Error('Unsupported content type in the received message.');
  return {type,length:new DataView(header.buffer,header.byteOffset,header.byteLength).getUint32(4)};
}
export function decodeContent(bytes,descriptor){
  if(!(bytes instanceof Uint8Array))throw Error('Content must be bytes.');
  if(!descriptor)throw Error('A monochrome content descriptor is required.');
  const {type,length}=descriptor;
  if(length!==bytes.length)throw Error('Content length does not match the received message.');
  return {type,...CONTENT_TYPES[type],bytes,...validate(bytes,type)};
}
/** A small PCM sample for an immediate audio round trip; no microphone required. */
export function sampleWav(){
  const sampleRate=8000,samples=960,bytes=new Uint8Array(44+samples),view=new DataView(bytes.buffer);
  const put=(offset,value)=>{for(let i=0;i<value.length;i++)bytes[offset+i]=value.charCodeAt(i);};
  put(0,'RIFF');view.setUint32(4,bytes.length-8,true);put(8,'WAVE');put(12,'fmt ');view.setUint32(16,16,true);
  view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,sampleRate,true);view.setUint32(28,sampleRate,true);
  view.setUint16(32,1,true);view.setUint16(34,8,true);put(36,'data');view.setUint32(40,samples,true);
  for(let i=0;i<samples;i++){const fade=Math.min(1,i/80,(samples-1-i)/160);bytes[44+i]=128+Math.round(50*fade*Math.sin(2*Math.PI*660*i/sampleRate));}
  return bytes;
}
export function samplePng(){
  // A 16 × 16, 94-byte PNG fixture. Generated once with stdlib PNG/zlib encoding.
  return Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAJUlEQVR4nGOQTfAmCTHQUcO1N2vwoIHV8B8JDFUNgydYB03iAwBwlHXYY9/N7QAAAABJRU5ErkJggg=='),c=>c.charCodeAt(0));
}
