/** Read one recording without creating a file-sized JavaScript string.
 * The schema is ordinary JSON. Only top-level records/diagnostics arrays are
 * split into individual JSON values; every value still uses JSON.parse.
 */
import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';

const whitespace=c=>c===' '||c==='\t'||c==='\r'||c==='\n';
const delimiter=c=>whitespace(c)||c===','||c==='}'||c===']';

class RecordingParser {
  constructor({maxTokenBytes,maxDepth,maxArrayItems,streamArrays=false,arrayKeys=['records','diagnostics']}){
    this.maxTokenBytes=maxTokenBytes;this.maxDepth=maxDepth;this.maxArrayItems=maxArrayItems;
    this.streamArrays=streamArrays;this.events=[];this.arrayKeys=new Set(arrayKeys);
    this.counts=Object.fromEntries(arrayKeys.map(key=>[key,0]));
    this.state='root';this.value={};this.keys=new Set();this.key=null;this.token=null;this.array=null;
  }
  fail(message='Malformed recording JSON.'){throw Error(message);}
  begin(role){this.token={role,kind:null,initial:true,inString:false,escaped:false,stack:[],parts:[],bytes:0};}
  append(text){
    if(!text)return;
    this.token.bytes+=Buffer.byteLength(text,'utf8');
    if(this.token.bytes>this.maxTokenBytes)this.fail('Recording JSON value exceeds the per-value byte limit.');
    this.token.parts.push(text);
  }
  field(value){
    // Define an own property even for names such as __proto__.
    Object.defineProperty(this.value,this.key,{value,writable:true,enumerable:true,configurable:true});
    this.key=null;this.state='after-value';
  }
  finishToken(){
    const token=this.token;this.token=null;
    let value;
    try{value=JSON.parse(token.parts.join(''));}catch{this.fail();}
    if(token.role==='key'){
      if(typeof value!=='string')this.fail();
      if(this.keys.has(value))this.fail('Duplicate recording JSON field.');
      this.keys.add(value);this.key=value;this.state='colon';
    }else if(token.role==='array'){
      if(this.counts[this.key]>=this.maxArrayItems)this.fail('Recording JSON array exceeds its item limit.');
      this.counts[this.key]++;
      if(this.streamArrays)this.events.push({key:this.key,value});else this.array.push(value);
      this.state='array-separator';
    }else this.field(value);
  }
  feed(text){
    let i=0;
    while(i<text.length){
      if(this.token){
        const token=this.token,start=i;let finished=false;
        while(i<text.length){
          const c=text[i];
          if(token.initial){
            token.initial=false;
            if(c==='"'){token.kind='string';token.inString=true;}
            else if(c==='{'||c==='['){token.kind='compound';token.stack.push(c);}
            else {
              if(delimiter(c))this.fail();
              token.kind='primitive';
            }
            i++;continue;
          }
          if(token.kind==='primitive'){
            if(delimiter(c)){finished=true;break;}
          }else if(token.inString){
            if(token.escaped)token.escaped=false;
            else if(c==='\\')token.escaped=true;
            else if(c==='"'){
              token.inString=false;
              if(token.kind==='string'){i++;finished=true;break;}
            }
          }else if(c==='"')token.inString=true;
          else if(c==='{'||c==='['){
            token.stack.push(c);
            if(token.stack.length>this.maxDepth)this.fail('Recording JSON nesting exceeds its limit.');
          }else if(c==='}'||c===']'){
            if(token.stack.pop()!==(c==='}'?'{':'['))this.fail();
            if(!token.stack.length){i++;finished=true;break;}
          }
          i++;
        }
        this.append(text.slice(start,i));
        if(finished){this.finishToken();continue;}
        return;
      }
      const c=text[i];
      if(whitespace(c)){i++;continue;}
      switch(this.state){
        case 'root':if(c!=='{')this.fail('Recording JSON must be an object.');this.state='first-key';i++;break;
        case 'first-key':if(c==='}'){this.state='done';i++;break;} // otherwise same as key
        case 'key':if(c!=='"')this.fail();this.begin('key');break;
        case 'colon':if(c!==':')this.fail();this.state='value';i++;break;
        case 'value':
          if(this.arrayKeys.has(this.key)&&c==='['){this.array=[];this.state='array-first';i++;}
          else this.begin('field');
          break;
        case 'array-first':
          if(c===']'){const value=this.array;this.array=null;this.field(value);i++;break;}
          this.begin('array');break;
        case 'array-next':if(c===']')this.fail();this.begin('array');break;
        case 'array-separator':
          if(c===','){this.state='array-next';i++;}
          else if(c===']'){const value=this.array;this.array=null;this.field(value);i++;}
          else this.fail();
          break;
        case 'after-value':
          if(c===','){this.state='key';i++;}
          else if(c==='}'){this.state='done';i++;}
          else this.fail();
          break;
        default:this.fail('Trailing data after recording JSON.');
      }
    }
  }
  finish(){
    if(this.token?.kind==='primitive')this.finishToken();
    if(this.token||this.state!=='done')this.fail('Truncated recording JSON.');
    return this.value;
  }
}

/** Object memory remains proportional to retained captures, but parsing avoids
 * the much lower engine limit on a single string containing the complete file.
 * The per-value limit bounds one capture, including its optional original PNG.
 */
async function readRecording(path,{maxBytes=804*1024*1024,highWaterMark=64*1024,
  maxTokenBytes=16*1024*1024,maxDepth=64,maxArrayItems=1024,onRecord,onDiagnostic,onItem,
  streamArrays=false,arrayKeys=['records','diagnostics']}={}){
  for(const [name,value] of Object.entries({maxBytes,highWaterMark,maxTokenBytes,maxDepth,maxArrayItems}))
    if(!Number.isSafeInteger(value)||value<1)throw Error(`Invalid ${name} limit.`);
  if(highWaterMark>1024*1024)throw Error('Recording JSON stream chunks must be at most 1 MiB.');
  if(!Array.isArray(arrayKeys)||arrayKeys.some(key=>typeof key!=='string'||!key)||new Set(arrayKeys).size!==arrayKeys.length)throw Error('Invalid streamed array keys.');
  if((await stat(path)).size>maxBytes)throw Error('Recording JSON file exceeds its byte limit.');
  const parser=new RecordingParser({maxTokenBytes,maxDepth,maxArrayItems,streamArrays,arrayKeys}),digest=createHash('sha256');
  const drain=async()=>{for(const {key,value} of parser.events){if(onItem)await onItem(key,value);else await (key==='records'?onRecord:onDiagnostic)?.(value);}parser.events=[];};
  // ignoreBOM:true preserves a leading BOM, which is not part of JSON grammar.
  const decoder=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true});
  let bytes=0;
  for await(const chunk of createReadStream(path,{highWaterMark})){
    bytes+=chunk.length;digest.update(chunk);
    if(bytes>maxBytes)throw Error('Recording JSON file exceeds its byte limit.');
    parser.feed(decoder.decode(chunk,{stream:true}));
    await drain();
  }
  parser.feed(decoder.decode());
  const metadata=parser.finish();await drain();
  return {metadata,sha256:digest.digest('hex'),counts:parser.counts};
}

export async function readRecordingJSON(path,options={}){
  return (await readRecording(path,options)).metadata;
}
/** Await one record at a time and discard its large base64 arrays afterwards.
 * Metadata is returned after the complete JSON is validated, regardless of key order.
 */
export async function visitRecordingJSON(path,options={}){
  return readRecording(path,{...options,streamArrays:true});
}
/** Same JSON parser for offline object manifests, with independent resource limits.
 * Recording callers retain their existing browser-capture limits above.
 */
export async function readObjectJSON(path,options={}){
  return readRecording(path,options);
}

