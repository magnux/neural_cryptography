import {Assembler,sameBytes} from './format.js';
import {decodeContent} from './content.js';

/** One receiver's state. Live camera and image lab each own a separate instance. */
export class ReceiveSession {
  constructor(){this.epoch=0;this.reset();}
  reset(){
    this.epoch++;this.assembler=new Assembler();this.captures=0;this.validCaptures=0;
    this.received=null;this.content=null;this.contentError=null;this.lastResult=null;this.exact=false;
  }
  accept(result,expected=null){
    this.captures++;this.lastResult=result;
    if(!result.ok)return {valid:false};
    const progress=this.assembler.accept(result.info,result.chunk);this.validCaptures++;
    const fresh=Boolean(progress.complete)&&this.received===null;
    if(fresh){
      this.exact=expected?.id===result.info.id&&sameBytes(progress.complete,expected.payload);
      if(expected?.id===result.info.id&&!this.exact)throw Error('Byte-for-byte comparison failed despite checksums.');
      this.received=progress.complete;
      try{this.content=decodeContent(this.received,result.info.content);}catch(error){this.contentError=error.message;}
    }
    return {valid:true,...progress,fresh,content:this.content,contentError:this.contentError,exact:this.exact};
  }
}
