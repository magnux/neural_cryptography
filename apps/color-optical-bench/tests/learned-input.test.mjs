import test from 'node:test';
import assert from 'node:assert/strict';
import {FORMAT,PALETTE,capacity,encodeSequence,phaseFrames,renderFrame,parseHeader} from '../dist/js/format.js';
import {extractCapture,decodeImage,decodeScores,simulateCapture} from '../dist/js/vision.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';
import {extractLearnedInput,normalizeCalibratedRGB,logitsToLogicalScores} from '../dist/js/learned-input.js';

const id=Uint8Array.of(1,4,7,10,13,16,19,22);
function make(options={}){
  const config={size:64,colors:16,gridBlock:8,parity:32,phases:4,permute:true,...options};
  const bytes=Uint8Array.from({length:Math.min(500,capacity(config.size,config.colors,config))},(_,i)=>(i*31+7)&255);
  return {bytes,frames:phaseFrames(encodeSequence(bytes,{...config,id}))};
}
// Deliberately perfect labels used ONLY in tests to test the score/codec boundary.
// They are never supplied to dense capture, a model, or a runtime recovery path.
function idealLogits(frame){
  const count=frame.size**2,logits=new Float32Array(16*count).fill(-20);
  frame.symbols.forEach((physical,cell)=>{logits[physical*count+cell]=20;});
  return logits;
}
function checkCell(result,frame,x,y,tolerance=.015){
  const side=4*frame.size,at=(4*y+2)*side+4*x+2,color=PALETTE[frame.symbols[y*frame.size+x]];
  for(let channel=0;channel<3;channel++){
    const expected=color[channel]/127.5-1,actual=result.input[channel*side*side+at];
    assert.ok(Math.abs(actual-expected)<tolerance,`cell ${x},${y} channel ${channel}: ${actual} vs ${expected}`);
  }
}

test('RGB normalization fixes CHW order and clipping using independent numeric values',()=>{
  const input=normalizeCalibratedRGB(Float32Array.of(0,127.5,255,-255,255,510),2,1);
  assert.deepEqual(input,Float32Array.of(-1,-1.5,0,1,1,1.5));
  assert.throws(()=>normalizeCalibratedRGB(new Float32Array(6),513,1),/raster/);
  assert.throws(()=>normalizeCalibratedRGB(Float32Array.of(NaN,0,0),1,1),/Non-finite/);
  assert.throws(()=>normalizeCalibratedRGB(new Float32Array(2),1,1),/raster/);
  assert.throws(()=>normalizeCalibratedRGB([0,0,0],1,1),/raster/);
});

test('a rendered header produces a dense payload-only tensor with data and reference cells',()=>{
  const frame=make().frames[2],image=renderFrame(frame,1024),result=extractLearnedInput(image);
  assert.equal(result.ok,true,result.error);
  assert.deepEqual(result.dims,[1,3,256,256]);
  assert.equal(result.info.phase,2);assert.equal(result.info.id,parseHeader(frame.header).id);
  assert.equal(result.input.length,3*256*256);
  assert.ok(result.stats.pixelsPerCell>5);assert.ok(result.stats.colorRMSE<2);
  checkCell(result,frame,0,0); // A reference cell is retained in the model raster.
  checkCell(result,frame,10,10);
  checkCell(result,frame,40,40);
  assert.ok(result.input.every(Number.isFinite));
});

test('the 128 grid shares the same preprocessing contract and uses a bounded larger shape',()=>{
  const frame=make({size:128,colors:4}).frames[0],result=extractLearnedInput(renderFrame(frame,1280));
  assert.equal(result.ok,true,result.error);assert.deepEqual(result.dims,[1,3,512,512]);
  assert.equal(result.input.byteLength,3*512*512*4);assert.equal(result.info.colors,4);
  checkCell(result,frame,15,15);checkCell(result,frame,75,75);
});

test('a rotated perspective capture returns canonical dense cell coordinates',()=>{
  const frame=make().frames[2],image=simulateCapture(renderFrame(frame,1024),{preset:'rotated'});
  const result=extractLearnedInput(image);
  assert.equal(result.ok,true,result.error);assert.equal(result.info.phase,2);
  checkCell(result,frame,10,10,.08);checkCell(result,frame,40,40,.08);
});

test('current-image palette correction ignores cached RGB and diagnostic payload truth',()=>{
  const frame=make().frames[0],image=renderFrame(frame,1024),capture=extractCapture(image,{preview:false});
  capture.samples.fill(NaN);capture.info={size:999};
  Object.defineProperty(capture,'chunk',{get(){throw Error('Do not access payload truth.');}});
  Object.defineProperty(capture,'symbols',{get(){throw Error('Do not access labels.');}});
  for(let i=0;i<image.data.length;i+=4){
    image.data[i]=.8*image.data[i]+10;
    image.data[i+1]=.9*image.data[i+1]+5;
    image.data[i+2]=.7*image.data[i+2]+20;
  }
  const result=extractLearnedInput(image,capture);
  assert.equal(result.ok,true,result.error);assert.equal(result.info.size,64);
  checkCell(result,frame,10,10);checkCell(result,frame,40,40);
});

test('a formerly valid capture cannot bypass a damaged current monochrome header',()=>{
  const frame=make().frames[0],image=renderFrame(frame,1024),capture=extractCapture(image,{preview:false});
  const h=FORMAT.header;
  for(let y=h.y*4;y<(h.y+h.rows*h.cellHeight)*4;y++)for(let x=h.x*4;x<(h.x+h.cols*h.cell)*4;x++){
    image.data.set([255,255,255,255],4*(y*image.width+x));
  }
  const result=extractLearnedInput(image,capture);
  assert.equal(result.ok,false);assert.equal(result.stage,'header');assert.equal(result.input,undefined);
});

test('unsupported profiles and malformed geometry fail before allocating model input',()=>{
  for(const size of [32,256]){
    const frame=make({size,colors:4}).frames[0],result=extractLearnedInput(renderFrame(frame,1024));
    assert.equal(result.ok,false);assert.equal(result.stage,'profile');assert.equal(result.input,undefined);
    assert.match(result.error,/classical decoder/);
  }
  const frame=make().frames[0],image=renderFrame(frame,1024);
  const transform=[1,0,0,0,1,0,0,0,NaN];
  assert.equal(extractLearnedInput(image,{transform}).ok,false);
  for(const bad of [null,{}, {width:64,height:64}, {width:64,height:64,data:new Uint8Array(3)}]){
    assert.equal(extractLearnedInput(bad).ok,false);
  }
});

test('failed payload CRC does not prevent creation of a correctly identified camera tensor',()=>{
  const frame=make({parity:0}).frames[0],image=renderFrame(frame,1024);
  // Alter a payload data cell without touching its header or reference palette.
  const layout=gridLayout(frame.size,frame.colors,frame.gridBlock),cell=layout.cells[0],x=cell%frame.size,y=Math.floor(cell/frame.size);
  const p=FORMAT.payload,scale=4,edge=p.side/frame.size;
  const rgb=PALETTE[(frame.symbols[cell]+1)%frame.colors];
  for(let yy=Math.ceil((p.y+y*edge)*scale);yy<Math.floor((p.y+(y+1)*edge)*scale);yy++){
    for(let xx=Math.ceil((p.x+x*edge)*scale);xx<Math.floor((p.x+(x+1)*edge)*scale);xx++)image.data.set([...rgb,255],4*(yy*image.width+xx));
  }
  const classical=decodeImage(image,{preview:false});
  assert.equal(classical.ok,false);assert.equal(classical.stage,'payload');
  const result=extractLearnedInput(image);
  assert.equal(result.ok,true,result.error);assert.equal(result.info.chunkCRC,frame.chunkCRC);
});

test('physical logits undo every palette phase and recover bytes through normal RS/CRC',()=>{
  for(const colors of [4,16]){
    const {bytes,frames}=make({colors,phases:16});
    for(const frame of frames){
      const info=parseHeader(frame.header),scores=logitsToLogicalScores(idealLogits(frame),info);
      assert.equal(scores.length,gridLayout(info.size,info.colors,info.gridBlock).cells.length*colors);
      const decoded=decodeScores(info,scores,{}, {recovery:'erasures'});
      assert.equal(decoded.ok,true,decoded.error);assert.deepEqual(decoded.chunk,bytes);
      assert.equal(decoded.stats.rawCRC,true);
    }
  }
});

test('inactive color classes are masked before softmax and reference scores are excluded',()=>{
  const frame=make({colors:4}).frames[3],logits=idealLogits(frame),count=frame.size**2;
  logits.fill(1e30,4*count); // Impossible colors must not reduce valid-class probability.
  const layout=gridLayout(frame.size,frame.colors,frame.gridBlock);
  for(const cell of layout.pilots)for(let c=0;c<4;c++)logits[c*count+cell]=-1e30;
  const scores=logitsToLogicalScores(logits,frame);
  for(let i=0;i<layout.cells.length;i++){
    const logical=paletteMap(frame.symbols[layout.cells[i]],frame);
    assert.ok(scores[i*4+logical]>254.99);
  }
  assert.equal(decodeScores(frame,scores).ok,true);
});

test('soft scores remain normalized and temperature adjusts confidence without changing phase mapping',()=>{
  const frame=make({colors:4}).frames[1],count=frame.size**2,logits=new Float32Array(16*count);
  logits.fill(2,0,count);logits.fill(1,count,2*count);
  const cold=logitsToLogicalScores(logits,frame,{temperature:.5}),hot=logitsToLogicalScores(logits,frame,{temperature:2});
  const preferred=paletteMap(0,frame);
  assert.ok(cold[preferred]>hot[preferred]);
  for(const scores of [cold,hot])assert.ok(Math.abs(scores.subarray(0,4).reduce((a,b)=>a+b,0)-255)<1e-4);
  logits.fill(3e38);
  assert.ok(logitsToLogicalScores(logits,frame).every(v=>Math.abs(v-63.75)<1e-5));
});

test('invalid logits, unsupported profiles, and bad confidence settings cannot enter recovery',()=>{
  const frame=make().frames[0],logits=idealLogits(frame);
  assert.throws(()=>logitsToLogicalScores(logits.subarray(1),frame),/shape/);
  assert.throws(()=>logitsToLogicalScores(new Float64Array(logits.length),frame),/shape/);
  for(const value of [NaN,Infinity,-Infinity]){
    const bad=logits.slice();bad[0]=value;
    assert.throws(()=>logitsToLogicalScores(bad,frame),/Non-finite/);
  }
  for(const temperature of [0,-1,NaN,Infinity])assert.throws(()=>logitsToLogicalScores(logits,frame,{temperature}),/temperature/);
  for(const change of [{size:512},{colors:8},{phase:9},{gridBlock:2},{permute:'true'},{phases:0}]){
    assert.throws(()=>logitsToLogicalScores(logits,{...frame,...change}),/profile/);
  }
});

test('confident incorrect neural evidence still fails CRC instead of becoming accepted bytes',()=>{
  const {frames}=make({parity:0}),frame=frames[0],logits=idealLogits(frame),layout=gridLayout(frame.size,frame.colors,frame.gridBlock);
  const cell=layout.cells[0],count=frame.size**2,old=frame.symbols[cell];
  logits[old*count+cell]=-20;logits[((old+1)%frame.colors)*count+cell]=20;
  const result=decodeScores(frame,logitsToLogicalScores(logits,frame));
  assert.equal(result.ok,false);assert.match(result.error,/checksum/);
});
