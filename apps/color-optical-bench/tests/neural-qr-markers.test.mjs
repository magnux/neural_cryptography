import test from 'node:test';
import assert from 'node:assert/strict';
import {findMarkers,quadProposals} from '../dist/js/vision.js';
import {markerPasses} from '../dist/js/neural-qr/markers.js';
import {renderNeuralQR,NQR_LAYOUT} from '../dist/js/neural-qr/layout.js';

const source=()=>renderNeuralQR({size:16,colors:16,phases:4,phase:0,gridBlock:8,pixelsPerCell:4,
  symbols:Uint8Array.from({length:256},(_,i)=>i%16)},512);

function scene(width,height,side,transform=value=>value){
  const original=source(),data=new Uint8ClampedArray(4*width*height);
  for(let at=0;at<data.length;at+=4){data[at]=data[at+1]=data[at+2]=90;data[at+3]=255;}
  const x0=Math.floor((width-side)/2),y0=Math.floor((height-side)/2);
  for(let y=0;y<side;y++)for(let x=0;x<side;x++){
    const from=4*(Math.floor(y*512/side)*512+Math.floor(x*512/side)),to=4*((y+y0)*width+x+x0);
    for(let k=0;k<3;k++)data[to+k]=transform(original.data[from+k],x,y,side);
  }
  return {width,height,data};
}

function checkCenters(markers,width,height,side,tolerance=1.3){
  const x0=Math.floor((width-side)/2),y0=Math.floor((height-side)/2);
  for(const [x,y] of NQR_LAYOUT.centers){
    const wanted=[x0+x*side/256,y0+y*side/256];
    assert.ok(markers.some(marker=>Math.hypot(marker.x-wanted[0],marker.y-wanted[1])<=tolerance),`Missing finder near ${wanted}`);
  }
}

test('clean frames use the first lazy global pass and shared defaults retain their behavior',()=>{
  const image=source(),iterator=markerPasses(image),first=iterator.next().value;
  assert.equal(first.mode,'global');assert.equal(first.proposals.length,1);assert.equal(first.markers.length,4);
  assert.deepEqual(findMarkers(image),findMarkers(image,{thresholdMode:'global',maxSide:1000}));
  assert.equal(quadProposals(first.markers,image.width,image.height).length,1);
  assert.equal(iterator.return().done,true);
});

test('local thresholding recovers corners with incompatible global black/white levels',()=>{
  // Dark-side black/white are 8/55, bright-side 146/239: no global threshold
  // can classify both. The full-resolution local finder ROIs retain their centers.
  const image=scene(1024,720,512,(value,x,y,side)=>x<side/2?.2*value+5:.4*value+140),before=image.data.slice();
  const passes=[...markerPasses(image)];
  assert.equal(passes[0].markers.length,2);assert.equal(passes[0].proposals.length,0);
  assert.equal(passes[1].mode,'adaptive');assert.equal(passes[1].markers.length,4);assert.equal(passes[1].proposals.length,1);
  checkCenters(passes[1].markers,1024,720,512);
  assert.deepEqual(image.data,before,'Detection must not modify RGB evidence for phase reading or the model.');
});

test('a small phone screen is not rejected solely for occupying under 1.2 percent of the camera',()=>{
  const image=scene(1920,1080,160),first=markerPasses(image).next().value;
  assert.equal(first.markers.length,4);
  assert.equal(quadProposals(first.markers,image.width,image.height).length,0,'Existing OBM area bound remains unchanged.');
  assert.equal(first.proposals.length,1);checkCenters(first.markers,1920,1080,160);
});

test('bounded higher-resolution fallback recovers finders lost by the 1000px camera reduction',()=>{
  const image=scene(1920,1080,128),passes=[...markerPasses(image)];
  assert.equal(passes[0].markers.length,0);assert.equal(passes[1].markers.length,0);
  assert.equal(passes.length,3);assert.equal(passes[2].mode,'adaptive-hires');
  assert.equal(passes[2].stats.detectionWidth,1600);assert.equal(passes[2].stats.detectionHeight,900);
  assert.equal(passes[2].markers.length,4);assert.equal(passes[2].proposals.length,1);
  checkCenters(passes[2].markers,1920,1080,128);
  assert.ok(passes.every(pass=>pass.markers.length<=16&&pass.proposals.length<=24));
});

test('blank and smoothly lit images produce no quadrilateral; small images skip a duplicate high-res search',()=>{
  for(const gradient of [false,true]){
    const width=512,height=512,data=new Uint8ClampedArray(4*width*height);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const at=4*(y*width+x),value=gradient?20+200*x/(width-1):200;
      data[at]=data[at+1]=data[at+2]=value;data[at+3]=255;
    }
    const passes=[...markerPasses({width,height,data})];assert.equal(passes.length,2);
    for(const pass of passes){assert.equal(pass.markers.length,0);assert.equal(pass.proposals.length,0);}
  }
});

test('detector and quadrilateral options enforce bounded work',()=>{
  const image=source();
  for(const options of [{maxSide:2048},{maxSide:64},{thresholdMode:'unknown'},{adaptiveWindow:512},{adaptiveWindow:32}])
    assert.throws(()=>findMarkers(image,options),/bounds/);
  assert.throws(()=>quadProposals([],512,512,{minAreaFraction:-1}),/bound/);
});
