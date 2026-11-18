import test from 'node:test';
import assert from 'node:assert/strict';
import {renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {gridLayout} from '../dist/js/grid.js';
import {createTimingTrial,TimingScanner,TimingPeer,compareTimingEcho,makeTimingEcho} from '../dist/roundtrip-core.js';

const request=(size=64,sequence=1,run=0xfacedead)=>createTimingTrial({size,sequence,run});
const scan=frame=>new TimingScanner().scan(renderNeuralQR(frame,768));
for(const [i,size] of [16,32,64,128].entries())test(`one exposure of G${size} is enough for each leg, including the independent echo ID`,()=>{
  const orig=request(size,i+1),received=scan(orig);
  assert.equal(received.ok,true,received.error);assert.equal(received.kind,'request');assert.equal(received.run,orig.run);
  assert.equal(received.sequence,i+1);assert.equal(received.frame.phase,i);assert.equal(received.timings.detectionMs>=0,true);
  assert.equal(compareTimingEcho(orig,received),null,'A sender cannot see its own request as an echo.');
  const peer=new TimingPeer(),echo=peer.accept(received),back=scan(echo),result=compareTimingEcho(orig,back);
  assert.equal(back.kind,'echo');assert.ok(result.cells>0&&result.cells<size**2);
  assert.equal(result.cer,0);assert.equal(peer.accept(back),null);assert.equal(peer.accept(received),null);
});

test('two-leg CER counts observed wrong cells even when almost every test symbol is wrong',()=>{
  const orig=request(),bad={...orig,symbols:orig.symbols.slice()},g=gridLayout(orig.size,16,8);
  for(const p of g.cells){
    const x=p%64,y=p>>6;
    if(x<24&&y<16||x>=40&&y>=48)continue; // both independent trial-ID pilot regions
    bad.symbols[p]=(bad.symbols[p]+1)&15;
  }
  const received=scan(bad),echo=makeTimingEcho(received);
  // Corrupt a previously changed real data cell back to its original symbol
  // only on the return trip. The end-to-end score must reflect both legs.
  const p=g.cells.find(i=>i%64>=24&&i%64<40);
  echo.symbols[p]=orig.symbols[p];
  const result=compareTimingEcho(orig,scan(echo));
  assert.ok(result.cells>100);assert.equal(result.errors,result.cells-1);
});

test('tag CRC handles a single damaged copy, rejects two bad or contradictory copies',()=>{
  const orig=request(32),other=request(32,5),mutate=()=>({...orig,symbols:orig.symbols.slice()});
  const a=mutate();a.symbols[0]^=1;
  assert.equal(scan(a).ok,true);
  a.symbols[32*24+20]^=1;assert.equal(scan(a).ok,false);
  const b=mutate();
  for(let y=24;y<32;y++)b.symbols.set(other.symbols.subarray(32*y+20,32*y+32),32*y+20);
  assert.equal(scan(b).ok,false);
  assert.equal(scan({...orig,phase:1}).ok,false,'matrix held across torn phase cues cannot match ID');
});

test('stale sequences and foreign runs cannot confirm a trial or replace a held return',()=>{
  const p=new TimingPeer(),newer=request(32,5),older=request(32,1),foreign=request(32,6,123);
  const echo=p.accept(scan(newer));assert.ok(echo);
  assert.equal(p.accept(scan(older)),null);assert.equal(p.accept(scan(foreign)),null);
  assert.equal(compareTimingEcho(older,scan(echo)),null);
  assert.equal(compareTimingEcho(foreign,scan(echo)),null);
  p.reset();assert.ok(p.accept(scan(foreign)));
});
