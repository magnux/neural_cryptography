import test from 'node:test';
import assert from 'node:assert/strict';
import {rsEncode,rsDecode} from '../dist/js/fec.js';

// Independent bitwise field arithmetic checks each codeword's root syndromes.
function multiply(a,b){let value=0;while(b){if(b&1)value^=a;b>>>=1;a<<=1;if(a&256)a^=0x11d;}return value;}
function rootSyndromes(bytes,parity){let root=1;const out=[];for(let j=0;j<parity;j++){let s=0;for(const b of bytes)s=multiply(s,root)^b;out.push(s);root=multiply(root,2);}return out;}
let state=1234;const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state;};

test('RS codewords are systematic and satisfy independent field root checks',()=>{
  for(const parity of [16,32])for(const length of [1,17,100,255-parity]){
    const data=Uint8Array.from({length},()=>random()>>>24),code=rsEncode(data,parity);
    assert.deepEqual(code.subarray(0,length),data);assert.ok(rootSyndromes(code,parity).every(v=>v===0));
    assert.deepEqual(rsDecode(code,parity),{bytes:code,corrected:0});
  }
});
test('RS recovers arbitrary data and parity errors through the promised per-block limit',()=>{
  for(const parity of [16,32])for(const length of [1,18,117,255-parity])for(let trial=0;trial<12;trial++){
    const data=Uint8Array.from({length},()=>random()>>>24),expected=rsEncode(data,parity),damaged=expected.slice(),positions=new Set();
    const count=trial%2?parity/2:1;
    while(positions.size<count)positions.add(random()%damaged.length);
    for(const p of positions)damaged[p]^=1+(random()%255);
    const decoded=rsDecode(damaged,parity);assert.equal(decoded.corrected,count);assert.deepEqual(decoded.bytes,expected);
  }
});
test('RS rejects invalid codeword dimensions and typical over-limit corruption',()=>{
  for(const p of [0,1,17,66])assert.throws(()=>rsEncode(Uint8Array.of(1),p));
  assert.throws(()=>rsDecode(new Uint8Array(16),16));
  const code=rsEncode(Uint8Array.from({length:223},(_,i)=>i),32);for(let i=0;i<40;i++)code[i]^=91+i;
  assert.throws(()=>rsDecode(code,32),/RS/);
});

test('RS corrects mixed errors and erasures at the shortened-block bound',()=>{
  let seed=0x9e3779b9;
  const next=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
  for(const parity of [2,4,16,32,64])for(const length of [1,11,255-parity])for(let trial=0;trial<8;trial++){
    const expected=rsEncode(Uint8Array.from({length},()=>next()>>>24),parity),damaged=expected.slice();
    const erasedCount=trial===0?parity:2*(next()%(parity/2+1));
    const unknownCount=(parity-erasedCount)/2,positions=[];
    while(positions.length<erasedCount+unknownCount){const p=next()%expected.length;if(!positions.includes(p))positions.push(p);}
    const erased=positions.slice(0,erasedCount),beforeErasures=erased.slice();
    for(const p of positions)damaged[p]^=1+(next()%255);
    const before=damaged.slice(),decoded=rsDecode(damaged,parity,erased);
    assert.deepEqual(decoded.bytes,expected,`parity ${parity}, length ${length}, ${unknownCount} errors + ${erasedCount} erasures`);
    assert.equal(decoded.corrected,positions.length);assert.deepEqual(damaged,before);assert.deepEqual(erased,beforeErasures);
    assert.notEqual(decoded.bytes,damaged);assert.ok(rootSyndromes(decoded.bytes,parity).every(v=>v===0));
  }
});

test('RS accepts odd erasure counts and erasures of already correct symbols',()=>{
  for(const parity of [16,32])for(const length of [1,37,255-parity]){
    const code=rsEncode(Uint8Array.from({length},(_,i)=>(i*67+19)&255),parity);
    // Include the first data byte, the last parity byte, and correct parity bytes.
    const erased=[0,...Array.from({length:6},(_,i)=>code.length-1-i)],damaged=code.slice();
    damaged[0]^=211;damaged[code.length-1]^=59;
    const unknownCount=Math.floor((parity-erased.length)/2);
    for(let i=1;i<=unknownCount;i++)damaged[i]^=i+31;
    const decoded=rsDecode(damaged,parity,Uint8Array.from(erased));
    assert.deepEqual(decoded.bytes,code);assert.equal(decoded.corrected,unknownCount+2);
    assert.deepEqual(rsDecode(code,parity,erased),{bytes:code,corrected:0});
  }
});

test('RS validates erasures even when all syndromes are zero',()=>{
  const code=rsEncode(Uint8Array.from({length:37},(_,i)=>i),16);
  for(const erased of [null,{},'0',[0,0],[-1],[code.length],[1.5],[NaN],[Infinity],Array.from({length:17},(_,i)=>i)]){
    assert.throws(()=>rsDecode(code,16,erased),/RS erasure/);
  }
  assert.deepEqual(rsDecode(code,16,[]),{bytes:code,corrected:0});
});

test('RS rejects a mixed over-bound corruption instead of returning unverified bytes',()=>{
  const code=rsEncode(Uint8Array.from({length:223},(_,i)=>(i*41+3)&255),32),damaged=code.slice();
  const erased=Array.from({length:12},(_,i)=>i*7);
  for(const p of erased)damaged[p]^=157;
  // Twelve erased symbols plus eleven unknown errors exceed 2e+s <= 32.
  for(let i=0;i<11;i++)damaged[150+i]^=i+1;
  const before=damaged.slice();assert.throws(()=>rsDecode(damaged,32,erased),/RS/);assert.deepEqual(damaged,before);
});
