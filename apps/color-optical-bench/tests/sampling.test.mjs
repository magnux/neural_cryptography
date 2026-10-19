import test from 'node:test';
import assert from 'node:assert/strict';
import {filterSamples,robustSamples} from '../dist/js/sampling.js';

const bytes=(...values)=>Uint8Array.from(values);
function approximate(actual,expected,tolerance=1e-5){
  assert.equal(actual.length,expected.length);
  actual.forEach((value,i)=>assert.ok(Math.abs(value-expected[i])<tolerance,`${value} != ${expected[i]} at ${i}`));
}

test('robust sampling selects different observations for different cells',()=>{
  const result=robustSamples([
    bytes(255,0,255,0),bytes(255,0,0,255),bytes(0,255,0,255),
  ],2,2);
  assert.deepEqual(Array.from(result.scores),[255,0,0,255]);
  assert.equal(result.stats.trimmedCells,2);
  assert.equal(result.stats.meanSelectionDistance,0);
  assert.equal(result.stats.meanDiscardedDistance,2);
});

test('five-of-seven rejects two contradictory outliers without hard-deciding the consensus',()=>{
  const good=bytes(240,10,5,0),bad=bytes(0,0,0,255);
  const result=robustSamples([good,bad,good,good,bad,good,good],4,5);
  approximate(result.scores,good);
  assert.equal(result.stats.kept,5);
  assert.equal(result.stats.trimmedCells,1);
  assert.equal(result.stats.ambiguousCells,0);
  assert.equal(result.stats.meanSelectionDistance,0);
  // The independent ordinary mean remains a different candidate.
  assert.ok(filterSamples([good,bad,good,good,bad,good,good],'mean')[3]>70);
});

test('each observation is normalized and zero evidence remains ambiguous',()=>{
  const result=robustSamples([bytes(1,1,0,0),bytes(100,100,0,0),bytes(0,0,0,0)],2,3);
  approximate(result.scores,[127.5,127.5,127.5,127.5]);
  assert.equal(result.stats.ambiguousCells,2);
  assert.equal(result.stats.trimmedCells,0);
  assert.equal(result.stats.meanSelectionDistance,0);
  assert.equal(result.stats.meanDiscardedDistance,0);
});

test('equal-distance medoids and subset membership have stable input-order ties',()=>{
  const samples=[bytes(255,0,0,0),bytes(0,255,0,0),bytes(0,0,255,0)];
  approximate(robustSamples(samples,4,1).scores,[255,0,0,0]);
  approximate(robustSamples(samples,4,2).scores,[127.5,127.5,0,0]);
});

test('FIFO replacement can displace an initially wrong consensus',()=>{
  const wrong=bytes(0,255),correct=bytes(255,0);
  const pool=[wrong,wrong,wrong,wrong,wrong,correct,correct];
  approximate(robustSamples(pool,2,5).scores,[0,255]);
  for(let i=0;i<3;i++){pool.shift();pool.push(correct);}
  approximate(robustSamples(pool,2,5).scores,[255,0]);
  assert.equal(pool.length,7);
});

test('robust sampling supports all palette sizes and leaves source evidence unchanged',()=>{
  for(const colors of [2,4,8,16]){
    const sample=Uint8Array.from({length:colors*3},(_,i)=>(i*37)%256);
    const before=sample.slice(),result=robustSamples(Array(11).fill(sample),colors,9);
    assert.deepEqual(sample,before);
    for(let cell=0;cell<3;cell++){
      let total=0;for(let k=0;k<colors;k++)total+=result.scores[cell*colors+k];
      assert.ok(Math.abs(total-255)<2e-5);
    }
    assert.equal(result.stats.trimmedCells,0);
  }
});

test('robust sampling rejects invalid settings and incompatible evidence',()=>{
  for(const [samples,colors,keep] of [
    [[],2,1],[null,2,1],[Array(12).fill(bytes(1,2)),2,5],
    [[bytes(1,2)],3,1],[[bytes(1,2)],2,0],[[bytes(1,2)],2,2],
    [[bytes(1,2)],2,1.5],[[bytes(1,2,3)],2,1],[[new Uint8Array()],2,1],
    [[bytes(1,2),bytes(1,2,3,4)],2,1],[[new Float32Array(2)],2,1],
    [[null],2,1],[[[1,2]],2,1],
  ])assert.throws(()=>robustSamples(samples,colors,keep));
});
