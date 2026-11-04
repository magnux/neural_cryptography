import test from 'node:test';
import assert from 'node:assert/strict';
import {createCaptureReference,selectCaptureReplacement} from '../dist/js/dataset/diversity.js';

const signature=percent=>new Float32Array(3072).fill(percent/50);
const pool=(count,percent=0)=>Array.from({length:count},()=>signature(percent));
const rng=seed=>()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};

test('central reference excludes a minority of terrible warm-up frames',()=>{
  const signatures=[...pool(26),...pool(6,40)],reference=createCaptureReference(signatures);
  assert.equal(reference.ready,true);assert.equal(reference.coherentSamples,26);
  assert.deepEqual(reference.limits,[1,2,3]);
  const selected=selectCaptureReplacement(signatures,signature(0),{reference,random:()=>0});
  assert.equal(selected.reason,'replace-outlier');assert.equal(selected.replaceIndex,26);
  assert.equal(selected.counts[3],6);
  assert.equal(selectCaptureReplacement(signatures,signature(20),{reference}).reason,'isolated-candidate');
});

test('ambiguous warm-up clusters cannot produce a trusted center',()=>{
  const reference=createCaptureReference([...pool(16),...pool(16,40)]);
  assert.equal(reference.ready,false);assert.equal(reference.reason,'no-majority-cluster');
  assert.equal(selectCaptureReplacement([...pool(16),...pool(16,40)],signature(0),{reference}).replaceIndex,null);
});

test('retention converges to core/shoulder/tail budgets rather than maximum distance',()=>{
  const signatures=pool(32),reference=createCaptureReference(signatures),random=rng(42);
  const limits=reference.limits.slice();let final;
  for(let i=0;i<800;i++){
    const incoming=signature((random()-.5)*5.9);
    final=selectCaptureReplacement(signatures,incoming,{reference,random});
    if(final.replaceIndex!==null)signatures[final.replaceIndex]=incoming;
  }
  final=selectCaptureReplacement(signatures,signature(20),{reference,random});
  assert.deepEqual(final.target,[22,8,2]);
  assert.deepEqual(final.counts,[22,8,2,0]);
  assert.deepEqual(reference.limits,limits,'retention must not contract its own reference');
  assert.equal(final.reason,'isolated-candidate');
});

test('six-frame pools retain a central majority without rounding a large hard tail',()=>{
  const signatures=pool(6),reference=createCaptureReference(signatures),random=rng(55);
  for(let i=0;i<200;i++){
    const incoming=signature((random()-.5)*5.9);
    const result=selectCaptureReplacement(signatures,incoming,{reference,random});
    if(result.replaceIndex!==null)signatures[result.replaceIndex]=incoming;
  }
  const result=selectCaptureReplacement(signatures,signature(2.8),{reference,random});
  assert.deepEqual(result.target,[4,2,0]);assert.deepEqual(result.counts,[4,2,0,0]);
  assert.equal(result.reason,'band-budget-zero');
});

test('near-duplicate frames do not churn, but valid variation gets randomized refresh',()=>{
  const signatures=pool(32),reference=createCaptureReference(signatures);
  assert.equal(selectCaptureReplacement(signatures,signature(.001),{reference}).reason,'near-duplicate');
  assert.equal(selectCaptureReplacement(signatures,signature(.5),{reference,random:()=>.9}).reason,'kept-band');
  const result=selectCaptureReplacement(signatures,signature(.5),{reference,random:()=>0});
  assert.equal(result.reason,'refresh-band');assert.equal(result.replaceIndex,0);
  assert.throws(()=>selectCaptureReplacement(signatures,signature(.5),{reference,random:()=>1}),/randomness/);
});

test('missing deviation bands do not prevent maintaining central samples',()=>{
  const signatures=pool(16),reference=createCaptureReference(signatures);
  for(let i=1;i<12;i++){
    const incoming=signature(i/20),result=selectCaptureReplacement(signatures,incoming,{reference,random:()=>0});
    if(result.replaceIndex!==null)signatures[result.replaceIndex]=incoming;
  }
  const result=selectCaptureReplacement(signatures,signature(20),{reference});
  assert.deepEqual(result.counts,[16,0,0,0]);assert.deepEqual(result.target,[11,4,1]);
});

test('invalid signatures and references are rejected',()=>{
  assert.equal(createCaptureReference([]).ready,false);
  assert.throws(()=>createCaptureReference([Float32Array.of(NaN)]),/signature/);
  assert.throws(()=>selectCaptureReplacement(pool(6),signature(0),{reference:{ready:true,center:signature(0),limits:[2,1,3]}}),/reference/);
});
