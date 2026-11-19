import test from 'node:test';
import assert from 'node:assert/strict';
import {sweepCellSketch} from '../dist/sweep-cell-sample.js';
import {gridLayout} from '../dist/js/grid.js';
import {scoreSweepCells} from '../../../scripts/score-sweep-cells.mjs';

const cfg={size:128,ackSize:16,startPhaseMs:500,startCaptureMs:250,decrementMs:50,localColorCorrection:false};
const id='12'.repeat(16),sketch=(w,s)=>'s1:'+w+':'+s;
const sample=(w,s)=>sketch(w.repeat(96),s.repeat(256));
const row=(step,cellSample,inferences={K4:1,K8:0})=>({step,phaseMs:500-50*(step-1),captureMs:250-50*(step-1),cellSample,reader:{inferences}});
const report=(role,rows)=>({format:'neural-qr-corrected-timing-sweep',version:1,runId:id,configuration:{role,...cfg},steps:rows});

test('sample looks only at distinct nonpilot wire nibbles; exactly 256 held-out and 96 witness nibbles',()=>{
  for(const size of [32,64,128]){
    const x=new Uint8Array(size*size),{pilots,cells}=gridLayout(size,16,8);
    const blank=sweepCellSketch(x,size);
    assert.equal(blank,sample('0','0'));
    for(const c of pilots)x[c]=15;
    if(cells.length%2)x[cells.at(-1)]=15;
    assert.equal(sweepCellSketch(x,size),blank);
    // A unique position can never influence both witness and scoring tracks.
    if(size===32){
      let marked=0;
      for(const c of cells.subarray(0,cells.length&~1)){
        x[c]=1;
        const n=sweepCellSketch(x,size).slice(3).split('1').length-1;
        assert.ok(n<=1);marked+=n;x[c]=0;
      }
      assert.equal(marked,352);
    }
  }
});

test('failed packets with a clear witness can be scored, while stale and ambiguous steps explicitly abstain',()=>{
  const sender=report('sender',[row(1,{reference:sample('a','0')}),row(2,{reference:sample('4','f')})]);
  const receiver=report('receiver',[
    row(1,{K4:sketch('a'.repeat(60)+'c'.repeat(36),'0'.repeat(192)+'2'.repeat(64)),status:{K4:'unconfirmed'}}),
    row(2,{K4:sample('a','0'),K8:sketch('a'.repeat(47)+'4'.repeat(49),'f'.repeat(256)),status:{K4:'unconfirmed',K8:'unconfirmed'}},{K4:1,K8:1})]);
  receiver.runId=null;
  const a=scoreSweepCells(sender,receiver);
  assert.match(a.runBinding,/witness-only/);assert.equal(a.byCapture.K4.sampledCER,.25);
  assert.equal(a.byCapture.K4.likely,1);assert.equal(a.byCapture.K4.otherStep,1);
  assert.equal(a.byCapture.K8.ambiguous,1);assert.equal(a.byCapture.K8.noInference,1);
  assert.equal(a.byCapture.K8.sampledCER,null);
});

test('missing, weak and known foreign samples are not scored; configs and conflicting run IDs reject',()=>{
  const sender=report('sender',[row(1,{reference:sample('1','2')}),row(2,{reference:sample('3','4')})]);
  const receiver=report('receiver',[row(1,{K4:sample('a','2'),K8:sample('1','2'),status:{K4:'unconfirmed',K8:'foreign'}}),
    row(2,{K4:sample('3','4'),status:{K4:'matched'}},{K4:1,K8:1}),row(3,{K4:sample('a','b'),status:{K4:'unconfirmed'}})]);
  const a=scoreSweepCells(sender,receiver);
  assert.equal(a.byCapture.K4.validated,1);assert.equal(a.byCapture.K4.sampledCER,0);
  assert.equal(a.byCapture.K4.weakWitness,1);assert.equal(a.byCapture.K4.missingReference,1);
  assert.equal(a.byCapture.K8.foreign,1);assert.equal(a.byCapture.K8.missingSketch,1);
  assert.throws(()=>scoreSweepCells(sender,{...receiver,runId:'cd'.repeat(16)}),/run IDs/);
  assert.throws(()=>scoreSweepCells(sender,{...receiver,configuration:{...receiver.configuration,size:32}}),/configuration/);
  assert.throws(()=>scoreSweepCells(sender,{...receiver,steps:[row(1,{K4:'not a sketch'})]}),/format/);
});
