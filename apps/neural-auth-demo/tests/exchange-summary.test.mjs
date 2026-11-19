import test from 'node:test';
import assert from 'node:assert/strict';
import {ExchangeStepClock,buildExchangeSummary,recordTrainedInference,recordTrainedAcceptance} from '../dist/exchange-summary.js';

const settings={role:'verifier',displayName:'My phone',statistics:true,phaseMs:350,captureMs:250,
  outgoingSize:64,incomingMaxSize:128,optimized:true,repetitions:32};
const auth={result:true,resultDelivered:true,peerDisplayName:'Other phone',repetitions:64,proofBytes:7620};

test('five local step intervals add to the completion time with startup broken out; waiting for Finish does not add time',()=>{
  const clock=new ExchangeStepClock(100);
  clock.observe(1,200);
  clock.ready(300.4);clock.ready(3000);
  for(const [step,at] of [[2,800],[3,900],[4,1200],[4,1400],[3,1500],[5,9210]])clock.observe(step,at);
  clock.finish(9810);clock.finish(1_000_000);clock.observe(2,1_000_001);
  const s=clock.snapshot();
  assert.equal(s.elapsedMs,9710);assert.equal(s.setupMs,200);
  assert.deepEqual(s.steps.map(r=>r.label),['Hello','Identity','Challenge','Proof','Result']);
  assert.deepEqual(s.steps.map(r=>r.durationMs),[500,100,300,8010,600]);
  assert.equal(s.steps.reduce((n,r)=>n+r.durationMs,s.setupMs),s.elapsedMs);
});

test('duplicate or invalid callbacks neither advance nor manufacture a missed boundary',()=>{
  const clock=new ExchangeStepClock(10);clock.ready(25);
  for(const [step,at] of [[2,9],[2,20],[0,25],[6,50],[2,NaN],[3,50],[2,55],[3,58],[4,60],[5,70]])clock.observe(step,at);
  clock.finish(67);assert.equal(clock.snapshot().elapsedMs,null);
  clock.finish(80);
  assert.deepEqual(clock.snapshot().steps.map(x=>x.durationMs),[null,null,10,10,10]);
});

test('short export distinguishes role outcomes, excludes identities/packet traces and uses actual repetitions',()=>{
  const clock=new ExchangeStepClock(10);clock.ready(20);clock.finish(100);
  for(const role of ['verifier','prover'])for(const accepted of [true,false]){
    const state={...auth,result:accepted,resultDelivered:role==='verifier',peerReported:role==='prover',
      pairingCode:'sensitive',localKeyId:'sensitive',proofDigest:'sensitive',sessionId:'sensitive'};
    const s=buildExchangeSummary({settings:{...settings,role,cameraId:'sensitive'},auth:state,clock,
      started:'2026-11-18T20:00:00.000Z',captureAttempts:121,acceptedMatrices:23,inference:{count:92,meanMs:116.75},
      camera:{width:1920,height:1080,frameRate:29.97,deviceId:'sensitive',label:'sensitive'},
      model:{effectiveBackend:'wasm',modelProfile:'VW',manifest:'sensitive'},
      packetRtt:{count:22,minMs:602,meanMs:1342,maxMs:2911,records:['sensitive']}});
    assert.equal(s.outcome,role==='verifier'?(accepted?'verified-locally':'rejected-locally'):
      (accepted?'peer-reported-accepted':'peer-reported-rejected'));
    assert.deepEqual(s.camera,{width:1920,height:1080,frameRate:30});
    assert.equal(s.repetitions,64);assert.equal(s.proofBytes,7620);assert.equal(s.backend,'wasm');
    assert.deepEqual(s.packetRtt,{count:22,minMs:602,meanMs:1342,maxMs:2911});
    assert.deepEqual(s.inference,{count:92,meanMs:116.8});assert.equal(s.captureAttempts,121);
    assert.equal(JSON.stringify(s).includes('sensitive'),false);
  }
  for(const role of ['prover','verifier']){
    assert.equal(buildExchangeSummary({settings:{...settings,role},auth:{result:false}}),null);
  }
});

test('statistics off skips timing access; completion and peer name are still reportable',()=>{
  const s=buildExchangeSummary({settings:{...settings,statistics:false},auth,
    clock:{snapshot(){throw Error('Disabled statistics were accessed.');}},captureAttempts:6,acceptedMatrices:5,
    camera:{width:1920,height:1080,frameRate:60},packetRtt:{count:2,meanMs:700},inference:{count:2,meanMs:85}});
  assert.equal(s.outcome,'verified-locally');assert.equal(s.peerName,'Other phone');
  for(const key of ['setupMs','elapsedMs','captureAttempts','acceptedMatrices','camera','packetRtt','inference','trainedRecovery'])assert.equal(s[key],null);
  assert.deepEqual(s.steps,[]);
});

test('short report retains only usable camera dimensions and a known frame rate',()=>{
  assert.equal(buildExchangeSummary({settings,auth,camera:{width:1920,height:Infinity}}).camera,null);
  assert.deepEqual(buildExchangeSummary({settings,auth,camera:{width:1280,height:720,frameRate:0}}).camera,
    {width:1280,height:720,frameRate:null});
});

test('short reports identify the selected ACK path and include only applicable receipt counts',()=>{
  const reads={scans:38,decoded:16,matched:15,unmatched:1,digest:'private'};
  for(const [ackSize,statistics] of [[16,true],[32,true],[16,false]]){
    const report=buildExchangeSummary({settings:{...settings,ackSize,statistics},auth,receiptReads:reads});
    assert.equal(report.settings.ackSize,ackSize);
    assert.deepEqual(report.receiptReads,ackSize===16&&statistics?{scans:38,decoded:16,matched:15,unmatched:1}:null);
  }
});

test('live recovery separates failed inference attempts from CER on matched packets',()=>{
  const rows=[];
  function scan(size,expected,inputCaptures,stats,acceptance){
    const row=recordTrainedInference(rows,expected,{info:{size},inputCaptures,stats});
    if(row)recordTrainedAcceptance(row,acceptance,stats);
    return row;
  }
  const failed={rawCRCValid:0,rsDecoded:0,crcAccepted:0};
  const fixed={rawCRCValid:0,rsDecoded:1,crcAccepted:1,correctedBytes:8,wireCellErrors:10,wireCells:16000};
  assert.equal(scan(128,'data',0,{}),null); // still collecting
  assert.equal(scan(16,'ack',1,fixed,{accepted:true}),null); // classical digest reader
  scan(128,'data',4,failed);
  scan(128,'data',8,{...failed,rsDecoded:1});
  scan(128,'data',8,fixed,{accepted:true,reason:'delivered'});
  scan(128,'data',4,fixed,{accepted:true,reason:'already-delivered'});
  scan(128,'data',4,fixed,{accepted:false,reason:'foreign-session'});
  scan(32,'request',4,{...fixed,rawCRCValid:1,correctedBytes:0,wireCellErrors:0,wireCells:800},{accepted:true});
  const report=buildExchangeSummary({settings:{...settings,localColorCorrection:true},auth,trainedRecovery:rows});
  assert.equal(report.settings.localColorCorrection,true);
  assert.deepEqual(report.trainedRecovery.map(r=>[r.expected,r.grid]),[['request',32],['data',128]]);
  assert.deepEqual(report.trainedRecovery[1],{expected:'data',grid:128,inferences:5,rawCRCPassed:0,rsCRCPassed:3,
    rsFailures:1,crcFailures:1,correctedBytesOnCRCPass:24,transportAccepted:1,duplicates:1,
    acceptedWireCellErrors:10,acceptedWireCells:16000,acceptedPacketsWithCER:1});
  assert.equal(report.trainedRecovery[0].acceptedWireCellErrors,0);assert.equal(report.trainedRecovery[0].acceptedWireCells,800);
});
