import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {mkdir,mkdtemp,rm,readFile,access} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createSender} from '../dist/js/dataset/contract.js';
import {captureFrame,CaptureBuffer} from '../dist/js/dataset/capture.js';
import {renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {recordingBlob} from '../dist/js/dataset/recording-json.js';
import {fetchCollectionSession} from '../dist/js/dataset/upload.js';
import {joinUploads} from '../../../scripts/corrector-dataset.mjs';

const root=fileURLToPath(new URL('../../../',import.meta.url));
test('actual acquisition, HTTP staging/finalization and offline joining preserve current and smaller recordings',async()=>{
  const scratch=join(root,'validation');await mkdir(scratch,{recursive:true});
  const dir=await mkdtemp(join(scratch,'session-stack-'));
  const child=spawn('python3',['-u','-c',`
import sys,runpy
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
m=runpy.run_path('apps/color-optical-bench/serve.py')
store=m['RecordingStore'](Path(sys.argv[1])/'recordings',Path('apps/color-optical-bench/dist'))
server=ThreadingHTTPServer(('127.0.0.1',0),partial(m['BenchHandler'],directory='apps/color-optical-bench/dist',recordings=store))
print(server.server_address[1],flush=True)
server.serve_forever()
`,dir],{cwd:root,stdio:['ignore','pipe','pipe']});
  let stderr='';child.stderr.on('data',chunk=>{stderr=(stderr+chunk).slice(-4000);});
  try{
    const port=await new Promise((resolve,reject)=>{
      let text='';const timer=setTimeout(()=>reject(Error('Server startup timed out: '+stderr)),10000);
      child.once('error',error=>{clearTimeout(timer);reject(error);});
      child.once('exit',code=>{clearTimeout(timer);reject(Error('Server exited '+code+': '+stderr));});
      child.stdout.on('data',chunk=>{text+=chunk;if(text.includes('\n')){clearTimeout(timer);resolve(Number(text.trim()));}});
    });
    const origin=`http://127.0.0.1:${port}`;
    const config=await (await fetch(origin+'/api/recordings/config')).json();
    const post=async(path,value)=>{
      const response=await fetch(origin+path,{method:'POST',headers:{'Content-Type':'application/json','X-Upload-Token':config.uploadToken},body:recordingBlob(value)});
      const body=await response.json();assert.ok(response.ok,JSON.stringify(body));return body;
    };
    const {manifest,frames}=await createSender({size:8,seed:554,sessionId:'stack-sender'});
    manifest.datasetSplit='train';
    const staged=await post('/api/collection-session?split=train',manifest);
    assert.ok(staged.storedPath.endsWith('.json.tmp'));
    const assignment=await fetchCollectionSession({config,fetcher:(url,options)=>fetch(origin+url,options)});
    assert.equal(assignment.id,staged.session.id);
    assert.equal(assignment.collectionCode,manifest.collectionCode);
    assert.equal(assignment.payloadBase64,undefined);assert.equal(assignment.payloadId,undefined);
    const buffer=new CaptureBuffer({maxBytes:160*1024*1024}); // Previously recorded budget stays valid.
    for(const frame of frames){
      const timestamp=1000+frame.phase*2000;
      const captured=captureFrame(renderNeuralQR(frame,512),{timestamp,profile:assignment.profile,collectionCode:assignment.collectionCode});
      assert.equal(captured.ok,true,captured.error);
      buffer.add(captured);assert.equal(buffer.add({...captured,timestamp:timestamp+600}).accepted,true);
    }
    const recording={...buffer.export({sessionId:'stack-receiver'}),datasetSplit:'train',collectionSessionId:assignment.id};
    const saved=await post('/api/recordings?split=train',recording);
    assert.equal(saved.sessionCompleted,true);
    await assert.rejects(access(join(dir,'recordings',staged.storedPath)));
    assert.deepEqual(JSON.parse(await readFile(join(dir,'recordings',staged.storedPath.slice(0,-4)),'utf8')),manifest);
    assert.equal(await fetchCollectionSession({config,fetcher:(url,options)=>fetch(origin+url,options)}),null);
    const duplicate=await post('/api/recordings?split=train',recording);
    assert.equal(duplicate.duplicate,true);
    const smaller={...recording,sessionId:'previous-manual-receiver'};delete smaller.collectionSessionId;
    await post('/api/recordings?split=train',smaller);
    // A pending next matrix must be excluded from finalized dataset discovery.
    const next=await createSender({size:8,seed:555,sessionId:'stack-sender-next'});
    await post('/api/collection-session?split=train',{...next.manifest,datasetSplit:'train'});
    const joined=await joinUploads({'recordings-root':join(dir,'recordings'),out:join(dir,'joined')});
    assert.equal(joined.datasets.length,1);assert.equal(joined.datasets[0].samples,8);
    const dataset=JSON.parse(await readFile(join(dir,'joined/train/g8/manifest.json'),'utf8'));
    for(const sample of dataset.samples){
      assert.deepEqual(new Uint8Array(await readFile(join(dir,'joined/train/g8',sample.labels))),frames[sample.info.phase].symbols);
    }
  }finally{
    const closed=child.exitCode===null?once(child,'close'):Promise.resolve();child.kill();await closed;
    await rm(dir,{recursive:true,force:true});
  }
});
