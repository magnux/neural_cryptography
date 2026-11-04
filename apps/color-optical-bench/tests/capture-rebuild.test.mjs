import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,writeFile,readFile,rm,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {MANAGED_OUTPUTS,planRebuild,rebuild} from '../../../scripts/rebuild-corrector-data.mjs';
const base=fileURLToPath(new URL('../../../validation/',import.meta.url));

test('unavailable multires rebuild never alters existing datasets, recordings or checkpoints',async()=>{
  await mkdir(base,{recursive:true});const root=await mkdtemp(join(base,'rebuild-guard-'));
  try{
    for(const name of MANAGED_OUTPUTS){
      if(name.endsWith('.json'))await writeFile(join(root,name),'existing groups');
      else{await mkdir(join(root,name));await writeFile(join(root,name,'keep'),'existing dataset');}
    }
    await mkdir(join(root,'recordings'));await writeFile(join(root,'recordings','capture.json'),'existing capture');
    const before=(await readdir(root)).sort();
    for(const action of [planRebuild,rebuild])await assert.rejects(action({dataRoot:root}),/Nothing was deleted or rewritten[\s\S]*join-uploads/);
    assert.deepEqual((await readdir(root)).sort(),before);
    for(const name of MANAGED_OUTPUTS)assert.equal(await readFile(join(root,name,...(name.endsWith('.json')?[]:['keep'])),'utf8'),name.endsWith('.json')?'existing groups':'existing dataset');
    assert.equal(await readFile(join(root,'recordings','capture.json'),'utf8'),'existing capture');
  }finally{await rm(root,{recursive:true,force:true});}
});
