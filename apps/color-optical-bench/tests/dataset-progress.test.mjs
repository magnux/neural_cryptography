import test from 'node:test';
import assert from 'node:assert/strict';
import {progress} from '../../../scripts/lib/progress.mjs';

test('offline progress prints initial/final counts and throttles intermediate updates',t=>{
  const lines=[];let now=0;t.mock.method(Date,'now',()=>now);
  const tick=progress('Captures',100,{write:line=>lines.push(line)});
  tick(0);tick(1);now=4999;tick(50);assert.equal(lines.length,1);
  now=5000;tick(51,'G64');assert.equal(lines.length,2);
  now=5001;tick(100);tick(100);assert.equal(lines.length,3);
  assert.match(lines[0],/^Captures: 0\/100 \(0%\)/);
  assert.match(lines[1],/^Captures: 51\/100 \(51%\) \| G64/);
  assert.match(lines[2],/^Captures: 100\/100 \(100%\)/);
});

test('a long file reports growing capture count without finishing the file',t=>{
  const lines=[];let now=0;t.mock.method(Date,'now',()=>now);
  const tick=progress('Files',2,{write:line=>lines.push(line)});
  tick(0);tick(0,'1 capture checked');assert.equal(lines.length,1);
  now=5000;tick(0,'100 captures checked');assert.equal(lines.length,2);
  assert.match(lines[1],/Files: 0\/2.*100 captures checked/);
});
