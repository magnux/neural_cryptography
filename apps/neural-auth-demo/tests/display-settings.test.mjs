import test from 'node:test';
import assert from 'node:assert/strict';
import {bindDisplaySettings,readDisplaySettings} from '../dist/display-settings.js';

test('the visible timing mode supplies all scheduler limits; hidden drafts never affect a run',()=>{
  const fields=new Map(),$=id=>{
    if(!fields.has(id))fields.set(id,{addEventListener(type,fn){this[type]=fn;}});
    return fields.get(id);
  };
  for(const [id,value] of Object.entries({phaseCadence:'progressive',phaseMs:'',phaseMinMs:'100',phaseStepMs:'25',phaseMaxMs:'300'}))$(id).value=value;
  bindDisplaySettings($);assert.ok($('phaseMsField').hidden);assert.ok($('phaseMs').disabled);
  assert.deepEqual(readDisplaySettings($),{phaseCadence:'progressive',phaseMs:100,phaseMinMs:100,phaseStepMs:25,phaseMaxMs:300});
  $('phaseMaxMs').value='90';assert.throws(()=>readDisplaySettings($),/at least as long/);
  $('phaseMaxMs').value='300';$('phaseStepMs').value='0';assert.throws(()=>readDisplaySettings($),/Increment/);
  $('phaseCadence').value='fixed';$('phaseCadence').change();$('phaseMs').value='350';
  assert.ok(!$('phaseMsField').hidden);assert.ok($('phaseStepMs').disabled);
  assert.deepEqual(readDisplaySettings($),{phaseCadence:'fixed',phaseMs:350,phaseMinMs:350,phaseStepMs:0,phaseMaxMs:350});
});
