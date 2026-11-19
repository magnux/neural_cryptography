// Only the active timing mode participates in validation. Switching modes
// preserves the other mode's values for a repeatable two-phone comparison.
export function bindDisplaySettings($){
  const mode=$('phaseCadence');
  function showFields(){
    const fixed=mode.value==='fixed';
    for(const id of ['phaseMs','phaseMinMs','phaseStepMs','phaseMaxMs']){
      const hide=id==='phaseMs'?!fixed:fixed;
      $(id+'Field').hidden=hide;
      $(id).disabled=hide;
    }
  }
  mode.addEventListener('change',showFields);showFields();
}

export function readDisplaySettings($){
  const phaseCadence=$('phaseCadence').value;
  function ms(id,label,min=50){
    const n=Number($(id).value);
    if(!Number.isSafeInteger(n)||n<min||n>10000)throw Error(label+' must be a whole number from '+min+' to 10000 ms.');
    return n;
  }
  if(phaseCadence==='fixed'){
    const phaseMs=ms('phaseMs','Fixed phase interval');
    return {phaseCadence,phaseMs,phaseMinMs:phaseMs,phaseStepMs:0,phaseMaxMs:phaseMs};
  }
  if(phaseCadence!=='progressive')throw Error('Choose progressive or fixed outgoing phase timing.');
  const phaseMinMs=ms('phaseMinMs','First phase interval'),phaseStepMs=ms('phaseStepMs','Increment',1),
    phaseMaxMs=ms('phaseMaxMs','Longest phase');
  if(phaseMaxMs<phaseMinMs)throw Error('The longest phase must be at least as long as the first phase interval.');
  return {phaseCadence,phaseMs:phaseMinMs,phaseMinMs,phaseStepMs,phaseMaxMs};
}
