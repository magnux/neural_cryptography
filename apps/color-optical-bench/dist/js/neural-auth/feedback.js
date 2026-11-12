/** Acquisition feedback is distinct from packet integrity and peer acceptance. */
export function captureFeedback(result){
  const acknowledged=result.expected==='ack',kind=acknowledged?'ACK':'Data';
  const candidates=result.markers?.length??0;
  let label;
  if(result.ok)label=`${kind} RS+CRC passed · checking transfer`;
  else if(result.stage==='profile')label=result.error||'Neural-QR grid exceeds the receive limit';
  else if(!result.info){
    const stages={quad:'Marker arrangement uncertain',contrast:'Finder contrast insufficient',
      orientation:'Orientation uncertain',size:'Grid-size cues uncertain',phase:'Phase cues uncertain'};
    const reason=typeof result.error==='string'?result.error.slice(0,240):'';
    label=stages[result.stage]?`${stages[result.stage]}${reason?` · ${reason}`:''}`:
      `Looking for Neural-QR ${acknowledged?'ACK':''} · ${candidates} marker candidates${reason?` · ${reason}`:''}`;
  }else if(result.error&&result.method!=='neural-corrector')label=`G${result.info.size} acquired · ${result.error}`;
  else if(result.method==='neural-corrector')label=`G${result.info.size} acquired · collecting more samples for packet recovery`;
  else label=`G${result.info.size} · phases ${(result.phaseCounts??[0,0,0,0]).join(' / ')} · ${result.collected??0} captures`;
  const validated=Boolean(result.info)&&result.stage!=='profile';
  return {label,validated,corners:result.corners?.length===4?result.corners:result.tentativeCorners??[],markers:result.markers??[]};
}
