/** Acquisition feedback is distinct from packet integrity and peer acceptance. */
export function captureFeedback(result){
  const acknowledged=result.expected==='ack',kind=acknowledged?'ACK':'Data';
  const candidates=result.markers?.length??0;
  let label;
  if(result.ok)label=`${kind} RS+CRC passed · checking transfer`;
  else if(result.stage==='profile')label=result.error||'Neural-QR grid exceeds the receive limit';
  else if(!result.info){
    const cue=['orientation','size','phase','header'].includes(result.stage);
    label=cue?`Markers found · ${result.stage==='header'?'acquisition cues':result.stage} unreadable`:
      `Looking for Neural-QR ${acknowledged?'ACK':''} · ${candidates} marker candidates`;
  }else if(result.error&&result.method!=='neural-corrector')label=`G${result.info.size} acquired · ${result.error}`;
  else if(result.method==='neural-corrector')label=`G${result.info.size} acquired · collecting more samples for packet recovery`;
  else label=`G${result.info.size} · phases ${(result.phaseCounts??[0,0,0,0]).join(' / ')} · ${result.collected??0} captures`;
  const validated=Boolean(result.info)&&result.stage!=='profile';
  return {label,validated,corners:result.corners?.length===4?result.corners:result.tentativeCorners??[],markers:result.markers??[]};
}
