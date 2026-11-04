/** Keep acquisition feedback distinct from payload and transport acceptance. */
export function captureFeedback(result){
  const qr=result.expected==='ack';
  let label;
  if(qr)label=result.ok?'QR ACK read · checking transfer':result.error||'Looking for a QR ACK';
  else if(!result.info)label=result.stage==='header'
    ?'Markers found · header unreadable'
    :`Looking for matrix · ${result.markers?.length??0} marker candidates`;
  else if(result.stage==='profile')label='Header read · waiting for the expected data matrix';
  else if(result.error&&result.method!=='neural-corrector')label=`Header read · ${result.error}`;
  else if(result.ok)label='Data CRC passed · checking transfer';
  else if(result.method==='neural-corrector')label='Header read · collecting more samples for payload recovery';
  else label=`Header read · phases ${(result.phaseCounts??[0,0,0,0]).join(' / ')} · ${result.collected??0} captures`;
  const validated=Boolean(result.info)||(qr&&result.ok);
  return {label,validated,corners:result.corners?.length===4?result.corners:result.tentativeCorners??[],markers:result.markers??[]};
}
