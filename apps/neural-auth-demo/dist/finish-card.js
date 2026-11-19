// Render the same compact object that the download contains. Display names
// come from the peer. Insert them as text, never as markup or filenames.
function duration(ms){
  if(!Number.isFinite(ms))return '—';
  if(ms<1000)return Math.round(ms)+' ms';
  if(ms<60000)return (ms/1000).toFixed(1)+' s';
  const sec=Math.round(ms/1000);return Math.floor(sec/60)+'m '+String(sec%60).padStart(2,'0')+'s';
}
export function createFinishCard({byId,createElement,urlAPI=URL}){
  let url=null;
  function reset(){
    byId('exchangeSummary').hidden=true;
    const link=byId('downloadSummary');link.hidden=true;
    if(url){link.removeAttribute('href');urlAPI.revokeObjectURL(url);url=null;}
  }
  function node(tag,text){const el=createElement(tag);el.textContent=text;return el;}
  function show(summary){
    reset();if(!summary)return;
    const {role,peerName,statisticsEnabled:s,settings}=summary;
    const outcomes={'verified-locally':'Proof verified locally','rejected-locally':'Proof rejected locally',
      'peer-reported-accepted':'Peer reported acceptance','peer-reported-rejected':'Peer reported rejection'};
    byId('summaryOutcome').textContent=outcomes[summary.outcome]+(peerName?' · '+peerName:'');
    byId('summaryDuration').textContent=summary.elapsedMs==null?'':duration(summary.elapsedMs);
    byId('summarySteps').replaceChildren(...summary.steps.map(step=>{
      const li=createElement('li');li.append(node('span',step.label),node('strong',duration(step.durationMs)));return li;
    }));
    const metrics=[];
    function metric(label,value){
      const row=createElement('div');row.append(node('dt',label),node('dd',value));metrics.push(row);
    }
    if(s&&summary.setupMs!=null)metric('Camera & model setup',duration(summary.setupMs));
    if(summary.repetitions!=null)metric('Proof',summary.repetitions+' repetitions'+(summary.proofBytes!=null?' · '+summary.proofBytes.toLocaleString()+' B':''));
    if(settings.ackSize)metric('Acknowledgments',settings.ackSize===16?'G16 · static digest':'G32 · four-phase corrector');
    metric('Trained input color',settings.localColorCorrection?'Border + local grid':'Border only');
    for(const row of summary.trainedRecovery??[]){
      if(row.expected!=='data')continue;
      metric('G'+row.grid+' DATA · CRC before / after RS',row.rawCRCPassed+' / '+row.rsCRCPassed+' of '+row.inferences+' inference attempts');
      if(row.acceptedWireCells>0)metric('G'+row.grid+' DATA · cell errors on accepted packets',
        (100*row.acceptedWireCellErrors/row.acceptedWireCells).toFixed(2)+'% · '+row.acceptedPacketsWithCER+' packets');
    }
    metric('Outgoing phases',settings.phaseCadence==='progressive'
      ?settings.phaseMs+' → '+settings.phaseMaxMs+' ms (+'+settings.phaseStepMs+'/pass)':settings.phaseMs+' ms fixed');
    metric('Camera search',settings.captureMs+' ms · '+(settings.firstCycle?'after first pass':'while sending'));
    metric('After lock',settings.phaseWatch?'Watch for phase changes':'Timed capture');
    if(s&&summary.packetRtt)metric('Packet RTT · min / mean',duration(summary.packetRtt.minMs)+' / '+duration(summary.packetRtt.meanMs));
    if(s&&summary.inference)metric('Mean inference',duration(summary.inference.meanMs));
    if(s&&summary.acceptedMatrices!=null)metric('Matrices accepted',String(summary.acceptedMatrices));
    byId('summaryMetrics').replaceChildren(...metrics);
    const note=byId('summaryNote');note.hidden=false;note.textContent=s
      ?'Local timings end '+(role==='verifier'?'at the final result acknowledgment.':'when the result is received.')
      :'Timing statistics were off for this exchange.';
    const link=byId('downloadSummary');
    url=urlAPI.createObjectURL(new Blob([JSON.stringify(summary,null,2)],{type:'application/json'}));
    link.href=url;link.download='neural-key-summary-'+role+'-'+summary.started.replace(/[:.]/g,'-')+'.json';
    link.hidden=false;byId('exchangeSummary').hidden=false;byId('summaryTitle').focus?.({preventScroll:true});
  }
  return {show,reset};
}
