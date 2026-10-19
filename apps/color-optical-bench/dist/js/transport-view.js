/** UI only. Error measurements are produced after classification in the worker. */
export class TransportView {
  constructor(container){
    container.innerHTML=`<div class="panel-heading"><div><span class="section-number">LINK</span><h2>Transport comparison</h2></div><span class="mode-label">SAME CAPTURES · BEFORE / AFTER FEC</span></div>
      <p class="cal-progress" aria-live="polite"></p><p class="cal-detail"></p>
      <div class="comparison-scroll"><table><thead><tr><th>Decoder</th><th>Raw CRC / attempts</th><th>Ordinary RS / attempts</th><th>Extra erasure passes</th><th>Raw cell errors</th><th>Corrected / erased bytes</th><th>First verified</th><th>Verified frames</th></tr></thead><tbody></tbody></table></div>
      <div class="error-maps"><figure><figcaption>Border evidence · raw errors</figcaption><canvas width="256" height="256" aria-label="Border decoder error map"></canvas></figure><figure><figcaption>Local evidence · raw errors</figcaption><canvas width="256" height="256" aria-label="Local decoder error map"></canvas></figure><p>Diagnostic only. Green = correct; red = raw errors; gray = no data. Maps show the latest robust estimate when available, otherwise ordinary fusion. Table totals include overlapping attempts. Erasure passes are additional recoveries after ordinary RS failed; erased-byte totals count chosen uncertain positions, which may have been correct.</p></div>`;
    this.root=container;this.clear();
  }
  clear(){this.render({methods:{}});}
  render(report){
    const p=report.profile,last=report.history?.at(-1),t=report.temporal??last?.temporal,counts=t?.counts??last?.phaseSamples;
    this.root.querySelector('.cal-progress').textContent=p?`${p.size} × ${p.size} · ${p.colors} colors · ${p.gridBlock?`${p.gridBlock}-cell reference blocks`:'border only'} · RS ${p.parity||'off'} · ${p.phases} ${p.permute?'permuted':'repeated'} phase(s)${p.holdMs?` · ${(p.holdMs/1000).toFixed(2)} s/phase`:''}`:'References are measured in every capture. Evidence keeps refreshing until content verifies.';
    const limits=t?` ${t.eligiblePhases.length}/${t.minPhases} sufficiently sampled phases; ${t.target} samples from ${t.poolCapacity} slots per phase. ${t.memoryLimited?`Memory limit: requested ${t.requested} samples / ${t.requestedPool} slots; retaining at most ${t.retainedPhaseLimit}/${t.phases} phases. `:''}Retained evidence ${(t.memoryBytes/1048576).toFixed(1)} MiB; expiry ${(t.ageLimitMs/1000).toFixed(0)} s.`:'';
    this.root.querySelector('.cal-detail').textContent=counts?`Samples by phase: ${counts.join(' / ')}.${limits} CRC is mandatory for every accepted candidate.`:'Choose Diagnostic content to measure actual errors, including failed frames. Ordinary RS includes raw-valid captures; extra erasure passes count only additional recoveries.';
    const tbody=this.root.querySelector('tbody');tbody.replaceChildren();
    for(const [key,label] of [['global','Border · one exposure'],['local','Local · one exposure'],['fusedGlobal','Border · ordinary fusion'],['fusedLocal','Local · ordinary fusion'],['robustGlobal','Border · robust subset'],['robustLocal','Local · robust subset'],['accepted','Accepted · CRC selection']]){
      const row=report.methods[key],tr=document.createElement('tr');
      for(const value of [label,row?`${row.rawPasses} / ${row.attempts}`:'—',row?`${row.hardPasses??0} / ${row.attempts}`:'—',row?.erasurePasses??'—',row?.cells?`${row.cellErrors} / ${row.cells} (${(100*row.symbolErrorRate).toFixed(3)}%)`:'Diagnostic needed',row?`${row.correctedBytes} / ${row.erasedBytes??0}`:'—',row?.firstVerifiedMs!=null?`${(row.firstVerifiedMs/1000).toFixed(1)} s · #${row.firstVerifiedCapture}`:'—',row?.frames?.length??0]){const td=document.createElement('td');td.textContent=value;tr.append(td);}tbody.append(tr);
    }
    const canvases=this.root.querySelectorAll('canvas');this.paint(canvases[0],(report.methods.robustGlobal??report.methods.fusedGlobal)?.last?.errors?.heatmap);this.paint(canvases[1],(report.methods.robustLocal??report.methods.fusedLocal)?.last?.errors?.heatmap);
  }
  paint(canvas,map){
    const ctx=canvas.getContext('2d');ctx.fillStyle='#e1e4df';ctx.fillRect(0,0,canvas.width,canvas.height);if(!map)return;
    const step=canvas.width/map.side;
    for(let i=0;i<map.errors.length;i++){
      const rate=map.totals[i]?map.errors[i]/map.totals[i]:0;
      ctx.fillStyle=!map.totals[i]?'#e1e4df':rate?`rgb(${Math.round(210+40*rate)},${Math.round(170*(1-rate))},${Math.round(145*(1-rate))})`:'#cfe6d8';
      ctx.fillRect((i%map.side)*step,Math.floor(i/map.side)*step,Math.ceil(step),Math.ceil(step));
    }
  }
}
