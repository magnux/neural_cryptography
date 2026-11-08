/** Sparse stderr progress for long offline preparation loops. */
export function progress(label,total,{write=line=>console.error(line),intervalMs=5000}={}) {
  const started=Date.now();let last=started,previous=-1,previousDetail='';
  return (done,detail='',force=false)=>{
    const now=Date.now();
    if(!force&&done===previous&&detail===previousDetail)return;
    if(!force&&previous>=0&&done!==total&&now-last<intervalMs)return;
    const count=Number.isSafeInteger(total)?`${done}/${total}`:`${done}`;
    const percent=total>0?` (${Math.floor(100*done/total)}%)`:'';
    write(`${label}: ${count}${percent}${detail?` | ${detail}`:''} | ${((now-started)/1000).toFixed(1)}s`);
    last=now;previous=done;previousDetail=detail;
  };
}
