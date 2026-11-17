/** Bounded aggregate timings. Disabled collection never reads the clock. */
export class StageProfiler {
  constructor(enabled=false,{now=()=>performance.now(),maxStages=32}={}) {
    this.enabled=enabled===true;this.now=now;this.maxStages=maxStages;this.rows=new Map();
  }
  record(name,ms) {
    if(!this.enabled||typeof name!=='string'||!Number.isFinite(ms)||ms<0)return;
    let row=this.rows.get(name);
    if(!row){if(this.rows.size>=this.maxStages)return;row={count:0,totalMs:0,minMs:Infinity,maxMs:0};this.rows.set(name,row);}
    row.count++;row.totalMs+=ms;row.minMs=Math.min(row.minMs,ms);row.maxMs=Math.max(row.maxMs,ms);
  }
  measure(name,fn) {
    if(!this.enabled)return fn();
    const start=this.now();
    try {
      const value=fn();
      if(value&&typeof value.then==='function')return Promise.resolve(value).finally(()=>this.record(name,this.now()-start));
      this.record(name,this.now()-start);return value;
    }catch(error){this.record(name,this.now()-start);throw error;}
  }
  snapshot(){return this.enabled?Object.fromEntries([...this.rows].map(([name,row])=>[name,{...row,avgMs:row.totalMs/row.count}])):undefined;}
  merge(timings){
    if(!this.enabled||!timings)return;
    for(const [name,row] of Object.entries(timings)){
      let target=this.rows.get(name);
      if(!target){if(this.rows.size>=this.maxStages)continue;this.rows.set(name,{count:row.count,totalMs:row.totalMs,minMs:row.minMs,maxMs:row.maxMs});continue;}
      target.count+=row.count;target.totalMs+=row.totalMs;target.minMs=Math.min(target.minMs,row.minMs);target.maxMs=Math.max(target.maxMs,row.maxMs);
    }
  }
}
