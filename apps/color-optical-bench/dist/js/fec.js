/** Shortened systematic Reed–Solomon over GF(256), primitive polynomial 0x11d.
 * Generator roots alpha^0 .. alpha^(parity-1), highest-degree byte first.
 * Errors and erasures: remove known locations from the syndrome sequence,
 * Berlekamp–Massey, Chien search, then a small GF linear solve.
 * The transport always checks CRC after decoding, including a zero syndrome.
 */
const exp=new Uint8Array(512),log=new Uint8Array(256);
let x=1;for(let i=0;i<255;i++){exp[i]=x;log[x]=i;x<<=1;if(x&256)x^=0x11d;}
for(let i=255;i<512;i++)exp[i]=exp[i-255];
const mul=(a,b)=>a&&b?exp[log[a]+log[b]]:0;
const div=(a,b)=>{if(!b)throw Error('Singular RS system.');return a?exp[(log[a]-log[b]+255)%255]:0;};
const evaluate=(p,x)=>p.reduce((s,c)=>mul(s,x)^c,0);
const generators=new Map();
function generator(parity){
  if(generators.has(parity))return generators.get(parity);
  let g=Uint8Array.of(1);
  for(let i=0;i<parity;i++){const next=new Uint8Array(g.length+1);for(let j=0;j<g.length;j++){next[j]^=g[j];next[j+1]^=mul(g[j],exp[i]);}g=next;}
  generators.set(parity,g);return g;
}
function validate(length,parity){if(!Number.isInteger(parity)||parity<2||parity>64||parity%2||length<=parity||length>255)throw Error('Invalid RS block.');}
export function rsEncode(data,parity){
  validate(data.length+parity,parity);
  const work=new Uint8Array(data.length+parity);work.set(data);const g=generator(parity);
  for(let i=0;i<data.length;i++){const lead=work[i];if(lead)for(let j=1;j<g.length;j++)work[i+j]^=mul(g[j],lead);}
  work.set(data);return work;
}
export function rsDecode(input,parity,erasures=[]){
  validate(input.length,parity);
  if(!Array.isArray(erasures)&&!(ArrayBuffer.isView(erasures)&&typeof erasures.length==='number'))throw Error('Invalid RS erasure positions.');
  const erased=Array.from(erasures),known=new Set(erased);
  if(erased.length>parity||known.size!==erased.length||erased.some(p=>!Number.isInteger(p)||p<0||p>=input.length))throw Error('Invalid RS erasure positions.');
  const bytes=input.slice(),syndromes=Uint8Array.from({length:parity},(_,j)=>evaluate(bytes,exp[j]));
  if(syndromes.every(v=>v===0))return {bytes,corrected:0};
  // S_j = sum(E_i * X_i^j). For each known X, the recurrence
  // S'j = S_(j+1) + X*S_j removes its contribution without knowing E_i.
  // The remaining p-s terms locate at most floor((p-s)/2) unknown errors.
  let reduced=syndromes.slice();
  for(const p of erased){
    const location=exp[bytes.length-1-p],next=new Uint8Array(reduced.length-1);
    for(let j=0;j<next.length;j++)next[j]=reduced[j+1]^mul(location,reduced[j]);
    reduced=next;
  }
  let c=new Uint8Array(reduced.length+1),b=new Uint8Array(reduced.length+1),degree=0,shift=1,last=1;c[0]=b[0]=1;
  for(let n=0;n<reduced.length;n++){
    let discrepancy=reduced[n];for(let i=1;i<=degree;i++)discrepancy^=mul(c[i],reduced[n-i]);
    if(!discrepancy){shift++;continue;}
    const old=c.slice(),scale=div(discrepancy,last);
    for(let i=0;i+shift<c.length;i++)c[i+shift]^=mul(scale,b[i]);
    if(2*degree<=n){degree=n+1-degree;b=old;last=discrepancy;shift=1;}else shift++;
  }
  if(2*degree+erased.length>parity)throw Error('RS correction limit exceeded.');
  const positions=erased.slice(),locations=erased.map(p=>exp[bytes.length-1-p]);
  for(let i=0;i<bytes.length;i++){
    if(known.has(i))continue;
    const power=bytes.length-1-i,z=exp[(255-power)%255];let value=0;
    for(let j=degree;j>=0;j--)value=mul(value,z)^c[j];
    if(!value){positions.push(i);locations.push(exp[power]);}
  }
  if(positions.length!==degree+erased.length)throw Error('RS error locations are inconsistent.');
  const count=positions.length;
  if(!count)throw Error('RS correction limit exceeded.');
  const rows=Array.from({length:count},(_,r)=>{
    const row=new Uint8Array(count+1);for(let k=0;k<count;k++)row[k]=r?exp[(log[locations[k]]*r)%255]:1;row[count]=syndromes[r];return row;
  });
  for(let col=0;col<count;col++){
    let pivot=col;while(pivot<count&&!rows[pivot][col])pivot++;
    if(pivot===count)throw Error('Singular RS errors.');[rows[col],rows[pivot]]=[rows[pivot],rows[col]];
    const lead=rows[col][col];for(let j=col;j<=count;j++)rows[col][j]=div(rows[col][j],lead);
    for(let r=0;r<count;r++)if(r!==col){const scale=rows[r][col];for(let j=col;j<=count;j++)rows[r][j]^=mul(scale,rows[col][j]);}
  }
  let corrected=0;
  positions.forEach((p,i)=>{const magnitude=rows[i][count];if(magnitude){bytes[p]^=magnitude;corrected++;}});
  if(Array.from({length:parity},(_,j)=>evaluate(bytes,exp[j])).some(Boolean))throw Error('RS verification failed.');
  return {bytes,corrected};
}
