/** Repeatable test content only; never use this generator for protocol secrets. */
export function seededRandom(seed){
  let state=seed>>>0;
  return ()=>{state=(state+0x6d2b79f5)>>>0;let n=state;n=Math.imul(n^(n>>>15),n|1);n^=n+Math.imul(n^(n>>>7),n|61);return ((n^(n>>>14))>>>0)/4294967296;};
}
export function diagnosticPayload(length=165000,seed=1234){
  if(!Number.isSafeInteger(length)||length<0||length>8*1024*1024)throw Error('Payload bytes must be 0..8388608.');
  if(!Number.isSafeInteger(seed)||seed<0||seed>0xffffffff)throw Error('Seed must be 0..4294967295.');
  const random=seededRandom(seed^0xa511e9b3),bytes=new Uint8Array(length);
  for(let i=0;i<length;i++)bytes[i]=Math.floor(random()*256);
  return bytes;
}
