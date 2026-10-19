import {decodeText} from './content.js';

export function hexPreview(bytes){
  const lines=[],limit=Math.min(256,bytes.length);
  for(let i=0;i<limit;i+=16)lines.push(`${i.toString(16).padStart(4,'0')}  ${Array.from(bytes.subarray(i,Math.min(i+16,limit)),b=>b.toString(16).padStart(2,'0')).join(' ')}`);
  if(bytes.length>limit)lines.push(`… ${(bytes.length-limit).toLocaleString()} more bytes`);
  return lines.join('\n')||'(empty content)';
}
export function downloadBlob(blob,filename){
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),5000);
}

/** Identical content display for the live receiver and the independent image lab. */
export class ContentView {
  constructor(root){
    root.innerHTML=`<div class="content-heading"><div><span class="content-type" data-node="kind">NO CONTENT YET</span><span data-node="size"></span></div><div class="content-actions"><button class="text-button" data-node="copy" hidden>Copy text</button><button class="text-button" data-node="asText" hidden>View as UTF-8</button><button class="text-button" data-node="download" disabled>Save content ↓</button></div></div>
      <p class="content-empty" data-node="empty">The decoded content appears here after every frame has arrived.</p>
      <textarea class="decoded-text" data-node="text" readonly aria-label="Decoded UTF-8 text" hidden></textarea>
      <img class="decoded-image" data-node="image" alt="Received image" hidden>
      <audio class="decoded-audio" data-node="audio" controls preload="metadata" hidden aria-label="Received audio"></audio>
      <pre class="decoded-bytes" data-node="raw" hidden></pre>
      <p class="content-note" data-node="note" aria-live="polite"></p>
      <details class="byte-details" data-node="details" hidden><summary>Inspect content bytes</summary><pre data-node="hex"></pre></details>`;
    this.nodes=Object.fromEntries([...root.querySelectorAll('[data-node]')].map(el=>[el.dataset.node,el]));
    this.url=null;this.content=null;this.filename='received.bin';
    this.nodes.download.onclick=()=>{if(this.content)downloadBlob(new Blob([this.content.bytes],{type:this.content.mime}),this.filename);};
    this.nodes.copy.onclick=async()=>{
      try{if(!navigator.clipboard?.writeText)throw Error();await navigator.clipboard.writeText(this.nodes.text.value);this.nodes.note.textContent='Text copied.';}
      catch{this.nodes.text.focus();this.nodes.text.select();this.nodes.note.textContent='Text selected. Use your device’s Copy command.';}
    };
    this.nodes.asText.onclick=()=>{
      try{this.showText(decodeText(this.content.bytes));this.nodes.note.textContent='Interpreted as UTF-8; the transmitted type remains raw bytes.';}
      catch{this.nodes.note.textContent='These bytes are not valid UTF-8.';}
    };
  }
  clear(){
    const n=this.nodes;n.audio.onerror=null;n.image.onerror=null;n.audio.pause();n.audio.removeAttribute('src');n.audio.load();n.image.removeAttribute('src');
    if(this.url)URL.revokeObjectURL(this.url);this.url=null;this.content=null;
    for(const key of ['text','image','audio','raw','copy','asText','details'])n[key].hidden=true;
    n.empty.hidden=false;n.kind.textContent='NO CONTENT YET';n.size.textContent='';n.note.textContent='';n.text.value='';
    n.download.disabled=true;n.download.textContent='Save content ↓';n.details.open=false;
  }
  showText(text){this.nodes.text.value=text;this.nodes.text.hidden=false;this.nodes.copy.hidden=false;this.nodes.raw.hidden=true;this.nodes.empty.hidden=true;}
  show(content,sequenceId){
    this.clear();this.content=content;this.filename=`received-${sequenceId.slice(0,8)}.${content.extension}`;
    const n=this.nodes;n.empty.hidden=true;n.kind.textContent=content.label.toUpperCase();n.size.textContent=`${content.bytes.length.toLocaleString()} bytes`;
    n.download.disabled=false;n.download.textContent=`Save .${content.extension} ↓`;n.hex.textContent=hexPreview(content.bytes);n.details.hidden=false;
    if(content.type==='text')this.showText(content.text);
    else if(content.type==='png'||content.type==='jpeg'){
      this.url=URL.createObjectURL(new Blob([content.bytes],{type:content.mime}));n.image.hidden=false;
      n.image.onerror=()=>{n.note.textContent='Bytes arrived intact, but the browser could not display this image. You can still download it.';};n.image.src=this.url;
    }else if(content.type==='wav'){
      this.url=URL.createObjectURL(new Blob([content.bytes],{type:content.mime}));n.audio.hidden=false;
      n.audio.onerror=()=>{n.note.textContent='Bytes arrived intact, but this browser could not play the audio. You can still download the WAV.';};n.audio.src=this.url;
      n.note.textContent=`${content.audio.channels===1?'Mono':'Stereo'} · ${content.audio.sampleRate.toLocaleString()} Hz · ${content.audio.bits}-bit PCM · ${content.audio.duration.toFixed(2)} s. Press Play to listen.`;
    }else{n.raw.textContent=hexPreview(content.bytes);n.raw.hidden=false;n.asText.hidden=false;}
  }
  error(message,bytes,sequenceId){
    this.show({type:'raw',label:'Unsupported content',mime:'application/octet-stream',extension:'bin',bytes},sequenceId);
    this.nodes.note.textContent=message;this.nodes.asText.hidden=true;this.nodes.download.textContent='Save received bytes ↓';
  }
}
