// The screen/camera are doubles. Real browser controllers, auth messages, proof,
// chunking, turns, result and final ACK run in-process at the packet boundary.
export function browser(t,role,statistics) {
  const values={role,backend:'wasm',repetitions:'32',resolution:'720',outgoingSize:'64',incomingSize:'64',ackSize:'16',
    phaseCadence:'fixed',phaseMs:'350',phaseMinMs:'200',phaseStepMs:'50',phaseMaxMs:'500',captureMs:'250',camera:''};
  const elements=new Map(),listeners=new Map(),track={stop(){this.stopped=true;},addEventListener(){},getSettings(){return {};}};
  class Node {
    constructor(id){this.id=id;this.hidden=['exchangeSummary','downloadSummary'].includes(id);this.value=values[id]??'';
      this.checked=id==='statistics'&&statistics;this.style={getPropertyValue(){return '';},setProperty(){}};this.dataset={};this.children=[];}
    textContent='';setAttribute(k,v){this[k]=v;} removeAttribute(k){delete this[k];}
    addEventListener(k,fn){listeners.set(this.id+':'+k,fn);}click(){listeners.get(this.id+':click')?.();}
    replaceChildren(...children){this.children=children;}append(...children){this.children.push(...children);}appendChild(child){this.append(child);return child;}
    getContext(){return {clearRect(){},putImageData(){}};}focus(){} requestVideoFrameCallback(){return 1;} cancelVideoFrameCallback(){} play(){}
  }
  const get=id=>{if(!elements.has(id))elements.set(id,new Node(id));return elements.get(id);};
  const document={body:get('body'),documentElement:get('html'),getElementById:get,createElement:k=>new Node(k),addEventListener(){}};
  const globals={document,window:{innerHeight:800,addEventListener(){}},navigator:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>[track],getVideoTracks:()=>[track]})}},
    requestAnimationFrame:cb=>setTimeout(()=>cb(performance.now()),16),cancelAnimationFrame:id=>clearTimeout(id),
    isSecureContext:true,ImageData:class{constructor(data,width,height){Object.assign(this,{data,width,height});}},Worker:class{}};
  globals.window.requestAnimationFrame=globals.requestAnimationFrame;globals.window.cancelAnimationFrame=globals.cancelAnimationFrame;
  for(const [key,value] of Object.entries(globals)){
    const prior=Object.getOwnPropertyDescriptor(globalThis,key);
    Object.defineProperty(globalThis,key,{configurable:true,value});
    t.after(()=>prior?Object.defineProperty(globalThis,key,prior):delete globalThis[key]);
  }
  return {get,track};
}
