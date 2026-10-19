import {runSimulation} from './simulation.js';
let busy=false;
self.onmessage=async({data})=>{
  if(data?.type!=='run')return;
  const {id}=data;
  if(busy){self.postMessage({type:'error',id,error:'A transport simulation is already running.'});return;}
  busy=true;
  try{
    const report=await runSimulation(data.options??{},progress=>self.postMessage({type:'progress',id,progress}));
    self.postMessage({type:'result',id,report});
  }catch(error){self.postMessage({type:'error',id,error:error?.message??String(error)});}
  finally{busy=false;}
};
