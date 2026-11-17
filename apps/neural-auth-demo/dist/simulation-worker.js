import {runSimulationComparison} from './simulation-core.js';
import {createSimulationCorrector} from './simulation-runtime.js';
let busy=false;
self.onmessage=async({data})=>{
  if(busy)return;busy=true;let corrector;
  const onProgress=message=>postMessage({type:'progress',message});
  try{
    corrector=await createSimulationCorrector({onProgress,statistics:data.statistics});
    const report=await runSimulationComparison({...data,corrector,onProgress});
    postMessage({type:'result',report});
  }catch(error){postMessage({type:'error',message:error.message});}
  finally{await corrector?.close();busy=false;}
};
