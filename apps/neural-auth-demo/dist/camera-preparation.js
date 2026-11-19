// One trained transport and one keyless crypto worker. Start joins the same
// promise as page load; neither graph bytes nor compiled sessions cross workers.
import {CorrectorClient} from './js/corrector-lab/client.js';
import {CryptoClient} from './js/crypto-lab/client.js';

export class CameraPreparation {
  #entry=null;
  constructor({makeCorrector=options=>new CorrectorClient(options),makeCrypto=options=>new CryptoClient(options),onState=()=>{},onLost=()=>{}}={}) {
    Object.assign(this,{makeCorrector,makeCrypto,onState,onLost});
  }
  clear() {
    const previous=this.#entry;this.#entry=null;
    previous?.corrector?.cancel();previous?.cryptoClient?.destroy();
    this.onState('idle');
  }
  prepare({size,backend}) {
    const current=this.#entry;
    if(current&&current.size===size&&current.backend===backend)return current.promise;
    this.clear();
    const entry=this.#entry={size,backend};
    const discard=error=>{
      if(this.#entry!==entry)return;
      this.clear();this.onState('error',error?.message||String(error));
    };
    const lost=error=>{
      if(this.#entry!==entry)return;
      discard(error);this.onLost(error);
    };
    this.onState('loading');
    entry.promise=(async()=>{
      try{
        entry.corrector=this.makeCorrector({onInvalidate:lost,onProgress:message=>{
          if(this.#entry===entry)this.onState('loading',message);
        }});
        entry.cryptoClient=this.makeCrypto({onInvalidate:lost});
        const [model,crypto]=await Promise.all([
          entry.corrector.call('transport-init',{size,backend,statistics:false}),
          entry.cryptoClient.request('selfTest',{}), // loads/executes the proof engine on a public fixture; creates no key
        ]);
        if(this.#entry!==entry)throw new DOMException('Preparation superseded.','AbortError');
        if(!model?.ready||!model.offlineReady||model.modelStatus!=='trained'||!model.parity?.passed||!crypto?.passed)
          throw Error('This configuration did not pass the on-device readiness checks.');
        this.onState('ready');
        return {corrector:entry.corrector,cryptoClient:entry.cryptoClient,model};
      }catch(e){discard(e);throw e;}
    })();
    return entry.promise;
  }
}
