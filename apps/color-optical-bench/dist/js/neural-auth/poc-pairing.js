/** This demo deliberately skips human comparison; the shared protocol stays explicit. */
export class POCAutoPairing {
  constructor({isActive=()=>true,onError=()=>{}}={}){
    this.isActive=isActive;this.onError=onError;this.requested=false;this.closed=false;this.pending=null;
  }
  update(session,state){
    if(this.closed||this.requested||!state?.canConfirm||!this.isActive())return this.pending;
    this.requested=true;
    // Avoid re-entering AuthSession while it publishes the pairing snapshot.
    this.pending=Promise.resolve().then(()=>{
      if(this.closed||!this.isActive())return false;
      return session.confirmPairing();
    }).catch(error=>{
      if(!this.closed&&this.isActive())this.onError(error);
      return false;
    });
    return this.pending;
  }
  close(){this.closed=true;}
}
