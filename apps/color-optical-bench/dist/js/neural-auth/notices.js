/** Short handoff notices observe the protocol; they never advance it. */
export const NOTICE_DURATION_MS=3000;
const notice=(title,message,tone='action')=>({title,message,tone});

export class TurnNotices {
  // Browser timers require the global receiver; do not store them as bare methods.
  constructor({onChange,setTimer=(callback,delay)=>globalThis.setTimeout(callback,delay),
    clearTimer=id=>globalThis.clearTimeout(id)}={}){
    this.onChange=onChange;this.setTimer=setTimer;this.clearTimer=clearTimer;
    this.previous={};this.hasSentRound=false;this.timer=null;this.revision=0;
  }
  reset(){
    this.previous={};this.hasSentRound=false;this.dismiss();
  }
  dismiss(){
    this.revision++;
    if(this.timer!==null)this.clearTimer(this.timer);this.timer=null;this.onChange(null);
  }
  show(value){
    if(this.timer!==null)this.clearTimer(this.timer);
    const revision=++this.revision;this.onChange(value);
    this.timer=this.setTimer(()=>{
      if(revision!==this.revision)return;
      this.timer=null;this.onChange(null);
    },NOTICE_DURATION_MS);
  }
  update(auth={},link={}){
    auth??={};link??={};
    const prior=this.previous,current={state:auth.state,result:auth.result,peerReported:auth.peerReported,
      resultDelivered:auth.resultDelivered,mode:link.mode,outgoing:link.outgoing?.messageId,
      outgoingIndex:link.outgoing?.index,incoming:link.incoming?.messageId,
      incomingChunks:link.incoming?.receivedChunks??0,
      received:link.receivedMessages??0,canSend:link.canSendNext,preparing:link.preparing};
    const delivered=Boolean(prior.outgoing&&!current.outgoing&&link.lastTransfer?.state==='delivered'&&
      link.lastTransfer.messageId===prior.outgoing);
    if(delivered)this.hasSentRound=true;
    current.pairing=Boolean(auth.canConfirm&&this.hasSentRound&&!current.outgoing&&!current.canSend&&!current.preparing);
    this.previous=current;
    // Only transport progress identifies fresh traffic. Repeated captures and
    // duplicate packets must not replace a notice or extend its lifetime.
    const ackAdvanced=Boolean(current.outgoing&&current.outgoing===prior.outgoing&&
      Number.isInteger(prior.outgoingIndex)&&current.outgoingIndex>prior.outgoingIndex);
    const incomingStarted=Boolean(current.incoming&&current.incoming!==prior.incoming);
    const chunkReceived=Boolean(current.incoming&&current.incoming===prior.incoming&&
      current.incomingChunks>prior.incomingChunks);
    let value;
    if(current.resultDelivered&&!prior.resultDelivered){
      value=notice(auth.result?'Authentication complete':'Rejection delivered',
        'The other device acknowledged the result. You can save both reports.',auth.result?'success':'error');
    }else if(current.peerReported&&!prior.peerReported){
      value=notice(auth.result?'Verifier reports acceptance':'Verifier reports rejection',
        'Keep this final ACK visible until the verifier confirms delivery.',auth.result?'success':'error');
    }else if(current.canSend&&!prior.canSend){
      value=notice('Your next turn is ready',current.mode==='ack'
        ?'Keep this ACK visible until the other device advances, then tap “Send next turn”.'
        :'Tap “Send next turn” and show this screen to the other camera.');
    }else if(current.outgoing&&current.outgoing!==prior.outgoing){
      value=notice('Sending this round','A new message-start matrix is displayed. Show it to the other camera and wait for its ACK.','info');
    }else if(ackAdvanced){
      value=notice(`ACK received · sending chunk ${current.outgoingIndex+1}/${link.outgoing.chunkCount}`,
        'The next data matrix is displayed. Aim the other camera at this screen.','success');
    }else if(current.pairing&&!prior.pairing){
      value=notice('Compare the pairing codes',current.mode==='ack'
        ?'Let the other device read this ACK, then compare and confirm the codes on both screens.'
        :`${delivered?'ACK received; this round is delivered. ':''}Check both codes and confirm only if they match.`);
    }else if(delivered){
      value=notice('ACK received · round delivered','Look at the other device for its next action. This camera is ready for the reply.','success');
    }else if(current.received>(prior.received??0)){
      value=notice('Round received · show this ACK',
        `${link.incoming?.chunkCount?`Chunk ${current.incomingChunks}/${link.incoming.chunkCount} received. `:''}Let the other camera read this final ACK before switching turns.`);
    }else if(incomingStarted){
      value=notice('Message start received · ACK ready','Show this ACK to the sender, then aim back at its first data matrix.');
    }else if(chunkReceived){
      value=notice(`Chunk ${current.incomingChunks}/${link.incoming.chunkCount} received · ACK ready`,
        'A new ACK is displayed. Show it to the sender, then aim back at its next data matrix.');
    }else if(current.mode==='ack'&&prior.mode!=='ack'){
      value=notice('ACK ready · show this screen','Turn the phone so the sender’s camera can read this ACK.');
    }else if(current.state!==prior.state){
      if(current.state==='proving')value=notice('Generating the proof','Keep this ACK visible while the device prepares its reply.','info');
      else if(current.state==='verifying')value=notice('Checking the proof','Keep this ACK visible while verification runs.','info');
      else if(current.state==='verified'||current.state==='rejected')value=notice(
        current.state==='verified'?'Proof verified on this device':'Proof rejected on this device',
        'The result still needs to be sent to the other device.',current.state==='verified'?'success':'error');
      else if(current.state==='awaiting-hello')value=notice('Ready to receive','Aim this camera at the verifier’s matrix.','info');
    }
    if(value)this.show(value);
  }
}
