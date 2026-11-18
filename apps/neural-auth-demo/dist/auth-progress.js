// Logical messages in optical order; ACKs, REQUESTs, chunks and palette phases
// are part of delivering a message, never extra authentication steps.
export const STEPS = Object.freeze(['Hello', 'Identity', 'Challenge', 'Proof', 'Result']);

export function exchangeStep(role, link) {
  if (!['verifier', 'prover'].includes(role)) throw Error('Unknown authentication role.');
  // A verifier sends the odd steps (1,3,5); a prover sends the even ones
  // (2,4). Each completed incoming message moves its next send ahead by two.
  const nextSend = (role === 'verifier' ? 1 : 0) + 2 * (link?.receivedMessages ?? 0);
  let current = nextSend;
  if (link?.waitingForRequest) {
    // Still displaying the final ACK. Even a fully prepared reply is not a
    // new turn until the other side has acknowledged delivery with a REQUEST.
    current--;
  } else if (link?.requestedReply || ['receiving', 'verifying'].includes(link?.incoming?.state)) {
    // After the sender reads the final ACK, it requests and waits for the
    // next incoming turn, staying on that step as its chunks arrive.
    current++;
  }
  const number = Math.max(1, Math.min(STEPS.length, current));
  return {number, total: STEPS.length, label: STEPS[number - 1]};
}
