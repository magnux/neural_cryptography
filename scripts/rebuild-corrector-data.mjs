#!/usr/bin/env node
/** Guard the former fixed-G64 rebuild during the Neural-QR collection transition.
 * Existing derived OBM5 datasets remain usable by their independent training
 * tools. No recording, dataset, checkpoint or lock is changed by this command.
 */
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

export const MANAGED_OUTPUTS=Object.freeze(['real-train','real-validation','target-clean-train',
  'target-clean-validation','target-synthetic-train','target-synthetic-validation','session-groups.json']);
const NEXT=`Neural-QR multires synthetic/training integration is not yet implemented. Nothing was deleted or rewritten.
Join current uploads without altering recordings:
  node scripts/corrector-dataset.mjs join-uploads --recordings-root local-data/recordings --out local-data/multires-joined
For downloaded Neural-QR recordings, use corrector-dataset.mjs join --senders FILE[,FILE] --captures FILE[,FILE] --split train --out NEW_DIRECTORY.
Existing prepared OBM5 datasets can still be used with their established training scripts. Earlier recording formats are not accepted by the current collector/join.`;

export async function planRebuild(){throw Error(NEXT);}
export async function rebuild(){throw Error(NEXT);}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  if(process.argv.slice(2).includes('--help'))console.log(NEXT);
  else rebuild().catch(error=>{console.error(`Corrector rebuild unavailable: ${error.message}`);process.exitCode=1;});
}
