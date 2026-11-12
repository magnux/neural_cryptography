/** Bounded detection passes. Yielding keeps all fallbacks lazy: a clean frame
 * pays for only the existing coarse global search. These grayscale operations
 * never modify the camera RGB image used by phase calibration or the model.
 */
import {findMarkers,quadProposals} from '../vision.js';

const PASSES=Object.freeze([
  Object.freeze({mode:'global',thresholdMode:'global',maxSide:1000}),
  Object.freeze({mode:'adaptive',thresholdMode:'adaptive',maxSide:1000}),
  Object.freeze({mode:'adaptive-hires',thresholdMode:'adaptive',maxSide:1600}),
]);

export function* markerPasses(image){
  let candidates=[];
  for(const pass of PASSES){
    if(pass.mode==='adaptive-hires'&&Math.max(image.width,image.height)<=1000)continue;
    const detected=findMarkers(image,pass),merged=detected.markers.map(marker=>({...marker}));
    // Keep earlier candidates that this threshold missed. Prefer the current
    // pass for coincident centers, since adaptive passes refine in a local ROI.
    for(const marker of candidates)if(!merged.some(other=>Math.hypot(other.x-marker.x,other.y-marker.y)<Math.max(other.module,marker.module)*1.5))merged.push(marker);
    candidates=merged.sort((a,b)=>b.module*Math.sqrt(b.hits)-a.module*Math.sqrt(a.hits)).slice(0,16);
    const proposals=quadProposals(candidates,image.width,image.height,{minAreaFraction:.001});
    yield {markers:candidates,proposals,mode:pass.mode,stats:{detectionMode:pass.mode,
      detectionMaxSide:pass.maxSide,detectionWidth:detected.detectionWidth,detectionHeight:detected.detectionHeight,
      detectionThreshold:detected.threshold,quadCount:proposals.length}};
  }
}
