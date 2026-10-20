import {extractDensePayload} from './vision.js';
import {gridLayout,paletteMap} from './grid.js';

/** M1 is deliberately bounded to the two exported single-frame models. */
export const LEARNED_INPUT = Object.freeze({sizes:Object.freeze([64,128]),colors:Object.freeze([4,16]),
  pixelsPerCell:4,channels:3,classes:16});

/** Interleaved calibrated camera RGB -> contiguous float32 CHW.
 * RGB values are in the nominal 0..255 range, but color correction can overshoot.
 * This exact affine normalization/clipping is also used by the Python contract.
 */
export function normalizeCalibratedRGB(rgb,width,height){
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>512||height>512||
     !(rgb instanceof Float32Array)||rgb.length!==width*height*3)throw Error('Invalid calibrated RGB raster.');
  for(const value of rgb)if(!Number.isFinite(value))throw Error('Non-finite calibrated RGB value.');
  const pixels=width*height,input=new Float32Array(3*pixels);
  for(let i=0;i<pixels;i++)for(let channel=0;channel<3;channel++){
    input[channel*pixels+i]=Math.max(-1.5,Math.min(1.5,rgb[3*i+channel]/127.5-1));
  }
  return input;
}

/** Read a header-validated dense payload; no diagnostic labels enter this path.
 * Returns {ok, info, input, dims, stats, colorTransform}; failures have no input.
 * An optional extractCapture result saves finder detection but is revalidated.
 */
export function extractLearnedInput(image,capture=null){
  const start=performance.now(),raster=extractDensePayload(image,capture);
  if(!raster.ok)return {...raster,stats:{...raster.stats,preprocessMs:performance.now()-start}};
  try{
    const input=normalizeCalibratedRGB(raster.rgb,raster.width,raster.height);
    return {ok:true,info:raster.info,input,dims:[1,3,raster.height,raster.width],
      colorTransform:raster.colorTransform,
      stats:{...raster.stats,inputBytes:input.byteLength,preprocessMs:performance.now()-start}};
  }catch(error){return {ok:false,stage:'normalize',error:error.message,
    stats:{...raster.stats,preprocessMs:performance.now()-start}};}
}

function validateProfile(info){
  if(!info||!LEARNED_INPUT.sizes.includes(info.size)||!LEARNED_INPUT.colors.includes(info.colors)||
     ![0,4,8,16].includes(info.gridBlock)||![1,2,4,8,16].includes(info.phases)||
     !Number.isInteger(info.phase)||info.phase<0||info.phase>=info.phases||typeof info.permute!=='boolean'){
    throw Error('Unsupported learned score profile.');
  }
}

/** Physical NCHW logits -> logical data-cell scores accepted by decodeScores.
 * Reference cells are omitted. Inactive palette classes are masked before
 * softmax; only then is the current public palette permutation undone.
 */
export function logitsToLogicalScores(logits,info,{temperature=1}={}){
  validateProfile(info);
  const cells=info.size*info.size;
  if(!(logits instanceof Float32Array)||logits.length!==16*cells)throw Error('Invalid learned logits shape.');
  if(!Number.isFinite(temperature)||temperature<=0)throw Error('Invalid learned confidence temperature.');
  for(const value of logits)if(!Number.isFinite(value))throw Error('Non-finite learned logits.');
  const layout=gridLayout(info.size,info.colors,info.gridBlock),scores=new Float32Array(layout.cells.length*info.colors);
  const weights=new Float64Array(info.colors);
  for(let i=0;i<layout.cells.length;i++){
    const cell=layout.cells[i];let maximum=-Infinity,sum=0;
    for(let physical=0;physical<info.colors;physical++)maximum=Math.max(maximum,logits[physical*cells+cell]);
    for(let physical=0;physical<info.colors;physical++){
      weights[physical]=Math.exp((logits[physical*cells+cell]-maximum)/temperature);sum+=weights[physical];
    }
    for(let logical=0;logical<info.colors;logical++){
      scores[i*info.colors+logical]=255*weights[paletteMap(logical,info)]/sum;
    }
  }
  return scores;
}
