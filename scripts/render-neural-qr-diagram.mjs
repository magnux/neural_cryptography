#!/usr/bin/env node
/** Regenerate the documentation figure from the executable acquisition renderer. */
import {writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {createSender} from '../apps/color-optical-bench/dist/js/dataset/contract.js';
import {NQR_LAYOUT,renderNeuralQR} from '../apps/color-optical-bench/dist/js/neural-qr/layout.js';
import {PALETTE} from '../apps/color-optical-bench/dist/js/format.js';
import {gridLayout} from '../apps/color-optical-bench/dist/js/grid.js';

const {frames}=await createSender({size:32,seed:23,sessionId:'documentation-neural-qr-layout'});
const frame=frames[2], rendered=renderNeuralQR(frame,1024);
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const hex=rgb=>'#'+Array.from(rgb).map(v=>v.toString(16).padStart(2,'0')).join('');
const a=[];
const text=(x,y,s,size=18,weight=400,color='#334155')=>a.push(`<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" fill="${color}">${escape(s)}</text>`);
const rect=(x,y,w,h,fill,more='')=>a.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" ${more}/>`);
const label=(x,y,s,color)=>{rect(x-12,y-15,25,25,color,'rx="5"');text(x-6,y+4,s,17,700,'#fff');};
a.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1120" viewBox="0 0 1200 1120" role="img" aria-labelledby="title desc"><title id="title">Neural-QR version 2: implemented optical layout</title><desc id="desc">Actual G32 phase-2 collection rendering with four finders, asymmetric bottom-right satellites, duplicated binary size codes, paired color phase cues, fixed palette strips and grid-8 references. The 32 by 32 matrix is rectified to 128 by 128 pixels. Colored outlines and labels are explanatory overlays, not transmitted marks. Four phases repeat one logical payload; G16 carries 128 gross bytes before reference and packet overhead.</desc><g font-family="DejaVu Sans, Arial, sans-serif">`);
rect(0,0,1200,1120,'#f3f6fa');
text(40,52,'Neural-QR · version 2',32,700,'#172c48');
text(40,87,'Implemented collection layout · 16 colors · four palette phases · 4×4 camera pixels per cell',18);
rect(28,116,624,680,'#fff','rx="14" stroke="#d7e0ea"');
text(48,149,'G32 example → 128×128 RGB crop · phase 2',18,600);
// Lossless run-length vectorization of the real renderer, merging equal adjacent rows.
// No manually reconstructed markers or alternate rendering specification.
a.push('<g transform="translate(48 176) scale(.5625)" shape-rendering="crispEdges">');
let previous=null,rowStart=0;
function flush(row,end){if(!row)return;for(const [x,w,color] of row)rect(x,rowStart,w,end-rowStart,color);}
for(let y=0;y<=rendered.height;y++){
  let row=null;
  if(y<rendered.height){row=[];let x=0;while(x<rendered.width){const i=4*(y*rendered.width+x),color=hex(rendered.data.subarray(i,i+3));let end=x+1;
    while(end<rendered.width){const j=4*(y*rendered.width+end);if(rendered.data[j]!==rendered.data[i]||rendered.data[j+1]!==rendered.data[i+1]||rendered.data[j+2]!==rendered.data[i+2])break;end++;}
    row.push([x,end-x,color]);x=end;}}
  if(JSON.stringify(row)!==JSON.stringify(previous)){flush(previous,y);previous=row;rowStart=y;}
}
a.push('</g>');
const ox=48,oy=176,s=2.25;
function outline(x,y,w,h,c){rect(ox+x*s,oy+y*s,w*s,h*s,'none',`stroke="${c}" stroke-width="2" stroke-dasharray="5 3"`);}
const colors=['#2660a5','#c15b16','#8550ad','#087f87','#566a7f','#ae345d'];
for(const [x,y] of NQR_LAYOUT.centers)outline(x-15,y-15,30,30,colors[0]);
outline(209,209,38,38,colors[1]);
for(const points of NQR_LAYOUT.sizeCopies){let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);outline(Math.min(...xs)-5,Math.min(...ys)-5,Math.max(...xs)-Math.min(...xs)+10,Math.max(...ys)-Math.min(...ys)+10,colors[2]);}
for(const pair of NQR_LAYOUT.phasePairs)outline(pair[0][0]-11,pair[0][1]-11,50,22,colors[3]);
for(const x of NQR_LAYOUT.referenceX)outline(x-9,47,18,160,colors[4]);
outline(43,43,170,170,colors[5]);
label(ox+8,oy+4,'A',colors[0]);label(ox+245*s,oy+251*s,'B',colors[1]);label(ox+89*s,oy+13*s,'C',colors[2]);label(ox+148*s,oy+20*s,'D',colors[3]);label(ox+8,oy+130*s,'E',colors[4]);label(ox+210*s,oy+52*s,'F',colors[5]);
text(48,779,'Dashed outlines and letter labels are explanatory overlays.',14,400,'#64748b');
const descriptions=[
 ['Four finder cores','Black/white squares locate the plane.','Their shape stays fixed across phases.'],
 ['Asymmetric bottom-right corner','Six satellite patches distinguish rotation','and reflection; the square core is retained.'],
 ['Grid size, repeated twice','Coarse black/white code selects G16–G128.','Both copies must be clear and agree.'],
 ['Phase cue pair, repeated twice','Colors identify phase 0, 1, 2 or 3.','Top and bottom must agree in this exposure.'],
 ['Fixed palette on both sides','All 16 known colors measure camera color','distortion; these strips do not cycle.'],
 ['Logical matrix and grid references','Random data cells change color by phase.','Known grid-8 reference rows/columns stay fixed.'],
];
for(let i=0;i<descriptions.length;i++){let y=151+i*103;label(695,y,String.fromCharCode(65+i),colors[i]);text(720,y+4,descriptions[i][0],21,700,'#172c48');text(720,y+34,descriptions[i][1],17);text(720,y+58,descriptions[i][2],17);}
text(40,842,'Four observations of the same logical payload',24,700,'#172c48');
for(let p=0;p<4;p++){
 const x=40+285*p;rect(x,863,266,89,'#fff','rx="10" stroke="#d7e0ea"');
 text(x+14,888,`Phase ${p}`,17,700);
 for(let j=0;j<2;j++)rect(x+14+j*32,901,26,26,hex(PALETTE[(p+j)%4]),'stroke="#c4cdd8"');
 text(x+88,918,`Data XOR mask: ${[0,15,7,8][p]}`,15);
}
text(40,981,'Phases add evidence, not four independent payloads. Reference cells keep their known colors.',17);
text(40,1024,'G16 capacity: 16 × 16 cells × 4 bits = 1,024 bits = 128 bytes gross',22,700,'#172c48');
const cells=gridLayout(16,16,8).cells.length;
text(40,1057,`Grid-8 leaves ${cells} data cells: 84 whole bytes + 4 bits, before packet framing and error correction.`,17);
text(40,1089,'4×4 camera pixels measure one color symbol. Sixteen colors encode 4 bits, not 16 bits, per symbol.',17);
a.push('</g></svg>');
const out=fileURLToPath(new URL('../docs/images/neural-qr-layout.svg',import.meta.url));
await writeFile(out,a.join('\n')+'\n');
console.log(out);
for(const g of NQR_LAYOUT.sizes){const n=gridLayout(g,16,8).cells.length;console.log(`G${g}: ${g*g/2} gross bytes; ${n} data cells; ${Math.floor(n/2)} whole bytes + ${n%2*4} bits before packet/ECC overhead.`);}
