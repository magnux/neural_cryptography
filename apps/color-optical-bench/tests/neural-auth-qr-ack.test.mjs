import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {renderACK,readACK} from '../dist/js/neural-auth/qr-ack.js';
import {encodePacket,decodePacket} from '../dist/js/link/protocol.js';
import {encodeQr,renderQr} from '../dist/js/qr-core/js/qr.js';

function ack(status='progress') {
  return encodePacket({kind:'ack',messageType:'proof',sessionId:'e2'.repeat(16),messageId:'b9'.repeat(8),
    digest:'d4'.repeat(32),base:42,bitmap:1,status});
}
function rotate(image) {
  const data=new Uint8ClampedArray(image.data.length),side=image.width;
  for(let y=0;y<side;y++)for(let x=0;x<side;x++)
    data.set(image.data.subarray(4*(y*side+x),4*(y*side+x)+4),4*(x*side+side-1-y));
  return {width:side,height:side,data};
}

test('static QR ACKs preserve every OLM1 field in binary byte mode',()=>{
  for(const status of ['ready','progress','complete']) {
    const bytes=ack(status),image=renderACK(bytes),result=readACK(image);
    assert.equal(bytes.length,84);assert.equal(image.qrVersion,5);assert.equal(image.qrModules,37);
    assert.equal(image.ecc,'M');assert.equal(image.width,(37+8)*16);assert.equal(image.height,image.width);
    assert.equal(result.ok,true,result.error);assert.equal(result.method,'qr-ack');
    assert.deepEqual(result.chunk,bytes);assert.deepEqual(decodePacket(result.chunk),decodePacket(bytes));
    assert.equal(result.corners.length,4);assert.ok(result.corners.flat().every(Number.isFinite));
  }
});

test('rotated ACK display remains readable without color references or custom header',()=>{
  const bytes=ack(),result=readACK(rotate(renderACK(bytes)));
  assert.equal(result.ok,true,result.error);assert.deepEqual(result.chunk,bytes);
});

test('blank camera image yields a detection miss without accepted bytes',()=>{
  const result=readACK({width:320,height:240,data:new Uint8ClampedArray(320*240*4).fill(255)});
  assert.equal(result.ok,false);assert.equal(result.stage,'detect');assert.equal(result.chunk,undefined);
  assert.deepEqual(result.corners,[]);
});

test('QR error correction does not bypass OLM1 CRC and packet kind validation',()=>{
  const damaged=ack();damaged[10]^=1;
  const data=encodePacket({kind:'data',messageType:'proof',sessionId:'e2'.repeat(16),messageId:'b9'.repeat(8),index:0,data:Uint8Array.of(1,2,3)});
  for(const bytes of [damaged,data,Uint8Array.of(1,2,3)]) {
    assert.throws(()=>renderACK(bytes));
    const result=readACK(renderQr(encodeQr(bytes,{ecc:'M'})));
    assert.equal(result.ok,false);assert.equal(result.stage,'payload');assert.equal(result.chunk,undefined);
    assert.equal(result.corners.length,4);assert.ok(result.error);
  }
});

test('prepared QR codec and licenses match the canonical QR application',async()=>{
  const root=new URL('../../',import.meta.url);
  for(const path of ['js/qr.js','js/qr-capacities.js','vendor/jsQR.js','vendor/qrcodegen.js',
    'vendor/jsQR-LICENSE.txt','vendor/qrcodegen-LICENSE.txt','vendor/README.md','vendor/manifest.json']) {
    const [canonical,prepared]=await Promise.all([
      readFile(new URL(`qr-neural-key/dist/${path}`,root)),
      readFile(new URL(`color-optical-bench/dist/js/qr-core/${path}`,root))]);
    assert.deepEqual(prepared,canonical,path);
  }
});
