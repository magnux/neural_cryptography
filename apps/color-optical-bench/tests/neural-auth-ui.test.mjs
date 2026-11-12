import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
const root=new URL('../dist/',import.meta.url),html=readFileSync(new URL('neural-zkboo.html',root),'utf8');
const app=readFileSync(new URL('js/neural-auth/app.js',root),'utf8');
test('separate Neural-ZKBoo page binds every control and local asset',()=>{
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(ids.length,new Set(ids).size);
  for(const [,id] of app.matchAll(/\$\('([^']+)'\)/g))assert.ok(ids.includes(id),`missing ${id}`);
  for(const [,url] of html.matchAll(/(?:href|src)="(\.\/[^"#]+)"/g))assert.ok(existsSync(new URL(url,root)),`missing ${url}`);
  assert.match(html,/id="phaseMs"[^>]+value="150"/);assert.match(html,/id="captureMs"[^>]+value="100"/);
  assert.match(html,/id="nextTurn"/);assert.match(html,/32 · quick test/);
  assert.ok(!/https?:\/\//.test(html));assert.ok(!/qr-client|qr-worker/.test(app));
});


test('Neural-QR UI separates screen geometry from proof role and retains full camera frames',()=>{
  assert.match(html,/id="deviceProfile"/);assert.match(html,/id="outgoingSize"/);assert.match(html,/id="role"/);
  assert.match(html,/id="nextGeometry"/);assert.match(html,/between outgoing messages/);
  assert.match(html,/value="1440"/);assert.match(html,/value="2160"/);assert.match(html,/value="max"/);
  assert.match(app,/openCaptureCamera/);assert.match(app,/captureDimensions/);
  assert.match(app,/size:config.incomingMaxSize/);assert.match(app,/outgoingSize:config.outgoingSize/);
  assert.match(app,/setOutgoingSize/);assert.match(app,/renderNeuralQR/);
  assert.doesNotMatch(app,/qr-ack|renderACK|renderFrame/);
  assert.doesNotMatch(app,/1920\/video.videoWidth|1080\/video.videoHeight/);
  assert.match(html,/G32 ACK/);assert.match(html,/RS\+CRC-valid/);
  assert.doesNotMatch(html,/QR ACK|readable header|unreadable header/);
});

test('live camera controls are separate from locked setup and retain acquisition diagnostics without frames',()=>{
  assert.ok(html.indexOf('id="cameraControls"')>html.indexOf('</fieldset>'));
  assert.match(html,/Camera exposure &amp; focus/);
  assert.match(html,/id="cameraControlStatus"[^>]+role="status"/);
  assert.match(app,/captureIntervalMs:config.captureMs/);
  assert.match(app,/cameraControls\.close\(\)/);
  assert.match(app,/report\.cameraControls=cameraControls\.snapshot\(\)/);
  assert.match(app,/report\.lastAcquisition=acquisitionDiagnostics\(result\)/);
  assert.match(app,/controller!==cameraControls/);
  assert.doesNotMatch(app,/ctx\.filter|brightness\(/);
});
