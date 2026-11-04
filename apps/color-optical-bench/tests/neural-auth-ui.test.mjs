import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
const root=new URL('../dist/',import.meta.url),html=readFileSync(new URL('neural-zkboo.html',root),'utf8');
const app=readFileSync(new URL('js/neural-auth/app.js',root),'utf8');
test('separate Neural-ZKBoo page binds every control and local asset',()=>{
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(ids.length,new Set(ids).size);
  for(const [,id] of app.matchAll(/\$\('([^']+)'\)/g))assert.ok(ids.includes(id),`missing ${id}`);
  for(const [,url] of html.matchAll(/(?:href|src)="(\.\/[^"#]+)"/g))assert.ok(existsSync(new URL(url,root)),`missing ${url}`);
  assert.match(html,/id="phaseMs"[^>]+value="95"/);assert.match(html,/id="captureMs"[^>]+value="100"/);
  assert.match(html,/id="nextTurn"/);assert.match(html,/32 · quick test/);
  assert.ok(!/https?:\/\//.test(html));assert.ok(!/qr-client|qr-worker/.test(app));
});
