"""Merge scripts/velocity_grid.py outputs into one self-contained page with a weight slider.
Usage: python scripts/velocity_grid_html.py out.html a.json [b.json ...]"""
import json, sys

profiles, meta = [], None
for path in sys.argv[2:]:
  d = json.load(open(path))
  meta = meta or {k: d[k] for k in ("vx", "wz", "reps")}
  for m, p in d["profiles"].items():
    label = "No box, walk base only" if m == "none" else f"{float(m):g} kg box"
    profiles.append({"label": label, "kg": -1 if m == "none" else float(m), "ckpt": d["checkpoint"], **p})
profiles.sort(key=lambda p: p["kg"])
PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>K1 velocity profile vs box weight</title><style>
.viz-root{color-scheme:light;--s:#fcfcfb;--t:#0b0b0b;--t2:#52514e;--line:#e4e3de}
@media (prefers-color-scheme:dark){.viz-root{color-scheme:dark;--s:#1a1a19;--t:#f0efec;--t2:#c3c2b7;--line:#3a3a37}}
body{margin:0}.viz-root{background:var(--s);color:var(--t);font:14px system-ui,sans-serif;padding:20px;min-height:100vh;box-sizing:border-box}
h1{font-size:20px;margin:0 0 4px}p{color:var(--t2);margin:0 0 12px;max-width:900px}
.ctl{display:flex;gap:14px;align-items:center;margin:12px 0}input[type=range]{width:360px}#lab{font-weight:600;min-width:190px}
.leg{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 12px;color:var(--t2)}.leg b{display:inline-block;width:18px;height:18px;line-height:18px;text-align:center;border-radius:4px;color:#fff;font-size:11px;margin-right:5px}
#tip{position:fixed;display:none;background:var(--s);border:1px solid var(--line);border-radius:6px;padding:8px 10px;pointer-events:none;box-shadow:0 2px 8px #0003;white-space:pre}
table{border-collapse:collapse;margin-top:16px}td,th{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right}th:first-child,td:first-child{text-align:left}th{color:var(--t2);font-weight:500}tr.on td{font-weight:600}
</style></head><body><div class="viz-root">
<h1>K1 velocity profile vs box weight</h1>
<p id="sub"></p>
<div class="ctl"><label for="s">Weight</label><input id="s" type="range" step="1"><span id="lab"></span></div>
<div class="leg" id="leg"></div><svg id="g"></svg><div id="tip"></div><table id="tab"></table>
</div><script>
const D=__DATA__;
const ST=[["T","Tracks vx and yaw","#1baf7a","#fff"],["V","Tracks vx only","#2a78d6","#fff"],["Y","Tracks yaw only","#e87ba4","#333"],["N","Up, tracks neither","#c3c2b7","#333"],["F","Fell","#2b2b29","#fff"],["D","Dropped box","#eda100","#333"]];
const C=20,L=58,B=44,T=8,nv=D.vx.length,nw=D.wz.length,svg=document.getElementById('g'),tip=document.getElementById('tip');
svg.setAttribute('width',L+nv*C+10);svg.setAttribute('height',T+nw*C+B);
document.getElementById('leg').innerHTML=ST.map(s=>`<span><b style="background:${s[2]};color:${s[3]}">${s[0]}</b>${s[1]}</span>`).join('');
document.getElementById('sub').textContent=`${D.reps} robots per cell in the training sim (mjlab). 1.5 s stand, ramp 1 m/s² and 2 rad/s², speed averaged over the last 4.5 s of 10 s. Tracks = error within max(0.1 m/s, 20%) for vx, max(0.2 rad/s, 20%) for yaw. Fell or dropped = at least half the robots in the cell. Hover a cell for numbers.`;
const NS='http://www.w3.org/2000/svg',el=(n,a,t)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(t!==undefined)e.textContent=t;svg.appendChild(e);return e};
const css=getComputedStyle(document.querySelector('.viz-root')),ink=css.getPropertyValue('--t2');
const cells=[];
for(let i=0;i<nv;i++){cells[i]=[];for(let j=0;j<nw;j++){const x=L+i*C,y=T+(nw-1-j)*C;
 const r=el('rect',{x:x+1,y:y+1,width:C-2,height:C-2,rx:3}),t=el('text',{x:x+C/2,y:y+C/2+4,'text-anchor':'middle','font-size':10,'pointer-events':'none'});
 r.addEventListener('mousemove',ev=>show(ev,i,j));r.addEventListener('mouseleave',()=>tip.style.display='none');cells[i][j]=[r,t];}}
D.vx.forEach((v,i)=>{if(i%5==0)el('text',{x:L+i*C+C/2,y:T+nw*C+16,'text-anchor':'middle','font-size':12,fill:ink},v.toFixed(1))});
el('text',{x:L+nv*C/2,y:T+nw*C+36,'text-anchor':'middle','font-size':13,fill:ink},'Commanded vx (m/s)');
D.wz.forEach((w,j)=>{if(j%4==0)el('text',{x:L-6,y:T+(nw-1-j)*C+C/2+4,'text-anchor':'end','font-size':12,fill:ink},w.toFixed(2))});
el('text',{transform:`translate(14,${T+nw*C/2}) rotate(-90)`,'text-anchor':'middle','font-size':13,fill:ink},'Commanded yaw rate (rad/s)');
const s=document.getElementById('s');s.min=0;s.max=D.profiles.length-1;s.value=0;
function show(ev,i,j){const p=D.profiles[+s.value];tip.style.display='block';tip.style.left=Math.min(ev.clientX+14,innerWidth-250)+'px';tip.style.top=ev.clientY+14+'px';
 tip.textContent=`${ST[p.status[i][j]][1]}\ncommand  vx ${D.vx[i].toFixed(1)}  yaw ${D.wz[j].toFixed(2)}\nmeasured vx ${p.vx[i][j].toFixed(2)}  yaw ${p.wz[i][j].toFixed(2)}\nfell ${(p.fell[i][j]*100).toFixed(0)}%  dropped ${(p.dropped[i][j]*100).toFixed(0)}%`}
function draw(){const p=D.profiles[+s.value];document.getElementById('lab').textContent=p.label;
 for(let i=0;i<nv;i++)for(let j=0;j<nw;j++){const a=ST[p.status[i][j]],[r,t]=cells[i][j];r.setAttribute('fill',a[2]);t.setAttribute('fill',a[3]);t.textContent=a[0]}
 [...document.querySelectorAll('#tab tr')].forEach((r,k)=>r.classList.toggle('on',k-1==+s.value))}
const z=D.wz.indexOf(0);
const rows=D.profiles.map(p=>{const c=[0,0,0,0,0,0];p.status.flat().forEach(v=>c[v]++);let mv=null;for(let i=0;i<nv;i++)if(p.status[i][z]==0)mv=D.vx[i];
 return `<tr><td>${p.label}</td>${c.map(x=>`<td>${x}</td>`).join('')}<td>${mv===null?'none':mv.toFixed(1)}</td></tr>`});
document.getElementById('tab').innerHTML=`<tr><th>Cells per status</th>${ST.map(a=>`<th>${a[1]}</th>`).join('')}<th>Highest tracked vx at yaw 0 (m/s)</th></tr>`+rows.join('');
s.oninput=draw;draw();
</script></body></html>"""
data = {**meta, "profiles": profiles}
open(sys.argv[1], "w").write(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
