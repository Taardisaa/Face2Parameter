"""Local-file interactive explorer; catalog and independently checked data come from exporter."""

TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HS2 头部表面位移</title><style>
:root{color-scheme:dark;font-family:system-ui,"Microsoft YaHei",sans-serif;background:#111822;color:#e6edf5}*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:25px;margin:0 0 8px}h2{font-size:18px}p{line-height:1.55}.muted{color:#a5b6c9;font-size:13px}.panel{background:#1a2533;border:1px solid #344356;border-radius:10px;padding:16px;margin:14px 0}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end}label{display:flex;flex-direction:column;gap:6px;font-size:13px}select,input{background:#111822;color:inherit;border:1px solid #536579;padding:8px;border-radius:5px;max-width:100%}select#control{min-width:280px}input[type=checkbox]{accent-color:#62d4e7}.inline{flex-direction:row;align-items:center}.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.view{background:#111822;border:1px solid #344356;border-radius:8px;padding:10px}.view h2{font-size:14px;margin:2px 0 8px}canvas{width:100%;height:360px;display:block}#status{min-height:24px;color:#62d4e7}.error{color:#ffb8a5!important}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px}.metric{background:#111822;padding:12px;border-radius:6px}.metric b{display:block;font-size:19px;margin-bottom:5px}.legend{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:12px}.gradient{width:180px;height:9px;background:linear-gradient(90deg,#365d90,#48e2da,#ffdd61,#ff785a)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111822;padding:12px;border-radius:6px;font-size:12px;max-height:380px;overflow:auto}summary{cursor:pointer;font-weight:600;padding:6px 0}.warn{border-color:#866146;background:#30261f;color:#ffe1b4}#failures li{margin:8px 0;overflow-wrap:anywhere}#suggestion{line-height:1.6}button{background:#244456;color:#e6edf5;border:1px solid #62a1b9;padding:8px 12px;border-radius:5px;cursor:pointer}@media(max-width:950px){.views{grid-template-columns:1fr}canvas{height:330px}main{padding:14px}}
</style></head><body><main>
<h1>HS2 头部参数 → 实际表面位移</h1>
<p class="muted">范围：o_head 头部曲面；顶点投影显示全部曲面点，会透视到背侧点，不是遮挡后的皮肤渲染。没有经过对应关系验证的解剖区域标签。单位是记录坐标系中的游戏几何单位，不能解释为毫米。</p>
<section class="panel controls" aria-label="参数选择">
<label>输入配置<select id="baseline"></select></label>
<label>控制类别<select id="kind"><option value="native">原生 59 参数（含扩展范围）</option><option value="abmx">ABMX 骨骼控制</option></select></label>
<label>参数<select id="control"></select></label>
<label>实际采样<select id="sample" disabled></select></label>
<label class="inline"><input id="vectors" type="checkbox">位移线</label>
<label>位移线显示倍数<input id="gain" type="number" min="1" max="100" step="1" value="1"></label>
</section>
<p id="status" role="status" aria-live="polite">选择参数后加载本地证据。</p>
<div id="content" hidden>
<section class="panel"><div class="metrics" id="metrics"></div><p class="muted" id="sample-note"></p>
<div class="legend"><span>灰点：基线</span><span>彩点：实际候选位置</span><div class="gradient"></div><span id="color-max"></span></div></section>
<section class="views"><div class="view"><h2>XY 平面 · X 水平 / Y 垂直</h2><canvas id="xy" aria-label="XY 顶点投影"></canvas></div><div class="view"><h2>ZY 平面 · Z 水平 / Y 垂直</h2><canvas id="zy" aria-label="ZY 顶点投影"></canvas></div><div class="view"><h2>XZ 平面 · X 水平 / Z 垂直</h2><canvas id="xz" aria-label="XZ 顶点投影"></canvas></div></section>
<section class="panel"><h2>局部步长估算</h2><p class="muted">根据该配置实测局部正 / 负方向最大位移斜率估算；限于实际局部探针区间。端点或 SliderUnlocker 的扩展采样不能证明区间内处处线性。此处不会自动修改游戏参数。</p>
<div class="controls"><label>目标最大位移（头部包围盒对角线 %）<input id="target" type="number" value="0.1" min="0.000001" step="0.01"></label><label>参数方向<select id="sign"><option value="1">增加</option><option value="-1">减少</option></select></label></div><p id="suggestion"></p></section>
<section class="panel"><details><summary>完整输入配置、来源与校验范围</summary><pre id="identity"></pre><pre id="verification"></pre></details></section>
</div>
<section class="panel warn"><details id="failure-details"><summary id="failure-summary">保留的未通过项</summary><p>头部曲面通过不代表全部捕获的表面均通过。以下失败不会被隐藏或按通过处理。</p><ul id="failures"></ul></details></section>
<section class="panel"><details><summary>数据包来源</summary><pre id="provenance"></pre></details></section>
</main><script>
"use strict";
const catalog = CATALOG_JSON;
const $ = id => document.getElementById(id);
let payload = null, wanted = null, generation = 0;
const cache = new Map();
const finite = x => typeof x === "number" && Number.isFinite(x);
const fmt = x => finite(x) ? (Math.abs(x)>=0.01 && Math.abs(x)<1000 ? x.toPrecision(6) : x.toExponential(5)) : "未记录";
const vec = x => Array.isArray(x) ? "["+x.map(fmt).join(", ")+"]" : "未记录";
const diagonal = () => Number(catalog.baselines[payload.baseline_name]?.head_diagonal || payload.stats?.[0]?.baseline_bbox_diagonal);
function option(select,value,label){const o=document.createElement("option");o.value=value;o.textContent=label;select.appendChild(o);return o;}
function status(message,error=false){$("status").textContent=message;$("status").classList.toggle("error",error);}
function clear(){payload=null;$("content").hidden=true;$("sample").replaceChildren();$("sample").disabled=true;}
function validPayload(p){
 if(!p || !Array.isArray(p.baseline)||!p.baseline.length||!Array.isArray(p.deltas)||!p.deltas.length||!Array.isArray(p.stats)||p.stats.length!==p.deltas.length||!Array.isArray(p.levels)||p.levels.length!==p.deltas.length||!Array.isArray(p.roles)||p.roles.length!==p.deltas.length)throw Error("数据数组或采样数量不一致");
 if(!Array.isArray(p.fixed_min)||!Array.isArray(p.fixed_max)||p.fixed_min.length!==3||p.fixed_max.length!==3||![...p.fixed_min,...p.fixed_max,p.max_distance].every(finite)||p.max_distance<0||p.fixed_max.some((v,i)=>v<p.fixed_min[i]))throw Error("固定绘图区间无效");
 const points=[p.baseline,...p.deltas];for(const a of points){if(a.length!==p.baseline.length||a.some(v=>!Array.isArray(v)||v.length!==3||!v.every(finite)))throw Error("顶点对应或数值无效");}
 if(!p.levels.every(finite))throw Error("采样参数值无效");
}
window.HS2_RESPONSE = function(p){
 try{validPayload(p);cache.set(p.id,p);if(wanted&&p.id===wanted.id){if(p.baseline_name!==wanted.baseline_name||p.kind!==wanted.kind)throw Error("数据配置与当前选择不匹配");show(p);}}
 catch(e){if(wanted&&p?.id===wanted.id){clear();status("加载失败："+e.message,true);}}
};
function refreshControls(){
 generation++;wanted=null;clear();const select=$("control"),previous=select.value;select.replaceChildren();
 const entries=catalog.entries.filter(e=>e.baseline_name===$("baseline").value&&e.kind===$("kind").value);
 for(const e of entries)option(select,e.id,e.label);
 select.disabled=!entries.length;
 if(!entries.length){option(select,"","此配置没有此类别的实测采样");status("未采集当前配置的 ABMX 数据；不会借用其他配置。");return;}
 if(entries.some(e=>e.id===previous))select.value=previous;
 else if($("kind").value==="native"){const defaultEntry=entries.find(e=>Number(e.control)===30);if(defaultEntry)select.value=defaultEntry.id;}
 load();
}
function load(){
 const token=++generation;clear();wanted=catalog.entries.find(e=>e.id===$("control").value);if(!wanted){status("没有对应的数据项。",true);return;}
 status("正在加载："+wanted.label+" · "+wanted.baseline_name);
 if(cache.has(wanted.id)){try{const p=cache.get(wanted.id);if(p.baseline_name!==wanted.baseline_name||p.kind!==wanted.kind)throw Error("缓存配置不匹配");show(p);}catch(e){clear();status(e.message,true);}return;}
 const file=wanted.file;if(typeof file!=="string"||!/^data\/[A-Za-z0-9_.-]+\.js$/.test(file)){status("拒绝无效的本地数据路径。",true);return;}
 const script=document.createElement("script");script.src=file;script.async=true;
 script.onerror=()=>{if(token===generation){clear();status("无法加载 "+file+"。请保留 index.html 与 data 目录的相对位置。",true);}script.remove();};
 script.onload=()=>{if(token===generation&&!payload){status("文件未提供有效的 HS2_RESPONSE 数据。",true);}script.remove();};
 document.head.appendChild(script);
}
function roleLabel(role){const labels={local_minus:"局部减少 · 实测",local_plus:"局部增加 · 实测",native_min:"原生下界 0 · 实测",native_max:"原生上界 1 · 实测",extended_min:"SliderUnlocker 扩展下界 · 实测",extended_max:"SliderUnlocker 扩展上界 · 实测",minus:"局部减少 · 实测",plus:"局部增加 · 实测"};return labels[role]||String(role)+" · 实测";}
function sampleRoleLabel(index){if(payload.kind==="abmx")return "ABMX "+(payload.levels[index]>payload.baseline_level?"增加":"减少")+" · 实测";return roleLabel(payload.roles[index]);}
function show(p){payload=p;$("sample").replaceChildren();p.roles.forEach((role,i)=>option($("sample"),String(i),sampleRoleLabel(i)+" / 值 "+fmt(p.levels[i])));$("sample").disabled=false;const plus=p.kind==="abmx"?p.levels.findIndex(level=>level>p.baseline_level):p.roles.indexOf("local_plus");if(plus>=0)$("sample").value=String(plus);$("content").hidden=false;$("identity").textContent=JSON.stringify({head_id:catalog.head_id,units:catalog.units,baseline:p.baseline_name,baseline_identity:catalog.baselines[p.baseline_name]?.identity,entry_identity:p.identity},null,2);$("verification").textContent=JSON.stringify(p.verification,null,2);status(p.label+" · "+p.baseline_name+" · o_head 已记录的实际采样");render();suggest();}
function metric(value,label){const div=document.createElement("div");div.className="metric";const b=document.createElement("b");b.textContent=value;div.appendChild(b);const s=document.createElement("span");s.className="muted";s.textContent=label;div.appendChild(s);$("metrics").appendChild(div);}
function color(t){t=Math.min(1,Math.max(0,t));const stops=[[54,93,144],[72,226,218],[255,221,97],[255,120,90]],x=t*3,k=Math.min(2,Math.floor(x)),a=x-k;return "rgb("+stops[k].map((v,i)=>Math.round(v*(1-a)+stops[k+1][i]*a)).join(",")+")";}
function draw(id,axes,deltas){
 const c=$(id),rect=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=rect.width,h=rect.height;c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const [a,b]=axes,lo=payload.fixed_min,hi=payload.fixed_max,margin=25,scale=Math.min((w-2*margin)/Math.max(hi[a]-lo[a],1e-10),(h-2*margin)/Math.max(hi[b]-lo[b],1e-10));
 const project=v=>[w/2+(v[a]-(lo[a]+hi[a])/2)*scale,h/2-(v[b]-(lo[b]+hi[b])/2)*scale];
 ctx.fillStyle="#8292a34d";for(const v of payload.baseline){const q=project(v);ctx.fillRect(q[0],q[1],1.1,1.1);}
 const gain=Math.min(100,Math.max(1,Number($("gain").value)||1));
 if($("vectors").checked){ctx.strokeStyle="#62d4e755";ctx.lineWidth=.6;ctx.beginPath();payload.baseline.forEach((v,i)=>{const d=deltas[i],q=project(v),r=project(v.map((x,j)=>x+d[j]*gain));ctx.moveTo(...q);ctx.lineTo(...r);});ctx.stroke();}
 payload.baseline.forEach((v,i)=>{const d=deltas[i],dist=Math.hypot(...d),q=project(v.map((x,j)=>x+d[j]));ctx.fillStyle=color(payload.max_distance?dist/payload.max_distance:0);ctx.fillRect(q[0],q[1],1.5,1.5);});
 ctx.fillStyle="#a5b6c9";ctx.font="11px system-ui";ctx.fillText("固定边界 · 彩点位置为实测 · 线可放大",10,h-8);
}
function render(){if(!payload)return;const i=Number($("sample").value),s=payload.stats[i],d=payload.deltas[i],diag=diagonal();$("metrics").replaceChildren();metric(fmt(s.displacement?.max),"最大位移 · 游戏几何单位");metric(fmt(s.displacement?.max/diag*100)+" %","最大位移 / 基线头部对角线");metric(fmt(s.displacement?.rms),"RMS 位移 · 游戏几何单位");metric(String(s.affected_vertex_count??"未记录")+" / "+payload.baseline.length,"超过实测阈值的顶点数量");metric(vec(s.centroid_shift),"平均位移向量 [X,Y,Z] · 非解剖区域标签");const noise=catalog.baselines[payload.baseline_name]?.repeat_drift_normalized;metric(fmt(noise*100)+" %","重复基线最大漂移 / 头部对角线");if(finite(payload.verification?.display_rounding_max_l2))metric(fmt(payload.verification.display_rounding_max_l2),"仅显示用数值舍入最大 L2 误差");$("color-max").textContent="颜色：0 → "+fmt(payload.max_distance)+"（同一参数各采样共用）";$("sample-note").textContent="基线值 "+fmt(payload.baseline_level)+" → 采样值 "+fmt(payload.levels[i])+"；"+sampleRoleLabel(i)+"。影响阈值 "+fmt(s.effect_threshold_absolute)+"（"+fmt(s.effect_threshold_relative)+" × 头部对角线）。影响区域由彩点 / 位移线定位。"+(payload.kind==="native"?"原生正常范围 0..1；扩展范围必须按实际采样标记理解。":"ABMX 仅显示 o_head 的响应；此处接近零不代表其他表面无变化。");draw("xy",[0,1],d);draw("zy",[2,1],d);draw("xz",[0,2],d);}
function gainNumber(x){return finite(x)?x:(finite(x?.max)?x.max:(finite(x?.max_units_per_input_unit)?x.max_units_per_input_unit:NaN));}
function suggest(){
 if(!payload)return;const r=payload.local_response||{},sign=Number($("sign").value),percent=Number($("target").value),gain=gainNumber(sign<0?r.left_gain:r.right_gain),probe=Math.abs(Number(sign<0?r.minus_step:r.plus_step)),diag=diagonal(),out=$("suggestion");
 if(!finite(percent)||percent<=0||!finite(diag)||diag<=0||!finite(gain)||gain<=0||!finite(probe)||probe<=0){out.textContent="当前方向没有可用局部增益，或目标值无效；不能给出可靠步长。";return;}
 const raw=percent/100*diag/gain,interval=r.valid_interval;let limit=probe;if(Array.isArray(interval)&&interval.length===2&&interval.every(finite)&&finite(payload.baseline_level))limit=Math.min(limit,Math.max(0,sign<0?payload.baseline_level-interval[0]:interval[1]-payload.baseline_level));
 if(finite(catalog.baselines[payload.baseline_name]?.repeat_drift_normalized)&&percent/100<=catalog.baselines[payload.baseline_name].repeat_drift_normalized){out.textContent="目标低于或等于重复基线最大漂移；无法区分位移与噪声，不提供步长。";return;}
 const step=Math.min(raw,limit),value=payload.baseline_level+sign*step,predicted=step*gain/diag*100,noise=catalog.baselines[payload.baseline_name]?.repeat_drift_normalized;
 const unit=payload.kind==="native"?"原生参数数值":((wanted?.channel==="rotation")?"ABMX 角度（度）":((wanted?.channel==="position")?"ABMX 位置单位":"ABMX 倍率数值"));
 out.textContent="局部建议："+(sign>0?"+":"−")+fmt(step)+" "+unit+"；候选值 "+fmt(value)+"。线性估算最大位移 "+fmt(predicted)+" % 头部对角线。"+(raw>limit?"目标超出局部探针区间，已限制到实测局部边界；该步长不会达到原目标。 ":"")+"适用参数区间 "+vec(interval)+"。"+(r.estimate_verified===true?"独立采样验证已记录；详情见来源与校验范围。":"估算尚未由独立采样验证，不能视为实测结果。")+(finite(noise)&&percent/100<=noise?" 目标低于或等于重复基线漂移，难以从噪声中分辨。":"");
}
for(const name of Object.keys(catalog.baselines))option($("baseline"),name,name);
if(catalog.baselines.card_input)$("baseline").value="card_input";
$("provenance").textContent=JSON.stringify(catalog.provenance,null,2);
const failures=catalog.failures||[];$("failure-summary").textContent="保留的未通过项（"+failures.length+"）";for(const f of failures){const li=document.createElement("li");li.textContent=f.name+"："+f.reason;$("failures").appendChild(li);}if(failures.length)$("failure-details").open=true;else{const li=document.createElement("li");li.textContent="当前数据包没有列出的失败。此信息只适用于包内记录的范围。";$("failures").appendChild(li);}
$("baseline").addEventListener("change",refreshControls);$("kind").addEventListener("change",refreshControls);$("control").addEventListener("change",load);$("sample").addEventListener("change",render);$("vectors").addEventListener("change",render);$("gain").addEventListener("input",render);$("target").addEventListener("input",suggest);$("sign").addEventListener("change",suggest);window.addEventListener("resize",render);
refreshControls();
</script></body></html>'''
