"""Export a fixed-axis, standalone HTML point-surface displacement viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def viewer_data(atlas_path, control, mesh="o_head"):
    atlas_path = Path(atlas_path).resolve()
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    rows = [row for row in atlas["controls"] if row["index"] == control]
    if len(rows) != 1 or mesh not in atlas["mesh_layout"]:
        raise ValueError("Control or mesh is missing from this atlas")
    row = rows[0]
    with np.load(atlas_path.parent / "baseline.npz") as baseline_file:
        baseline = baseline_file[mesh]
    with np.load(row["vertex_effects_path"]) as effects:
        delta = effects[mesh + "__displacement"]
    if delta.shape[1:] != baseline.shape or not np.isfinite(delta).all():
        raise ValueError("Displacement geometry is invalid or mismatched")
    candidates = baseline[None] + delta
    # One frame over all levels, never independently recenter/resize a selected sample.
    bounds = np.concatenate([baseline[None], candidates])
    return {
        "head_id": atlas["head_id"],
        "profile": atlas["sampling_profile"],
        "mesh": mesh,
        "control": control,
        "baseline_level": atlas["baseline_native_input"][control],
        "baseline_input_sha256": atlas.get("baseline_provenance", {}).get(
            "native_input_sha256"
        ),
        "levels": atlas["levels"],
        "baseline": baseline.tolist(),
        "deltas": delta.tolist(),
        "fixed_min": bounds.min(axis=(0, 1)).tolist(),
        "fixed_max": bounds.max(axis=(0, 1)).tolist(),
        "max_distance": float(np.linalg.norm(delta, axis=2).max()),
        "stats": [sample["meshes"][mesh] for sample in row["samples"]],
    }


TEMPLATE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native head surface displacement</title>
<style>
body{margin:24px;background:#141922;color:#e0e6ee;font:15px system-ui}h1{font-size:21px}
.views{display:flex;gap:16px;flex-wrap:wrap}canvas{background:#080d15;border:1px solid #354157;max-width:100%}
label{margin-right:20px}select{padding:6px;background:#252f40;color:inherit}p{max-width:1000px;line-height:1.5}
.note{color:#a6b5c8}#numbers{font-variant-numeric:tabular-nums;color:#79d6e5}
</style>
<h1 id="title"></h1>
<p>灰色：baseline。彩色：采样后顶点；暖色表示更大位移。三个坐标平面共享固定尺度，切换数值不自动对齐或缩放。</p>
<label>采样系数 <select id="level"></select></label>
<label><input id="arrows" type="checkbox" checked>位移线（每20个顶点画一条）</label>
<p id="numbers"></p><div class="views"><canvas width="420" height="420"></canvas><canvas width="420" height="420"></canvas><canvas width="420" height="420"></canvas></div>
<p class="note">这是完整顶点的坐标投影，包含背面点；不表示不透明皮肤外观，也不命名解剖区域。距离是未校准的资产单位。此图仅包含 native 形状响应，未包含 ABMX／实时表情。</p>
<p class="note" id="provenance"></p>
<script>
const d=DATA_JSON;
const select=document.querySelector('#level');
d.levels.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v;select.appendChild(o)});
document.querySelector('#title').textContent=`Head ${d.head_id} · ${d.profile} · Control ${d.control} · ${d.mesh}`;
document.querySelector('#provenance').textContent=`Baseline coefficient: ${d.baseline_level}; input SHA-256: ${d.baseline_input_sha256||'older atlas without input hash'}`;
const planes=[[0,1,'X–Y'],[2,1,'Z–Y'],[0,2,'X–Z']];
const size=Math.max(...d.fixed_max.map((v,i)=>v-d.fixed_min[i]),1e-12);
const center=d.fixed_min.map((v,i)=>(v+d.fixed_max[i])/2);
function draw(){
 const index=Number(select.value),delta=d.deltas[index],s=d.stats[index];
 document.querySelector('#numbers').textContent=`Δ coefficient: ${(d.levels[index]-d.baseline_level).toPrecision(5)} | max: ${s.displacement.max.toPrecision(5)} | RMS: ${s.displacement.rms.toPrecision(5)} | max / baseline bbox: ${(100*s.normalized_displacement.max).toPrecision(4)}% | affected vertices: ${s.affected_vertex_count} / ${s.vertex_count}`;
 document.querySelectorAll('canvas').forEach((canvas,view)=>{
  const ctx=canvas.getContext('2d'),[a,b,label]=planes[view];ctx.clearRect(0,0,420,420);
  const point=p=>[210+(p[a]-center[a])*350/size,210-(p[b]-center[b])*350/size];
  ctx.fillStyle='#627183';d.baseline.forEach(p=>{const [x,y]=point(p);ctx.fillRect(x,y,1.2,1.2)});
  if(document.querySelector('#arrows').checked){ctx.strokeStyle='#7bdcea80';ctx.lineWidth=.7;ctx.beginPath();for(let i=0;i<delta.length;i+=20){const p=d.baseline[i],q=p.map((v,j)=>v+delta[i][j]);const start=point(p),end=point(q);ctx.moveTo(...start);ctx.lineTo(...end)}ctx.stroke()}
  delta.forEach((v,i)=>{const distance=Math.hypot(...v),t=Math.min(1,distance/(d.max_distance||1));ctx.fillStyle=`hsl(${195-190*t} 85% ${45+15*t}%)`;const [x,y]=point(d.baseline[i].map((p,j)=>p+v[j]));ctx.fillRect(x,y,1.6,1.6)});
  ctx.fillStyle='#dce5ef';ctx.font='15px system-ui';ctx.fillText(label+' · fixed asset axes',15,25);
 });
}
select.addEventListener('change',draw);document.querySelector('#arrows').addEventListener('change',draw);draw();
</script>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--control", type=int, required=True)
    parser.add_argument("--mesh", default="o_head")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = viewer_data(args.atlas, args.control, args.mesh)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(TEMPLATE.replace("DATA_JSON", payload), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
