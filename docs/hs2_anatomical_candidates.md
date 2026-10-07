# 原图语义假设到固定材料点

本工具把**明确但未通过的解剖假设**放到可审查的真实截图与三角形上。几何对应和语义验收分别记录；没有把最大支持方向顶点自动叫鼻尖，也没有修改既有FAN失败结果。所有文件新增于 `tools/anatomical_candidates/` 和 `outputs/anatomical_candidates_20261005/`，不操作游戏，不改原图、core、缓存或旧报告。

## 已实际执行的旧head2样例

输入为 `HS2Mod/artifacts/infrastructure_live_20261004/paired_capture_aa1/response.json`、同目录geometry、`outputs/unity_parity_20261004/paired_capture_aa1.json`、`outputs/pixel_calibration_20261004/report_v3.json`。七yaw是 −90／−60／−30／0／30／60／90。工具重新计算actual bone／cached source LBS，验证报告snapshot SHA、renderer path、source SHA和候选residual，不只相信passed字段。相机必须通过原pixel certificate，且每图与geometry同SHA／pose／frame。inactive renderer明确排除，重复 `o_tang`不按名字覆盖；active scene使用完整renderer path。

已有FAN30／48／54的heldout误差分别约2.196／3.138／2.270px，仍是failed；眼角仍insufficient visibility。新工具只是把这些旧状态带入报告作不可变对照，FAN不参与新点构造。`seed_head2_aa1.json`由agent看**未加overlay的正面原图**给出六个像素假设，身份、PNG／capture／geometry SHA均固定。

输出 `outputs/anatomical_candidates_20261005/head2_old_baseline/`：

| ID | 未通过的语义假设 | 正面seed xy | 原o_head triangle |
|---|---|---|---:|
| N | 鼻部中央凸起skin dome，须与鼻孔／subnasale区分 | 254,351 | 6049 |
| AL | 正面图左鼻翼外侧皮肤边缘，须与鼻孔内缘／阴影区分 | 233,355 | 1237 |
| AR | 正面图右鼻翼外侧皮肤边缘 | 276,355 | 6001 |
| ML | 正面图左上下唇汇合点，须与化妆延长线区分 | 218,401 | 6625 |
| MR | 正面图右上下唇汇合点 | 289,401 | 2088 |
| C | 下巴前方中线隆起，须与最低轮廓menton区分 | 254,448 | 5637 |

六seed都命中可见o_head几何，但**六个语义全部未通过**。triangle/barycentric来自标定相机ray的前表面交点，并绑定ordered vertex IDs／renderer／source SHA，随后固定这些权重投影到七角度。未在每个视角重找极值，也未用变动轮廓当固定点。单图ray回投命中seed是构造恒等关系，报告明确不是独立验证。图像左右没有擅自当解剖左右。

`visual_review/`含每图原分辨率overlay、无重采样局部overlay crop、原图crop，以及 `overlay_seven_view_sheet.png`／`original_seven_view_sheet.png`。彩色圈表示无几何blocker的候选投影，红圈表示遮挡／裁剪等几何拒绝；两者都不是解剖通过标志。每张原PNG不变且SHA保存。实际profile显示部分鼻翼、唇角和下巴点被遮挡，保留拒绝信息，不以FAN分数补可见性。

## 执行和审查流程

```powershell
.venv/Scripts/python.exe -m tools.anatomical_candidates.run --capture ../HS2Mod/artifacts/infrastructure_live_20261004/paired_capture_aa1/response.json --geometry ../HS2Mod/artifacts/infrastructure_live_20261004/paired_capture_aa1/geometry.json --lbs-report outputs/unity_parity_20261004/paired_capture_aa1.json --pixel-report outputs/pixel_calibration_20261004/report_v3.json --seeds tools/anatomical_candidates/seed_head2_aa1.json --prior-consensus outputs/surface_consensus_20261005/report.json --out outputs/anatomical_candidates_20261005/<new-directory>
```

先由独立reviewer看原图，在新副本 `independent_review_template.json`记录各点visible／occluded／ambiguous和原图像素。看投影后照抄坐标不算独立观察。看不清的点用null；不要猜遮挡点。然后看overlay确认假设是否其实落到纹理、唇妆或鼻孔阴影。记录reviewer identity、`annotation_method=original_png_before_overlay`、`skin_vs_texture_anchor`、语义decision和notes。**当前模板完全pending，未填独立标注。**

```powershell
.venv/Scripts/python.exe -m tools.anatomical_candidates.validate_review --candidate-report outputs/anatomical_candidates_20261005/head2_old_baseline/report.json --review <filled-new-review.json> --out outputs/anatomical_candidates_20261005/<new-review-report.json>
.venv/Scripts/python.exe -m unittest tools.anatomical_candidates.test_review -v
```

验收器锁定原candidate report／各source／PNG SHA，不能看完review residual后改seed或重拟合材料点。每点至少三条独立可见原图标注、至少一个30°或更斜角，固定材料点各误差≤2px；鼻尖／下巴前隆起额外需要60°以上独立可见语义检查。独立rays还需角度≥10°、condition≤100；这是数值可观测性，不是detector准确率。所有门限在本轮填标注前固定，改变门限须新协议／新报告。

几何gate通过仍要求独立reviewer显式接受同一语义定义、skin anchor明确。最多输出 `review_supported_material_hypothesis`／`accepted_for_named_material_candidate`，全局 `anatomical_correspondence_validated=false`、shader material visibility仍false。代码无法认证人类身份或确实先看原图；报告坦白这些是reviewer声明，不能把模板checkbox当临床真值。未知shader alpha／depth、图像阴影与材质边缘仍是限制。

八个新tests通过：独立身份合同、自验身份拒绝、overlay-first拒绝、报告变动拒绝、漏点拒绝、别的capture seed拒绝，以及解析三相机／真实三角形的exact projection不能把pending semantics通过、已拒绝seed即使勾接受也不得通过。实际空模板CLI也明确拒绝 `Independent reviewer identity required`，未生成伪造通过报告。

## 后续新采集如何使用

旧seed严格绑定旧capture／geometry／PNG，不能直接套到新bridge或新图片。新capture先完成自己的pixel／LBS certificate，再新建seed JSON并在原图上标注。工具接受路径参数，不依赖FAN模型或下载；camera／world space未知时直接拒绝。当前所有material只属于head2资产和该snapshot；跨native／ABMX形变的固定三角权重可交给既有 `surface_consensus.transport` 独立认证，但解剖意义在表情、拓扑、底模或纹理改变后仍需复审。

若要减少猜测，下一批最好在同freeze拍额外pitch／斜角，让鼻翼外缘、唇角和下巴前方隆起可见，并提供head-only／带眼睫完整两套图帮助区分skin与遮挡。仅yaw相机不应靠投影失败强制把namedpoint移到另一顶点。鼻尖和下巴都需要定义到底是固定skin材料点、某方向的瞬时支撑点，还是图像轮廓点；这三类任务的验收不能混用。
