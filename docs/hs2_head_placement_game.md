# 缩小／低头候选进入真正游戏底模

2026-10-08 用户要求将已评审网格真实反映在游戏里。本次选择
`placement_review_final_v1/chenger/size_and_pitch`：整体缩至 94%、低头 3°。
该版本已构建为独立 Sideloader 头部资产，并在当前 Maker 中载入。

## 接入机制

`build.py --authored-neck` 读取候选及已通过的静态验收，绑定原始 img-002
数据和安装版 head2 的摘要，直接写入 prefab 的真实 `o_head`。
`asset_placement.py` 处理当前 source／native collar／六排 bridge 布局，
不再依赖未完成的 `neck_surface.py` 集成草稿。

- 源面、裁切后源面及连接面的顶点和三角形直接来自已评审候选；
  重新读取序列化包后，真实 `o_head` 与候选位置逐项相等（float32）。
  Build 没有再次局部改脸；receipt 的“外带位置未改”相对于声明的
  整体 similarity 而言，不应解释为头部未缩小／旋转。
- 使用安装版 bone palette、59 项 0.5 的源码还原骨矩阵及其逆 bindpose。
  原生接口／连接带在头侧绑定 `cf_J_FaceRoot_s`；实际身体接口绑定
  `cf_J_Head_s`，依据已审查的两帧重合合同。身体没有裁切或改网格。
- UV／普通面部 skin 是资产属性转移，不用于改变脸的顶点。
  原生 collar 的属性从后部重塑前的实际 native 面读取；已重塑部分用
  最终曲面法线，原生固定接口保留 native 法线场。
- 使用 `--native-eyes`，以原生半球／UV 眼部资产按源眼球中心和尺寸摆放，
  并应用同一低头方向。这一选择改变源眼球几何，不能把整头含眼球都
  宣称为未修改原始 FLAME 顶点。眼部外观和 gaze 尚未验收。
- 仍使用 plain native skin，未恢复身份贴图。未兼容的睫毛、眼膜、口腔
  等沿用先前底模构建的零索引处理，表情没有假装兼容。

实际安装包为 `E:/HoneySelect2_ArcticFox/mods/Codex/Chenger.MICA.NativeHead.zipmod`，
GUID `codex.chenger.mica.nativehead`、prefab `p_cf_head_chenger_mica`。
原安装包备份在 ignored `outputs/native_head_20261008/placement_native_v2/previous_installed.zipmod`。

现有加载器缓存需更新，因此保存当前人物后只重启了一次游戏，没有改 DLL，
没有用 Computer Use。新人物卡采用标准 head／skin GUID 引用，移除旧 source
显示记录；游戏保存并重载后 source overlay 仍为 inactive。

注意：`ChaControl.ChangeHeadAsync` 使用 `createName="ct_head"` 装入所选 prefab，
不能用运行时 GameObject 根名称等于包内 prefab 名来证明来源。验收直接
检查真实 renderer 的完整网格、蒙皮、bindpose 和法线数组等于序列化资产。

## 有限游戏收尾

`accept_placement_asset.py` 在一次同帧 `o_head`／`o_body_cf` 导出中通过：
实际源数组等于新包，源身体摘要与候选引用相同，source overlay 关闭，
加载后的中性头部等于评审摆放，28 点接口实际对齐。Raw BakeMesh 的
scale-free TRS 分支与实际骨矩阵 LBS 逐顶点核对；不是盲选坐标转换候选。
详细浮点误差和摘要见 [轻量记录](hs2_head_placement_game_receipt.json)。

游戏原生三视角图位于 HS2Mod ignored
`artifacts/chenger/placement_native_20261008/native_sheet.png`。
游戏最终人物卡为
`E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/程儿_缩小低头底模_游戏保存_20261008.png`，
已通过 native full-card 重载；游戏保持打开，显示这份底模。
更新前人物保存在同目录 `程儿_摆放更新前_20261008.png`。

**本次完成程儿这一摆放、中性卡及实际 BP 身体的资产接入。** 色差、贴图、
眼部／表情和任意滑杆、ABMX 组合仍未完成；整体通用接口目标保持 active。
其他已存在的 attachment／rim／C# 草稿保留，未混入本次提交。

## 重现

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.export_game_mesh --manifest outputs/chenger_mica_review_20261007/raw/manifest.json --image-index 1 --head-local --source-obj C:/Users/13666/Workspace/smirk/assets/head_template.obj --source-mask C:/Users/13666/Workspace/smirk/assets/FLAME_masks/FLAME_masks.pkl --out outputs/native_head_20261008/source_img002_new.json
.venv/Scripts/python.exe -m tools.native_head.build --source outputs/native_head_20261008/source_img002_new.json --audit outputs/head_base_audit_20261007/prefab_contract_v1.json --scale 9.851049 --translation 0.00339057157 0.458031476 0.603983462 --authored-neck outputs/native_head_20261007/placement_review_final_v1/chenger/size_and_pitch/geometry.npz --plain-skin --native-eyes --out outputs/native_head_20261008/placement_native_new
.venv/Scripts/python.exe -m tools.native_head.accept_placement_asset --candidate outputs/native_head_20261007/placement_review_final_v1/chenger/size_and_pitch --built outputs/native_head_20261008/placement_native_v2 --source outputs/native_head_20261008/source_img002.json --game C:/Users/13666/Workspace/HS2Mod/artifacts/chenger/placement_native_20261008/native_head_body.json --state C:/Users/13666/Workspace/HS2Mod/artifacts/chenger/placement_native_20261008/native_state.json --installed E:/HoneySelect2_ArcticFox/mods/Codex/Chenger.MICA.NativeHead.zipmod --sheet C:/Users/13666/Workspace/HS2Mod/artifacts/chenger/placement_native_20261008/native_sheet.png --out outputs/native_head_20261008/placement_native_v2/game_acceptance_new.json
```

最后一条核对当前已安装包和既有收尾输入；自行安装重建包时需相应更新
`--built` 和游戏导出。生成数据／游戏资产仍保持忽略，不上传许可资产。
