# 导入头部的错位区域遮罩修正

2026-10-08。基础外观 v4 的头顶圆斑和眼睛外上方椭圆明暗区来自
原生 `cf_head_00_mask`，不是头模缺面、透明区域或没有头顶 UV。

## 安装版机制与修正

`data/hs2_head/shaders/AIT/Skin True Face.shader` 的安装版反汇编中：

- 1862 行以 UV0 读取 `_NailMask`。G 通道不仅控制微细节法线。
- 2116–2127 行将 G 的反值及 R/B 用于区域 gloss/metallic。
- 2141–2144 行将 G 用于附加光照项。因此 `_DetailNormalMapScale=0`
  和 `_Riality=0` 不会取消这张遮罩的全部作用。

v4 的原生游戏材质读回仍引用 `cf_head_00_mask`；该贴图有眼部黑椭圆、
唇部黑区和顶部黑圆区。新柱面 UV 不能沿用这些原生解剖分区。
`ChaControl.ChangeHeadAsync → InitBaseCustomTextureFace → SetFaceBaseMaterial`
从底模行的 MainAB/MatData 取得真正绘制材质，故修正在私有底模资产中完成。

`build.py --authored-skin` 现在沿实际 MatData 的 `_NailMask` PPtr 找到内部
贴图，替换为 RGBA `[0,255,0,255]` 的均匀皮肤区域，保留原生 shader、
肤色合成与皮肤高光功能。它不包含原生眼唇／头顶分区；这不是照片纹理推断。
微细节仍关闭。皮肤 normal/AO 已中性化；原生 weathering ranges/all 为 0。
头部、眼球、脖子几何和蒙皮不改。zipmod 外观版本为 0.1.4。

第一次 v5 输出被静态检查拒绝：UnityPy 在对象 save 后的 read_typetree
仍读取原始 reader 数据，随后的 CAB URI 改写覆盖了新像素。保留失败输出；
最终在 URI 改写之后写入遮罩，验证最终序列化材质引用及全部像素。

## 重现

```powershell
.venv/Scripts/python.exe -m tools.native_head.build --source outputs/native_head_20261008/source_img002.json --audit outputs/head_base_audit_20261007/prefab_contract_v1.json --scale 9.851049 --translation 0.00339057157 0.458031476 0.603983462 --authored-neck outputs/native_head_20261007/placement_review_final_v1/chenger/size_and_pitch/geometry.npz --authored-skin --native-eyes --out outputs/appearance_20261008/authored_skin_fresh
.venv/Scripts/python.exe -m tools.native_head.verify_appearance_assets --previous outputs/appearance_20261008/authored_skin_v4 --built outputs/appearance_20261008/authored_skin_fresh --out outputs/appearance_20261008/authored_skin_fresh/appearance_static_checks.json
```

本次输出 `outputs/appearance_20261008/authored_skin_v6`。所有已接受 mesh 的
位置、三角形、法线、骨权重和 bindpose 逐项不变；实际绘制材质所引用的
遮罩解码为均匀皮肤区域，资源引用完整。此前 v4 交付记录保留为历史。
游戏收尾证据另存 `hs2_head_region_mask_fix_receipt.json`。

本次只处理错误遮罩。后侧分面、完整肤质、身份 albedo、眉睫毛／口腔和
动态兼容性继续开放，不通过修改头形掩盖材质问题。
