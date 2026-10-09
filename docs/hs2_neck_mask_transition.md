# 原生颈部遮罩与独立贴图区

2026-10-08，私有底模 0.1.6。只完成先前源码审查中定位的区域遮罩回归，
完整 AO／法线衔接仍待实现。

当前已保留的原生颈圈和新建连接带不再使用全脸均匀遮罩。程序从实际身体
捕获确定网格、材质和接口，再沿安装身体材质的 PPtr 读取原生 `_NailMask`。
接口位置必须吻合，捕获的身体指纹必须与几何制作记录一致；当前来源支持
接口遮罩恒定的 `cf_body_00_mask`，其他遮罩会拒绝，不能默默套默认值。

颈圈使用该实际原生值。五行连接带按既有拓扑行号，从脸部现有遮罩到颈圈
作 smoothstep 过渡；这是一条明确的贴图制作规则，不是假装还原了某种游戏
自动接头算法。着色器仍使用游戏原有计算，尤其保留源码审查中确认的 A
通道透光用途。

## 为什么增加颈部贴图区

原 UV 中，连接带与部分源头部表面共享像素。只在这张图上改颈部遮罩，
也会影响其他区域，因此颈部使用单独的 UV 图块，放到当前外表面明确没有
使用的原生口腔 tile 中。原生眼球仍使用原来的独立网格／材质；没有把口腔
或眼球贴图当成脸部皮肤，也不宣称将来的口腔 graft 已兼容这个分配。

只为 `integrated_parent_face_ids < 0` 的原生颈圈／连接带展开 UV。使用
[xatlas Python bindings](https://github.com/mworchel/xatlas-python) 的原顶点
映射，必须保持每个三角形角的原顶点身份。脸部逐角 UV 完全保持。
xatlas 的绝对位置 epsilon 会忽略几条极薄连接三角形，因此仅在 UV 求解器
内部换用更小长度单位；写回游戏的顶点不缩放。压入小图块后，float32 UV
塌成点的薄三角形另放独立微图块，不删除面或移动顶点。

原生底图按旧颈部 UV 采样、重新烘到这个空图块。图块外的原生底图像素保持
字节相同；脸部没有重新着色。遮罩也只在这个图块改变，源面区仍保留当前
已经去掉圆斑的配置。图块带邻近像素延展，防止双线性采样读取其他区域。

本版不改顶点、三角形角顺序、几何法线、骨权重或 bindpose。AO／normal
仍使用先前的中性输入，身体资产不改，不能据此宣称色差已经完全消失。

## 重现

```powershell
.venv/Scripts/python.exe -m tools.native_head.build --source outputs/native_head_20261008/source_img002.json --audit outputs/head_base_audit_20261007/prefab_contract_v1.json --scale 9.851049 --translation 0.00339057157 0.458031476 0.603983462 --authored-neck outputs/native_head_20261007/placement_review_final_v1/chenger/size_and_pitch/geometry.npz --retarget-skin --native-eyes --neck-shader-capture ../HS2Mod/artifacts/chenger/native_skin_retarget_20261008/final_v6/head_body.json --out outputs/appearance_20261008/neck_mask_fresh
.venv/Scripts/python.exe -m tools.native_head.verify_appearance_assets --previous outputs/appearance_20261008/retarget_skin_v6 --built outputs/appearance_20261008/neck_mask_fresh --out outputs/appearance_20261008/neck_mask_fresh/appearance_static_checks.json
.venv/Scripts/python.exe -m tools.native_head.review_atlas --built outputs/appearance_20261008/neck_mask_fresh
```

实际输出 `outputs/appearance_20261008/neck_mask_final`。几何与蒙皮按完整三角形
角逐项比较；独立检查脸部 UV 和图块外底图像素保持，以及实际材质 PPtr
所指向的遮罩像素指纹。另有原生接口遮罩检查。记录见
`hs2_neck_mask_transition_receipt.json`；提取资产及生成图片继续 ignored。

真实游戏加载后，导出的实际源网格与打包资产各项相同，实际身体指纹保持，
新的遮罩已绑定且接口读回原生值。保存并重载
`程儿_颈部遮罩修正_20261008.png`，没有外部 source-head 依赖。运行时头身
`_Translucency` 均仍为原生 30；没有关闭透光来隐藏问题。
最终原生画面仍有头身明暗差异，本版不声称可见色差已经消除。

适用于现有五行连接带制作合同和固定原生 atlas 配置，不依赖程儿专属顶点
补丁；它不证明任意模型拓扑、任意身体皮肤或完整动态材质兼容。
