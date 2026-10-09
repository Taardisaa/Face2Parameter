# 颈部遮罩修正之后的受光差异

2026-10-08，当前原生包 0.1.6。遮罩已实际加载；肤色连续性仍未完成。
这次继续查已安装材质、编译 shader、已有几何捕获，以及六个只读材质值。
没有修改肤色、几何或游戏设置，没有采集新的截图或做参数扫描。

## 当前能确定什么

头身底色的合成材料 `create_skin_face` 和 `create_skin_body` 指向同一个
`Create/skin color` shader；`ChaControl.CreateFaceTexture/CreateBodyTexture`
都使用 `fileBody.skinColor`。这排除了“必然采用两套不同肤色合成算法”的
猜测，但不代表化妆、附加图层等所有输入已经相等。

身体 `detailId=0` 从 `ft_detail_b_00.unity3d` 加载 `_OcclusionMap`
(`cf_body_00_o`) 和 `_BumpMap2` (`cf_body_00_n`)。基础 `_BumpMap`
(`cf_body_base_n`) 则是 `mm_base` 绘制材质指向同一 detail bundle 的
外部 PPtr。不能把整个依赖归到 body skin 图上，也不能用未校准的游戏
纹理读回代替原始纹理数据。这次核对的是实际依赖 CAB、PathID 和捕获绑定。

## 所谓 AO 实际上是打包的受光输入

已安装 directional forward 分支中，头部 `_OcclusionMap` 位于 t10/s12，
身体位于 t7/s9。沿寄存器读取追踪：

- **R** 进入 `_Gloss`、`_ExGloss` 的计算，改变高光／粗糙度行为。
  头部 asm 第 328 行、身体第 240 行可直接看到 R 乘这两个常量；
  后续有 max/min 与其他遮罩合并，不能简化成“R 直接乘最后 RGB”。
- **G** 进入环境反射的遮蔽项，与 `_OcclusionStrength` 混合；
  头部第 358–362 行、身体第 265–269 行，随后分别在 479／401 行
  乘反射探针结果。本分支不能把它称为直接光整体变暗的系数。
- **B** 先乘 `_BumpScale2`，还参与细节着色和遮蔽修正。
  头部第 295–299 行、身体第 217–222 行分别读取 B。

当前头部把这张图替换成全图 `[255,255,0,255]`，身体接口则有原生 R/G
变化。两边的 `_OcclusionStrength` 实际均为 1，所以该输入差异不是被
强度关闭的无效分支。正确的局部修正应迁移各通道的原生输入；不能只取
灰度，也不能直接烘焙一个人为调暗的底色。

## 法线要区分两层

几何接口位置相合，但头部保留的 head2 几何法线与实际 BP 身体法线并不
完全一致。当前审查包括同位置的所有身体 UV 副本，不任取一个最近顶点。
它们在身体原生 UV 接缝上还可能具有不同贴图采样，因此下一版需要按
相邻三角形的侧别建立对应，而不是把所有副本强行平均。

切线空间法线贴图则是另一条路径。实际头部 `_BumpScale2=0`，身体为
`0.495652169`；头部第二法线未绑定，身体绑定了两张图。安装 shader 都按
`X = 2*(R*A)-1, Y = 2*G-1` 解码；法线强度作用于 XY，再重建 Z，基础和
细节按实际 shader 计算组合，最后进入网格切线基。

**接口原始 texel 上，身体两张法线图接近中性。** 因而不能仅凭“双法线图”
就断言它是主要成因。几何法线与打包 AO/高光图是已经明确的不连续输入；
各项在最终画面中的占比仍没有认证。

颈部 UV 已换成独立图块。如果迁移法线图，必须先得到源表面方向，再用
目标图块的切线基重新表达。直接复制 RGBA 会错误旋转法线；复制几何
法线也不能同时修好贴图法线。当前只读审查没有实施这两项修正。

## 下一项实现边界

1. 根据身体边界相邻三角形与 UV 副本，建立颈圈的贴图来源；逐通道迁移
   打包 AO/高光输入，在现有独立颈部图块内沿过渡带衰减到现有头部策略。
2. 让头侧接口的几何法线使用实际 BP 身体来源，并在过渡带内连续过渡。
   接口位置、脸型、下巴、后脑和身体顶点不移动。
3. 法线图迁移保留实际解码、强度和切线基语义；无法匹配的路径明确保持
   未完成。完成后才进行一个有限的游戏收尾，不开展截图／参数扫描。

## 重现与适用范围

```powershell
.venv/Scripts/python.exe -m tools.native_head.audit_neck_lighting --capture ../HS2Mod/artifacts/chenger/neck_mask_20261008/head_body.json --package outputs/appearance_20261008/neck_mask_final/Chenger.MICA.NativeHead.zipmod --scalars ../HS2Mod/artifacts/chenger/neck_mask_20261008/lighting_scalar_readback.json --out outputs/neck_lighting_followup_20261008/fresh
```

输出包含原生接口的全部 UV 副本、原始 RGBA、几何法线／切线、真实依赖
与两份 shader 反汇编。输入头部在实际注册 prefab 内定位，全部位置、
法线、UV、三角形和蒙皮数组必须与当前捕获逐值吻合，否则停止。

完整证据：`outputs/neck_lighting_followup_20261008/source_v4/audit.json`。
指纹概要：`hs2_neck_lighting_followup_receipt.json`。失败的早期输出目录保留。
这是当前 detailId=0、实际 BP 身体和 0.1.6 头部的源／资产审查；静态 bind
坐标、level-0 最近 texel 和 directional 分支不等于完整实际帧渲染认证。
没有宣称 AO／法线已修好，当前游戏仍是遮罩修正版。
