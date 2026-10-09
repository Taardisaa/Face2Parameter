# 当前脖子色差：安装资产与着色器审查

2026-10-08。对 0.1.5 程儿底模与已经捕获的实际 BP 身体进行只读审查。
没有调整肤色、几何或颈部材质，也没有进行截图搜索或参数扫描。
结论是**底色迁移已经成立，但颈部受光输入仍不连续**。

## 已确认的配置错误

之前为消除原生 atlas 错位的头顶圆斑／眼外侧阴影，把整张 `_NailMask`
改成 `[0,255,0,255]`。这是暂用的均匀外观策略，不能称为完整原生皮肤配置。
原生 head2 和实际身体在同一接缝圈的遮罩均为 `[0,0,0,153]`。
原生颈部的 G=0、A=0.6 都在这次统一化中丢失。

从实际头部私有包和安装版 `mm_base.unity3d` 的身体材质指针出发，提取
实际编译 D3D11 着色器，解压其程序，使用系统 `D3DDisassemble` 反汇编。
两边的 directional forward 分支均从 UV0 读取 `_NailMask`，并把其 **A 通道
乘入最后以 `_Translucency` 加到输出的皮肤透光项**。它不是最终透明度；
该分支输出 A=1。均匀遮罩会直接改变这一项，即使底图完全相同。
这条路径也不依赖已经关闭的微细节开关。

G 通道控制微细节及高光分支；但当前头部 `_DetailNormalMapScale=0`、
`_Riality=0`，序列化 `_FresnelSetting.w=0`，不能把所有涉及 G 的公式都
说成当前头部正在贡献色差。应区分实际打开的分支。

## 另外两项真实差异

- **阴影输入不同。** 当前头部 `_OcclusionMap` 是中性常量；身体 detailId=0
  从 `ft_detail_b_00.unity3d/cf_body_00_o` 取得带颈部明暗的 AO。游戏的
  `CreateBodyTexture` 由 detail 列表加载它，不能从 body skinId 列表猜。
- **接口法线不完全一致。** 接口位置已经相合，但当前头侧保留的原生 head2
  几何法线与实际 BP 身体对应顶点存在差别。审查考虑全部重合的身体 UV
  副本，而不是任取一个最近顶点。保持前版头部法线不变，不等于匹配 BP 法线。

还有身体基础／细节法线、两种皮肤 shader 的差别需要处理。本次没有分解
各项在最后图像中的占比，不宣称遮罩是唯一原因或色差已修好。

## 排除和更正

实际接缝 UV 对应的原生头、身底图 RGB 已很接近；头身也共用
`fileBody.skinColor` 和相同的 gloss 设置路径。分别盲调两边肤色会掩盖输入
差异，并破坏其他照明下的表现。

先前根据资产内 `_BumpScale2=1` 推测头部还有第二张法线图，不能成立：
现有运行时绑定记录明确显示头部 **`_BumpMap2=null`**。不能只看存储的强度
值就认定这张图参与了计算。身体则确实绑定 `cf_body_00_n`。

## 修正应如何落地

先限定在已保留的原生颈圈及新建过渡带，恢复原生颈部的遮罩输入，并向
当前头部配置平滑过渡。区域来自网格来源／接口拓扑，不按程儿坐标临时猜。
制作贴图时必须检查颈部和脸部是否共享目标 UV 像素，不能为了补颈部再
把旧 atlas 图案覆盖到额头和眼外侧。

接着按实际身体接口传递 AO 与法线：AO 的多个通道需沿实际 shader 用法
处理，几何法线与切线空间法线必须区分。只改接口受光，不移动已接受的
脸、下巴、后脑或身体顶点。仅复制底图或接缝 RGB 不足以完成它。

这是一份已完成的成因审查；上述局部材质修复尚未实现。

## 可重现依据

```powershell
.venv/Scripts/python.exe -m tools.native_head.audit_neck_shading --capture ../HS2Mod/artifacts/chenger/native_skin_retarget_20261008/final_v6/head_body.json --inputs ../HS2Mod/artifacts/chenger/native_skin_retarget_20261008/manifest.json --package ../HS2Mod/artifacts/chenger/neck_mask_20261008/before_0.1.5.zipmod --out outputs/neck_shading_audit_20261008/fresh
```

入口不调用游戏。输入为现有几何捕获、已提取且校验指纹的原生皮肤图、
当前 zipmod 和安装身体资产。接口用声明的 bindpose 映射到支持骨骼坐标，
必须实际吻合，否则拒绝审查；没有拟合比例或修正颜色。
输出原图逐点记录、全部重合顶点的法线对照、真实材质绑定、shader texture
寄存器与常量偏移，以及两份源 bytecode 的反汇编。

完整本地证据位于 `outputs/neck_shading_audit_20261008/source_final/`。
指纹与概要见 `hs2_neck_shading_audit_receipt.json`。
提取的着色器、贴图和网格保持 ignored；Git 只记录代码、机制和出处。
点纹理读数是源 RGBA8 的 level-0 最近 texel，不等同过滤后的 GPU 采样。
当前照明 variant 未独立确定；本次证明的是明确编译分支中的输入用途，
不认证完整帧渲染或某项色差占比。

以上审查对应修正前 0.1.5。0.1.6 已隔离颈部 UV 并修正遮罩，安装路径内容
已改变；重现本次成因审查应使用上述保存的 0.1.5 包。新包的对应检查见
[颈部遮罩修正](hs2_neck_mask_transition.md)，AO／接口法线仍未完成。
