# 原生底模基础外观修正

2026-10-08。用户接受当前静态头形，要求修正颜色，并询问眼睛／嘴巴是否
缺少贴图。本次保持已接受的缩小 94%＋低头 3° 网格，单独处理外观。

## 两个实际机制

之前的 `--plain-skin` 是几何评审皮肤：原生 albedo 的中位颜色、平坦法线和
中性 AO，没有程儿本人的 albedo。MICA 输出形状系数及几何，不输出身份贴图。
原生半球眼已经有材质、UV 和虹膜／瞳孔贴图引用，但引用存在构建错误。

头部 bundle 的 streamed texture URI 格式是
`archive:/CAB-.../CAB-....resS`。旧构建只替换最后的资源文件名，未替换
所属 archive 目录；实际眼白、法线等读回为黑色。修正完整 URI 中的 CAB 名，
资源字节及眼球网格保持。检查基于序列化资源引用和实际纹理读取，未扫描参数。

另一个问题是 UV 属性最近表面转移不能保证解剖对应。原生 head2 的完整
albedo／normal／AO 虽然原像素复制成功，游戏里嘴唇色落到下巴、眼周色
落到下眼皮外，头顶还有错误色块。失败 `native_skin_v1` 和截图保留，
**不能把 `--native-skin` 宣称为完成的 FLAME 解剖纹理转换**。

## 当前基础皮肤

`build.py --authored-skin` 使用现有头部的连续正面柱面 UV，依据原解码器已绑定
SHA 的 FLAME 68 点 embedding 的外唇缘 48–59 画基础唇色。宽泛的 `lips`
mask 包含嘴周皮肤，不用它充当真实唇缘；失败 `authored_skin_v2` 保留。
背部接缝处保持均匀底色。原生 collar 的
几何／蒙皮／法线保持，UV 纳入同一基础皮肤图。使用游戏原来的皮肤合成、
skinColor 和绘制材质；normal 和 AO 暂时保持几何评审的中性输入。

安装版 `Skin True Face` 的 UV1 眉毛高度差分乘 `1-color.b`，不乘眉毛颜色
alpha；故关闭眉毛颜色仍会留旧 atlas 的眉毛凹凸。基础皮肤将 vertex color
G/B 门设为 1，关闭旧 UV1 的 nipple／brow 高度分支；关闭旧 atlas 门控的
`_DetailNormalMapScale`／`_Riality`。几何法线和所有顶点保持，真正的眉骨
仍来自模型。未来贴图眉毛需要正确 atlas／显式纹理，不是假称兼容旧 UV。

这是一套通用基础肤色／唇色，**不是照片贴图、不是推断的程儿肤色，也不是
毛孔／眼周／眉毛／妆容完整复原**。背部均匀色的 UV 接缝不适用于直接换上
任意带花纹贴图。照片 albedo 及解剖 atlas 注册仍需单独实现。

人物卡使用 [可重放的 MCP 设置](../tools/native_head/appearance_profile.json)：
较小的虹膜／瞳孔、淡眼白、统一 body.skinColor、降低高光、关闭金属感和
晒痕。这些是外观设计选择，不是模型输出参数或经验拟合的游戏计算逻辑。
安装版 `ChaControl.ChangeEyesWH/ChangeBlackEyesWH` 对 rate 做实际 UV-scale
Lerp，`CreateFaceTexture` 使用 body.skinColor 合成头部皮肤。

原生睫毛、眼膜及牙齿／舌头仍未迁移；闭口唇部颜色不代表口腔已经完整。
表情、gaze 和所有滑杆组合没有在本次认证。
最终图仍可见头顶局部明暗斑和后侧分面阴影，其原因尚未定位；不把基础
颜色修正宣称为完整材质合格或人物相似度认证。

以上为 v4 历史交付。后续已定位头顶圆斑及眼旁明暗区是原生区域遮罩与
新 UV 错配；修正及当前交付见 [区域遮罩修正](hs2_head_region_mask_fix.md)。
后側分面问题仍独立开放。

之后的原生底色迁移与可复用固定区域，见
[原生皮肤适配](hs2_native_skin_retarget.md)。它保留原生底色像素，使用显式
区域／UV 对应；本文件的均匀肤色方案保留为历史。

## 重现与证据

```powershell
.venv/Scripts/python.exe -m tools.native_head.build --source outputs/native_head_20261008/source_img002.json --audit outputs/head_base_audit_20261007/prefab_contract_v1.json --scale 9.851049 --translation 0.00339057157 0.458031476 0.603983462 --authored-neck outputs/native_head_20261007/placement_review_final_v1/chenger/size_and_pitch/geometry.npz --authored-skin --native-eyes --out outputs/appearance_20261008/authored_skin_new
```

实际输出 `outputs/appearance_20261008/authored_skin_v4`。三个实际 mesh 的
positions／triangles／normals／骨权重／bindpose 与前版逐项一致；UV／tangent
允许外观修正。序列化内部 streamed texture 全部引用新 archive 和实际资源。
静态结果位于该目录 `appearance_static_checks.json`。

```powershell
.venv/Scripts/python.exe -m tools.native_head.verify_appearance_assets --previous outputs/native_head_20261008/placement_native_v2 --built outputs/appearance_20261008/authored_skin_v4 --out outputs/appearance_20261008/authored_skin_v4/appearance_static_checks_new.json
```

游戏收尾、保存重载和当前状态见相邻 `hs2_head_basic_appearance_receipt.json`；
游戏资产及生成图片保持本地 ignored，不上传许可资产。原生贴图失败版保留。
