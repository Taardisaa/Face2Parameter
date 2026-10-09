# 原生口腔与人物表情接入

2026-10-08。用户重新明确需要人物做表情；不能继续把现有静态头模当作
完整人物交付。本项完成源码和已安装资产审查、保留原生供体数据，尚未
实现口腔接合或表情重定向，游戏资产未更新。

## 原生资产的组成

已安装原生 head2 的实际 render prefab 包含：

- `o_head`：外部脸和口内相关表面，包含嘴部表情通道。
- `o_tooth`：独立牙齿网格，使用自己的材质、UV 和表情通道。
- `o_tang`：独立舌头网格，同样有自己的材质、UV 和表情通道。
- `o_namida`：原生嘴部 controller 还引用此目标，不能遗漏其通道合同。

牙齿和舌头都实际蒙皮到 `cf_J_MouthCavity`。它们的张嘴运动还来自
blendshape，不是只靠一个下颌骨旋转。完整顶点、法线、切线、UV、权重、
bindpose、全部帧和 Close/Open 表均已从供体保存到忽略目录。

## 游戏如何驱动

`ChaControl.ChangeHeadAsync` 获取 `FaceBlendShape`，将其 `MouthCtrl`
接给人物。`FaceBlendShape` 位于安装版 **IL.dll**，在 late update 调用
`FBSCtrlMouth.CalcBlend(voiceValue)`，继而进入 `FBSBase.CalculateBlendShape`。
它依据 pattern、OpenMin/OpenMax、FixedRate 和过渡状态，为每个目标
分别计算 Close/Open 权重，并调用 `SetBlendShapeWeight`。

不同 mesh 的 Close/Open 索引不一样。`ChaControl` 还负责舌头状态、
嘴宽调整骨骼和人物状态初始化，不能用一套自制 jaw-angle 增益替代。
原生头的默认闭口帧也存在非零位移；不能假设所有表情权重清零就是
原生闭嘴状态。迁移须明确以实际闭口参考和 B 的既定中性形状为基准。

## 当前安装包的实际缺口

对当前私有 prefab `p_cf_head_chenger_mica` 按 ancestry 定位，排除了
bundle 中仍包含的其他原生头和 hit 对象：

- `o_tooth`、`o_tang` 的三角形数和 blendshape channel 数均为零。
- `o_head` 没有表情 channel。
- `FaceBlendShape` 禁用，眼、眉、嘴三个 target 表均清空。

这些对象名仍在包里，不表示部件已接入。FLAME 原输出也没有独立的原生
牙齿、舌头；嘴部边界或唇色贴图不等于完整口腔。

## 推荐的原生接入路线

1. 建立原生 A 与导入 B 的嘴部三维表面对应：嘴角、内外唇缘、上/下唇和
   口内连接边界。把供体三角形/重心坐标及方向转换保存为可复用数据。
   已有二维底色 UV 锚点只能作为来源之一，不能直接冒充完整三维对应。
2. 复用原生牙齿、舌头及必要的口内表面，按 B 的实际口部坐标和边界
   制作适配；保留独立材质、原始帧和蒙皮语义。不要靠改 B 的外部脸型
   去容纳供体，也不能只恢复三角形便称已经接上。
3. 将 A 的嘴部表情形变重定向成 B 的 blendshape。三维对应可以共用，
   但顶点拓扑不同，不能复制 delta 的索引。B 的中性身份保持，表情中的
   唇缘、嘴角、下巴与口内表面同步；输出 normal/tangent 帧的合同也完整
   保留。不能以一个张嘴姿态替代原生全部已声明 pattern。
4. 接回原生 controller 与各 renderer 的目标/索引、骨骼和材质依赖。
   当前整个 FaceBlendShape 被关闭，直接启用不是修复；需检查眼眉口及
   泪液等所有引用，不用删除原生目标或关闭功能来掩盖不匹配。
5. 协调贴图区：当前颈部独立图块占用先前从外表 UV 中排除的区域，新增
   口内表面不能不加分析就占回同一区域。按实际语义/拓扑分割；单独一个
   UV 矩形不足以识别所有口内面。

控制器与原生供体有明确来源，能够复用。真正仍待实现的是 A→B 的
解剖/形变对应和部件接合；现有纹理映射尚未自动解决这些问题。

## 重现与证据

```powershell
.venv/Scripts/python.exe -m tools.native_head.audit_oral_integration --reference outputs/model_bridge_20261007/oral_asset_audit_v2.json --package E:/HoneySelect2_ArcticFox/mods/Codex/Chenger.MICA.NativeHead.zipmod --build-source tools/native_head/build.py --out outputs/oral_integration_20261008/fresh
```

当前结果：`outputs/oral_integration_20261008/source_v2/audit.json`。报告校验
已安装 Assembly-CSharp/IL.dll、旧反编译来源和原生 bundle 指纹，再读取
当前 zipmod 的实际 prefab；供体数据和详细数字保持在忽略目录。
这是静态源码/资产证据，不声称新口腔已在游戏中加载或表情兼容。
