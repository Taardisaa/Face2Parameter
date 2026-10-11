# 程儿美术风格版 → HS2 人物制作

2026-10-11。用户明确目标是在 HS2 中复现程儿，不能把评估流水线当作终点。
目前第一版原生人物已保存；身份特征仍需继续制作。

## 固定输入与流水线

用户接受的图为 Codex `exec-089e3e48-93c1-4474-87f3-cc25608471bd.png`，
已复制到 ignored `outputs/chenger_stylized_arcface_20261011/accepted_style_target.png`。
它是 HS2 风格的正面程儿肖像，不是原始照片，也不认证真实三维骨相。

源码实际名称是 ArcFace，不是 AceFace。当前 `infer.py` 调用 `predict_set`，
先以 MTCNN 五点相似变换对齐，使用冻结 backbone 和 MLP 得到205维向量，
再由 `FaceData.set_from_vector` 按统计 min/max 还原54项原生脸部参数与掩码骨骼，
镜像补充右侧参数，并经 `HS2ABMX.exe` 序列化写入模板副本。
耳部其余5项、head/skin、身体、头发与妆容均继承模板，模型不预测这些外观。
本次没有 desmile、neutralize、关闭检测器、夹值或重训练。

ArcFace：`release/head_arcface.pth` + 缓存的 `w600k_r50.onnx`；
DINOv2：`release/head_dinov2_vits14.pth` + 本地缓存的 ViT-S/14。
两次推理均完整加载权重且检测到输入人脸。采用 release 权重避免自动切换
本地实验 checkpoint。ArcFace 的 config `aug_prob=1`，而 DINOv2 混合游戏与
写实化数据；因此当前制作使用 DINOv2 起点，但这并不保证生成图片的域匹配。

```powershell
.venv/Scripts/python.exe infer.py --config arcface --head release/head_arcface.pth --image outputs/chenger_stylized_arcface_20261011/accepted_style_target.png --template ../HS2Mod/artifacts/hair_game_catalog_20261009/fixed_native02.png --out outputs/chenger_stylized_arcface_20261011 --name chenger_stylized_arcface_v1
.venv/Scripts/python.exe infer.py --config dinov2_vits14 --head release/head_dinov2_vits14.pth --image outputs/chenger_stylized_arcface_20261011/accepted_style_target.png --template ../HS2Mod/artifacts/hair_game_catalog_20261009/fixed_native02.png --out outputs/chenger_stylized_arcface_20261011 --name chenger_stylized_dinov2_v1
```

## 游戏接入与人工外观配方

原桥接完整读卡没有 Maker 插件加载事件，实际54项滑杆进入游戏，但30项
预测 ABMX 没有加载。该失败已保留在 `audit_receipt.json`，随后修复 HS2API
事件链路；这不是用常量替换预测的理由。完整机制见
[HS2 记录](../../HS2Mod/docs/chenger_stylized_reconstruction.md)。
对应 HS2Mod 读卡修复提交：`1945659d14c38d1bec7e49a4bcca66ded6cf1d6e`。

修复后读取 DINOv2 原始卡，游戏自动加载全部预测脸部骨骼；随后原生调用：

- 头发 front：`mod/ymz_anyi_hair/301/326`，本次 runtime id `100042154`；
  依据六向实拍选择，manifest 在 HS2 hair catalog 的 `d016133207ecbb6f563d1e5b` 行。
- back：`mod/com.saint.baldhead/300/1`（ASH None），本次 runtime id `100010342`。
  front 已包含完整短发，此为明确空后发资产，不用 native back 0（实际是长后发）。
  side/option 均为0。使用前者完整短发本身，未禁用部分头发来伪造截图。
- front 材质：base `[.025,.018,.018,1]`、top `[.016,.013,.013,1]`、
  under `[.055,.04,.035,1]`、metallic0、smoothness.25。
- eyebrow 保留17，颜色 `[.022,.014,.014,1]`；eyelashes3，颜色 `[.015,.012,.012,1]`。
- pupil 保留23，颜色 `[.08,.055,.045,1]`，w/h均.57；highlight alpha.42。
- lip4，颜色 `[.52,.09,.18,.72]`、gloss.35；eyeshadow24，
  颜色 `[.25,.15,.17,.2]`、gloss0。
- 人工修正 native index22/23/48/49 为 `.48/.36/.82/.62`。
  这些是眼宽、眼高、嘴宽、嘴高的原生外观选择，不能算模型输出或拟合证明。
- top 原模板背心，color0 `[.04,.23,.1,1]`、gloss.15、metallic0；款式尚未匹配。
- 名称 `程儿 美术风格复现 v1`。保存前恢复正常 blink/非 mouth_fixed 的原运行设置。

模型的非单位 Length 骨骼为 Chin_rs、CheekLow_L/R、Nose_tip，它们的 Position
均为零；未在这些骨骼上引入混合 Length/Position 的人工改动。没有承诺
全动态形变或任意参数兼容。没有源头壳、输入肖像贴图或自定义脸部网格。

## 当前卡与后续制作

`E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_stylized_reconstruction_v1.png`
经原生写卡及重载，`final_snapshot.json` 与 `final_abmx.json` 保存重载状态；
`roundtrip.json` 记录 all snapshot 等值且 ABMX 数据无变更。
真实截图在 HS2Mod `artifacts/chenger_style_20261011/chenger_reconstruction_v1_reloaded.png`。
旧卡、原失败卡和输入均保留。

当前不是完整身份复现。需要继续制作目标的眉眼、唇形、面部比例，以及准确
发型轮廓/圆领衣服。模型输出只作为制作起点，不以运行成功收尾，不以新的
模型评估项目替代这些具体人物差距。输入、许可模型、提取资产和输出均继续 ignored。

## v2：继续制作眼部与唇形

v1 保留，当前卡为同一游戏目录 `Chenger_stylized_reconstruction_v2.png`，
名称 `程儿 美术风格复现 v2`。两侧 black_w/h 为 .42，虹膜颜色
`[.16,.105,.085,1]`，睫毛6；native index23/48/51/53 为 `.45/.88/.50/.62`。
其余使用上述 v1 配方。保存重载 all snapshot 和 ABMX modifier 等值，
`v2_native_state.json` 保存修正配方及回执；截图为 HS2Mod
`artifacts/chenger_style_20261011/chenger_reconstruction_v2.png`。
这一步改善发黑的瞳孔及唇形，但人物辨识度仍未完成。
