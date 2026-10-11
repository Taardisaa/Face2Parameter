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

## v3：按部位继续人物制作

逐项真人/美术图/v2差异与原生配方见
[身份制作记录](../../HS2Mod/docs/chenger_identity_authoring.md)。当前人物卡
为游戏Codex目录 `Chenger_stylized_reconstruction_v3_final.png`，v2和v3均保留。
完整当前参数与ABMX在ignored输出 `v3_native_state.json`；初次选定候选的
安装版副本计算顶点为 `v3_source_forward.json`。最终嘴部3项的修正另有记录，
没有冒称它们也是最初forward的输入。人物仍未通过身份相似度完成判断。

原生眉/唇纹理通过ChaList实际源行提取，不猜测ID：

```powershell
.venv/Scripts/python.exe -m tools.character_authoring.source_texture_sheet --category st_eyebrow --out outputs/chenger_brow_source_fresh
.venv/Scripts/python.exe -m tools.character_authoring.source_texture_sheet --category st_lip --out outputs/chenger_lip_source_fresh
```

工具保存原始RGBA、原列表行、bundle/raw SHA，检查图按纹理实际通道显示alpha
或RGB；不会把带全红RGB的透明唇彩错误显示为空白。无纹理ID显式保留。
这是素材源检查，不模拟运行shader，不把裁剪检查图当作游戏渲染。
本轮实际输出为brow_source_v3_rgba与lip_source_v3_alpha。
# v4 identity authoring continuation

## v8 eye/brow/lip proportions

Actual game card Chenger_stylized_reconstruction_v8.png in the Codex female folder.
Native22=.62,23=.57,48=.65,51=.47; brow13; pupilcolor[.23,.15,.12,1];
both blackW/H=.32; lip3 color[.46,.23,.28,.54], gloss.56. ABMX unchanged.
Full before/after, ABMX, patch and successful snapshot/bone reload equality at
`outputs/chenger_stylized_arcface_20261011/v8_native_state.json`.
HS2Mod actual two-view captures `chenger_v8_final_0/1.png` retain full capture JSON.
Installed ChaControl writes pupil and black controls into shader layout lerps;
card values are not pixel radii and their quotient is not a pupil/iris size ratio.
No complete eye-shader reconstruction is asserted. Reference-photo/style review
motivated manual proportions. Identity goal remains active: hair dimensions,
crisp upper eye-tail and profile nose/lip character remain unresolved.

## v7 lower face and outfit

Actual card `E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_stylized_reconstruction_v7.png`.
Native4=.30,6=.68,11=.49; other native and ABMX inherited from v6.
Head2 source keys confirm lower-face X scale, jaw Y and chin-tip Y directions.
Top2 is actual native p_cf_top_tsyatu1, category240, chara/00/fo_top_00.unity3d.
Color slots0/1 [.045,.24,.09,1], gloss.12, metallic0, pattern0 produce green crewneck.
Full before/after snapshot, ABMX, clothing patch and reload equality receipt:
`outputs/chenger_stylized_arcface_20261011/v7_native_state.json`.
HS2Mod `chenger_v7_final_0/1.png` shows actual fixed-camera final portrait, including
collar, with retained native capture metadata. All snapshot and ABMX reload equal;
clothing colors explicitly retained separately because ordinary snapshot omits them.
Identity goal remains active; precise eye-tail, hair dimensions and face likeness unresolved.

## v6 brow and eye-tail authoring

Actual v6 card in the game Codex folder, full state evidence at
`outputs/chenger_stylized_arcface_20261011/v6_native_state.json`.
Native59 and ABMX unchanged; brow5 RGBA [.025,.016,.014,1]; eyeshadow7
RGBA [.012,.008,.009,.85], gloss0. Eyebrow8 was too faint. Eyeshadow17
was rejected after native rendering showed thick lower outer eyelid shadow.
Rejected and final captures are both retained in HS2Mod artifacts.
Final outer eye-tail is darker but blurry; precise raised eyeliner remains unresolved.
Full native snapshot and ABMX bones match after card reload; identity goal remains active.
Source texture tool now accepts st_eyeshadow (native category316), exercised with
`.venv/Scripts/python.exe -m tools.character_authoring.source_texture_sheet --category st_eyeshadow --out outputs/chenger_stylized_arcface_20261011/eyeshadow_source_v6`.
This preserves source RGBA and hashes, not a shader/render appearance claim.
Previous 4a0622c mixed contemporaneously staged mouth work into this task's commit;
history/files preserved. Subsequent commits use explicit `--only` task paths.

## v5 continuation evidence

Actual card `E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_stylized_reconstruction_v5_final.png`.
Changed native36=.50,45=.55,46=.36, eyebrow8; all ABMX retained.
Hair back `mod/GMCC.hairpack01/300/5` runtime100020305, based on existing
six-view native02 captures, not thumbnails. Full state and exact source binding:
`outputs/chenger_stylized_arcface_20261011/v5_native_state.json`.
v5 card reload matches all snapshot and ABMX bones. v5_final differs only in
thumbnail from v5; v5 uses an overly tight diagnostic frame, final uses the
HS2Mod `chenger_v5_portrait_0.png` complete-hair portrait.
Recovered head2 Nose_rx, Nose_tip and NoseBridge animation keys underpin the
control directions; selected values remain manual identity authoring choices.
HS2Mod identity-authoring notes contain exact recipe, capture hashes and gaps.
Goal remains active: open forehead improved, but hair is short/voluminous,
wisps cross brow/eye, brow is too faint, eyeliner and overall likeness unresolved.

Actual Maker card: `E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_stylized_reconstruction_v4.png`.
Preserved v3 and all existing ABMX modifiers. Native changes from v3:
23=.46, 49=.60, 51=.52, 52=.68, 53=.64; lip3 RGBA [.46,.18,.23,.62],
gloss .56; eyelash7 RGBA [.008,.006,.006,1]; iris width/height .51.
These are explicit visual authoring choices using recovered native control paths,
not inferred photo ground truth or an evaluation success claim.
Source lash assets inspected with
`.venv/Scripts/python.exe -m tools.character_authoring.source_texture_sheet --category st_eyelash --out outputs/chenger_stylized_arcface_20261011/lash_source_v4`.
Full before/after snapshot and ABMX, plus reload equality receipt, are retained in
`outputs/chenger_stylized_arcface_20261011/v4_native_state.json`.
Both full snapshot and ABMX bone arrays match after native card reload.
HS2Mod `docs/chenger_identity_authoring.md` records source mechanism, captures,
and remaining visible gaps. Identity goal remains active: lip thickness improved,
but nose, brow, open forehead, eyeliner and outfit still differ.

## Profile and installed rendering follow-up (2026-10-11)

The user reports that the frontal likeness is now close, but the iris differs
from the accepted artwork and the profile lacks depth. Native front/90-degree
captures, including temporarily hidden hair, are retained under HS2Mod
`artifacts/chenger_style_20261011/chenger_depth_unoccluded_0/1.png`.
The visible nose tip projects; orbital/cheek-to-mouth transitions remain shallow.
This is a visual authoring observation, not recovered 3D truth from the frontal
artwork. v8 native morphs and ABMX remain unchanged in this rendering milestone.

Installed Hanmen 1.8.5 Face/Body shaders and DHH 1.15.30 SSS were applied through
their actual persistence/configuration paths. HS2Mod's
`plugins/HS2_McpBridge/MakerRenderStyleService.cs` exposes direct HTTP state,
shader/float authoring and actual main-camera capture. Independent offscreen
captures omit the main-camera DHH chain and cannot validate SSS. See HS2Mod
`docs/chenger_identity_authoring.md` for DLL hashes, native callbacks, capture
focus/temporal handling, failure evidence and saved-card acceptance.

A full-card load defect was also corrected in MakerRenderService: native Reload
consumes nowCoordinate, so ChangeNowCoordinate must precede it after reading the
new file coordinate. Same-session clothing equality previously masked this;
loading v8 from a fresh Maker now correctly selects green top2. v9 Hanmen card
was switched away to v8 and reloaded, confirming saved Face/Body shaders while
preserving the separate liquid material. DHH SSS remains a global setting, not
part of the character card. Renderer changes do not claim profile geometry or
iris matching complete.

Final v10 card: `E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_stylized_reconstruction_v10_HanmenSSS.png`.
Face Gloss=.34/SecondaryGloss=.25; Body Gloss=.32/SecondaryGloss=.25 and
VeinsOpacity=.12, through MaterialEditor persistent properties. Switching to v8
and back verifies full snapshot equality and all 42 ABMX bone entries, saved
shader/float values and the separate liquid shader. SSS is explicitly true in
the final runtime readback. `chenger_v10_front.png` was a retained SSS-disabled
diagnostic, while `chenger_v10_sss_front.png` and `chenger_v10_sss_side_open.png`
are the final effects-chain captures. Blink was temporarily disabled for the
profile and restored. No geometry change is claimed. The final state, native
ABMX and acceptance metadata are in ignored `v10_native_state.json`.

