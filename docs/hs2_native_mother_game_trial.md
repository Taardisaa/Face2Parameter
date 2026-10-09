# 原生母版与程儿身份：游戏试用记录

2026-10-08，用户要求将局部修正版作为真正底模导入，再应用程儿的
MICA–FLAME 输出。本轮已实现这两项游戏试用，当前可见为程儿 v3。
原先私有程儿 0.1.6、此前人物卡和启动前完整状态均另存保留。

## 实际加载路径

完整 `local_contacts_v16` 闭嘴参考重新经过八部件、稀疏表情帧、法线／
切线、原生中性骨骼绑定和完整 bundle 写回。新增两个独立 Sideloader
GUID、头型及配套皮肤条目；通过普通人物卡、UniversalAutoResolver 和
`ChaControl.Reload/ChangeHead` 加载。不是挂接外部头套或隐藏原生部件。
SourceHead 的状态为 inactive，人物卡没有嵌入 source-head 记录。

母版 `codex.native.flame_native_mother_v16`；程儿试用版
`codex.native.flame_native_chenger_v3`。动态本地 ID 仅见证据，不硬编码进
生成器；人物卡用 GUID/Slot 注册引用。实际完整头资产仍是原生女性 02
号拓扑、UV、蒙皮、八部件和原始控制器／材质／纹理引用，没有删除面。

安装版 Sideloader 18.2 的 `Awake -> LoadModsFromDirectories` 在启动时扫描
注册资产；本轮保存原状态后只重启一次。没有使用 Computer Use。
静态母版适配后沿用原生 renderer 的运行时 bounds，使截图自动取景过宽；
最终头部截图用实际 BakeMesh 范围计算相机目标，没有修改人物来适应截图。

## 身份传递的数学与实际缺口

读取原始 `chenger_mica_review_20261007/raw/manifest.json` 的图像索引 1，
对应 `img-002.jpg`，不重新推理、训练或修改 MICA 的 300 维系数。
原始 FLAME 默认眼／颈姿态为零；原始输出与源码公式
`v_template + shapedirs[:, :, :300] @ beta` 一致。

固定一次母版／FLAME 位移对应：FLAME 头壳样本的 Delaunay 四面体内使用
P1 重心插值；凸包外使用已记录的耳／眼区域和方向约束下的曲面重心
插值。这个四面体是位移场的插值支架，没有替换原生网格，也没有变更
FLAME 面连接。母版原有几何偏移保留。物理眼开口沿用嵌入地标曲线，
颈圈硬锁定并在限定拓扑环带作位移修正。最终存储
`[4439, 3, 300]` 原生身份形状基，人物几何是母版加系数加权位移。
眼球位置／尺寸按当前身份参考和原始 FLAME 关节回归器适配。

**这些是新资产的确定性对应／插值，不是游戏滑杆，也不是两套连续曲面
完全等价的证明。** 全形状分量路径在 `chenger_identity_v2` 保留，闭嘴
薄层产生穿插，失败收据和图形数据没有删除或改判通过。

**游戏 v3 是明确不完整的嘴部诊断试用版。** 原生闭嘴唇层和口腔共用
FLAME 20 个嘴部地标位移的仿射分量，外圈平滑接回完整位移场。这保留
薄层顺序，但没有传递非仿射 FLAME 唇形残差，不能声称完整还原嘴唇。
当前代码默认保留完整位移；只有显式 `--mouth-detail-preview` 才启用此
诊断策略。它不作为交付方案，后续需要解决完整唇形与原生闭嘴表情的
耦合，而不能把仿射过滤当作已完成。

## 最终证据与限制

母版和程儿 v3 均通过完整静态头壳“无新增穿插”检查；原供体既有穿插
仍单独记录。原始八部件数组、面、UV／缺失 UV1 通道、权重、绑定矩阵、
骨骼名及表情通道在实际游戏加载后与打包数组一致。缺失 UV1 的 NPZ
便利零数组与 Unity 的空通道按原始 channel 描述核对，没有补造 UV。

游戏实景揭示了尚未解决的后脑勺到脖子轮廓台阶，以及皮肤受光／色差
和部分细节不自然。仅锁住原始颈圈不保证整个改造后的后颈轮廓自然。
全部原生表情帧仍在；本轮没有认证所有表情、注视、任意滑杆或 ABMX。
当前人物卡头部滑杆为 0.5，保留的 ABMX 条目只涉及胸部。

静态数组／几何来源、运行时读取、人物卡保存再加载成功等收据位于
ignored `outputs/native_mother_template_20261008/`；最终原生截图／实际
几何位于 HS2Mod ignored `artifacts/native_mother_20261008/`。本地路径、
SHA 与动态 ID 摘要见 [来源清单](hs2_native_mother_game_trial.json)。

人物卡：

- `UserData/chara/female/Codex/FLAME_native_mother_v16_01.png`
- `UserData/chara/female/Codex/Chenger_native_topology_v3_game_01.png`
- 恢复原状态：`UserData/chara/female/Codex/pre_native_mother_20261008_01.png`

## 可重现命令

在 Face2Parameter 根目录，用本仓库 `.venv/Scripts/python.exe`，每个输出
目录必须全新。全部大型／授权资产保持在既有 ignored 位置。

```powershell
$root = 'outputs/native_mother_template_20261008'
.venv/Scripts/python.exe -m tools.native_head.mother_component_adaptation --inputs "$root/inputs_v2" --candidate "$root/local_contacts_v16" --out "$root/components_fresh"
.venv/Scripts/python.exe -m tools.native_head.mother_reference_bindings --inputs "$root/inputs_v2" --components "$root/components_fresh" --out "$root/bindings_fresh"
.venv/Scripts/python.exe -m tools.native_head.mother_bundle_candidate --inputs "$root/inputs_v2" --bindings "$root/bindings_fresh" --asset-key flame_native_mother_v16 --out "$root/bundle_fresh"
.venv/Scripts/python.exe -m tools.native_head.mother_registration --bundle "$root/bundle_fresh" --title 'FLAME 原生母版' --out "$root/package_fresh"
.venv/Scripts/python.exe -m tools.native_head.mother_identity --inputs "$root/inputs_v2" --mother "$root/local_contacts_v16" --manifest outputs/chenger_mica_review_20261007/raw/manifest.json --image-index 1 --out "$root/full_identity_fresh"
# 明确不完整的闭嘴细节诊断；不是完整身份转换的默认行为。
.venv/Scripts/python.exe -m tools.native_head.mother_identity --inputs "$root/inputs_v2" --mother "$root/local_contacts_v16" --manifest outputs/chenger_mica_review_20261007/raw/manifest.json --image-index 1 --mouth-detail-preview --out "$root/preview_identity_fresh"
```

身份候选同样执行 component_adaptation、reference_bindings、bundle_candidate
和 registration，使用独立 asset-key，再用 `mother_card` 写 GUID 人物卡。
`mother_game_acceptance` 对最终实际游戏几何进行只读核对。
12 项身份／现有完整资产制作数学合同检查通过。显式诊断重放与已加载
v3 全部候选数组一致；不是通过参数扫描调出来的身份。
