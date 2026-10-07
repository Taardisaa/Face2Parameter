# ABMX Apply 的四种位置迭代状态

本文依据本机 HS2ABMX 4.4.6 的 [Apply 反编译源码](../../HS2Mod/tools/ABMX_BoneModifier.cs)、[HasLength/HasPosition 定义](../../HS2Mod/tools/ABMX_BoneModifierData.cs) 和 [installed_contract_v2](../outputs/abmx_replay_20261005/installed_contract_v2.json)，由 [独立 NumPy replay](../tools/abmx_replay/model.py) 执行数值检查。新工具在 [classify.py](../tools/abmx_stability/classify.py)，不会改变原核心或 writer 生命周期。

## 为什么 combined 可能持续漂移

以下只讨论：固定已解析 modifier，固定所有 baseline/history，clean 起始 flags，无任何 external writer/H-scene/exclusion，历史位置通道非零、当前 local position 非 Unity approximately-zero，Length≥.1、历史长度>0；每次始终走 current-direction 分支。设当前 local position 为 p，缓存 `_lenBaseline` 为 b，Length 为 L，Position offset 为 d，R=bL>0。

源码每次先取**当次** `BoneTransform.localPosition` 归一化到半径 R，再叠加 d。因此 Length!=1 且 Position!=0 时：

`p[n+1] = R · p[n]/||p[n]|| + d`。

这通常不幂等。偏移并不是每次简单 `p += d`，也不是每次从 `_posBaseline` 重新计算；上一次偏移已改变下一次的归一化方向。即使 private cache 的每个 field、public modifier 数字始终不变，local position 与子骨世界坐标仍可能随 Apply 次数改变。

当 d≠0，令 D=||d||，u=d/D。精确算术下正向固定点 `p+=(R+D)u`；若 R>D，还有反向固定点 `p-=-(R-D)u`。正向附近方向扰动的倍率 `R/(R+D)<1`，反向的倍率 `R/(R-D)>1`。非共线起始方向在上述固定分支条件下逐渐向 d 方向靠拢；这属于条件数学结论，不能证明游戏已稳定或变形质量良好。d 很小时收缩极慢，早晚截图可呈现明显不同状态。

R=D 且正好反向会得到零位置，下一调用进入 fallback，已超出此证明；Unity 的 approximate-zero 和 float32 舍入也可能提前触发分支变化。新工具遇到这种状态会标 unsupported，不用长跑数值曲线冒充当前分支的稳定证明。正/反向共线固定点也说明不能宣称所有 combined 初始状态都会漂移。

## 四 regime 的合同

| regime | Length channel | Position channel | 位置映射与数学结论 |
|---|---|---|---|
| length_only | L!=1 | d=0 | `p'=R normalize(p)`；clean `_changedPosition=false` 时第一次达到半径 R，随后数学幂等 |
| position_only | L=1 | d!=0 | `_lenModForceUpdate=false` 时 `p'=_posBaseline+d`；第一次后数学幂等 |
| scale_rotation_only | L=1 | d=0 | clean position flags 时位置不变；scale/rotation 至少一项非 identity，第一次后数学幂等 |
| combined | L!=1 | d!=0 | `p'=R normalize(p)+d`；通常为依赖次数的方向迭代，上述固定点例外 |

四组可以同时带固定 scale/rotation；两者始终取 `baseline_scale × modifier_scale` 与 `baseline_q × Euler(delta)`，不会因为重复 Apply 自己累乘。子骨必须在 parent-first FK 后评估，父骨持续移动会改变子骨世界位置，即使子骨 local 固定。

“一次后数学幂等”不代表所有 Unity float32 位完全相同，也不代表整个游戏全脸已稳定。所有 baseline 及外部 local writer 假设都是必需条件。若 length-only 起始 `_changedPosition=true`，安装代码会**覆盖刚计算的长度结果为 `_posBaseline`**并清 flag，已不是表中 clean regime。identity 清理时 `_forceApply`→`_lenModForceUpdate` 的 restore 序列同样不能套用表中结论。

以下在本分类中明确 unsupported：Length<.1、H-scene、current 或历史位置通道 Unity approximately-zero、非正历史长度、初始 dirty/restore/pending-force flags、额外 modifier/rotation-excluded 分支、完全无 active channel 的 identity。不是声称这些分支永远无法建模，而是本次不能自动泛化为已验证稳定控制。

## 缓存、metadata 与 actual timeline

`runtime_baseline.frame_count` 是读出 wrapper 的时间信息。它变化不等于 `_lenBaseline/_positionBaseline/_posBaseline/_sclBaseline/_rotBaseline` 或 flags 变化。[private_difference](../tools/abmx_stability/classify.py) 分别记录 `changed_private_fields` 与 `changed_runtime_metadata_keys`；不能用整个 wrapper 字典不相等作为 cache fields 漂移的证据。

actual 验收要保留每骨完整初始 before/cache，逐调用 NumPy replay 完整 stopped trace，并检查 inter-call continuity。观察早/晚/far60 时按各骨 exact cursor prefix 实际次数选状态；settle_frames=5/18/60 仅描述采样等待，不证明每骨恰好执行这些次数，也不认证共同 N。

实际比较需同时报告：每骨 local position/rotation/scale 及 private **fields** 的真实差异、窗口之间的 source/history/flags/参数是否相同、全部非覆盖 ancestors/native59/expression/blendshape nuisance，以及实际整头几何。不能从局部数学公式推断 mesh/BakeMesh 全脸稳定，更不能推断谁写入了一个外部边界。

## 执行与门槛

```powershell
& ./.venv/Scripts/python.exe -B -m unittest discover -s tools/abmx_stability -p test_stability.py -v
& ./.venv/Scripts/python.exe -B tools/abmx_stability/analyze_trace.py TRACE.json `
  --contract outputs/abmx_replay_20261005/installed_contract_v2.json `
  --out outputs/abmx_stability_20261005/new_prediction.json --count 512
```

分类器的 `classify(first_candidate_event, long_count=512)` 只将实际 first before/cache 当初始条件，独立 NumPy 向前执行固定输入；不读取 actual after 来生成候选 local。输出 `actual_runtime_stability_proven=false`，并标注长跑是条件预测。外层 actual 审核必须调用 `source_contract` SHA/MVID/IL 验证，不能复制旧绿色证书。

actual whole-head replay维持固定最大 L2 / actual bbox diagonal ≤1e-5；local/cache 维持原有固定容差。actual temporal surface 继续使用既有采样/面积权重、无拟合 scale/affine 的合同和 RMS≤.002、P95≤.005（其他已有 max/normal gate 不放宽）。早/晚/far60 三个窗口分别验证并比较 early↔late、late↔far60、early↔far60；失败可如实报告，不能放宽阈值或从 observed-before replay 通过推出 temporal 稳定。

原生 render 的 settle 上限为 60，本次实际采集声明为 early=5、late=18、far60=60。旧 proposed far90 未实际采集或验收，不能把被 clamp 的 60 宣称为 90；Apply 次数依然由 observer 记录。

完成实际恢复且根代理确认输入 ready 后运行：

```powershell
& ./.venv/Scripts/python.exe -B tools/abmx_stability/run.py `
  ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_stability_v1/live_cases.json `
  --contract outputs/abmx_replay_20261005/installed_contract_v2.json `
  --out outputs/abmx_stability_20261005/new_actual --prediction-count 512
```

输入是 head2 的四个 `case.regime`，预声明 `protocol.regimes[{name,patches}]`，各 case 保留 earlier identity `source_history` 与完整 candidate trace，窗口为 early/late/far60。恢复 flags 和实际 before/after 参数记录必须匹配；不以未恢复 progress.json 宣告验收。输出完整 actual before/cache/replay、每窗口独立全头、actual 三对 temporal surface 与字段/metadata 差异。

## 2026-10-05 实际四 regime 结果

新批次已完成，恢复 state/expression/all modifiers 的 flags 均为 true，实际 before/after 记录严格相等。[完整 summary](../outputs/abmx_stability_20261005/actual_v1/summary.json) 绑定 live manifest SHA256 `0c1dda10567e0ef9e4b3c19a5bf48b27ccedf63d4123ae68e98b17846ebfb154`、预声明 protocol、installed contract、所有预测源文件；审核结束源文件未变化。

范围仅为 head2、59 个原生值循环 `.48/.52/.47/.53`、Chin/ChinTip/左右 CheekUp 四骨、同一历史与固定 moderate scale/rotation，early5/late18/far60 三窗。独立逐调用 NumPy replay → parent-first FK/LBS 的 actual 全头匹配 **12/12 通过**，各组最大误差/bbox diagonal 均小于 `9.905e-8`。这证明当前输入及实际调用次数的模型预测准确；不证明这些状态随时间稳定。

| regime | 全头匹配 | 三对 temporal gate | 实际每骨 active Apply prefix：early / late / far60 |
|---|---|---|---|
| scale_rotation_only | 3/3 | 3/3 | 6 / 25 / 86 |
| length_only | 3/3 | 3/3 | 6 / 27 / 89 |
| position_only | 3/3 | 3/3 | 7 / 27 / 89 |
| combined | 3/3 | **0/3** | 6 / 25 / 88 |

每一窗这四骨碰巧观察到相同次数，但未控制或认证 common N。前三组仅在本次所观察窗口内稳定，temporal RMS≤`7.246e-7`、P95≤`1.363e-6`；selected local position 三对差异均为零。

combined 的 early→late、late→far60、early→far60 表面 RMS/P95 分别为 `.003718/.010834`、`.010516/.026817`、`.013971/.037842`，全部超过原有 RMS `.002` / P95 `.005` 门槛。early→far60 的 selected local position 最大分量漂移，Chin 为 `.101034`、ChinTip 为 `.019503`、左右 CheekUp 各 `.150641`。三对比较中四骨全部 private fields 均保持不变；只有 wrapper 时间 metadata 等发生变化，不能将漂移归因于 cache fields 改写。

因此此实际 combined 配置不能作为稳定、与等待次数无关的参数控制；不能因全头 replay 通过而认证稳定，亦不能仅凭本批次推断其他 native/head/history 的 combined 结果。补充冻结审核逐一校验 source/baseline/三候选几何的 native59、frame/pose 和 SHA，并将 private-field 差异与 metadata 差异分别保存；报告在 [actual_v1_supplemental_freeze.json](../outputs/abmx_stability_20261005/actual_v1_supplemental_freeze.json)。人物相似度、全局无限时间稳定性、arbitrary candidate 和共同 N 仍未认证。

10 个合成测试已通过，含 actual after 不作为长跑输入的检查。`outputs/abmx_stability_20261005/reference_combined_prediction.json` 是对既有 head2 fresh stopped trace 全部调用重新 replay 后的**条件预测**，用于说明 current-direction 迭代；它没有测量新的 early/late/far60 temporal surface，也没有宣告实际稳定。

## 可选的稳定执行与首次逻辑目标

另设 logical combined 控制，保留其 **FIRST clean Apply** 的目标 local TRS，再将实际执行换成 L=1 的 position-only 分支。此举改变后续控制定义：稳定停在首次目标，而不继续迭代原 logical combined 的方向。编译器由独立几何代理提供，见 [stable lowering 合同](hs2_abmx_stable_lowering.md)；本工具不会改变已安装 ABMX 或向 private baseline 写值。

[lowered_target.py](../tools/abmx_stability/lowered_target.py) 只接受冻结 earlier history、declared native59 与 logical patches。它独立通过 NumPy native FK 分解 TRS，再用 earlier persistent `_lenBaseline/_positionBaseline` 生成 clean cache，对 logical patch 仅独立 replay 一次，parent-first FK/LBS 生成整头目标。compiler 的数值目标仅用于比较；candidate after、candidate expression 或调用次数不参与目标构建。目标使用 earlier source 的真实 expression/blendshape nuisance，因此实际窗口若这些状态不同，固定目标比较应拒绝。

既有原始 combined source-only artifact 的预检中，独立预测四骨首次 TRS 与 compiler 完全一致，earlier identity 原生全头 guard max/bbox=`9.425e-8`。证据与生成目标在 [original_source_only_target](../outputs/abmx_stability_20261005/original_source_only_target/source_freeze.json)。这只是该次冻结 source 条件预测；实际 lowered 验收另用下一批新历史，不能套用本预检作为新 runtime 证书。

新的实际批次必须独立冻结 compiler artifact 与执行 protocol，并在写入 physical patches 前捕获 identity guard。完成恢复后：[lowered.py](../tools/abmx_stability/lowered.py) 先冻结 source-only target，再读取 candidate，验证完整 trace、三窗全头各≤1e-5、三对 temporal 原有 gate，并比较三窗 actual 全头与 FIRST logical target 各≤1e-5。

```powershell
& ./.venv/Scripts/python.exe -B tools/abmx_stability/lowered.py NEW_live_cases.json `
  --artifact NEW_compiled_execution.json `
  --contract outputs/abmx_replay_20261005/installed_contract_v2.json `
  --out outputs/abmx_stability_20261005/NEW_lowered_actual
```

`manifest` 沿用实际 stability schema，唯一 case 的 `regime='position_only'`，有完整 `source_history`、`identity_capture.paired_geometry`、candidate trace 和 early/late/far60 三窗。协议的 `patches` 与 compiler `executed_patches` 严格相等，`logical_patches` 与 compiler 同名字段严格相等，native/profile/门槛及 common-null/false 明确声明。case source trace/geometry SHA 必须与 compiler history 输入一致；本审核仍要求 source→candidate 骨/Modifier 实例不变，不能把允许重建实例的 compiler guard 当作放宽本 actual trace scope。

16 个离线测试通过：原 10 个迭代分支测试，加 6 个目标/source/native guard 测试（含世界父变换移除、shear/reflection 拒绝、stale hash、native 改动、compiler 数值目标篡改）。新 actual lowered 数据未输入时，不预宣告 runtime 稳定、共同 N 或人物相似度。

## LOWERED V2 实际验证

root 新采集的 lowered_v2 已成功恢复全部 public state、expression、modifiers，before/after 严格一致。live manifest SHA256 为 `ab4fc82a6f1dffe665936913f3970c599a13d3286f7012b399adeff121c37aed`，fresh compiler artifact SHA256 为 `3e1cb1c04b49279df502288b4b52d1837f52e1ab29aa776166a15f97932c55c2`。输入、installed/native/编译器/独立审计源码及输出目标绑定在 [lowered_actual_v2 summary](../outputs/abmx_stability_20261005/lowered_actual_v2/summary.json)，审核结束时所有输入/算法 hash 仍匹配。

| 独立 actual gate | 结果 | 实际最大残差 |
|---|---|---|
| 完整逐调用 replay → FK/LBS 全头 | 3/3 | max/bbox `1.043e-7`，门槛 `1e-5` |
| 三窗 actual 对 source-only FIRST logical target 全头 | 3/3 | max/bbox `1.049e-7`，门槛 `1e-5` |
| 三对 actual temporal surface | 3/3 | RMS `9.653e-7` / P95 `1.814e-6`，原门槛 `.002/.005` |

actual physical active Apply prefix 每骨为 early **6**、late **26**、far60 **96**；完整 candidate observer 的 1052 个事件包含长 identity guard 计算等待期间的调用，不能将总数或 settle 数当作 active N。selected 四骨的 local position/rotation/scale 三对差异均为零，全部 private fields 三对也无变化。微小整头表面残差仍如实保留，未拟合 scale/affine 或放宽门槛。

这是 head2、同一 mixed-native59、fresh measured history、四骨 fixed logical/physical controls 的实例：稳定实际执行位置分支，在已观察三窗内同时保持首次 logical combined 的整头目标。先生成并保存 source-only target，之后才读 candidate trace/geometry；actual candidate after、其表达/调用次数未用于生成目标。该实例没有认证其他 head/native/history、任意 logical controls、共同 N、无限时间稳定性或人物相似度。原始 combined 四组 actual temporal 失败仍有效，并未因为 optional lowering 通过而改成绿色。

## 独立的 relative quality 检查

稳定性与 FIRST 目标匹配不等于可用外观。新增 [lowered_quality.py](../tools/abmx_stability/lowered_quality.py) 对同批三窗 actual，使用 **earlier identity source** 作同 native/head/renderer/source/topology 的 baseline；复用原 `analyze_mesh`、`baseline_comparison`、`quality_gate`、certified cross-mesh 和 surface 库，没有放宽门槛或修改上述已冻结实际证明。

```powershell
& ./.venv/Scripts/python.exe -B tools/abmx_stability/lowered_quality.py `
  ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_lowered_v2/live_cases.json `
  --replay-dir outputs/abmx_stability_20261005/lowered_actual_v2 `
  --established-config outputs/stateful_base_comparison_20261005/comparison_config.json `
  --out outputs/abmx_stability_20261005/NEW_lowered_quality
```

[lowered_quality_v1](../outputs/abmx_stability_20261005/lowered_quality_v1/summary.json) 实际 o_head relative quality **3/3 通过**：无新增退化、非相邻 self-crossing、normal reversal、edge ratio 越过 `[.5,2]`、area ratio 越过 `[.25,4]`，无新 reflected/singular bone-world frames，四骨 actual modifier scale 均为正。相对 identity 的 actual surface RMS 约 `.001427`、P95 约 `.001739`，既有 surface diagnostic 三窗也通过。shape 改动允许与 identity 不同，该 diagnostic 不是目标相似度。

基线绝对缺陷仍明确保留：168 self-crossings、1205 contacts、1 nonmanifold vertex、6 aspect warnings；三窗这些 head 计数相同。相对 gate 的通过不自动接受已有绝对缺陷，也未验证解剖/审美正确性。

此外 active certified head cross-mesh baseline 原有 1652 个 crossing pairs，候选新增 proper-crossing pairs 为 early **8**、late **10**、far60 **10**，涉及 eyebase/eyeshadow/tooth 与 o_head。这些是单独几何诊断；既有 **o_head** relative gate 本身未将其纳入通过/失败，不能把本次3/3宣传为完整脸部无穿插、可见部分已可用。模块表面的设计关系、shader/occlusion 与实际可见问题需另行校准，不能未经验证把新增 pairs 自动视为容许或忽略。

本分析不识别用户手动修改下巴所见问题的精确触发：当前实验说明选定 combined 分支的迭代漂移，以及一个稳定替代执行实例；manual restore/native rebuild、其他 flags/history/base 路径仍需按真实事件另外复现。

## InGameEvaluator 消费端迁移（独立于上述冻结证明）

[hs2_ingame_eval.py](../src/hs2_ingame_eval.py) 现在通过 [hs2_ingame_semantics.py](../src/hs2_ingame_semantics.py) 在 native59 写入、warm capture、评分前读取完整继承 ABMX、actual geometry/head source、sampler capabilities 与公开参数。此节描述消费端保护，未把 mock 测试、provider 存在或文件 SHA 当作运行认证。

| 显式模式 | 消费端行为 |
|---|---|
| `native_only`（默认） | 拒绝全 actor 任意 nonidentity ABMX、pending restore/force flags，以及 **所有已有 head-descendant modifier，包含 empty identity**。保留 body 的 clean identity modifier，未建立 baseline 本身不等于 pending。未知 source/cache/hierarchy 拒绝；绝不自动 reset/remove 用户 modifier。 |
| `installed_stateful_diagnostic` | 保留已识别继承状态，记录 native/ABMX/cache/source 和实际多视图；不认证 stateful mixed Length+Position、common N、稳定性或人物相似度。 |
| `native_radial_target_v1` | 必须显式注入具有 `prepare/apply/validate_capture/restore` 的生命周期 provider，并声明 logical sidecar；本模块没有内建 live provider，也不会回退至其他模式。 |

默认拒绝 empty facial modifier 有具体生命周期原因：installed `Reset → native UpdateShape → CollectBaseline` 可按历史 `_lenBaseline` 归一化 `_posBaseline`，即使公开 controls identity 也可能改变姿态。不能只凭 identity/clean flags 认证 no-effect。消费端使用 actual head-root 的完整 descendant hierarchy 与 modifier transform ID/name 检查 membership。root 的该实际观察与上文四骨稳定性实验是不同路径，不能把症状直接归因给同一分支。

`range_mode='native'` 严格 `[0,1]`；`range_mode='installed'` 需显式选择并证明 loaded SliderUnlocker 18.2 / registered sampler patches / current installed bounds，声明 profile 必须匹配实际。两模式均拒绝非有限值、错误维数、bool/非整数/重复/越界索引；不 clip，不跳过 requested frozen indices。原 card `p0` 原样保存，native 模式中的原 out-of-range 项只能通过不包含这些索引的 partial write 保留；完整请求含 frozen 项会明确失败。无 installed capability 时不会冒用扩展范围。

context 绑定 actor/head/root/source mesh hash、game metadata、能力/profile、native59、完整公开 ABMX 与 non-face public config、实际 private cache provenance。实际未提供 bridge MVID 时记录 unavailable；不捏造 MVID。外部变动使 context 失效。自己的合法 native 更新可刷新 head native TRS baselines（仅显式 diagnostic/provider 路径，因为 native_only 拒绝 head modifiers），有意义的 persistent `_lenBaseline/_positionBaseline` 仍须保持；native_only 的 non-head cache 是 immutable。wrapper `frame_count` 不当作 cache field。

同一次 `evaluate(..., sets=('a','b'))` 将两组 yaw 合并成 **一个** native `/maker/render` POST，固定共同 framing/freeze/settle 与 paired geometry。HTTP 配置放在 query（native POST body 留给 visibility variant），沿用 native 当前自动 framing，不调用旧 `aligned_framing`。输入 context、response、PNG/geometry SHA、actual native readback、同 frame/pose/visibility signature 与各 state/restoration flags 均在 scorer 初始化前保存检查。任何缺失或不匹配阻止评分。默认 beauty scorer 的结果始终标记 `beauty_is_likeness=false`、`runtime_certified=false`，不会当作肖像相似度。

provider 接口如下；prepare 只是冻结请求，不得把 native 写入前的旧 offset 当作 candidate baseline：

```python
plan = provider.prepare(context, candidate_native59, logical_controls=sidecar)
# plan: mode, input_context_binding_sha256, sampler_profile；JSON 可冻结。
execution = provider.apply(plan, boundary)
# 必须自行完成 native write → identity recovery/settle → stopped COMPLETE ALL30
# identity history → source-only compile → fresh guard → physical apply。
# accepted, plan_binding_sha256, model_source 和 guard_report 的绝对 JSON path/SHA。
proof = provider.validate_capture(execution, capture_evidence)
# accepted/source_bound=true，实际 report_bindings 列表；无 proof 不评分。
restoration = provider.restore(original_context, boundary)
# 移除自身创建的 facial modifiers 必须发生在原 native 恢复之前；
# 保留用户继承状态，返回 restored。消费端另外读取公开参数验证。
```

model/guard/actual report 的 SHA 只冻结证据；消费端不据此声称数学或运行认证。provider 必须拥有整个公开生命周期、30 骨 history/ID/source 校验和实际采样 proof；它不能接受 actual candidate after 作为 source-only target 输入。每个新 native59 都使旧 physical offset 失效。

离线边界测试运行：

```powershell
& ./.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_ingame_semantics.py -v
& ./.venv/Scripts/python.exe -B -m ruff check src/hs2_ingame_eval.py src/hs2_ingame_semantics.py tests/test_ingame_semantics.py
```

目前 **15 项 mock boundary 测试通过**，覆盖 native/installed/no-clipping、继承 mixed/empty facial 拒绝、stale source/history、单滑条中途失败、restore、联合 capture 失败-before-score、provider request/report 缺失阻止评分，以及显式 provider 仍不提升 runtime certification。mock 不能替代 live ordinary actor 验收；需要 root 用无 facial modifier、body clean identity、一次 declared native change 测试多视图与公开恢复。

`restore()` 只恢复本消费者拥有的公开写入并返回明确报告，不承诺 private cache exact restoration。外部 source/history/ABMX 改动时拒绝覆盖，恢复失败记录而不掩盖原错误。旧 user-dirty `scripts/hs2_optimize_ingame.py` 本轮保持原样：它自己的 before/after capture 仍绕过此消费端，也未增加新模式/provider CLI 参数；不能声称整个旧 CLI 已迁移。默认旧 evaluator 构造参数仍支持合法 native、无 facial modifier 的状态，继承 unsafe 状态在首次 native write/warm 前明确拒绝。

### 实际 consumer 验收与 face dependency 只读调查

root 的 [native_only_v1 实际报告](../../HS2Mod/artifacts/infrastructure_live_20261005/ingame_consumer_native_v1/report.json) 保留了默认拒绝：`cf_J_Mune00_t_L` 虽然公开 controls identity，存在 pending cache flags；首次 native write/render/scoring 之前拒绝，公开状态保持、没有 owned writes。不是 native_only 成功验收。独立显式 [diagnostic_v1 实际报告](../../HS2Mod/artifacts/infrastructure_live_20261005/ingame_consumer_diagnostic_v1/report.json) 完成一次 native change、一次联合 A/B capture、10 次只读 PNG scorer 调用和公开恢复；private cache 相同亦为该实例测量，不等于解剖/稳定性/相似度认证。

当前普通 actor **不支持默认 native_only 路径**；diagnostic 仅通过实际 transport/capture/order/public-restoration 验收。native_v1 实际 POST 数 **0**、评分/PNG 读取 **0**；diagnostic_v1 实际 **3 POST**（full59 native batch、1 joint render、仅恢复 index0 的 native batch）、10 PNG 读取。未隐式 reset 或豁免 body modifier。

上述保存证据/source SHA-256 精确如下，后续文件变化需重新绑定，不能沿用本次实际验收：

```text
native_only_v1/report.json  f3c46c1a7ad25afd70d9eccce3ebf8cdf55e78d07e5e4ea894222a5fc91da85a
diagnostic_v1/report.json   b91c48d6a46b165c91939165938780052e841fc2888437a4d286a3b7600c3bc5
src/hs2_ingame_eval.py      1ce743dea4cbad566372c9e4a6f900f669d845a593629ce3de8786c81d90fd10
src/hs2_ingame_semantics.py 8ab92659461d4faca5ac09189da834b7273f7d80eeddbbccfd641efbdd0258d0
native live collector      da251fbef8bdb67d862ab0aaa84b0a740f566026243ccb20609eec2f6758a64b
diagnostic live collector  560dc665da32d14a85213da4fe1a2408344606e56819d5ca30022714e0ca28b4
ABMX_BoneController.cs     40d617f917730352070e78a540957a2195a5568acb43c87e4329ca2c5346143b
ABMX_BoneModifier.cs       a374d69f192bcd07e6de6869dbe477e46fd5853e684a192e14eb6869370aca8e
```

collector 位于 `HS2Mod/tests/geometry_export/live_ingame_consumer.py`；两次运行其 modes/config 不同，source hash 不同，报告分别冻结了当时实际 source。controller hash 绑定本地反编译源；逐骨 installed Apply 的独立 IL 合同仍以此前 `installed_contract_v2.json` 为准，不能从 controller 文本 hash 追加未测试的方法运行认证。

diagnostic 输入 [input_context.json](../../HS2Mod/artifacts/infrastructure_live_20261005/ingame_consumer_diagnostic_v1/consumer/native_consumer_49872482f3174101a9a31ee57aaea895/input_context.json)，SHA `81ac1e8c0e0f4dce6edbc132cf1f4304a46ab672f5a90b07a193c58130d12b21`，实际 19 个 public identity modifier 全部 `_forceApply=true`，多数另有 `_changedPosition/_changedScale/_changedRotation`。该文件证明实际 flags，不能只由 flags 识别 AdditionalBoneEffect 的具体提供者。

静态 installed source [BoneController.ApplyEffects](../../HS2Mod/tools/ABMX_BoneController.cs) 遍历 `AdditionalBoneEffects`，对 nonempty `GetEffect` 找到或创建 modifier，然后将 effects 传给 [BoneModifier.Apply/CombineModifiers/CanApply](../../HS2Mod/tools/ABMX_BoneModifier.cs)。`_forceApply` 是 combined effective data 非空产生的状态，而公开 `GetBones` 仅给 coordinate controls，identity 并不能证明 effective identity。另有明确 dynamic-body 分支：`cf_J_Mune00*`/`cf_J_SiriDam*` 在每次 ApplyEffects 前 `Reset+CollectBaseline`。controller `UpdateBaseline` 针对 face+body native 字典并调用 face/body UpdateShape，因此不能假定 body cache 在 face write 时一律 immutable。上述是源码可见机制；具体当前 effect provider/source 与 side effects 仍需 actual read-only metadata/trace。

建议未来显式 `face_dependencies_v1` 范围按 actual IDs 建立：完整 head-root descendants，全部请求脸部 renderer 的 skinning bones/root/renderer transform，以及这些 seed 的全部 ancestors；同时 source/topology/actor/head/native/profile 与 dependency 集合冻结。不能用 `Mune`/`Neck` 字符串白名单，不能因为 modifier 不在 `o_head.bone_names` 就排除 ancestor。缺失 ID/祖先链、重复/cycle、effect membership 未知时拒绝；scope 外 cache 可作为独立 nuisance 记录变化，但公开 controls/source/actor 仍须保留，不能宣称整个 actor 状态已认证。

对上述实际 snapshot（10 default meshes / 180 exported transforms / 141 head descendants）这样形成的 dependency closure 共 180 transforms。19 modifier 中 18 个在 closure 外，**`cf_J_Neck` 的实际 ID `-24796` 在 head-root ancestor 链中**，其 `_hasBaseline=true`、`_changedPosition=true`、`_forceApply=true`，`_lenBaseline=1.3990525`。因此即便缩小 body cache 范围，当前 neck 的实际 effective/pending 状态仍会阻止 native-only 的认证路径；只豁免乳房不能解决问题。本轮没有修改任何默认 guard 或增加此策略。

如要实现该显式 scope，最小追加证据应包括 controller baseline/update/hscene flags、AdditionalBoneEffects 的 type/assembly identity、实际 affected transform IDs，以及已发生 Apply 的 combined data/extra-effect 记录，并和 paired geometry 同 frame/actor/source 绑定。读取缓存或 observer 的实际事件，避免为了 snapshot 重新调用未知 `GetEffect` 引入副作用。对 dependency 内 effect 必须建立源码合同/actual replay，或明确拒绝；对 scope 外效应仍需调查 constraints/外部 writer 是否有跨 scope coupling。文件 SHA 只绑定记录，不证明没有这种 coupling。
