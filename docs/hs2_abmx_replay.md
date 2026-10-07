# Installed ABMX Apply 的只读状态回放

本工具验证**实际观测的函数调用**：从每次 `BoneModifier.Apply` 的 before local TRS、private cache、实际参数计算 predicted after，再对真实 after 比较。它不接触游戏，不 reset／rebase／collect baseline，不估计应用次数；不修改旧 full59 benchmark，也不将 cached-rest 静态整头模型自动标为修复。

工具仅新增于 `tools/abmx_replay/`。Python 一律 `.venv/Scripts/python.exe`。

## 输入和独立来源绑定

输入是 root 的 `HS2Mod/plugins/HS2_McpBridge/MakerAbmxTrace.cs` 被动 observer 导出的 stopped trace。顶层需 `metadata/events/active/observed_calls/dropped_events/pending_calls/observer_errors/trace_complete`。metadata 需 schema 1、session ID、started／stopped frame、plugin MVID／version、原 `Apply` 方法 IL SHA、pre-existing Harmony patch owners、requested names 与明确观察规则。

每 event 保留真实 `sequence/frame/completed_frame/bone_name/modifier_instance_identity/coordinate/coordinate_specific/resolved_modifier/additional_modifiers/no_rotation_excluded/is_during_h_scene`；before／after 含同一个 bone transform ID、raw local position/quaternion/scale、`cache` wrapper。cache 需 frame／bone ID／assembly MVID／modifier type／missing_fields／完整12项实际 private fields。

`installed_contract.ps1` 从磁盘上的实际 `E:/HoneySelect2_ArcticFox/BepInEx/Plugins/HS2ABMX.dll` 通过 .NET `PEReader/MetadataReader` 读取 MVID、version 与 `Apply` raw IL hash，**不加载／初始化插件程序集**。同时记录 DLL、Unity core 和已阅读的安装反编译 sources 文件哈希、rotation exclusion 集合。仅支持实际安装的 MVID `5442c72a-f463-4bf9-831a-247be87146c8`／version `4.4.6.0`。独立 source contract 不是游戏自己给自己声明通过。

验证拒绝：未停止、drop／pending／observer error、count不一致、乱序／缺少调用、frame跨越、source MVID／IL mismatch、未知已有 Apply patches、缺少cache、错误bone ID、未知modifier参数、非有限值、排除标记与独立 installed source 不一致。本版严格要求 pre-existing Apply patch owners `[]`，不能通过CLI白名单绕过未知patch语义。

运行示例（输出必须是新路径）：

```powershell
& ./tools/abmx_replay/installed_contract.ps1 -OutPath outputs/abmx_replay_20261005/installed_contract_v2.json
.venv/Scripts/python.exe tools/abmx_replay/validate_trace.py <stopped-trace.json> --contract outputs/abmx_replay_20261005/installed_contract_v2.json --out outputs/abmx_replay_20261005/<new-report>.json
.venv/Scripts/python.exe -m unittest discover -s tools/abmx_replay -p 'test_*.py' -v
```

`installed_contract_v2.json` 已实际生成；再次运行请选择另一个文件名，以保留证据。

## 独立模型覆盖的分支

`model.replay_apply` 实现安装反编译 `BoneModifier.Apply/GetModifier/CombineModifiers/CanApply`，包括：

1. missing Transform 和 null resolved modifier 的 early return；有附加modifier而base为null时从 identity组合。
2. 本 HS2 `IsCoordinateSpecific()` 恒 false，GetModifier总取slot0；回放使用观察到的 resolved modifier，拒绝其它variant。组合时scale／length相乘，position／Euler rotation相加；不会把Euler增量当成其它旋转序列。
3. nonempty modifier设 `_forceApply=true`；empty但原force为true时清force并设 `_lenModForceUpdate=true`；pending length force也允许apply，其余empty直接return，不能无条件恢复changed flags。
4. scale从 `_sclBaseline * modifier.scale` 写入；撤销时恢复 `_sclBaseline` 与changed flag。
5. rotation用 `_rotBaseline * Quaternion.Euler` 的 Z-X-Y等效转换；实际NoRotationBones禁止增量，而此前changed rotation仍触发baseline restore。
6. 有length（或force）且 `_positionBaseline != Vector3.zero` 时，使用**当前**local position方向。length `<0.1`、current approximate-zero或HScene时切换历史 `_positionBaseline` 并设置needs-position-restore。计算 `p / magnitude(p) * persistent_lenBaseline * modifier.length`，再按分支加offset或恢复位置。
7. length计算之后，若没有position modifier但 `_changedPosition`为true，安装代码会直接以 `_posBaseline` **覆写刚计算的length结果**。position-only分支则是 `_posBaseline + offset`；撤销时恢复cache，flags同步预测。
8. `_hasBaseline`没有被安装Apply用作guard，不能在离线模型额外加入该guard。

全部数值按 float32计算。只读反编译实际 `UnityEngine.CoreModule.dll` 验证了 Vector3 `==` 是 `SqrMagnitude(lhs-rhs) < 9.9999994E-11f`，不是逐component精确零；magnitude按Unity浮点求和／sqrt顺序，Quaternion乘法按安装源码。Quaternion.Euler 的native内核以独立Z-X-Y float32实现近似，数值容差固定而非每case调整。

## 有序调用链和验收

每条call独立比较predicted after与observed after，包括所有private cache numeric／boolean fields。local position和scale最大component容差 `1e-6`，quaternion符号等价component `2e-6`，rotation角 `0.001°`；cache numeric `1e-6`、boolean必须完全相同。输出保留raw quaternion误差；符号等价比较只消除相同旋转的±q表示，不拟合任何旋转／缩放／位置。

同modifier identity＋同bone ID依实际sequence串联。如果上一条actual after与下一条actual before吻合，传播上一条**预测**状态进行多步replay；如果有外部native／animation／baseline写入造成变化，明确记成 `inter_call_boundaries`，从新的实际before重新anchor。Apply-only observer没有证明这些写入来自哪个函数，报告不会擅自命名为native reset。实际调用次数来自events，绝不搜索六次／七次以匹配结果。

有界window外的调用尚未被观察；通过只能称为 `passed_specific_observed_calls`。丢失／未知source称为拒绝，数值失败称为 `failed_specific_observed_calls`，均保存结果并退出非零。输出记录exact trace／contract／validator／model哈希，拒绝覆写旧证据。

## 已完成和仍需实测

目前28项解析／拒绝tests通过，含3-4-5方向归一化、第二次调用方向改变、length小值／HScene／近零、empty CanApply分支、position restore覆写length、rotation exclusion、90°解析旋转、附加modifier组合、缺缓存／NaN／错误bool、乱序／drop／外部patch／IL mismatch、实际after或cache篡改，以及有外部输入边界的有序链。新增cursor tests验证早于trace末尾的精确时刻、当前frame LateUpdate尚未调用时的上一已完成call、错误session／futurecursor／bone／cache／pending拒绝。

这些测试没有使用 `replay_apply` 生成 expected after；关键结果手工给定或用简单解析几何计算，避免自我比较。实际结果见下一节，不能用离线测试代替runtime实测。

即使所有实际调用replay通过，现有 `src/hs2_mesh_deform.py::_fk_world`／`src/hs2_deform_torch.py::local_transforms` 的静态 `p*length+offset` 模型仍然没有重写。完整surface quality／人物相似度／其它ABMX bones／未观测modifier效果，需各自验收。

## 实际调用与状态条件整头验收结果

root完成并恢复了 `HS2Mod/artifacts/infrastructure_live_20261005/abmx_trace/four_heads/live_cases.json` 和 `branch_probes/live_cases.json`。独立安装contract的 `Apply` IL SHA为 `f6ff0735099f5471864d6d9df7f06099543d4f4f1b70ab232c8be2c01777a41f`，实际trace metadata与之匹配，已有Apply patches为空，drop／pending／observer errors为零。

四个底模真实调用分别 **22／22／19／18**，附加short-length与rotation-exclusion分别 **18／15**，合计 **114/114调用通过**。最大position component residual约 `1.49e-8`，最大quaternion component约 `5.96e-8`，scale／cache／flags通过原固定gate。汇总 `outputs/abmx_replay_20261005/call_series_final.json` 重新从实际trace调用独立模型，不是读旧报告的passed字段。

short-length实测覆盖九次 `<0.1` historical-direction fallback＋offset，随后一次 neutral force restore并以 `_posBaseline`覆写length结果，再八次CanApply false；rotation-exclusion实测有八条非零effective rotation被排除，随后恢复／empty分支。rotation-exclusion的 `cf_J_Head`出现六个inter-call状态边界，报告保留它们，没有把body／animation写入当作Apply累计。未实际观测的HScene／附加modifier／近零边界仍仅有解析tests，不能泛化成runtime覆盖。

四head producer manifest提供了trace SHA并逐一校验；branch manifest仅记录路径，未提供producer SHA。汇总对这两条明确记录 `producer_manifest_trace_sha_present=false`，独立保存实际读取文件SHA，不伪造先前producer签名。模型source／IL／完整call coverage与numerical gate仍严格验证。

实际overflow反例 `abmx_trace/overflow/trace.json`只有一个event，dropped七次、trace_complete=false。`overflow_rejection.json`明确为 `rejected_incomplete_or_unsupported_evidence`，原因 `Incomplete trace: dropped_events`，不能认证为只有一次真实调用。

`validate_geometry.py` 再用exact `snapshot.abmx_trace_cursor`选择**截图时刻**的已完成调用，选中序号分别 **9／8／8／7**，排除停止trace前后续 **13／14／11／11** 条调用。snapshot可能发生在当前frame的LateUpdate之前，因此不能强制last completed Apply与snapshot同frame；必须用cursor序号，要求call frame不晚于snapshot，并以同骨ID的实际snapshot local和private cache独立验证预测状态保持一致。最初过强同frame guard的四个拒绝报告仍保留于 `geometry/`，最终正确cursor合同见 `geometry_final/`。

重建用cached rig的native59预测其它骨局部TRS，**只用独立回放／传播预测结果**覆写ChinTip局部TRS，再重新FK／cached weights＋bindposes LBS完整o_head。实际after和snapshot local只参与误差验收，不能被拷贝作预测。额外需要actual expression frame delta与独立 recorded ancestor uniform scale，单位固定1，只移除proper rigid pose；无拟合scale／affine。

完整头部与经独立actual-bone LBS认证的Unity BakeMesh世界顶点 **4/4通过**，固定最大normalized gate `1e-5`：

| Head | 全头顶点数 | State-conditioned max normalized | World max L2 | World RMS | 保留的旧静态模型max normalized |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 4418 | 9.996e-8 | 2.990e-7 | 1.089e-7 | 9.501e-7 |
| 1 | 4438 | 9.607e-8 | 2.860e-7 | 1.113e-7 | 0.01609849 |
| 2 | 4439 | 1.026e-7 | 3.045e-7 | 1.150e-7 | 0.01547718 |
| 3 | 3650 | 9.508e-8 | 2.836e-7 | 1.146e-7 | 0.00673625 |

报告 `outputs/abmx_replay_20261005/geometry_final/geometry_summary.json` 保存source／topology／bonepalette／weight／bindpose验证、其它native skin-bone locals、所有head cache来源文件SHA、真实三视图和checkpoint SHA。原head0／1／2质量失败winner仍仅diagnostic，原selection和checkpoint未改变。state-conditioned parity不是安全fit、完整底模表达力或人物相似度结论。

复现：

```powershell
.venv/Scripts/python.exe tools/abmx_replay/validate_geometry.py ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_trace/four_heads/live_cases.json --contract outputs/abmx_replay_20261005/installed_contract_v2.json --out outputs/abmx_replay_20261005/<new-geometry-directory>
.venv/Scripts/python.exe tools/abmx_replay/summarize_calls.py --manifest ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_trace/four_heads/live_cases.json --manifest ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_trace/branch_probes/live_cases.json --contract outputs/abmx_replay_20261005/installed_contract_v2.json --out outputs/abmx_replay_20261005/<new-call-summary>.json
```

## 下一步严格stateful fitting的必要条件

候选参数到曲面的函数应显式写为 `mesh = F(head_asset, native59, modifier_commands, initial_persistent_cache, initial_local_state, ordered_external_updates, ordered_Apply_calls, expression, ancestors)`；不能仍以 `F(native59,ABMX)`声称完整runtime模型。

**可由native59预测的部分，需要限定更新协议。** 目前四个ChinTip mixed case的 `_posBaseline/_sclBaseline/_rotBaseline`在native shape写入／baseline refresh后吻合cache native FK，第一次trace的before也从已写native局部TRS开始。因此候选可以在明确“native UpdateShape→CollectBaseline完成”的边界计算这些字段，而不是拷贝旧候选的数字。一般骨受pose／animation／其它effects控制时，还需那些作用的独立输入；不能因为ChinTip四点匹配，就推广全ABMX骨。zero inter-call变化也不能证明没有外部函数调用，只表示该次调用没有观测到数值差异。

**必须显式保留的持久部分。** `_lenBaseline/_positionBaseline`是首次收集的长度／方向历史，本次不随native59／换头自动更新；不能换成cached rest或候选native长度。`_forceApply/_lenModForceUpdate/_lenModNeedsPositionRestore/_changedScale/_changedRotation/_changedPosition`和`_hasBaseline`须从实际初始状态继承并逐step更新。current local position可能已经包含之前length／offset效果，不能每次都用native值替代。resolved modifier／coordinate／additional effects／rotation exclusions须明确；只支持已验证安装variant和可观察输入。

**调用时序不能从framecount拟合。** 为候选定义可复现的native update、modifier command、baseline refresh和Apply调用顺序，初始state固定，优化器只改参数。snapshot cursor决定验收终点；既不能用trace最终call，也不能靠搜索六／七次让曲面更近。若现实HTTP等待期间Apply次数变化，使用实测cursor／event，或建立已验证的固定update协议；elapsed frames和real Apply calls并不自动一一对应。

**未建模外部边界阻断候选传播。** 本次rotation-exclusion实际六个边界说明Apply-only trace不覆盖native／animation／Reset／CollectBaseline／其它effect写入。复制一个真实after／下一before到候选会注入真实候选的答案。只有该writer被独立模型化、其输出可由候选和声明状态预测时，才能跨边界传播；否则限定到没有未知边界的segment，或增加被动writer观察，报告unsupported而非偷偷重新anchor后声称候选预测成功。当前replay的anchor用于验证具体观察到的calls，不是未来候选的免费输入。

最后，length==1与非1、position exact-zero、CanApply flags、Vector3 near-zero、length<0.1、coordinate或modifier增删等branch会改变过程；stateful differentiable proxy必须维护对应branch与状态，不能以连续平滑乘法替换而未经runtime验收。新候选仍须完整surface＋same-head quality gate，先前quality-invalid winner不能因state parity通过而被批准。

## 新Torch显式状态模型的独立只读审查

`tools/abmx_replay/audit_torch.py` 对 `src/hs2_abmx_torch.py` 做只读实际调用对照。审查版本SHA256为 `1c78c1e2bfdf9de07dca6b2daf8712fd4b58626320ff04d0e855c9297017892e`，报告分别为 `outputs/abmx_replay_20261005/torch_review_cpu_linear.json` 与 `torch_review_cuda_linear.json`。调用前重新校验installed assembly／Unity core／反编译source／所有trace SHA，实际trace先经过独立NumPy verifier，不依赖旧报告passed字段。

CPU及CUDA float32均 **114/114** predicted local TRS／private cache与actual after通过原固定gate，所有boolean flags完全匹配；**114/114** 对local TRS、modifier和cache numeric输入的反向梯度有限。实际最大position component误差 `1.4901161e-8`、quaternion符号等价component误差 `5.9604645e-8`。Torch与独立NumPy position误差为零，quaternion最大误差CPU `5.9604645e-8`、CUDA零。这是固定容差下符合安装语义，不是要求不同sin/cos实现逐bit相等。

3-4-5方向／length1.4／非零position offset的解析梯度另独立验收：`d(sum(position))/dp = [0.224,-0.168,1.4]`，length导数7、position offset导数各1；最大误差 `1.421e-7`，gate `2e-6`。这项提供一个平滑分支的梯度正确性证据；114条梯度有限不等于各分支梯度都已证明正确。七个正常解析分支探针（identity skip、零history、current零fallback、short length、HScene、position恢复覆盖length、无history但pendingforce）forward／flags／gradient有限性均通过；HScene等仍是解析而非新增runtime覆盖。

审查发现两个可复现的防御性缺口，报告与正常runtime结论分开：

1. `apply_transition`仅检查cache keyset，未验证直接传入tensor cache的shape／bool dtype／device／finite。把unused `_sclBaseline`设NaN，identity最终TRS仍finite，但modifier backward出现NaN。正常 `tensor_cache`入口会拒绝该输入；此反例模拟绕过入口的无效cache，不是114实际调用失败。建议共享实现同时约束直接tensor调用的cache合同。
2. 极端但finite的 `position=[1e20,0,0]`、`_lenBaseline=1e20`、`_positionBaseline=0`、identity modifier时，inactive length归一化divisor虽安全置1，但仍先计算 `direction * _lenBaseline * length`，产生inf。最终where前向仍返回正确position，length梯度却被未选分支的 `0*inf`污染。使用**线性**输出loss重复确认，排除平方loss溢出。数值远离本次runtime范围；建议在inactive长度乘法前把direction／baseline length／modifier length也替换为安全值，而不只mask sqrt。不得用扩大容差处理梯度NaN。

原 `torch_review_cpu.json` 是初次平方loss诊断，保留历史；上述结论优先引用两个线性loss报告。未修改共享Torch模型，未重新求解／替换任何full59 winner，未进行游戏操作。

优化器还需注意正确exact分支的后果：neutral identity且无pending force会直接skip，所有ABMX参数梯度为零；scale=1、position=0、rotation=0对应通道也会被Has*分支排除。不能把neutral梯度求解失败当底模不能表达。候选搜索需记录有界非零seed或离散branch探索，并对最终候选按真实分支／持久状态／实际调用协议验收；不应暗改branch来声称runtime一致。

`validate_trace.py` 已支持 `from tools.abmx_replay.validate_trace import validate, verify_trace_header`，相对 `.model` import优先，直接CLI时回退原 `model`；`validate_geometry.py`同理。包import与直接CLI help均已验证。此次兼容修改没有改变公式／阈值，旧报告source SHA仍代表旧验收版本，新审查报告记录新的source SHA。当前Torch验收用各条**真实before/cache**作单call输入，未知inter-call边界仍阻止未来候选预测；它不修复或认证无状态整头优化器。

root随后修复两处防御性缺口；修复版SHA256 `d446acfdf8c2f3b8a5cb0f637f5862dd118019c84b24228bf105fc39cb53b19a` 已再次独立跑CPU／CUDA相同114实际调用与两个反例，输出新 `torch_review_cpu_fixed.json`／`torch_review_cuda_fixed.json`，保留全部旧报告。两端均114/114 forward／cache flags／gradient有限性通过，actual及NumPy residual、解析gradient误差均未退化。极端inactive length乘法反例现在forward／cache吻合且所有梯度有限；直接NaN cache在运算前明确拒绝 `Nonfinite/invalid explicit cache value: _sclBaseline`。审查者只新增验证与文档，未修改共享core。

只读查看新 `tools/stateful_fit/adapter.py` 确认其范围是clean native-baseline protocol：候选native FK生成before及pos/scale/rotation baseline，实际persistent length／direction／flags保留，declared和observed call count分开，未知external boundary拒绝，actualafter只作证据guard而非候选输入。root已加入非零seed探索来避开identity零梯度。这些是实现合同审查；本节114单call／原四head state-conditioned验收不能代替新adapter任意候选的actual Unity验证，也不证明完整ABMX泛化／质量合格。

复现只读审查：

```powershell
.venv/Scripts/python.exe tools/abmx_replay/audit_torch.py --summary outputs/abmx_replay_20261005/call_series_final.json --contract outputs/abmx_replay_20261005/installed_contract_v2.json --out outputs/abmx_replay_20261005/<new-review>.json --device cuda
```
