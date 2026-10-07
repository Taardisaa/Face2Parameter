# 受控原生径向目标消费者

`src/hs2_controlled_radial.py` 为现有 `InGameEvaluator` 提供具体 `prepare/apply/validate_capture/restore` 生命周期。它复用已冻结 ALL30 编译器、原生 mouth/gaze guard、严格 V2 authored asset provider 和独立 NumPy 全可见头部审计；不是根据 JSON capability 字段声明成功的 stub。当前仅支持显式 head3 asset plan、实际 `slider_unlocker_18_2` 和原先不存在的 ALL30 修饰器。新增测试使用 fake transport/math，不能当作实机验收。

## 显式调用

```python
from src.hs2_controlled_radial import ControlledRadialProvider
from src.hs2_ingame_eval import InGameEvaluator

provider = ControlledRadialProvider(
    work_dir=r'C:\...\new_run',
    contract_path=r'C:\...\installed_contract_v2.json',
    driver_contract_path=r'C:\...\driver_contract.json',
    asset_plan=r'C:\...\head3_asset_plan_v1.json',
    strict_backend='identity_source_assets_strict_v2',
    outside_state_policy='source_bound_identity_nuisance_v1',
    resolution=512,
)
evaluator = InGameEvaluator(
    semantics_mode='native_radial_target_v1',
    range_mode='native', sampling_profile='slider_unlocker_18_2',
    radial_provider=provider, logical_controls=raw_all30_dictionary,
    scorer=explicit_benchmark_scorer,
)
# 参数必须完整59；raw_all30_dictionary 含完整30骨，每骨scale/length/position/rotation。
result = evaluator.evaluate(candidate_native59, sets=('a', 'b'))
```

现有 evaluator 由 root 单独整合；新 provider 要求以下接口，缺少接口时不能声明可执行：

- radial context 的实际几何 GET 使用 `include_actor_transforms=true`。所有 provider render POST query 都包含 `geometry_include_actor_transforms=true`；consumer supplied capture 也须包含此字段。旧 DLL 配对导出缺少18个待执行身体修饰器的实际局部变换，应拒绝，而不能把缓存基线当作当前姿态。
- `apply` 返回 `requires_actor_transform_coverage=true`、`fixed_framing={ortho_size,distance,target}`、明确 `owned_expression_patch_fields/owned_expression_values`。consumer 必须使用相同固定相机边界，允许且绑定这些具体表达驱动字段变化，不得忽略全部其他公共配置。
- `validate_capture(execution,evidence)` 接收 `evidence.render_response={path:absolute saved JSON,sha256}` 和 `evidence.geometry`。provider 独立重读实际 response/PNG/geometry，不接受仅有摘要布尔值。
- 验证成功返回 `restored_before_acceptance=true`、`restoration_report` 与数值报告文件绑定。consumer 此时重新读取并验证原始实际 context、清除 owned execution；随后才调用显式 scorer。不能仍把 active physical30 当作当前状态。

## 生命周期与来源

1. `prepare` 只冻结请求：完整59范围、raw ALL30、head/actor/profile/config source hashes，以及 inherited outside 公共/私有状态。已有 facial modifier 即使 public identity 也拒绝，避免历史半径 Reset 副作用。准备结果不认证尚未产生的候选基线。
2. `apply` 再读实际 context，拒绝其他 active observer；取得原始配对姿态后才允许写入。先写最终 native59，再设中性表达、关闭 mouth width adjustment，从实际 runtime pattern array 解析唯一 `NO_LOOK`，不硬编码索引3。最后创建 absent identity30。
3. 新建 identity observer、同 native/profile/driver 下取得配对 source 并停止 observer。之后才运行重型 authored asset stage、source-only compiler、全8可见 renderer native preflight 和 driver guard preparation。目标为 `p_native * logical_length + logical_position`，没有 candidate-after 输入。
4. 开始 bounded candidate observer，实际 fresh identity/asset/driver/outside guard 通过后才写 physical patches（Length 恒为1）。记录 early5/late18/far60 三次配对 capture。settle 只是等待配置，真实 Apply count 取 trace cursor，不能宣称共同 N。
5. consumer 同一 observer 下追加自己请求的一次多角度 capture。`validate_capture` 停止 observer，独立执行恢复，再重算内部三窗口数值审计。另存 NEW manifest，明确记录原 manifest/SHA 与 supplied→early 替换，重算 supplied 全可见头部/trace/driver/asset/temporal，而不是只检查图片哈希。
6. 数值接受仍使用全可见头部 `1e-5`、temporal RMS `.002` / P95 `.005`，没有放宽。失败禁止 scoring。新 protocol、source、compiler、guards、actual trace、response/geometry、恢复与数值报告均保存在唯一请求目录。

## Outside 状态与恢复范围

默认 `outside_state_policy='strict'` 拒绝任何待执行 outside 修饰器。显式 nuisance policy 仍要求每一个 outside 的公共值及所有坐标项 identity；绑定 modifier/bone IDs、插件来源、完整 private flags/fields，并逐项验证 source→fresh guard→三个窗口→supplied→recovery。所有 pending outside 必须在实际配对导出中具有本骨及完整祖先局部 TRS，否则拒绝。没有 Neck 名字白名单，也不自动 reset body。

原始请求前与候选 native 后 source 是不同锚点；候选基线正常更新不冒充外部漂移。candidate 阶段冻结 post-native source 的 outside 私有字段和全部非 ALL30 actual local transforms；恢复则比较原始公共 snapshot、10个表达字段、全 ABMX 公共配置、原始 outside 私有字段，以及每个导出的真实 skeleton ID/path/parent/local TRS，使用既有 local replay 容差。frame_count 不是私有 cache field，不因 frame 不同猜测缓存变化。

恢复逐步尝试停止 owned trace、identity owned30、移除新建30、恢复原 native59、恢复原表达；某一步失败仍尝试后续步骤。**移除30在恢复 native 之前**。新建30只能在最终 native 之后收集历史。无私有 baseline 写入，也不会删除用户已有修饰器。START 已被服务器接受但回复丢失时仍执行恢复 STOP；请求类型/SHA 改动在第一笔写入前拒绝。

`full_actor_pose_certified=false` 始终保留；未观测的非 pending 身体 local IDs 单列。即使新版导出扩大 actor transform 范围，也不据此泛化未识别外部 writer/无限时间稳定性、动画中的原始 gaze、全身动态姿态或 rendered RGB 恢复。真实原始 local 移动/恢复失败必须如实失败，不能改成只查 ALL30 后报告全身成功。

数值曲面目标和运输验证不等于 anatomical quality、真人相似度或 beauty。当前 acceptance 明确 `numerical_runtime_accepted=true` 与 `deformation_quality='not_certified_by_numerical_gate'`、`anatomy_certified=false`、`likeness_certified=false`；质量门槛没有以数值通过代替。首个实际用途是冻结源曲面目标的 geometry fitting benchmark，须显式 scorer。

## 本轮验证

离线20个独立 fake boundary/math tests覆盖生命周期顺序、source-only 输入、完整三窗口及 supplied 重算、实际 runtime gaze 索引、默认 pending拒绝、pending缺本骨/祖先观测、unpending缺body观测的范围记录、已有 facial identity拒绝、raw维度/范围/bool与不可变源拒绝、stale prewrite/其他observer、START响应丢失、非受控 local漂移、图片/JSON bytes篡改、数值失败不接受及恢复继续尝试，以及第一笔写入前必须返回整actor观测元数据。额外覆盖全部 outside 私有字段漂移拒绝、frame wrapper变化不假定cache变动、covered pending显式opt-in通过完整观察链，以及公共参数全恢复但非ALL30实际local未恢复时仍拒绝。provider 还独立对照 consumer 保存的每张 PNG path/SHA/尺寸，consumer→provider/scorer 之间的图片变更也拒绝；不据此认证像素坐标或解剖点。Ruff通过。未操作游戏，实机消费者接受/拒绝由 root 新实测报告给出。
