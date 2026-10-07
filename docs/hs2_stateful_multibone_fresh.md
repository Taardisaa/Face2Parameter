# Earlier history → fresh native59 四骨诊断

[adapter.py](../tools/stateful_multibone_fresh/adapter.py) 与 [validate_fresh.py](../tools/stateful_multibone_fresh/validate_fresh.py) 只读审核两个独立观察阶段。预测输入由**候选前** neutral trace 的持久历史、预声明 changed native59、四个预声明 patches、各骨实际 observer prefix 次数组成。candidate actual before、baseline TRS、after、BakeMesh 都是验证目标，不能替换预测值。

## 具体支持的生命周期

每 head 先设置 neutral native59=.5 与四 identity modifier，采集至少每骨两次的完整 trace 和 exact-cursor geometry，停止 source observer。随后保持同一 character/head/bone/modifier 实例，仅写预声明 native59，按实际生命周期自然刷新 `_posBaseline/_sclBaseline/_rotBaseline`。新的 candidate trace 先留下 clean identity calls，再统一写四 patches，拍 early/late paired 三视图，停 trace 后恢复原状态。

输入范围是 Chin_rs、ChinTip_s、CheekUp_L/R 四骨，以及显式 `slider_unlocker_18_2` sampler。本批 candidate native59 按索引循环 `.48,.52,.47,.53`，避开 `.5` keyframe knot；source neutral59 与原历史应保持独立。只改变 native59 并不自动证明生命周期合法；实际测量必须通过以下守卫：

- 完整 source 与 candidate trace 均用独立 NumPy installed replay 重算；完整 trace 不能包含外部 local/cache boundary、未知 Apply patch owner、额外 modifier、H-scene 或 exclusion 分支。
- 各 snapshot exact cursor 独立匹配相应 trace 的全局完成数，以及**每骨**最后完成调用的 local/cache。不能用最后一个 global call 覆盖其他骨状态；最多一帧年龄，pending/dropped/errors 必须为零。
- source observer 的 stopped frame 与 source geometry frame 必须早于 candidate started frame；两个 session 不同，character/head/bone_transform_id/modifier identity 一致。中间重载 head 或重建 modifier 会拒绝。
- source 整个 trace 必须 identity 且 clean flags；candidate 首个 before 也须实际 clean flags。禁止私写 flag，禁止在离线补 flag。
- 每骨 `_lenBaseline`、`_positionBaseline` 必须在 source 所有 before/after、candidate 所有 before/after及两个 snapshot 中与 earlier source 保持**相同 float32 值**，不能重新估计或容差吞掉生命周期变化。
- source native59 计算的 local TRS 与 source 首个 before/cache baseline TRS 匹配；candidate **预声明** native59 计算的 local TRS 与 candidate 首个 before/cache baseline TRS 匹配。使用现有固定 TRS 门槛；任一骨失败，整窗口拒绝，不缩减骨组来宣告整组通过。

`_positionBaseline` 是安装插件持久历史位置向量，不是目标表面点。它与 `_lenBaseline` 是明确 measured nuisance。候选 native local TRS 与新的三项 baseline 则由 native59 计算，不能从 candidate 实际 cache 注入。

## Manifest 最小合同

根节点必须含 `protocol_path/protocol_sha256`、四个 head case，以及实际 `state_restored/expression_restored/bone_restored=true`。预声明协议包含：

```json
{
  "heads": [0,1,2,3],
  "names": ["cf_J_Chin_rs","cf_J_ChinTip_s","cf_J_CheekUp_L","cf_J_CheekUp_R"],
  "source_history_expected_native59": "59 个 .5",
  "native59": "59 个预声明 candidate 数值",
  "candidate_expected_native59": "与 native59 完全一致",
  "baseline_expected_native59": "与 native59 完全一致",
  "patches": "恰好四骨 scale/length/position/rotation 字段",
  "sampling_profile": "slider_unlocker_18_2",
  "normalized_whole_head_gate": 0.00001,
  "common_apply_count": null,
  "common_count_runtime_certified": false
}
```

上面的字符串仅表示数组内容，实际 JSON 必须使用数值数组。各 case 形状：

```json
{
  "head_id": 0,
  "names": ["上述四骨名"],
  "source_history": {
    "trace": "earlier stopped identity trace 路径",
    "trace_sha256": "SHA256",
    "geometry": {"path":"earlier geometry 路径","sha256":"SHA256"},
    "capture": {"views":"三个 actual paired view 对象"},
    "native59": "earlier native59 记录"
  },
  "trace": "fresh candidate stopped trace 路径",
  "trace_sha256": "SHA256",
  "windows": [
    {"name":"head_0_early","window":"early","head_id":0,"geometry":{"path":"...","sha256":"..."},"capture":{"views":"三个 actual paired view 对象"}},
    {"name":"head_0_late","window":"late","head_id":0,"geometry":{"path":"...","sha256":"..."},"capture":{"views":"三个 actual paired view 对象"}}
  ]
}
```

SHA 负责文件证据绑定，不单独证明实际状态。源码绑定包含 ABMX/Unity Core DLL、decompiled source、MVID/IL 合同；所有输入及缓存 FK/LBS 表在报告中保留 SHA。报告不能使用未完成的 progress.json 当最终 restored manifest。

## 运行与验收

```powershell
& ./.venv/Scripts/python.exe -B -m unittest discover -s tools/stateful_multibone_fresh -p test_fresh.py -v
& ./.venv/Scripts/python.exe -B tools/stateful_multibone_fresh/validate_fresh.py `
  ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_multibone_fresh_v1/live_cases.json `
  --contract outputs/abmx_replay_20261005/installed_contract_v2.json `
  --out outputs/stateful_multibone_fresh_20261005/new_cpu --device cpu
# CUDA 使用独立未存在 output，--device cuda
```

`EarlierHistoryProtocol.from_manifest(...)` 保存 earlier source cache/history；`FreshObservedCountTorchRig(HeadRig(...,sampling_profile='slider_unlocker_18_2'), protocol, device=..., dtype=torch.float64)` 仅接受该批预声明 candidate 参数。source/candidate native guards 均通过后，从 source flags/history与 candidate 计算出的新 local/三个 TRS baseline，独立逐骨 `apply_sequence`（float32）→ parent-first FK/LBS（float64）。其他骨必须 identity，不能回落静态公式。

实际整头证据需要 actual renderer LBS/world candidate 认证、缓存 mesh/weight/bindpose 对应、非覆盖 skin ancestors、paired frame/pose/material sampled signature 等检查。整头最大 L2 / actual bbox diagonal 固定 ≤1e-5；local/cache 使用既有 position/scale ≤1e-6、quaternion sign-equivalent ≤2e-6、角度 ≤.001°、cache float ≤1e-6、flags 完全一致。只允许 proper rigid pose 与来自 actual 祖先矩阵的 uniform scale，禁拟合 scale/affine/count。实际 blendshape仍是声明 nuisance 输入。

## 次数与认证范围

本批 `settle_frames=5/18` 是采样窗口，不是 Apply 次数。模型使用每骨 exact cursor prefix 的实际次数，未优化或拟合。即使某窗口四骨次数巧合相同，也不会认证新的共同 N 调度；`counts_controlled=false`、`common_N_runtime_certified=false` 始终保留。

CPU/CUDA 数值通过仅能认证这批 fresh changed native59、四温和 patches、source-bound measured history 和 observed counts 的具体对应。任意候选泛化、其他骨、特殊 length<.1/zero-position 分支、未知外部写入、mesh quality、anatomical surface 与程儿 likeness 均不由此自动证明；`character_ready=false`。

2026-10-05 实际 fresh 验收：CPU/CUDA 各 8/8，最大整头归一化误差分别 `3.563778442e-7` / `3.563774760e-7`，固定门槛 `1e-5`。四骨 earlier-history/source/candidate flags、实例身份、baseline native guards 和逐调用 persistent history 检查全部通过。11 个新增测试通过。证据见 [本批汇总](../outputs/stateful_multibone_fresh_20261005/SUMMARY.md)；共同 N 与任意候选认证状态保持 false。
