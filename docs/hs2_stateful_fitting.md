# Native59＋ABMX 的状态条件拟合

安装的 ABMX 4.4.6 在 length 与 position 同时非中性时，按当前 local position 的方向逐次 Apply；`_lenBaseline` 与 `_positionBaseline` 又可跨 native 刷新持久保留。旧 `p*length+offset` 不是这个过程。已有静态 benchmark 保留为历史证据，不能以历史参数误差下降认证新候选。

`src/hs2_abmx_torch.py` 提供可微的显式状态转换，局部 TRS 和全部 private cache 同时输入、输出。它实现 installed CanApply、flags、近零比较、length fallback、位置恢复覆盖 length、rotation exclusion，保留 float32 计算与 exact identity 分支。输入 cache 不完整／非有限时拒绝；不活跃分支在算术前 mask，避免 `torch.where` 的非选中溢出污染梯度。

`tools/stateful_fit/adapter.py` 从完整 stopped trace、独立安装源码合同和准确几何 cursor 建立协议。它重新执行独立 NumPy 回放，验证 source SHA／MVID／IL、producer SHA、snapshot local/cache 一致性，并拒绝未知 external writer boundaries。当前支持四底模的 clean ChinTip 边界；其它骨骼需要自己的状态合同，禁止静态 fallback。

候选的完整59原生参数预测当前 native local TRS，作为 before 和 `_posBaseline/_sclBaseline/_rotBaseline`。历史 `_lenBaseline/_positionBaseline` 和实际 flags 保持测量值，随后传播每次 Apply。观察到的最终骨骼位置只用于验收，不能成为候选的输入。协议要求 native baseline refresh 已完成、无未知后续 writer；四个实际边界已验证，不能将此推为所有 runtime 状态的认证。

调用次数分开记录：几何 cursor 对应的实际次数与搜索声明的公共次数。新 benchmark 在四底模都声明8次；这不是拟合次数，也不是宣称客户端能恰好第8次截图。实际早／晚窗口分别使用真实 cursor，并测量时间漂移。新人物候选还必须验证正常运行的稳定性。

## 运行

Python 必须使用项目 `.venv/Scripts/python.exe`。所有验证输出选新路径，保留失败与历史产物。

```powershell
.venv/Scripts/python.exe -m unittest discover -s tools/stateful_fit -p 'test_*.py' -v
.venv/Scripts/python.exe tools/stateful_fit/validate_recorded.py <live-manifest.json> --contract <installed-contract.json> --out <new-report.json> --device cuda
.venv/Scripts/python.exe tools/stateful_fit/validate_gradients.py <live-manifest.json> --contract <installed-contract.json> --out <new-gradient-report.json>
.venv/Scripts/python.exe tools/stateful_fit/validate_rejections.py <live-manifest.json> --contract <installed-contract.json> --out <new-negative-report.json>
.venv/Scripts/python.exe tools/stateful_fit/benchmark.py outputs/base_comparison_full59_20261005/comparison_config.json --manifest <live-manifest.json> --contract <installed-contract.json> --apply-count 8 --out-dir <new-benchmark-dir>
.venv/Scripts/python.exe tools/stateful_fit/validate_candidate_series.py <early-late-manifest.json> --contract <installed-contract.json> --out <new-series-report.json>
```

`tools/base_comparison/search.py` 的 `native+ABMX` 模式要求配置：

```json
"abmx_protocol": {
  "manifest": "C:/.../four_heads/live_cases.json",
  "contract": "C:/.../installed_contract_v2.json",
  "apply_count": 8
}
```

无协议时直接拒绝。仅显式 `legacy_static_abmx_diagnostic: true` 可运行旧公式诊断，且它的候选 `accepted` 必为 false。新状态模型仍使用原有完整 `o_head` 面积加权曲面距离与 same-head 质量门槛；数值回放通过不能批准已知几何质量不合格的结果。

exact identity 可因 installed predicates 得到零 ABMX 梯度。状态搜索默认从许可 envelope 内1%的明确 seed 开始，不将游戏分支平滑化。梯度验证分别报告 predicate 与原生动画采样 knot；中央差分步长事先选为 `min(.001, 最近 knot 距离/4)`，不跨 knot 比较不同斜率。初次固定 `.001` 差分在20/47跨 `.5` 失败的报告保留。

## 当前证据与边界

- 独立 CPU／CUDA 各114次实际单调用和全部 cache flags 通过，反向梯度有限；见 `outputs/abmx_replay_20261005/torch_review_*_fixed.json`。
- 四组完整头部的候选初始与当前基线由 native59 推导，实测曲面最大归一化误差约 `3.9e-7`；见 `outputs/stateful_fit_20261005/recorded_forward_*.json`。
- 所有59原生与10个 ChinTip 维度在记录的 head3 非中性点通过数值梯度对照；见 `gradient_v2.json`。不是全局可微性证明。
- 9项解析／搜索入口测试及7项实际证据损坏拒绝通过；见 `protocol_rejections_v1.json`。
- 四底模×三模式、完整59维、原40步预算、原混合目标、原曲面／质量 gate 的新搜索已执行；见 `outputs/stateful_base_comparison_20261005/benchmark_summary.json`。输出仍明确新候选需独立实机验证。
- head3 新候选在真实第7次和第85次 Apply 配对几何中通过预测，窗口内最大归一化曲面漂移 `3.9e-7`；见 `head3_candidate_series_v1.json`。这不认证未采集时刻、其它 bones 或真人目标。

原 `src/hs2_mesh_deform.py`／`src/hs2_deform_torch.py` 的直接 ABMX 调用保留为 legacy 离线路径；本次受控搜索通过 `StatefulTorchHeadRig` 覆写状态阶段。既有的 ingame optimizer、ML 参数预测与卡片直接推理尚未迁移，不能借新 adapter 的数值验收声明那些路径已修复。完整解剖对应、眼部和 shader 遮挡、真人目标表达力仍未完成，程儿目标保持暂停。
