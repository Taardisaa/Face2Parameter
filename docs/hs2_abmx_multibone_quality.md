# 四骨四底模的实际曲面质量与时间漂移

本轮读取 `HS2Mod/artifacts/infrastructure_live_20261005/abmx_multibone_v1/live_cases.json` 的十二个实际 geometry：四个 all.5 native59／四骨 identity baseline，以及每底模的 early、late patch。游戏状态、expression 和 modifiers 的 before/after 记录逐项相等，三个 restoration flags 为 true。没有连接游戏、读新解剖留出图、改缓存或拟合代码。

结果保存在 `outputs/abmx_multibone_quality_20261005/actual_v2/`。既有四骨 numerical replay 的 8/8 pass，与下面的质量和时间结果分别记录，不能互相替代。

## 不变的验收合同

`tools/abmx_multibone_quality/run.py` 复用 `geometry_quality.mesh_quality.Thresholds`、`base_comparison.search.quality_gate` 和既有 comparison config，绑定其文件 SHA。相同 head、完整 renderer path、source SHA、ordered source/baked triangles、source vertices、bindposes、skin weights/indices 必须完全相符；不按名称回退。

相对质量 gate 仍拒绝新增退化三角形、非邻接自穿、相对 normal reversal、新反射／奇异骨 world frame、edge ratio 超出 `[0.5,2]`、area ratio 超出 `[0.25,4]`。normal reversal 是曲面法线方向检查，不是严格体积单元 inversion 证明。完整 baseline 的绝对异常和 triangle IDs 保留，不能因为“新增异常为零”就宣称 baseline 绝对有效。

每个 actual snapshot 都重新执行独立 skinning/LBS 对照，容差 `1e-5` normalized，并经既有 `load_certified_head` 严格绑定 snapshot 字节 SHA、frame、renderer/source、实际 skin influence 与唯一 world candidate。所有 enabled head-subtree renderer 的跨网格三角形诊断在这个共同认证空间执行；不混用各 mesh 的 raw 坐标。

完整 `o_head` 曲面距离沿用每个正面积三角形至少一条 area quadrature、4096 样本合同、seed7381，以及 RMS `.002`、P95 `.005`、max_sampled `.03`、normal P95 `45°` 的既有阈值。这里的 baseline 距离是形变幅度诊断，不是要求 ABMX patch 模仿 native baseline，也不是人物相似度验收。

曲面时间比较仅移除 **实际记录的 renderer translation/rotation**，不做 vertex-fit rotation、translation、scale 或 affine；比例变化不会被拟合掉。报告同时保留全部对应 vertex 的 raw 与记录 rigid frame 差异。标准 mesh-quality normal comparison 使用既有 proper rigid Kabsch，仅用于法线朝向相对比较，也没有 scale/reflection fit。

## 实际结果

八个 patch 的完整 `o_head` 相对 distortion gate 全部通过：没有新增上述异常。但四个底模早晚曲面均超过既有 RMS 与 P95 门限。

| head | baseline 自穿数 | baseline 非流形顶点数 | early→late RMS | P95 | max sampled | normal P95 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 171 | 8 | 0.00381140 | 0.01096141 | 0.01994072 | 4.97667° |
| 1 | 190 | 0 | 0.00393525 | 0.01125023 | 0.02125302 | 5.30260° |
| 2 | 164 | 1 | 0.00411956 | 0.01199548 | 0.02218423 | 5.46597° |
| 3 | 150 | 3 | 0.00404337 | 0.01162460 | 0.01985460 | 5.08779° |

以上距离为认证转换后的游戏单位，没有毫米标定。`max_sampled` 不是连续 Hausdorff 上界。四个 baseline 没有退化三角形、负 determinant 骨 world frame 或 near-singular 骨 frame；其余既有自穿、contact、aspect 和 topology 标志完整保留，尚未判断是否为允许的内部构造、seam 或可见缺陷。

从 baseline 到 patch 新增跨 renderer triangle crossing pairs，early 分别为 88/89/85/67，late 为 173/248/211/223。三角形 pair 数会随交线滑过网格改变；这些是实际跨网格几何诊断，**不能直接称为新增可见缺陷**。原 shader 的 depth/discard/alpha 和眼球/头部合法嵌套仍需要独立可见性审查，跨网格诊断不被偷偷加进或移出既有单 mesh gate。

## 参数一致仍有实际形状变化

四个 early/late case 都有完全一致的 native59、expression 配置、actual active blendshape、四骨参数、source、renderer identity/enable 状态、renderer rotation 与 lossy scale。renderer position 与 pose signature 不同；记录 position 的影响已通过上述 actual rigid removal 处理，曲面差异仍存在。

`cache_drift_v1.json` 另行绑定 early/late actual geometry，并检查 skin-palette 的全部 head-root 祖先 local TRS。每个底模只有四个选中骨 `cf_J_Chin_rs`、`cf_J_ChinTip_s`、`cf_J_CheekUp_L`、`cf_J_CheekUp_R` 的 actual local 改变，其余该范围祖先未变；选中四骨已导出的 private cache fields 在 early/late 之间也相等。这是实际状态差异的定位，不是对全游戏 writer 的因果识别。Trace cursor、call counts、完整字段与差值保留在报告中，没有拟合应用次数。

因此本轮可以说特定四骨快照的 numerical replay 和相对 mesh distortion 检查分别通过，不能说这个组合已经时间稳定。固定参数不能代替实际 pose/shape evidence。完整脸部解剖 region、跨底模语义、人物 likeness、shader 可见性与全部 ABMX 骨行为仍未验收，程儿目标不能据此恢复。

## 重现

始终使用 Face2Parameter 项目 Python，输出必须是新目录：

```powershell
.\.venv\Scripts\python.exe -m tools.abmx_multibone_quality.run --manifest C:\Users\13666\Workspace\HS2Mod\artifacts\infrastructure_live_20261005\abmx_multibone_v1\live_cases.json --replay-summary outputs/abmx_multibone_20261005/actual_v1/summary.json --established-config outputs/stateful_base_comparison_20261005/comparison_config.json --out-dir outputs/abmx_multibone_quality_20261005/<new-run>
.\.venv\Scripts\python.exe -m tools.abmx_multibone_quality.cache_drift --manifest C:\Users\13666\Workspace\HS2Mod\artifacts\infrastructure_live_20261005\abmx_multibone_v1\live_cases.json --out outputs/abmx_multibone_quality_20261005/<new-cache-report.json>
```

首个 `actual_v1` 因错误读取 blendshape 字段 `weight`（实际字段为 `current_weight`）在 baseline 审查之前失败，只保留 audit_contract；没有伪造通过报告。修正后 `actual_v2` 完成全部十二个 geometry，原失败目录保留不覆盖。
