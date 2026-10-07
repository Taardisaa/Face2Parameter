# Fresh 四骨、四 head 实际曲面质量审核

本工具仅审查 root 在 `HS2Mod/artifacts/infrastructure_live_20261005/abmx_multibone_fresh_v1/live_cases.json` 记录的几何、实际状态和时序差异。不操作游戏，不读取人物 PNG，不修改既有拟合工具、人物点标注或 goal。candidate 能否被 old protocol 预测由 parameter_agent 独立认证，当前工具永远不签发这个结论。

## 固定比较合同

`outputs/stateful_multibone_fresh_quality_20261005/threshold_binding_v1.json` 只从先前审核提取阈值和实现文件字节 hash，不复制先前样本的任何通过结果。运行时拒绝既有 config 或 geometry_quality / base_comparison / unity_parity / read-only helper 实现 SHA 变化。

同一 head 的基线是实际 variednative59 identity geometry，59 值按 index%4 为 [.48,.52,.47,.53]。neutral all .5 geometry 只用于 source_history 状态诊断。期望 native59 和 candidate scale / length / position / rotation 与实际读回值以 float32 精确等价比较；浮点十进制表示差异不会改变任何几何阈值。

保留原 Thresholds：edge ratio [.5,2]、area ratio [.25,4]、normal reversal cosine 0、relative area epsilon 1e-12、relative intersection epsilon 1e-7、aspect warning 10。相对质量拒绝新增退化、自交、反向法线、越界边/面积比及新 reflected/singular bone frames。基线已有 anomalies 完整保留；`baseline_absolute_quality_validated=false` 不变。

完整 o_head surface 使用相同每正面积三角形覆盖及面积加权采样，evaluation_samples=4096、seed=7381。原 common gates：RMS .002，P95 .005，max_sampled .03 game units，oriented normal P95 45°。采样最大值不代表连续 Hausdorff 上界。这里没有通过 RGB 认证的解剖 face region 或命名 curve。

每份 geometry 分别验证 bytes SHA、actual frame / pose、head、renderer/source ordered topology / bindposes / skin influences，再独立重建实际 LBS world certificate。完整曲面比较只移除每份实际记录的 renderer proper rigid translation / rotation，不拟合全局形状 scale、旋转或 affine。relative normal 审核保留原工具 rigid Kabsch 规则，不扩展到形状 surface gate。

## History 与 raw observables

每份 state 保留四骨完整 runtime records（含全部 private runtime_baseline fields）、完整 ABMX runtime、实际 skin palette ancestors、四骨 local transforms、native59、expression、active blendshapes、renderer rotation / position / lossy scale、pose/visibility signature 和 trace cursor。neutral→varied 和 early→late 比较保留变化记录两端的原值；观察 trace 以文件 SHA 绑定。参数相同不蕴含 pose 相同；不从 settle_frames 推算 Apply 次数，不归因外部 writer。

每 submesh 额外输出实际 BakeMesh raw xyz spans。它们是整个 asset submesh 在原始 renderer 单位中的 extents，不能称为解剖脸宽、脸长或实物距离。frame policy 只是原数据记录，未独立认证；此 batch 同时改变多个骨通道，不提供 single-factor physical response 或跨 base 等价结论。

## 执行与验证

```powershell
.venv\Scripts\python.exe -m tools.stateful_multibone_fresh_quality.run --manifest C:\Users\13666\Workspace\HS2Mod\artifacts\infrastructure_live_20261005\abmx_multibone_fresh_v1\live_cases.json --established-config outputs/stateful_base_comparison_20261005/comparison_config.json --out-dir outputs/stateful_multibone_fresh_quality_20261005/actual_v1
.venv\Scripts\python.exe -m tools.stateful_multibone_fresh_quality.verify --run-dir outputs/stateful_multibone_fresh_quality_20261005/actual_v1 --out outputs/stateful_multibone_fresh_quality_20261005/verification_v1.json
.venv\Scripts\python.exe -m unittest tools.stateful_multibone_fresh_quality.test_contract -v
```

7 项 synthetic tests 只证明 verifier 的拒绝与记录行为。`verify` 重新读取全部实际 geometries、native59、all registered baseline modifiers、SHA、cache/history 差异和完整 raw vertex temporal delta；它不重复已由 main 计算的 triangle intersections / full surface quadrature。输出以 exclusive-create 保存，失败或 partial run 保留，不覆盖为 green。

## 实际结果

实际 main、独立 verify、history investigation 全部 exit0。四 head 的 4 variednative identity +8 early/late primary geometry、4 neutral history geometry 都逐份重读，独立实际 LBS certificate 完成。`actual_v1/summary.json` SHA256 为 `5e01cdab1b7d5e0a73017bb0a8c5b677ec51a7fbe1d782b9da6387507c346c83`；fresh live manifest SHA256 为 `2815c277b513306c9c6b77524cd3d0cdfa8d12d5345ac655244624af398f4bbc`。

8/8 single-o_head **相对 distortion quality gates 通过**，4/4 early/late **完整曲面 temporal gates 失败**。它们是不同判断，不能合并成整体质量通过。

| Head | Temporal RMS | Temporal P95 | max_sampled | normal P95° | 固定时序 gate |
|---|---:|---:|---:|---:|---|
| 0 | .003784177 | .010855809 | .019704566 | 4.944933 | RMS/P95 fail |
| 1 | .003906568 | .011145318 | .020998193 | 5.270270 | RMS/P95 fail |
| 2 | .003699175 | .010751203 | .019885248 | 5.121728 | RMS/P95 fail |
| 3 | .003995637 | .011458025 | .019537146 | 5.050058 | RMS/P95 fail |

相对实际 variednative baseline 的 surface distance 仅 head1 early 满足全部 reference distance 阈值；head0/2/3 early RMS 超出 .002，四份 late RMS/P95/max_sampled 全超出固定门槛。这是改形之后相对 reference 的差异诊断，不是 target likeness。

基线异常也没有被清零：head0/1/2/3 self-crossing counts 为 171/190/168/150、nonmanifold vertices 为 8/0/1/3、aspect warnings 为 0/10/6/2；四份 baseline 的退化 triangles / negative bone frames / singular bone frames 均为0。跨 enabled head meshes 相对基线的新 crossing pair counts early 为 125/125/96/82、late 为 188/293/222/231。这些相交数据是实际几何诊断，未验证为材质可见缺陷。baseline_absolute_quality_validated 始终 false。

四个 head 一致：neutral→varied 的 48 个 skin palette ancestor local transforms 有变化；四 selected 私有缓存的 `_posBaseline` 改变，ChinTip 另有 `_sclBaseline`，Chin_rs 另有 `_sclBaseline` / `_rotBaseline`。这些差异保留实际原值，证明不能将 neutral state 当 variednative baseline。

early→late 的 native59、expression、active blendshape、四骨参数、source/renderer identity/visibility/rotation/lossy scale 相同；四骨私有 `runtime_baseline.fields` 全部相同，只有其 snapshot frame_count 元数据改变。实际 ancestors 的 local position 只在四个 selected bones 改变，其他 local transforms 全部同值。renderer_position 与 pose_signature 不同；移除实际记录的 renderer rigid pose 后曲面差异仍然超门槛。该证据不识别外部 writer，也不证明共同 Apply count 或全局 pose 等价。

独立 `verification_v1.json` 对 8 windows、12 primary +4 neutral geometries、全部 registered baseline modifier identity、实际 cache/history 和完整 raw temporal vertex delta 复查通过；`history_investigation_v1.json` 区分 private fields 与 metadata/world 字段。验证器不重复 main 的 intersections / quadrature。

质量报告不等于 anatomy、target likeness、shader 可见性或整个基础设施 goal 完成。此任务没有读取72张人物图，也没有恢复程儿任务。
