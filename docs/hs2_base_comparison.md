# 共同曲面评估与有界底模搜索

`tools/base_comparison/` 提供可执行的离线共同评估器、TorchHeadRig 有界搜索、实际缓存 benchmark。它比较完整 `o_head` 的所有三角形，包括耳朵、口腔内部和其它连通组件；不使用骨骼中心或当前候选的人脸关键点作为 3D 真值。

当前验收范围明确是 **o_head 曲面**。单独眼球、睫毛、泪线、牙齿等 Renderer 和多视角轮廓尚未合并成最终人物验收，因此这个 benchmark 不表示完整底模表达力比较已经完成。输入契约预留 `extra_groups`，报告其名称和未评估状态；当前不会偷偷把它们混入头部指标。

## 运行

在 Face2Parameter 根目录始终使用项目 Python，不连接游戏：

```powershell
.\.venv\Scripts\python.exe tools\base_comparison\test_base_comparison.py
.\.venv\Scripts\python.exe tools\base_comparison\benchmark_cached.py --device cpu --iterations 20
.\.venv\Scripts\python.exe tools\base_comparison\search.py C:\...\comparison_config.json --device cpu --out outputs\base_comparison_20261004\comparison.json
.\.venv\Scripts\python.exe tools\base_comparison\evaluate.py C:\...\source.json C:\...\target.json --baseline C:\...\same_head_baseline.json --out outputs\base_comparison_20261004\pair.json
```

`cpu` 是默认，benchmark 使用 2 个 CPU 线程；`cuda` 可显式开启。没有下载、训练或 live MCP 调用。

小搜索配置示例：

```json
{
  "target": {
    "kind": "rig_target",
    "head_id": 2,
    "native59": ["这里必须是完整59个实际数值"],
    "sampling_profile": "vanilla"
  },
  "heads": [0, 1, 2],
  "modes": ["native", "installed18.2", "native+ABMX"],
  "seed": 8245,
  "evaluation_seed": 7381,
  "search": {
    "iterations": 30,
    "restarts": 1,
    "optimization_samples": 128,
    "evaluation_samples": 4096,
    "learning_rate": 0.03,
    "normal_loss_weight": 0.0
  }
}
```

示例中的说明字符串不是合法数值输入；真实可运行配置由 `benchmark_cached.py` 写为 `outputs/base_comparison_20261004/known_reachable_config.json`。省略 `active_native` 时搜索全部 59 个实际原生值；也可显式选索引子集。其它原生值固定为 `initial_native59`，默认全 `0.5`。每个 restart 保存初值、实际 59 维结果、ABMX 实际值、上下界、seed、逐迭代损失和选中迭代；全局配置、阈值及底模缓存文件 SHA 一并保留。

## 输入、单位和作用范围

`rig_target` 由真实已提取的 `HeadRig(head_id)` 与 `TorchHeadRig` 生成，必须提供全部 59 个有限数值，原生采样行为通过 `sampling_profile` 显式选择。完整缓存文件和共享形变方程的哈希写入 provenance，不把不同 head ID 当成相同资产。

`full_mesh` schema 1 格式可由 `contracts.full_mesh_config(Mesh(...))` 产生：

```json
{
  "schema_version": 1,
  "kind": "full_mesh",
  "vertices": [[0,0,0], [1,0,0], [0,1,0]],
  "faces": [[0,1,2]],
  "metadata": {
    "head_id": 2,
    "units": "hs2_cache_units",
    "coordinate_frame": "hs2_cached_head_fk",
    "scope": {
      "surface_group": "o_head",
      "expression_blendshapes": "excluded",
      "external_ancestors": "excluded",
      "body_pose": "cached_rest"
    },
    "asset": {"sha256": "明确的资产证据哈希", "provenance": "来源与坐标标定证据"},
    "content_sha256": "surface.content_hash(vertices, faces)",
    "topology_sha256": "surface.topology_hash(faces)"
  },
  "extra_groups": []
}
```

上述哈希必须实际计算；内容和 topology 哈希不一致、资产证据缺失、未知空间／单位或 scope 不匹配直接拒绝。不同 source／target topology 可以在已声明共同空间中比较曲面；候选与质量 baseline 则必须同 head、同资产和完全相同 topology。外部 metadata 是来源提供者的声明，不是工具自行认证坐标／资产的证明，真人扫描输入必须先完成独立标定。

`hs2_cache_units` 是固定缓存游戏单位，不宣称毫米或米。TorchHeadRig 包含原生滑杆 → Update 方程 → ABMX → FK → LBS，**不包含表情 blendshape 或骨架以外的 ancestor 变换**。默认角色姿态为缓存 rest，而非游戏中性表情的证明。带实际表情、ancestor scale 或未知世界坐标的 Unity JSON 不直接作为当前 target；必须先通过独立数值认证和明确转换形成匹配 scope。真实人物目前没有可信的标定 3D 曲面真值。

可选 `to_comparison_rigid` 是输入来源明确声明的 4×4 rotation／translation，并必须附 `pose_provenance`。只接受正 determinant 的正交旋转；拒绝 scale、反射、shear。评估与搜索不拟合配准、不拟合 scale／仿射来隐藏形变。质量诊断内部原有的刚体 Kabsch 仅用于相对法线方向检查，不改变共同曲面距离。

## 共同距离与表面法线

最终每个正面积三角形至少有一个内部面积均匀采样点，若请求更多采样，再按面积分配。每个三角形的权重总和等于其实际面积。不同顶点密度不会凭空改变该区域的总权重。两个方向各贡献总损失的 `1/2`，不以某个底模的面积或顶点数给它额外权重。

每个 query 求到对方**完整三角形曲面**的距离，而不是最近顶点或最近采样点。KDTree 以三角形中心作候选加速：初始精确距离给上界，中心距离减三角形包围半径提供安全排除条件；其余候选执行平面内投影和三条边投影的精确窄阶段。长三角形也不能被只查固定 K 个中心漏掉。

报告包含 source→target、target→source 和合并双向面积权重的 mean、RMS、p95、`max_sampled`，以及两边完整面积、query 数量、采样方法和 seed。`max_sampled` 是当前全曲面 quadrature 的最大值，**不是连续 Hausdorff 上界**。默认 4096 请求对 8516 面头部实际产生约 8516 个 query，因为每面覆盖优先；提高采样可做分辨率复核。

法线来自几何绕序，按 source query 所在三角形与最近 target 三角形的有向法线夹角比较，双向同样面积加权，输出度数 mean／RMS／p95／max。没有用绝对点积隐藏方向反转。Seam、口腔重叠面、最近面的模糊对应及高曲率区可能造成法线角变化，指标不能自动命名解剖部位。

可选区域格式是 `{name, side: source|target, face_ids, topology_sha256, provenance}`。必须有来源说明、精确绑定该侧 topology 的哈希和对应面的合法索引；只输出该侧区域细分诊断，不改变全局共同损失。没有来源的鼻子、颧骨等 mask 会被拒绝；不能把一个底模的面编号自动借给另一底模命名区域。

## 有界搜索与 ABMX 限制

| 模式 | 原生59维范围 | 采样实现 | ABMX |
| --- | --- | --- | --- |
| `native` | `[0,1]` | `vanilla` | identity |
| `installed18.2` | `[-1,2]` | `slider_unlocker_18_2` | identity |
| `native+ABMX` | `[0,1]` | `vanilla` | 仅 `cf_J_ChinTip_s` |

不是把 vanilla 在范围外 clamp 当成安装的 SliderUnlocker。ABMX 按实际 scale xyz、length、position xyz、rotation xyz 的 10 维布局处理，其余骨骼严格 identity，非 ChinTip 输入直接拒绝。

ChinTip 默认范围只取 identity 到四个真实记录 probe 的逐轴包络：scale `[1,1,1]..[1.08,1.02,1.04]`、length `1..1.05`、position `[0,0,0]..[0.01,0.02,0]`、rotation `[0,0,0]..[2,0,1]` 度。证据来自 `outputs/unity_parity_20261004/recorded_abmx_cases.json` 及对应逐快照报告。**只有 head 2 的这些单独 probe 的局部作用得到 Unity 验证；head 0／1／3、包络内组合、其它幅度、其它骨骼和全 ABMX 泛化没有获得 runtime 通过声明。** 这个箱形范围是小搜索约束，不是全范围安全认证。

搜索使用 projected Adam；每次更新 nearest-triangle 及 barycentric 对应，再对实际 TorchHeadRig 曲面作可微距离。目标与共同坐标固定，source 的采样重要性权重跟随候选真实面积。目标对角线仅作为固定数值归一化分母，不缩放 mesh。可选法线损失权重默认 `0`，最终法线仍参与验收。优化 query 默认每方向 128，最终评估每面覆盖；有限搜索是共同评估器的候选生成手段，不是全局最优证明。

每个 restart 的最小 surrogate loss 候选与同 head 全 `0.5`／ABMX identity 基准都执行最终质量检测。优化中尚未逐迭代实施质量约束，质量无效的中间候选不会凭距离下降获得成功；也不能声称评估过所有中间 quality-valid 候选。

## 共同验收与质量

默认距离阈值为固定游戏单位 RMS `<=0.002`、p95 `<=0.005`、`max_sampled <=0.03`，有向法线 p95 `<=45°`。它们是显式可修改的起始验收阈值，不是已通过真人验证的相似度标准。搜索配置 `acceptance` 或 pair CLI `--acceptance` 可提供阈值 JSON。

候选末尾调用已有 `tools/geometry_quality/mesh_quality.py`，保留完整逐面／逐边和相交对报告。搜索的 rig adapter 提供当前骨世界矩阵；任意 full-mesh pair 的 surface-only adapter 没有骨矩阵证据，不宣称已检查其骨骼 determinant。

严格质量策略拒绝相对同 head baseline 新增退化面、非邻接穿越、法线方向反转、边长比 `<0.5`／`>2`、面积比 `<0.25`／`>4`、新增反射／近奇异骨架框架。既有 baseline 交集单独保留，不把它们全部视为新失败。方向变化、口腔资产内交集可能有合法情况；该策略是保守诊断门槛，不是已经完成的解剖／视觉质量认证。细节及共享顶点邻接排除限制见 `hs2_mesh_quality.md`。

只输出两种查找结果：`found_quality_valid_approximation` 或 `not_found_within_this_search`。后者包含有限搜索、已验收阈值和质量条件的作用，不能推断底模原生不能做，也不能据此自动建议创建新 base。局部 Jacobian、有限优化失败或其它底模当前残差较大都不构成全局表达力否定。

## 实际运行证据

11 个解析测试已通过：平行完整曲面、三角形内部最近点、KDTree 与穷举一致且涵盖长三角形、面积权重、缩放保留、绕序法线差异、未知 space／scope／asset／hash 拒绝、声明刚体与区域来源、零距离退化质量拒绝、baseline 原生交集与新增交集区分、完整 mesh pair 的 same-head baseline 契约。

`benchmark_cached.py` 实際载入 head 0／1／2 缓存，target 由 head 2 的原生第 0 控制 `0.8`、其它 58 个值 `0.5` 生成。benchmark 为可审查的小 harness 验证，只搜索原生索引 0（native+ABMX 额外允许上述 ChinTip 小箱），三个 bounds 模式一致使用完整最终曲面指标；这不是公平探索全部 59 控制后得出的底模能力排名。报告保存完整参数、seed、20 次迭代、阈值、质量和残差，位置为 `outputs/base_comparison_20261004/known_reachable_report.json` 和 `benchmark_summary.json`。

另将真实缓存 head 2 的一条边刻意坍缩，令候选与该损坏 target 完全相同。该 case 可以得到近零曲面距离，但同头 baseline 质量检查必须拒绝新增退化；保存为 `unsafe_exact_match_report.json`。它证明距离不覆盖质量，而非证明不存在任何安全近似。

2026-10-04 实际九组 CPU 运行已完成，20 次更新、每方向 64 个优化 query、最终每个正面积三角形一个 query。九组比较耗时 `673.28` 秒；代码后续已增加每组 checkpoint callback，避免再次只能等全部完成才查看结果。完整距离与质量报告约 15.7 MB，摘要约 36 KB。

| Head／mode | 双向 RMS | p95 | max_sampled | 法线 p95（°） | 质量 | 本次结果 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 0／native 或 installed18.2 | 0.00601401 | 0.01440253 | 0.05725871 | 13.4399 | valid | 未达到共同距离阈值 |
| 0／native+ChinTip | 0.00591492 | 0.01384553 | 0.05725774 | 13.3577 | valid | 未达到共同距离阈值 |
| 1／native 或 installed18.2 | 0.00342103 | 0.00846912 | 0.02906698 | 8.4999 | valid | 未达到共同距离阈值 |
| 1／native+ChinTip | 0.00336543 | 0.00827216 | 0.02906673 | 8.5276 | valid | 未达到共同距离阈值 |
| 2／native 或 installed18.2 | 0.00038076 | 0.00076033 | 0.00094543 | 0.0693 | valid | 找到质量有效近似 |
| 2／native+ChinTip | 0.00044006 | 0.00077507 | 0.00297980 | 0.1299 | valid | 找到质量有效近似 |

Head 2 native 的第 0 值为 `0.806925843`，已知 target 为 `0.8`；相同底模的曲面、共同指标和有界搜索链路得到实际验证。所有 9 个最终选中候选都没有新增内部穿越、退化、法线方向反转或默认异常拉伸，既有 baseline 穿越数分别为 head 0 的 `171`、head 1 的 `190`、head 2 的 `164`。这些既有资产交集保留在报告中，没有被删掉以使质量通过。

此 target 位于原生范围内，native 与 installed18.2 的优化轨迹没有走到范围外，二者残差相同不能证明外推范围没有价值。ChinTip 模式增添自由度后仍可能受有限采样、有限迭代影响，在同底模验证中略高的最终残差也不证明 ABMX 无效。Head 0／1 这里只自由调整了原生控制 0 与选定 ChinTip 小箱，必须保留 `not_found_within_this_search`，不得作为全 59 控制、其它骨骼或其它 base 的能力否定。

损坏 candidate 与损坏 target 的双向 RMS 为约 `1.02e-16`，但新退化面为 `[0,4]`，并触发边长与面积比阈值，质量明确拒绝；没有通过提高距离验收阈值绕过该失败。

## 四个底模的完整59维有限搜索

Head 3 的实际角色卡已核对并提取真实缓存，且集成端提供的 baseline／原生 0 与 24 的四个范围外快照通过独立 Unity 数值对照；详细范围见 `hs2_head3_coverage.md`。在这个证据之后，只在本工具的 admission 中加入 head 3，没有修改核心、ML 的 54／205 维配置或游戏参数。

可运行的固定预算作业：

```powershell
.\.venv\Scripts\python.exe tools\base_comparison\benchmark_full59.py --device cuda --iterations 40
```

输出目录固定为新的 `outputs/base_comparison_full59_20261005/`，已存在时拒绝覆盖；不会改前述单控制 benchmark。该脚本先保存资源与实现哈希，再独立做目标质量检查。第一候选为真实 cached head 3 原生 0=`0.65`、24=`0.60`、4=`0.55`，其余 56 个值=`0.5`，ABMX identity。这是缓存 rig 生成的已知可达完整曲面，不是真人扫描、骨中心或候选关键点真值。

这次第一目标已通过严格同头 baseline 质量门槛，无新增退化、穿越、方向反转或异常拉伸；没有使用更简单 fallback。目标完整 mesh、native59、缓存指纹与质量报告保存在 `target_mixed_0_24_4.json` 与 `selected_target_provenance.json`，之后整个作业不再改变目标。

脚本只在目标失败时按预先记录的较小混合 trial 和单控制 fallback 序列继续，逐个保留失败原因与完整目标；全部失败则保存失败报告，不启动 fitting 或把 unsafe target 宣称为 safe。实际本轮没有触发该路径。

实际资源检查为 RTX 4080 Laptop GPU，CUDA 可用、初始可用显存约 11.6 GB；使用 float64 TorchHeadRig，CPU 线程数 2。搜索包含 head 3／0／1／2 × native／installed18.2／native+ChinTip，共 12 组。每组所有原生索引 `0..58` 均参与 projected Adam，不设置 `active_native` 子集。预算每组 40 次更新、1 restart、每方向 128 个优化 query，最终评估仍是同一完整曲面 quadrature、固定单位及 same-head 质量门槛。

原生两种模式有 59 个有界维度；ChinTip 模式有 59+10 个布局维度，其中 2 个 ChinTip 轴边界相等，合计 67 个非恒定范围维度。每个 checkpoint 保留完整 59 维结果、ChinTip 值、active indices、上下界、seed 和逐迭代记录。原生 profile 为 `[0,1]`，installed profile 为 `[-1,2]`；不能将全59维参数启用理解成所有参数在本次姿态都有非零梯度，或已证明全域最优。

不同模式完全相同的 baseline 曲面／骨矩阵，只在完整快照内容哈希相等后复用诊断；最终距离只在顶点＋topology 内容相等时复用。目标、区域、种子与阈值在整个调用中固定，复用不改变共同指标，也不借其它 head 的质量基准。

每组结束立即写 `head_<id>_<mode>.json` 与 `progress.json`。结束后另写 `comparison_report.json`、`benchmark_summary.json`。RMS 下降而质量失败的 winner 保留在 checkpoint，结果依旧只能是当前预算内没找到质量有效近似，不得宣称底模不能做、自动换 base 或程儿已重建。

2026-10-05 的 12 组实际作业已完成，CUDA 运行约 `561.20` 秒。只检查已完成的实际 artifact，`completion_audit.json` 确认第一混合 target 通过、没有 fallback、target 与实现哈希未变、每组 all59 active／40 次更新／41 次 loss evaluation、最终参数全部符合对应边界。每组 winner 的 59 个原生值都实际偏离 `0.5` 超过 `1e-8`；这证明完整参数搜索确实发生，不等于已测得 59 个相互独立的表达自由度。

下面列出的 RMS／p95／max 是**优化 winner** 的完整曲面残差，包含质量失败者，避免只显示 baseline 而隐藏优化进展：

| Head／mode | Winner RMS | p95 | max_sampled | 新增索引穿越对 | 反转面／异常边／异常面积面 | 质量与当前结果 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 3／native 或 installed18.2 | 0.00092984 | 0.00138494 | 0.01328793 | 0 | 0／0／0 | valid，找到近似 |
| 3／native+ChinTip | 0.00096126 | 0.00155909 | 0.01327608 | 0 | 0／0／0 | valid，找到近似 |
| 0／native 或 installed18.2 | 0.00504073 | 0.01220558 | 0.04858215 | 26 | 0／0／0 | rejected，本次没找到 |
| 0／native+ChinTip | 0.00500368 | 0.01207377 | 0.04848012 | 26 | 0／0／0 | rejected，本次没找到 |
| 1／native | 0.00558077 | 0.01201493 | 0.05457976 | 26 | 10／62／16 | rejected，本次没找到 |
| 1／installed18.2 | 0.00558420 | 0.01199511 | 0.05457647 | 26 | 10／62／16 | rejected，本次没找到 |
| 1／native+ChinTip | 0.00556540 | 0.01201571 | 0.05463156 | 26 | 10／62／16 | rejected，本次没找到 |
| 2／native | 0.00528704 | 0.01190751 | 0.05554775 | 18 | 0／10／0 | rejected，本次没找到 |
| 2／installed18.2 | 0.00528696 | 0.01190464 | 0.05554763 | 18 | 0／10／0 | rejected，本次没找到 |
| 2／native+ChinTip | 0.00526836 | 0.01186827 | 0.05543604 | 18 | 0／10／0 | rejected，本次没找到 |

Head 3 的 native／installed winner 有向法线 p95 为 `1.3516°`，ChinTip 为 `1.4512°`；都满足预先固定的距离与法线阈值，质量 gate 没有新增失败项。RMS 达标不意味着精确恢复：最大采样误差仍约 `0.0133` 固定缓存单位，稀疏区域与 quadrature 分辨率仍需独立复核。

质量失败的 head 0／1／2，正式选中候选保留为各自 quality-valid neutral baseline，RMS 分别为 `0.01309897`、`0.01299444`、`0.01391000`，不把被拒绝的更低 RMS winner 当成成功。新增索引交集可能包含原有口腔交线滑过 triangulation 的变化，严格 gate 的拒绝不是 26／18 个已证实视觉缺陷的数量。

Head 1 installed winner 的原生值范围实际达到 `[-0.03405,1.01247]`，索引 26／28／38／51 超出原生 rail；head 2 installed 索引 26 为约 `-0.009887`。因此本轮确实使用了部分额外范围，而非只把 profile 名换掉；仍没有达到共同距离与质量验收。Head 0／3 的 installed winner 保持原生 rail 内，不能据相同残差推断外推无用。

这次已知 target 是 head 3 缓存的三控制联合形变，不含实际人物身份信息；该 benchmark 完成当时，head 3 的真实 runtime parity 只覆盖已记录的原生 0／24 五个快照。后续完整组合实测单独记录在下面，不覆盖原始 benchmark。Head 0／1／2 本次失败只描述这个 seed、40 次更新、固定 query 和严格 gate 下的有限搜索，不证明这些底模无法在更完整模型／算法／质量策略下近似 target。

## 2026-10-05 完整组合的独立 Unity 验收

`tools/base_comparison/runtime_validation.py` 只读取 root 导出的 `HS2Mod/artifacts/infrastructure_live_20261005/full59_runtime/live_cases.json`，不接触游戏。输入是四个同头 `.5` baseline，以及原 12 组 checkpoint 的 `bounded_search_winner`；原 quality-invalid winner 作为 diagnostic，原正式 selection 不变。运行方式：

```powershell
.venv/Scripts/python.exe tools/base_comparison/runtime_validation.py ../HS2Mod/artifacts/infrastructure_live_20261005/full59_runtime/live_cases.json --out outputs/base_comparison_runtime_20261005/strict_coverage_v2
.venv/Scripts/python.exe tools/base_comparison/test_runtime_validation.py
.venv/Scripts/python.exe tools/base_comparison/chin_runtime_diagnosis.py ../HS2Mod/artifacts/infrastructure_live_20261005/full59_runtime/live_cases.json --out outputs/base_comparison_runtime_20261005/chin_stateful_diagnosis.json
```

每组绑定实际 geometry SHA、完整 59 值回读、实际 ChinTip 四字段与 coordinate modifier、checkpoint SHA 与原参数、实际三张 PNG 的同帧与 pose signature。报告保留实际 animator/expression/blendshape 输入。独立 `unity_parity.analyze_snapshot` 用实际骨 world、源 weights/bindposes、blendshape deltas 重建每个 renderer，再对 Unity BakeMesh 两种候选比较；visible head 必须有唯一认证世界变换、真实 skin-quality 且不 override。按 renderer path 加 source SHA 区分重复 `o_tang`，不把 inactive body/silhouette mesh 当头部证据。

缓存模型比较覆盖完整 `o_head` 顶点和精确 topology；拒绝 source、palette、weights、bindposes 不匹配，验证整个 skin-bone palette 的实际 local position/rotation/scale 与 parent。实际 expression delta 先作用到源顶点，独立记录的 uniform ancestor scale 再作用到缓存 FK；单位固定 1，只以 proper rigid 旋转／平移移除姿态，不拟合 scale／affine。保留不补 ancestor 的误差与 suggested scale diagnostic，但不应用 suggested scale。

固定 gate：engine 与 offline 最大对应顶点误差／目标 bbox 对角线均 `<=1e-5`；skin-bone local position 与 scale L2 均 `<=1e-5`，rotation `<=0.001°`；readback 最大绝对误差 `<2e-6`。这不是人物相似度阈值，也不是 surface quadrature fit acceptance。脚本保存 validator 与共享比较依赖文件哈希、原 benchmark 全部文件哈希；输入或数值失败保留报告，退出非零。输出目录必须不存在，不能覆写历史证据。

实际 16 组全部 10 renderer 的 actual-bone LBS 通过；每组 visible head subtree 的 8 个 renderer 唯一选中 `scale_free_trs`，两个 inactive 额外舌 renderer 仍不区分 scale convention。缓存 native→actual 完整头部 gate **13/16 通过**：四 baseline、八 native/installed 与 head0 ChinTip 通过，最大 normalized residual 在 `8.35e-7–1.12e-6`。head1／2／3 的 ChinTip 组合分别失败为 `0.01329522`、`0.01312170`、`0.00795049`，下巴 local position 同时失败，rotation/scale 基本一致。因此“actual bone 输入能够解释 BakeMesh”和“当前离线原生＋ABMX 参数模型准确”是两项不同验收；不能用前者掩盖后者失败。

`chin_stateful_diagnosis.json` 保留根因证据：安装的 `BoneModifier.Apply` length 分支实际为 `p_next = normalize(current_local_p) * persistent_lenBaseline * L + offset`，而当前 NumPy `_fk_world` 与 Torch `local_transforms` 都用 `p * L + offset`。`CollectBaseline` 仅在没有 length baseline 时收集 `_lenBaseline/_positionBaseline`，普通 shape baseline refresh 不保证更新这两项。

三组失败从 live local minus offset 推得共同 historical length 约 `0.12083234`，准确对应 manifest.before 原 head2／原59值的 ChinTip 长度 `0.12083234752688482`。四个同头 `.5` identity baseline 和 cached rest 的长度约 `0.14142137`，两者 local position 相差仅约 `1.1e-8`；因此根因不是提取了错误的 rest position。head3 原生 post-shape local 为 `[~0,-0.10133619,0.09880002]`，离线乘法预测 ABMX 后 `[~0,-0.09483762,0.10187660]`，实际却为 `[~0,-0.04019047,0.11419033]`，local L2 差 `0.05601732`。

插件每 LateUpdate 在 current local 上 normalize 后再次加 offset，若 native shape 不重新写，方向也随重复应用改变。用独立 before 长度递推，对 head1／2 六次、head3 七次分别与实际 local 相差约 `1.68e-8`、`1.95e-8`、`1.87e-9`。这些次数是离散回溯诊断，不是观测计数，不得变成隐藏拟合参数。head0 的 length 恰为 1，走 position-only baseline 分支，当前模型该组通过。

最小修复应先记录实际 private cache 与 apply timing，确定可复现合同后才同时更新 NumPy／Torch。仅把 `p*L` 换成 `normalize(p)*rest_length*L` 不能解释历史 length 和 repeated offset。后续 root 新 private-cache 快照仍需独立验收；本轮不改 shared core、不调松阈值、不覆盖 benchmark，也不把失败 ChinTip 推广为 runtime-valid ABMX。

随后 root 在独立测试进程重拍同样 16 组，geometry 新增只读 `abmx_runtime`；输入 `full59_runtime_baseline/live_cases.json` 完成并恢复后，严格报告保存于 `outputs/base_comparison_runtime_baseline_20261005/runtime_summary.json`。仍为 **13/16**，head1／2／3 ChinTip 最大 normalized residual 为 `0.01329533`、`0.01312175`、`0.00673608`，其他组合通过；实际 160 renderer LBS 仍全部通过。head3 与第一批误差不同，不能认为相同参数会在未知 ABMX update 历史下给出同一曲面。

新 `chin_actual_private_cache_diagnosis.json` 对同帧、同骨 transform ID、实际 modifier readback 绑定后直接读到四个 head 的 `_lenBaseline=0.120832346`、`_positionBaseline≈[0,-0.07261762,0.09657711]`，安装 ABMX assembly MVID `5442c72a-f463-4bf9-831a-247be87146c8`，private fields 无缺失。缓存 length 与独立 before59 FK 长度差 `1.53e-9`，历史 position 与 before FK local 差 `3.25e-9`；当前 `_posBaseline` 则与各 head mixed59 native local 差 `<=3.54e-9`。四个 head 的 modifier 仍引用同一实际骨 transform ID，直接支持 length baseline 没随 current native baseline／换头更新这一点。

用**实测 private length**进行同一离散回溯，第二批 head1／2／3 都在六次应用处对应实际 local，残差约 `1.76e-8`、`2.05e-8`、`8.22e-9`；第一批 head3 是七次。length 与原始 position 历史现在是观测事实，应用次数仍是诊断推断。报告明确分开二者；head0 的 length=1 时不运行这条递推，另外验证 `_posBaseline + offset`。这些证据支持后续显式 state/timing 合同的修复方向，并没有使当前静态模型的三个失败转为成功。
