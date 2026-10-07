# Unity 人物网格质量检查

`tools/geometry_quality/mesh_quality.py` 读取 HS2 MCP `MakerGeometryService` 的 schema 1 JSON，执行完整三角形、拓扑、法线、骨骼矩阵和逐底模 baseline 诊断。脚本只读取文件，不连接游戏，不修改卡、参数、资产或模型。

这些指标用于阻止优化器拉坏脸部结构，**不测量真人相似度，也不证明人物制作已完成**。输出保留具体三角形／边／骨骼索引，不把多个问题压成一个没有解释的总分。

## 可执行接口

从 Face2Parameter 仓库运行，始终使用项目 Python：

```powershell
.\.venv\Scripts\python.exe tools\geometry_quality\test_mesh_quality.py

.\.venv\Scripts\python.exe tools\geometry_quality\mesh_quality.py C:\...\geometry.json --out outputs\geometry_quality_20261004\initial.json

.\.venv\Scripts\python.exe tools\geometry_quality\mesh_quality.py C:\...\candidate.json --baseline C:\...\neutral.json --meshes o_head --out outputs\geometry_quality_20261004\comparison.json
```

也可在 Python 中调用：

```python
from tools.geometry_quality.mesh_quality import analyze_snapshot, Thresholds
report = analyze_snapshot(snapshot, baseline, Thresholds(), space="baked_raw",
                          mesh_names={"o_head"}, visible_only=False)
```

默认处理所有选中的 Renderer，保留 `renderer_path/enabled/active_in_hierarchy`。同名 Renderer 不被字典覆盖；例如三个 `o_tang` 可能分别属于实际头部、inactive body 和 inactive silhouette。`--visible-only` 只保留 `enabled && active_in_hierarchy`，这仍不等于通过相机视锥、遮挡和 shader 可见性验证。

默认 `--space baked_raw`，逐 Renderer 使用原始 BakeMesh 坐标。可选 `world_renderer` 和 `world_scale_free` 对应导出器的两个世界转换候选；工具不认证它们的 scale 约定。应先完成 Unity／离线数值对照，才能把候选世界空间用于跨 Renderer 的几何解释。

上述逐网格工具不提供跨网格穿插结论。默认 raw 空间来自不同 Renderer，不能直接拼在一起求相交；经过独立 LBS 认证的共同世界坐标可使用下面的 `cross_mesh.py` 扩展。

## 可选的认证跨网格检查

```powershell
.\.venv\Scripts\python.exe tools\geometry_quality\test_cross_mesh.py

.\.venv\Scripts\python.exe tools\geometry_quality\cross_mesh.py C:\...\candidate.json --certificate outputs\unity_parity_20261004\paired_base_cases\candidate.json --baseline C:\...\baseline.json --baseline-certificate outputs\unity_parity_20261004\paired_base_cases\baseline.json --out outputs\geometry_quality_20261004\candidate_cross_mesh.json
```

`--certificate` 是 `tools/unity_parity/compare_export.py` 独立数值重建产生的报告。工具严格绑定输入文件字节的 SHA-256、每个 `(mesh_name, renderer_path)`、源几何哈希和顶点数，要求快照帧一致、没有覆盖实际皮肤权重数、认证误差不高于 `1e-5` 归一化容差，并且世界矩阵候选唯一。候选缺失、模糊、认证失败或旧文件证书会直接拒绝。Baseline 也必须提供其自身证书，不能借用候选或其它案例的证书。

选择范围固定为 `enabled && active_in_hierarchy` 且位于导出 `character.head_root_transform_id` 子树下的 Renderer。没有依赖名称猜测头部；同名路径分别保留，inactive body／silhouette 在认证前排除。至少需要两个合格 Renderer。输出保留选中及跳过的路径、原因、矩阵候选和误差下限。

每个 Renderer 的 raw BakeMesh 顶点通过其**自身已认证矩阵**转换到世界空间，随后才求不同 Renderer 的真实三角形交集。来自不同网格的相同顶点编号没有邻接含义，不因此排除。边、点和共面薄片接触仍单独计数，不作为内部穿越。每一对的长度容差为 `1e-7 * combined_bounds_diagonal + residual_A + residual_B`；每个 residual 是证书 LBS 最大距离与世界坐标序列化一致性最大距离之和。这是保守的数值误差处理，不是毫米精度或解剖置信区间。

跨网格 baseline 比较要求完全相同的 Renderer 身份集合、源哈希、顶点数和三角形索引。与逐网格诊断不同，此处不在路径变化或源资产变化时回退对应；输出 `renderer_selection_mismatch` 或 `source_or_topology_mismatch` 并停止新增对比较。通过时报告 `new_crossing_pairs`、`resolved_crossing_pairs` 和未变对数，每一对包含两个路径及各自的局部三角形索引。

眼球嵌在眼眶、牙齿穿过口腔内壁、睫毛／泪线叠层等可能是资产的正常构造。三角形对数量也会在交线滑过网格时变化，不能将新增索引对直接等同新增视觉缺陷。当前工具不判断 shader 透明度、遮挡、解剖区域或真人相似度。证书是独立提供的数值证据，并非加密签名的安全证明；认证只作用于对应快照，不能推断全游戏统一矩阵约定。

## 指标与阈值

每个网格分别输出分布、具体异常索引、完整拓扑和相交记录。长度与面积为所选空间的游戏单位；没有毫米标定。

| 项目 | 默认阈值／定义 | 如何解释 |
| --- | --- | --- |
| 退化三角形 | `area <= 1e-12 * bounds_diagonal²` | 无量纲相对面积阈值；报告索引，不将它们送入普通三角形相交算法 |
| Aspect | `sqrt(3) * longest_edge² / (4 * area)`；提示值 `>10` | 等边为 `1`；过大表示细长。资产原本可能包含细长三角形，不自动失败 |
| 相交长度容差 | `1e-7 * bounds_diagonal` | 与 Unity float32 来源相称的初始数值容差，可修改；不是解剖误差阈值 |
| 边界 | 一条索引边只属于一张面 | 开放头部、口腔和 UV seam 可能有合法边界；不能直接理解为漏洞 |
| 非流形边 | 一条索引边属于 `>2` 张面 | 区分真实拓扑连接与几何重合的独立顶点 |
| 非流形顶点 | 顶点 link 不是单个 cycle 或单个 path | 检出 bow-tie 及不连续面扇；不只检查边的 incidence |
| Winding | 二面共享边的有向使用方向相同 | 指示局部面绕序不一致；单独保留边 ID |
| 法线方向 | `dot(geometric_cross_normal, mean_vertex_normal)` | 报告分布和负值索引；不先验宣称整个资产哪面为外侧 |
| Bone determinant | 实际世界矩阵线性块 determinant `<0`；近奇异为 `abs(det)<=1e-12` | 负值说明该坐标框架反射，可能来自刻意镜像；不是表面必然翻转的证明 |
| Baseline 边长比 | 提示 `<0.5` 或 `>2` | 比较同一索引边，保留整体缩放，报告每条边的比值 |
| Baseline 面积比 | 提示 `<0.25` 或 `>4` | 整体放大三倍会有边长比 `3`、面积比 `9`，不会被相似配准消掉 |
| Baseline normal reversal | 移除整体刚体运动后，法线 cosine `<0` | 表示相对参考的方向反转；不等于三维体积单元 inversion 的严格证明 |

所有分布报告 min、p01、p50、p95、p99、max。Aspect 退化项单独计为 infinity 数量，JSON 不写非标准 `NaN/Infinity`。

阈值全部通过 CLI 对应参数开放，例如 `--max-edge-ratio 1.5`、`--relative-intersection-epsilon 1e-8`。应由已验收人物、合法表情和游戏数值噪声建立项目阈值；这些默认值是诊断起点，不是全游戏通用的放行标准。

## 真正三角形相交，而不是包围盒重叠

检测分两阶段：

1. 沿包围盒跨度最大的轴 sweep，利用三维 AABB 筛选候选。
2. 对**不共享任何顶点索引**且非退化的候选执行 narrow phase。

非共面三角形分别与对方平面求交，投影到两平面的交线并计算线段区间重叠。共面情况按法线主轴投影到二维，以三条半平面裁剪凸多边形，计算实际重叠面积。AABB 计数只存在于 `aabb_candidate_pairs`，不进入真实相交计数。

分类如下：

- `proper_crossing`：交线有超过容差的长度，且对方平面穿过两个三角形的内部。
- `coplanar_overlap`：共面多边形有超过面积容差的正面积重叠。
- `edge_contact`、`point_contact`、`coplanar_contact`：边／点／共面薄片接触，单独计数，不并入内部穿越。

共面面积容差使用 `epsilon * projected_perimeter_bound / 2`，防止 seam 处小于长度容差的薄片被夸大为确定面积重叠。共面面积计算先减去局部原点，避免世界坐标较大时 shoelace 的消减误差。

每条记录带三角形索引、相交长度或面积、连通组件 ID。组件自身穿越和组件之间穿越分别计数；不同组件刻意相接也可能存在真实穿越，不能把数量直接当作视觉缺陷。

严格按要求排除了共享顶点索引的邻接对。因此相邻面即使折叠后在共享顶点之外相交，也不会由这个检测器报告。重复面的索引组另由拓扑检查输出。UV seam 的不同索引顶点可能几何重合，仍进入 narrow phase，通常作为接触报告；工具不焊接顶点改变原有编号。

这是带显式数值容差的浮点算法，不是无限精度 exact predicate。接近共面、非常狭窄或距原点极远的案例应调整容差并人工复核。没有把“候选没有命中”宣传成全部几何无缺陷。

## Baseline 比较契约

优先通过 `(mesh_name, renderer_path)` 匹配；路径变化时，只在该名称对应唯一 Renderer 的情况下回退。多份同名网格无法唯一匹配时返回 `unmatched_ambiguous`，不会任意选择 inactive silhouette。

逐 mesh 要求相同顶点数和完全相同的三角形索引顺序。不同底模或改了 topology 时返回 `topology_mismatch`，不计算逐点形变。如果源资产哈希不同，报告仍会明确 `source_hash_identical=false`；相同 topology 本身不能证明不同资产的顶点语义一一对应，需额外标定。

整体配准使用 determinant 为正的 Kabsch 刚体旋转和位移，不去掉缩放、不允许用反射隐藏负向变形。报告配准后的位移和法线方向变化、原始边长／面积比，以及相交集合相对 baseline 的新增和消失三角形对。

改变表情、眨眼、骨骼姿态或皮肤资产可能同时改变这些指标。比较诊断时应固定参数之外的条件，在相同中性表情与冻结／settle 姿态下导出。服务本身不冻结，工具不会自行把差异归因于某个滑杆。

## 已运行的验证

2026-10-04 的 18 个解析测试全部通过，包括：

- 等边三角形面积与 aspect、退化三角形、非流形边和 bow-tie 顶点。
- 正确三维穿越、共面正面积重叠、合法共享边、共面边接触、平行分离。
- AABB 重叠但三角形实际分离的反例。
- 旋转／平移／缩放下的穿越，较大平移下的共面面积。
- 负 determinant、局部折叠后的相对法线反转、刚体运动不会产生翻转，三倍放大保留边／面积比。
- Topology 不一致停止逐点比较，同名 visible／inactive Renderer 不被覆盖。

同时检查了真实 Unity 导出 `HS2Mod/artifacts/infrastructure_live_20261004/head2_initial_fixed/geometry.json`，不是离线伪造的游戏快照。全部 `10` 个 Renderer 在约 `4` 秒内完成，结果位于 `outputs/geometry_quality_20261004/head2_initial_all.json`。

该快照的 `o_head` 有 `4439` 顶点、`8516` 三角形，退化面 `0`，边界边 `366`，非流形边 `0`、非流形顶点 `1`，winding 不一致边 `0`，连通组件 `3`。Aspect 中位数约 `1.64`，max 约 `17.64`，默认提示 `6` 张细长面；导出法线与三角形几何方向没有负点积，骨骼世界矩阵没有负 determinant。

Narrow phase 检出 `168` 对非邻接内部穿越；牙齿网格检出 `158` 对，主要发生在两个牙齿组件之间。**这不表示这些都是本次调参导致的缺陷。** 当前快照不是已验收的中性基准，口腔／牙齿内部及资产原生结构需要结合位置和 baseline 检查。接触数另外保留，不混作穿越数。

对同一真实头网格的 baseline 自比较，逐边比值均为 `1`，方向反转与新增穿越均为 `0`。结果文件是 `outputs/geometry_quality_20261004/head2_self_comparison.json`。

### 原生外推与 ABMX 的 15 个真实导出案例

另对集成端提供的 `head2_cases/` 做了批量离线分析，以 `baseline.json` 的原生 `59` 个值全部 `0.5` 为共同参考：

```powershell
.\.venv\Scripts\python.exe tools\geometry_quality\audit_case_set.py C:\Users\13666\Workspace\HS2Mod\artifacts\infrastructure_live_20261004\head2_cases --out-dir outputs\geometry_quality_20261004\head2_cases
```

批量脚本只读取 schema 1 几何文件，跳过 `live_cases.json` 等操作记录；缓存不可变 baseline 的完整诊断，逐案例保存报告和 `case_index.json`。工具本身没有操纵游戏。

全部 `15` 个案例的头部 topology 匹配，源哈希和表情配置相同。该中性 baseline 自身存在 `164` 个内部穿越对，应扣除其集合来判断新增问题：

| Case | 新增穿越对 | 消失穿越对 | 法线方向反转面 | 默认异常边数 |
| --- | ---: | ---: | ---: | ---: |
| baseline 自比较 | 0 | 0 | 0 | 0 |
| 四个 ABMX length／position／rotation／scale 轻微扰动 | 各 0 | 各 0 | 各 0 | 各 0 |
| `native_0_-0.25`／`native_0_1.25` | 0／0 | 0／0 | 0／0 | 4／2 |
| `native_24_-0.25`／`native_24_1.25` | 0／0 | 0／0 | 0／0 | 8／2 |
| `native_47_-0.25`／`native_47_1.25` | 6／4 | 8／4 | 4／0 | 20／11 |
| `native_4_-0.25`／`native_4_1.25` | 0／0 | 0／0 | 0／0 | 0／0 |
| `native_54_-0.25`／`native_54_1.25` | 0／0 | 0／0 | 0／2 | 2／2 |

`native_47_-0.25` 的穿越总数从 `164` 降到 `162`，但仍**新增了 6 对**，说明单看总数量会漏掉新问题。该案例最小边长比约 `0.153`，最小面积比约 `0.141`。`native_24_1.25` 有三角形面积比约 `0.103`。原生范围外仍能产生有效且局部激烈的形变，需要逐控制的质量约束。

这些是当前底模、特定姿态和具体扰动的几何诊断。耳部 `native_54_1.25` 的方向反转提示也可能涉及允许的局部转动，不能仅凭法线角就宣布体积翻转或视觉失败。四个轻微 ABMX 扰动未触发这些指标，不代表全部 ABMX 范围安全。表情配置一致也不替代完整 pose／成对拍摄一致性验证。

### 三个底模的严格冻结成对导出

对 `paired_base_cases/` 的 13 个快照按各自底模 baseline 重新验证，逐网格结果保存在 `outputs/geometry_quality_20261004/paired_base_cases/head_0/`、`head_1/`、`head_2/` 的 `case_index.json`。捕获与几何导出由集成端在同一冻结窗口完成；本工具仍只离线读文件。

| 底模／控制 | -0.25／1.25 新增内部穿越对 | -0.25／1.25 法线方向反转面 | -0.25／1.25 异常边 |
| --- | ---: | ---: | ---: |
| head 0／native 0 | 0／0 | 0／0 | 4／2 |
| head 0／native 24 | 0／0 | 0／0 | 16／12 |
| head 1／native 0 | 0／0 | 0／0 | 4／2 |
| head 1／native 24 | 0／0 | 0／0 | 10／8 |
| head 2／native 47 | 6／4 | 4／0 | 20／11 |

head 2／native 47 的新增内部穿越和方向变化在严格成对冻结导出中重复出现，不能仅用先前拍摄与导出之间的姿态漂移解释。这里只验证该底模的两个扰动点，不推广为所有控制或底模的全范围结论。

认证跨网格检查另有 10 个解析测试通过，覆盖不同 raw 空间的旋转／非均匀缩放、同名路径保留、合法跨 Renderer 接触、逐文件与源哈希及路径拒绝、模糊候选拒绝、宽松容差与影响数覆盖拒绝、inactive／头部之外排除、以及单独认证 baseline 的新增对和源变化停止比较。它们验证读取与几何算法行为；实际游戏的 LBS 认证来自独立报告，不由这些合成证书代替。

head 2／native 47 `-0.25` 的真实认证检查选择 8 个 active 头部 Renderer，排除 2 个 inactive body／silhouette。Baseline 跨网格内部穿越对为 `1651`，候选为 `1662`，新增索引对 `330`、消失 `319`；其中 head／tooth 新增 `322`、消失 `310`。这些主要是口腔／牙齿交集及其位置变化，不能称为 330 个新缺陷。完整逐对报告保存在 `head_2_native_47_-0.25_cross_mesh.json`，包含相交和接触的分类及数值容差。

同底模 `1.25` 的跨网格内部穿越对为 `1677`，新增索引对 `150`、消失 `124`，接触另计 `16`；报告为 `head_2_native_47_1.25_cross_mesh.json`。两份候选均使用各自逐文件证书以及 baseline 自身证书，源与 topology 匹配。Baseline 和候选已有大量设计内交集，仍需要区域对应和图像检查后判断局部缺陷。

仍需扩大原生范围、外推范围和 ABMX 扰动的控制／幅度覆盖，并形成区域／语义对应。当前检查不覆盖闭合体积校验和所有 shader 位移，也不证明程儿的身份特征已重建。
