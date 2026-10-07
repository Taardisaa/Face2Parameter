# 暂停角色实际 head 3 的基础设施覆盖

2026-10-04 只读检查角色卡 `E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/程儿_蓝裙_v1.png`，实际 `headId=3`。先前的参数 atlas、真实 Unity 对照和共同底模 benchmark 只覆盖 head 0／1／2；不能将它们直接推广到暂停人物实际使用的底模。

检查前 `data/hs2_head/head_3/` 不存在。卡片 SHA-256 为 `42d2aaa1797ac305a1aa16636fa7c53d80249c5717720107bf2d05768bec3901`，提取与 sanity 后仍完全一致。没有操作游戏、改卡、模型配置或核心代码。

## 卡片驱动的实际资产提取

使用未修改的 `scripts/hs2_extract_head.py` 的 `main()` 与 `--card`，通过游戏 ChaList 解析，而非用 head ID 猜 bundle：

| 字段 | 实际列表解析结果 |
| --- | --- |
| Head ID | 3 |
| Bundle | `E:/HoneySelect2_ArcticFox/abdata/chara/50/fo_head_50.unity3d` |
| Prefab | `p_cf_head_03` |
| Shape animation | `cf_anmShapeHead_03` |
| Head material | `cf_m_skin_head_03` |

提取脚本默认还写共享 customhead、textures、composite 与 card manifest。为保持这次任务范围，仅在运行时把 `OUT` 指向 `outputs/head3_coverage_20261004/extracted/`；复用复制到 staging 的现有 ChaList 缓存，并将其缓存写入位置隔离。脚本源码未修改，没有枚举提取其它 head。

Fresh staging 首次执行暴露了脚本假设 `textures/` 已存在的问题；创建该 staging 目录后原脚本重跑成功，没有改核心逻辑。最终只把新 `head_3/` 目录复制到 `data/hs2_head/head_3/`。新共享 texture／composition／card manifest 留在 staging，未覆盖已有共享缓存。生成的 customhead 表与现有 canonical 表逐内容相等。

新 head 3 几何缓存包含：`3650` 顶点、`6952` 三角形、`43` skin bones、`74` skeleton bones、`7` 个独立 submeshes、`8` 份材质和实际 `cf_anmShapeHead_03` 的动画表。只选中渲染 prefab 下的 `o_head`，没有选用 `_hit` 碰撞网格。

资产解析记录为 `outputs/head3_coverage_20261004/resolved_asset.json`；提取的卡片 manifest、appearance 资源在其 `extracted/` 下。canonical 新目录的文件哈希保留在 sanity JSON。

## 有限离线 sanity

始终使用项目 `.venv/Scripts/python.exe`，专属只读验证脚本为 `outputs/head3_coverage_20261004/validate_head3.py`：

```powershell
.\.venv\Scripts\python.exe outputs\head3_coverage_20261004\validate_head3.py
```

输入为原生完整 59 维全 `0.5`，ABMX identity，`vanilla` profile，CPU float64。

| 检查 | 实际结果 |
| --- | ---: |
| NumPy HeadRig 与 TorchHeadRig 最大逐顶点距离 | `4.47545e-16` |
| 全部曲面 self-match 双向面积加权 RMS | `1.00982e-16` |
| 全部曲面 self-match 最大采样距离 | `5.04635e-16` |
| Same-head baseline 相对质量 | valid，无新增退化／穿越／反转／异常拉伸 |
| Baseline 本身退化面 | 0 |
| Baseline 本身非邻接内部交集 | 150 对 |
| Boundary edges | 354 |
| Nonmanifold edges／vertices | 0／3 |

距离按每个正面积三角形至少一个 query、每面权重和等于面积；不是只比较骨中心或高密度顶点。浮点自匹配近零证明输入、缓存与共同曲面指标接通，不证明完整底模表达力、视觉质量或人物身份。Baseline 已有的 150 对交集与 3 个非流形顶点明确保留；相对质量 valid 不等于资产没有任何几何问题。

完整结果为 `outputs/head3_coverage_20261004/head3_sanity.json`，摘要为 `head3_summary.json`。此阶段没有 head 3 的实际 Unity 快照，因此**不宣称 Unity runtime parity**。TorchHeadRig 仍不含 expression blendshape 与外部 ancestor；没有验证 head 3 的范围外采样或任何 ABMX probe。

在本节离线提取阶段，共同底模 CLI 的显式 head admission 仍为 0／1／2，尚未改工具或模型配置；HeadRig／TorchHeadRig 本身已经可以直接读取这个新 head 3 缓存。随后取得下节的实际 Unity 数值证据，再在独立完整59维任务中将 head 3 纳入工具并完成四底模有界搜索，结果见 `hs2_base_comparison.md`。核心和 ML 配置未修改，程儿人物制作目标保持暂停。

## 五个同次冻结 Unity 快照的独立数值对照

随后集成端在独立测试游戏采集 head 3 baseline 与原生控制 0／24 的 `-0.25`、`1.25` 四个扰动，并在同一冻结窗口导出多视角图像和几何。该 agent 只读保存文件，没有调用游戏。开始分析前确认 `HS2Mod/artifacts/infrastructure_live_20261004/paired_head3_cases/live_cases.json` 的 `snapshot_restored=true` 与 `expression_restored=true`；集成端也明确确认测试完成。

使用现有、未修改的独立工具，显式选择安装的 SliderUnlocker 18.2 采样约定：

```powershell
.\.venv\Scripts\python.exe tools\unity_parity\compare_series.py C:\Users\13666\Workspace\HS2Mod\artifacts\infrastructure_live_20261004\paired_head3_cases\live_cases.json --out outputs\unity_parity_20261004\head3_cases --profile slider_unlocker_18_2 --renderer-uniform-scale
```

该命令退出 `0`，每个输入文件的 SHA 与 manifest 核对一致。五组快照都 frame-stable，10 个 Renderer 的实际骨矩阵 LBS 重建与 BakeMesh 都在工具 `1e-5` 归一化容差内。8 个 active 头部 Renderer 唯一匹配 `scale_free_trs`；2 个 inactive body／silhouette 的 `o_tang` 匹配不唯一，保留路径和模糊状态，不能据此推广其世界坐标约定。

缓存／实际 o_head 同为 `3650` 顶点和 `6952` 面，拓扑顺序、bone palette、权重对应检查通过；source 坐标最大 component 差 `5.83801e-8`，bindpose 最大差 `5.00641e-8`，在独立 cache 对照阈值内。

| Case | 最大 world 距离误差 | 最大 bbox 归一化误差 |
| --- | ---: | ---: |
| head 3 baseline | `2.62565e-6` | `8.80736e-7` |
| native 0 = -0.25 | `2.60049e-6` | `9.01594e-7` |
| native 0 = 1.25 | `2.62615e-6` | `8.50771e-7` |
| native 24 = -0.25 | `2.77909e-6` | `9.32207e-7` |
| native 24 = 1.25 | `2.71785e-6` | `9.11666e-7` |

比较实际应用捕获的 `head.e00_defo=100` blendshape 帧，再使用从 renderer 世界矩阵独立记录的均匀 ancestor 因子 `0.91525143822642`，移除 proper rigid pose／translation。祖先证据含 `cf_N_height=0.9` 与 `cf_J_Head_s=1.0169462`；这不是从顶点残差拟合的新 scale。`unit_scale=1`，未应用报告内的 suggested fitted scale，没有 shear／仿射补丁。未补祖先因子的约 `4%` normalized rigid 残差仍保留在原报告，不能悄悄丢弃。

头部 skin-bone 的最大 local position 误差为 `1.19209e-8`；非头部 skin palette 的 eye-look 骨有动画差异，单独记录，不误解为这五组原生 o_head 形变失败。其它头部 Renderer 的实际骨矩阵 LBS 认证与 offline 原生参数→o_head 对照是两类证据，不能把前者当作已验证原生参数→全部眼球／睫毛的模型。

每组逐文件报告与 `series.json` 位于 `outputs/unity_parity_20261004/head3_cases/`；另有 `head3_verification_summary.json` 保留输入／报告哈希、恢复状态、候选约定、source／bindpose 误差、表情与 ancestor 因子。

这是对 **这五个 head 3 快照** 的真实 Unity 数值 parity 支持，补足暂停人物实际底模的必要覆盖。仍未验证 head 3 其它 57 个控制、其它幅度、任何 ABMX、完整多视角轮廓验收或人物相似度。后续完成的已知缓存 target 全59维拟合 benchmark 见 `hs2_base_comparison.md`；它没有为新的参数组合增加 Unity 认证。不能把这个小集或有限搜索推广为 head 3 全表达力结论或宣布程儿已完成。
