# 原生参数曲面效果 atlas

`tools/parameter_atlas/` 把“滑杆数值”转换为可检查的实际曲面响应，读取已有
HeadRig／TorchHeadRig 缓存，不连接游戏、不写人物卡、不训练或下载模型。
它衡量形变和局部控制条件，不计算真人相似度，也不判断某底模全局上做不到某张脸。

## 执行与产物

始终使用项目解释器：

```powershell
.venv/Scripts/python.exe tools/parameter_atlas/test_atlas.py -v
.venv/Scripts/python.exe -m unittest discover -s tools/parameter_atlas -p 'test_*.py' -v
.venv/Scripts/python.exe tools/parameter_atlas/generate.py --out outputs/parameter_atlas --device cpu
.venv/Scripts/python.exe tools/parameter_atlas/inspect_atlas.py outputs/parameter_atlas/manifest.json --out outputs/parameter_atlas/exterior_effects.json
.venv/Scripts/python.exe tools/parameter_atlas/view_displacement.py outputs/parameter_atlas/head_2/slider_unlocker_18_2/atlas.json --control 30 --out outputs/parameter_atlas/control_30.html
```

默认处理缓存中的底模0／1／2、全部59个原生参数（包含耳朵），逐个测试
`-.25,0,.25,.5,.75,1,1.25`。其余参数保持 `.5`。
新增 `--baseline` 后，可以保持当前人物的其余58个参数，而不必回到全 `.5`。
同时生成 `vanilla` 与 `slider_unlocker_18_2` 两种**离线采样语义**。
MCP 的 native 范围模式只是接受写入值的校验策略，不会卸载游戏中已经安装的
SliderUnlocker；不能把范围内写入理解为实际游戏退回了无插件模式。

可重复指定 `--head-id`／`--profile`，或改变 `--levels=...`、`--step`、
`--effect-threshold`、`--device` 和 `--batch-size`。默认数值差分步长为 `1e-3`，
顶点效果阈值为该网格 baseline bbox 对角线的 `1e-6`。

### 当前人物的完整 baseline

`--baseline current.json` 接受以下三种明确格式：完整59个 JSON 数字的数组、
`{"head_id": 2, "native_input": [59个数值]}`，或原生桥导出的几何 snapshot
（读取 `character.head_id` 与 `character.shape_value_face`）。拒绝54维ML向量、
缺项／多项、字符串、布尔值和非有限数值；不会用 `.5` 静默补齐。

```powershell
.venv/Scripts/python.exe tools/parameter_atlas/generate.py --out outputs/current_character_atlas --baseline current_geometry_snapshot.json --profile slider_unlocker_18_2 --device cpu
```

baseline 中记录了 head ID 时，默认只运行该底模；显式 `--head-id` 与记录不一致
会拒绝执行。只提供数值数组时仍可显式选择底模，省略则使用所有已有缓存。
这不表示一个卡的形状向量经过验证可移植到其他底模。

每份 manifest／atlas 保存原始输入文件 SHA-256、实际使用字段、解析后59维向量
的规范 JSON SHA-256，以及全部 baseline 数值。每个样本还记录该控制的 baseline
数值与 `native_delta`；逐参数采样仍使用 `--levels` 的绝对系数，其余58项保持输入值。
`schema_version=2` 增加这些字段，默认 `.5` 行为、缓存和网格测量定义保持一致。

读取游戏 snapshot **仅提取原生头部形状**，不会复制 ABMX、表情、场景姿态、
纹理或祖先缩放。因此这是“当前人物参数向量的离线响应”，并非该人物的完整
游戏状态重建。任意 baseline 输入支持也不等于任意人物组合已通过实机验证。

### 直观检查曲面位移

`view_displacement.py` 导出无需服务或外部依赖的单文件 HTML。可选控制编号和网格，
在同一页面切换其采样系数；X–Y／Z–Y／X–Z三个投影保持相同资产轴、等比例尺度和
固定 framing，不因某次采样自动对齐、居中或缩放。显示 baseline 点、候选点、位移
颜色及稀疏位移线，同时列出系数差、max／RMS、bbox 归一化位移和受影响顶点数。
颜色范围固定为该控制全部样本的最大位移，切换样本不重新归一化。

投影使用该网格全部顶点，包括背面顶点；它是曲面响应的检查工具，不模拟游戏
皮肤遮挡、灯光和纹理，也不把未经认证的顶点区域命名为眼角或颧骨。

### 已验证的游戏曲面查看

`view_live_response.py` 使用原生游戏实际导出的顶点，并复用相同固定轴投影方式：

```powershell
.venv/Scripts/python.exe tools/parameter_atlas/view_live_response.py <游戏sweep的manifest.json> --report <独立审核report.json> --baseline card_input --control 30 --out outputs/game_control_30.html
```

独立 report 必须以 `source_manifest_sha256` 绑定当前 manifest 的原始文件字节。
工具要求 collection 完整、所选 renderer 的59控制 coverage 与重复漂移通过、
所选 case 明确可信，且编号、baseline、采样角色和数值与审核一致。它重新读取
baseline、该控制全部 case 和重复 baseline 的 geometry receipts，逐个核验 SHA、
源拓扑／权重／bindposes／renderer身份，检查 native59、固定 ABMX／表情及祖先
坐标稳定，再用 LBS 与 BakeMesh 一致的曲面重算位移和重复漂移。

这张图包含所捕获的固定 ABMX／表情状态；它不自动宣称其他 renderer、底模或
人物配置同样通过，也不将点云投影包装成游戏皮肤截图。HTML 中保存输入 manifest、
report 与 geometry receipt 的来源及 SHA，用于追溯显示的实际样本。

若独立报告明确标注 `analysis_mesh_scope=["o_head"]`，viewer 仅允许选择 `o_head`，
重建时也使用该组件审核范围；仍保留原始完整 source、身份与表情检查。
它不能用头部组件报告查看眼部，不能把未通过的完整网格报告默认转成“全部通过”。

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 所有底模／配置 atlas 的索引 |
| `head_<id>/<profile>/atlas.json` | 参数完整输入、逐网格测量、源骨骼驱动行、缓存 SHA-256、局部诊断 |
| `baseline.npz` | 固定 baseline 的所有网格顶点及 native 输入 |
| `control_<index>.npz` | 每个采样值的完整顶点位移与对应顶点 effect mask，float64／bool |
| `jacobian.npz` | 原始与归一化 Jacobian、左右斜率、coupling、SVD 控制方向 |
| `exterior_effects.json` | 范围外点相对0／1端点究竟新增了多少几何变化 |

这些是曲面几何数据，不是纹理截图。完整同索引位移可以叠加到 `baseline.npz`
重建每个样本；它没有依照关键点裁脸，也没有对各样本分别对齐／缩放。

## 测量定义

对头部、眼部、睫毛、眼影、泪水、牙齿和舌头的实际缓存网格分别记录：

- bbox 最小／最大值、三轴尺寸、对角线及尺寸变化。
- 曲面顶点 centroid 及位移向量／长度。
- **所有顶点**相对 baseline 的位移最大值、均值、RMS、p50／p95／p99。
- 用该网格 baseline bbox 对角线归一化的相同统计。
- 受影响顶点数、占比、效果阈值，以及最大变化顶点的具体索引。

默认只保留**源网格身份与顶点索引**。某片区域受影响，并不意味着它自动被命名为
颧骨、眼角或下颌；骨骼名称也不能代替解剖皮肤位置。

`inspect_atlas.py` 比较同一参数的外侧采样点与有界端点，避免把“相对 `.5` 有变化”
误当成“解锁范围确实新增变化”。这也是逐网格判断，例外旋转保留 clamped 行为时，
同一参数的位移通道仍可能继续外推。

## 坐标、姿态和单位

所有底模统一使用**缓存 prefab 原生 FK 坐标，场景／body 祖先 scale 固定为1**。
没有 ABMX，没有动态表情 blendshape deltas，没有角色站姿、头部朝向和场景位移。
该“neutral”是确定的离线测量约定，不代表游戏当前所有表情权重恰好为零。
例如实机中 `head.e00_defo` 可以在中性表情时权重100，Unity parity 工具会使用
实际导出的 deltas；此 atlas 默认使用缓存 bind surface，不混用两者。

距离仍是 Unity 资产单位，不能解释为毫米或厘米。把 atlas 与实机几何比较前，
必须核对同一个 head ID、源网格／骨骼／bindposes，实际表情，以及记录的祖先缩放。
`tools/unity_parity/` 已提供该验证流程；禁止用任意拟合 scale 隐藏形变偏差。

## 局部 Jacobian、耦合与 SVD

每个参数在指定 baseline（默认 `.5`）周围单独取 `±step`，从完整曲面位移计算 Jacobian。
记录中央差分以及左右单边斜率。`.5` 可能是动画关键帧的 knot：中央差分是两边
响应的平均，不应宣称它就是唯一的可微导数。左右斜率差异单独输出。

输入 probe 不裁到 `[0,1]`，是否发生几何 clamp／外推由所选采样语义决定。
例如 vanilla 的0边界，负向 probe 可以保持端点曲面、左斜率为0；右斜率仍保留
真实响应，不能把中央差分当成边界的唯一导数。现在同时记录完整 probe 输入、
实际可表示的左右系数间隔，并按实际间隔计算斜率，避免边界裁切后仍误用
`2*step`。极大 baseline 导致步长无法在 float64 中表示时会拒绝采集。

为避免大网格或较小 submesh 的原始尺度主导诊断，每个网格的 Jacobian 块除以
`baseline_bbox_diagonal * sqrt(vertex_count)` 后拼接。原始单位 Jacobian 同时保存，
归一化规则明确记录，不能把另一种测量 metric 下的条件数直接混在一起比较。

coupling 是两列响应的 cosine，可以揭示参数在当前姿态／曲面集合上产生近似同向
或反向的效果。SVD 的奇异值、数值秩、非零子空间条件数及控制方向也保留。
默认相对秩阈值为 `1e-7`；零列只描述**该 baseline 与所选网格集合**，不说明参数
在别处或饰品上无效。头部单独的诊断也输出，避免和含眼部／牙齿的集合混淆。

这些结果是局部诊断：即使秩不足或目标残差暂时不在某个 Jacobian 列空间内，也不能
证明非线性形变在其他位置不可达。反之，满秩也不说明能够拟合任意真实人脸。
边界、三维质量、目标观测、相机、表情及多起点搜索仍需要独立检验。

## 可选的真实区域 masks

`--regions masks.json` 接受经过独立验证、与缓存源文件匹配的顶点 mask／对应表。
格式示例仅展示结构，不是已认证的人脸区域：

```json
{
  "heads": {
    "2": {
      "o_head": {
        "source_file_sha256": "<atlas中的o_head源npz SHA-256>",
        "provenance": {
          "kind": "verified_vertex_mask",
          "verified": true,
          "source": "<实际已检查的对应表／标注文件>"
        },
        "regions": {"<已验证的区域名称>": [1, 2, 3]}
      }
    }
  }
}
```

也支持 `kind="verified_surface_correspondence"`。工具检查源 hash、明确来源、确认标记
和顶点索引有效性；**`verified=true` 本身不完成实际解剖验证**，来源必须先经过真实
表面／像素对应或 mask 验证。默认未提供此文件，所有 atlas 都不生成解剖区域名称。

## 已生成的结果与限制

`outputs/parameter_atlas_20261004/` 已生成6份 atlas：3个底模×2种采样语义，
354个参数／配置组合、2,478个完整 native 样本、8个网格，约161MB。
源网格、动画、骨骼、驱动表和参数输入均有索引／hash记录。

在固定 `.5` baseline 和当前几何 metric 下，三个底模的头部／完整网格集均得到
59个局部方向，完整集合的条件数约509–523。响应列长度差异明显；head2 最大／最小
约247倍，不能把不同滑杆的同一个 `0.1` 视为相同的几何调整量。
head2 控制1与21的列 cosine 约 `.9982`，是当前 metric 下的强耦合例子，不是全局
等价控制的结论。

旧文档对27／29／36／38“不会影响头部曲面”的笼统说法不适合当前 `.5` baseline：
直接 head2 控制27增加 `.01`，头部顶点最大变化约 `5.61e-4` 资产单位。
不能从一张卡的零梯度或特定骨骼名称推断全局无效。

相对端点测量确认：vanilla 的全部外侧点保持端点几何；18.2配置的59个控制在所测
外侧点上都有至少一个网格新增几何变化。这只覆盖这里的缓存和采样值，不能免除
例外通道、质量检查或更极端范围的验证。

七项合成测试验证了统计量、索引效果 mask、有限差分、knot 两侧斜率、coupling／秩、
范围外相对端点比较，以及来源不匹配／未验证语义区域的拒绝行为。此产物解决参数数字缺乏几何标度的部分
问题；它不是程儿人物验收，也不替代固定多角度游戏截图与真人留出评估。

新增八项 baseline 测试覆盖严格59维输入、原始与规范 hash、游戏 snapshot 字段提取、底模身份匹配拒绝、
所有 probe 保持其余58项、不裁边界输入、clamp 两侧斜率与实际不等间隔。
完整合成生成测试还检查59个控制 archive 的输入向量和已知非线性响应的 Jacobian。
