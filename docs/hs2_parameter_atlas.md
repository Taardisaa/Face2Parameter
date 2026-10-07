# 原生参数曲面效果 atlas

`tools/parameter_atlas/` 把“滑杆数值”转换为可检查的实际曲面响应，读取已有
HeadRig／TorchHeadRig 缓存，不连接游戏、不写人物卡、不训练或下载模型。
它衡量形变和局部控制条件，不计算真人相似度，也不判断某底模全局上做不到某张脸。

## 执行与产物

始终使用项目解释器：

```powershell
.venv/Scripts/python.exe tools/parameter_atlas/test_atlas.py -v
.venv/Scripts/python.exe tools/parameter_atlas/generate.py --out outputs/parameter_atlas --device cpu
.venv/Scripts/python.exe tools/parameter_atlas/inspect_atlas.py outputs/parameter_atlas/manifest.json --out outputs/parameter_atlas/exterior_effects.json
```

默认处理缓存中的底模0／1／2、全部59个原生参数（包含耳朵），逐个测试
`-.25,0,.25,.5,.75,1,1.25`。其余参数保持 `.5`。
同时生成 `vanilla` 与 `slider_unlocker_18_2` 两种**离线采样语义**。
MCP 的 native 范围模式只是接受写入值的校验策略，不会卸载游戏中已经安装的
SliderUnlocker；不能把范围内写入理解为实际游戏退回了无插件模式。

可重复指定 `--head-id`／`--profile`，或改变 `--levels=...`、`--step`、
`--effect-threshold`、`--device` 和 `--batch-size`。默认数值差分步长为 `1e-3`，
顶点效果阈值为该网格 baseline bbox 对角线的 `1e-6`。

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

每个参数在 `.5` 周围单独取 `±step`，从完整曲面位移计算 Jacobian。
记录中央差分以及左右单边斜率。`.5` 可能是动画关键帧的 knot：中央差分是两边
响应的平均，不应宣称它就是唯一的可微导数。左右斜率差异单独输出。

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
