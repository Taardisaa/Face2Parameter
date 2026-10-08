# 头部整体缩放／低头候选

2026-10-07 用户同意先缩小完整原始头部、再试轻微顺时针低头，固定身体
接口并重新生成颈部过渡。本次生成两版供评审：`size_only` 等比缩至 94%，
`size_and_pitch` 在此基础上绕头部中心低头 3°。这些是明确的设计候选，
不是对原生脸型拟合出的参数，也不代表已证明更忠实于照片。

## 实际实现

- 从 MICA 原始输出重新读取完整、未裁切的 FLAME 顶点，校验原始数据、
  照片及 manifest 的摘要和既有裁切来源。没有累积变形此前已经收口的头。
- 中心取源头面部／头皮／耳朵语义区域的包围盒中心，排除颈部。
  只做等比缩放和刚性旋转；在剖面横轴 +Z 向右、纵轴 +Y 向上的约定下，
  正 X 轴旋转即顺时针低头。没有局部改五官，也没有单独压薄后脑。
- 源裁切配方继续复用。仿射变换与按原边插值的裁切可交换，校验完整原头
  与保留源网格的关系。实际原生接口和 collar 保持原位置。
- 六排连接面重新按三次 Hermite 曲线生成。源端方向由完整源网格法线沿
  实际裁切边插值得到，随整体旋转转动；原生端使用安装版 `p_cf_head_02`
  网格保存的法线，沿其真实上沿边插值。后部按已提交的长贝塞尔曲面规则
  重新铺顺，解剖标记读取整体变换后的完整原始头部。

连接带早期失败的具体原因是从孤立的原生末端小带重新计算法线，丢失了
完整头部的相邻曲面方向，导致前部曲线列交叉。最终恢复实际资产保存的
方向场，保留三次曲线。没有以直线连接或调低检查标准掩盖交叉。
安装 bundle 与缓存 rig 的顶点、面及法线逐项核对；来源沿用
[原生接口审查](hs2_native_neck_interface_receipt.json)。这里的法线用于
连接曲面的几何创作，不声称已经实现游戏实时蒙皮／着色法线管线。

## 结果与边界

程儿 `img-002` 和另一份已有身份 AF1 400016 共用同一规则，分别产出两版。
四份最终网格都通过静态收尾：整体变换是等比相似变换，受保护面部顶点
等于变换后的原始输出，原生接口逐点保持，拓扑保持，没有新增真实穿插，
连接带无真实三角面穿插。源保留区域原有的眼／嘴附近穿插仍存在。

完整剖面显示，单纯缩小使下巴底面的下沉减轻；低头版使鼻部下降，
下巴比单纯缩小版略低。这是摆放变化，不能据此判断照片身份准确度。

**本次是实际网格的离线候选，尚未装入游戏底模。** 原生 prefab、蒙皮、
渲染法线、材质及游戏验收仍未完成；总体目标继续 active。
生成时 receipt 的 `unfinished` 是待验收状态，独立生成的 `validation.json`
记录后续实际验收结果；轻量汇总见 [receipt](hs2_head_placement_receipt.json)。

产出目录为 `outputs/native_head_20261007/placement_review_final_v1/`，各身份下
有 `size_only/`、`size_and_pitch/` 的 `geometry.npz`、`head.obj`、receipt 和
validation，以及 `complete_placement_sections.png` 和 `placement_3d.png`。
数据和许可网格保持 Git 忽略。中间失败／探索输出保留在
`placement_review_v1` 至 `placement_review_v4`，不能作为最终结果。

## 重现

在 Face2Parameter 中执行：

```powershell
.venv/Scripts/python.exe -m tools.native_head.placement_review --baseline outputs/native_head_20261007/interface_authoring_v3/chenger --current outputs/native_head_20261007/posterior_bezier_review_v1/chenger/geometry.npz --body outputs/native_head_20261007/neck_local_v1/native_body.json --output outputs/native_head_20261007/placement_review_new/chenger
.venv/Scripts/python.exe -m tools.native_head.validate_placement --baseline outputs/native_head_20261007/interface_authoring_v3/chenger --candidate outputs/native_head_20261007/placement_review_new/chenger/size_only
.venv/Scripts/python.exe -m tools.native_head.validate_placement --baseline outputs/native_head_20261007/interface_authoring_v3/chenger --candidate outputs/native_head_20261007/placement_review_new/chenger/size_and_pitch
```

另一身份将所有 `chenger` 路径段替换为 `af1_400016`。输出必须使用新目录。
工具依赖现有已审查的 native-band baseline 输入布局及 FLAME masks；不是
任意拓扑、任意身体或从照片直接安装 zipmod 的通用完整工作流。
此前未完成的 `build.py`、`rim.py`、attachment 及 C# 集成草稿没有混入此次提交。
