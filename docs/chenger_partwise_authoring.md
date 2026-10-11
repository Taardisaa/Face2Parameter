# 程儿：逐部位量化与锁定

制作TODO在HS2Mod `docs/chenger_authoring_todo.md`，当前只处理下巴。
用户于2026-10-11否定v14方硬/突出轮廓，要求一个部位通过后不再随意改动。
v14历史文件记录成功保存重载，但其外观不再视为已通过。

## 修正验收覆盖

`jaw_trace_compare.py`原来只检查嘴线到颏底25/50/75%的三处宽度。
它们接近并不能覆盖颏底形状。现在保留这三处，并增加10/90/95%，
逐侧坐标与完整原图轮廓；旧25–75%割线角的计算定义保持不变。
统一眼距缩放/去滚转，嘴线对齐，不拉伸图像。截面按各自嘴到颏高度
定义，必须连同实际长度差与完整曲线解释，不能当成相同绝对高度。
人工边界和残余头姿/透视仍有误差，尤其末端小宽度的相对百分数更敏感。

复用已保存v14原图与SHA绑定完整贴边trace，不重新扫掠参数：
v14旧25/50/75%宽度约-3/-7/-5%，新增90/95%约+7/+29%。
这说明旧指标遗漏末端偏宽，不能据此说用户观察有误。
输出保留在 `outputs/chenger_stylized_arcface_20261011/v14_jaw_tip_review/`。

## 一次仅一个部位

上一轮运行试改眼23、鼻45、上唇51、下唇52已通过重载完整v14卡恢复。
它们未进入新正式制作配方；原始试图与源控制证据保留在忽略输出目录。
此后v15_chin_candidate只改已有ChinTip_s的X Scale1.10→.88，保持其
YZ、Position、Length、Rotation及其余42 modifier结构和原生59滑条。
配方为 `tools/character_authoring/chenger_v15_chin_recipe.json`。

源路径仍是安装ABMX4.4.6.0的Apply/CollectBaseline和实际原生头2：
ChinTip_s是Chin_rs子骨；本地Scale按ABMX已有baseline乘入。
原生10同时改变XYZ，所以本次不使用它替代单轴X操作。
已有源网格直接权重显示Tip206顶点，Low289、rs757；直接权重范围
不能证明屏幕边界只变末端，也不排除父骨与混合权重的影响。
源几何/安装DLL哈希沿用v14记录与忽略证据
`v13_jaw_shape_review/source_weight_review.json`。

## 当前结果：不通过，未锁定

v15完整人工贴边图经可视检查保留；原图SHA绑定的新trace在
`tools/character_authoring/chenger_v15_chin_trace.json`。

| 截面/长度 | v14 | v15候选 |
|---|---:|---:|
| 10%宽度 | -3% | -5% |
| 25%宽度 | -3% | -4% |
| 50%宽度 | -7% | -11% |
| 75%宽度 | -5% | -19% |
| 90%宽度 | +7% | 约0% |
| 95%宽度 | +29% | +24% |
| 下唇下缘到颏底 | -3% | -3% |
| 嘴线到颏底 | -5% | -6% |

候选虽缩减底部部分宽度，但上方收得过窄，底缘仍平；不能锁定或推进
下一部位。这是有限最终观察返回源支持范围的失败，不是拟合游戏变形逻辑。
下一步核对实际顶点的混合权重与父链，对应完整两侧/底缘而非骨名猜测。
没有接受美术图的侧面，不能宣称已量化目标三维前突或伪造其侧面真值。

卡保存并重载成功，实际snapshot+ABMX+renderstyle对v14 pre-trial状态差异
仅ChinTip X（含coordinate副本）及meta.fullname，眼鼻唇试改未残留。
记录在 `v15_chin_state_diff.json`，它证明参数保留，不证明外观通过。
当前普通原生人物为v15候选，旧v14及所有失败证据保留；无诊断壳。

## 复现命令与锁定条件

```powershell
.venv/Scripts/python.exe tools/character_authoring/jaw_trace_compare.py --measurements outputs/chenger_stylized_arcface_20261011/v14_reloaded_proportion_review/measurements.json --trace tools/character_authoring/chenger_v14_jaw_trace.json --out outputs/chenger_stylized_arcface_20261011/v14_jaw_tip_review_repeat
.venv/Scripts/python.exe tools/character_authoring/jaw_trace_compare.py --measurements outputs/chenger_stylized_arcface_20261011/v15_chin_proportions/measurements.json --trace tools/character_authoring/chenger_v15_chin_trace.json --out outputs/chenger_stylized_arcface_20261011/v15_chin_jaw_review_repeat
```

当前无已锁定部位。验收同时覆盖全部局部指标/完整轮廓和用户已指出的
突出、方硬、平底问题，不能靠选择通过的截面或单个百分数宣布完成。
通过时记录完整卡SHA、当前原生/ABMX字段与共享父骨依赖、材质和图证据；
之后所有作者修改都核对锁定值及共享依赖。共享父骨被改会触发形状复核，
不以子骨字段相同代替几何不变。
