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

## 恢复任务后的源支持核对与v16

Codex内部goal已重新明确创建为active，当前仅下巴项。
游戏未运行时启动一次并等到Maker ready，加载暂停卡SHA
`343d5d550af3c096bbfa16679a39c61b2d4ea5bb4657690a0efac746f0936148`。
安装ABMX DLL实际SHA仍与v14相同：
`9224876ce931eb3c40a347ba57abebcca515bde58af2a6b9802b367a5b3563af`。
全局SSS重新开启；实际Bridge0.31.13。原生普通头2，未加载诊断壳。

新工具`tools/character_authoring/jaw_weight_audit.py`读取现有原生导出的
完整source vertices、indices/weights、骨名，不运行游戏采样、不拟合形变。
明确源空间底部区域Y<-.30/Z>.50；中央abs(X)<.08，侧部.15<=abs(X)<.35。
保存全部选中顶点和实际混合权重，不能把这些诊断选择当解剖标签或最终算法。
中央40点平均Tip约.286、rs约.242、Low约.194；侧部24点Tip约.153、
rs约.388、Low约.196。可见Tip不是只作用末端，实际底部混合多个驱动。
尤其源最低前端1184的Tip/FaceLow/rs/Low权重约.313/.250/.239/.198。
因此只按Tip骨名收窄不具有局部圆度保证。源parent链及插件Apply支持
分开作者修改中心与底部的offset；没有用平均权重计算或拟合参数增益。

注意旧`src/hs2_mesh_deform.py`在ABMX阶段仍直接`pos*length+offset`，
安装插件实际使用当前position方向及cached length，且position-only路径
使用cached pos；前者不是本轮实际缓存状态的合格预测器。未修改该旧模块，
不以其结果声称安装版本兼容。制作仍由实际游戏原生/安装插件执行。

v16以暂停卡为基线：Tip X .88→1.10，Y offset -.09→-.13；
Low X1.16→1.00，Y offset-.03→0；两者Length=1且原YZ/旋转保留。
Chin_rs及全59原生参数保持，不联动眼鼻唇。配方为
`chenger_v16_chin_recipe.json`，实际截图SHA绑定trace为
`chenger_v16_chin_trace.json`，同协议最终观察只采本候选正面与侧面。

完整贴边图v16依然未通过：10/25/50/75/90/95%宽度差约
-5/-6/-11/-15/-2/+16%，唇下长度-1%，嘴线长度-4%。
底部偏宽改善但仍平，75%区域仍窄；含人工误差，不能当精确改善幅度。
保存、重载snapshot/ABMX/renderstyle全等，暂停→v16只两modifier的
X/Y字段（及coordinate副本）和版本名变动。未锁定下巴。

本轮还定位主相机捕获流程缺少PoseFreeze；固定眼距和去滚转不能排除
残余yaw/pitch。这是验收证据不足，不应通过反复作者调值补偿。
`MakerRenderService.cs`离屏流程已有PoseFreeze源实现，而实际SSS捕获
`MakerRenderStyleService.Capture`没有使用。下一步合入固定姿态/恢复，
再验收下巴；不能用普通离屏图冒充SSS主相机图，也不在该缺口下锁定。

```powershell
.venv/Scripts/python.exe tools/character_authoring/jaw_weight_audit.py --geometry outputs/chenger_stylized_arcface_20261011/v12_head_geometry.json --out outputs/chenger_stylized_arcface_20261011/resume_jaw_weight_audit_repeat.json
```

证据输出：`resume_pause_state.json`、`resume_jaw_weight_audit.json`、
`v16_chin_saved_state.json`、`v16_chin_reloaded_state.json`、
`v16_chin_state_acceptance.json`、`v16_chin_jaw_review/`。
卡为 `E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_stylized_reconstruction_v16_chin_candidate.png`。
当前v16普通原生候选，暂停及v14/v15旧卡保留，目标仍active。

## 主相机固定姿态里程碑（未完整验收）

HS2Mod新增真实main camera冻结相位与恢复报告，详细实现和版本保留见
`../HS2Mod/docs/chenger_main_camera_pose.md`。本仓库新增
`main_capture_acceptance.py`，只做两个未改人物的捕获/状态恢复/非法请求检查；
任一声明门槛失败即退出，不据子集通过推进人物部位。
`capture_bridge_il_audit.py`验证原安装0.31.13源基线及补丁，仅两个捕获类型
改变；原119顶层类型的117个其它类型IL保持。四项独立guard测试通过。
仅忽略PE placement标签/comment，不忽略opcode、属性或data bytes。

V1固定head TRS重复差0、公开状态恢复通过，但全节点精确恢复失败。
源修正版V2避免重复写入未改变的TRS，head重复差仍0，公开状态仍相等。
首次916节点中只有cf_J_SiriDam_R和cf_J_Mune00恢复不等，第二次零个；
完整脚本仍失败。没有降低门槛，也没有把两个身体节点直接略去宣布通过。
原图像素最大差分别158/173，不虚构像素一致。head矩阵不等于实际脸表面相同，
下一步诊断恢复分量并补蒙皮/表情的同次记录。这不是人物参数扫掠。

运行实例：
```powershell
.venv/Scripts/python.exe -m unittest tools.character_authoring.test_capture_bridge_il_audit -v
.venv/Scripts/python.exe tools/character_authoring/main_capture_acceptance.py --out outputs/chenger_stylized_arcface_20261011/main_pose_capture_acceptance_next
```

V1/V2失败都保留在对应输出目录；实验后原生重载v16，未修改脸参数。
当前SSS开启、普通原生头2、完整卡可恢复，无诊断壳；下巴不锁定，目标active。

## V3诊断与V4量测条件

`main_capture_acceptance.py`新增八个真实面部BakeMesh表面的严格相等检查。
V3原失败保留：部分身体/手指的旋转逐分量差约1e-8；左右o_eyebase顶点hash
不等而blendshape权重相等，o_head等其余表面相等。未放宽精确恢复门槛。
安装EyeLookController→EyeLookCalc源码及实际eyeTypeStates明确TARGET有
Time.deltaTime角度插值，index3的NO_LOOK则写缓存fixAngle。临时通过原生
表达接口设置index3后，V4两次未改人物捕获的全部原门槛通过，包括八表面
严格相等、head world TRS实差0、全部916节点与blink即时恢复、公开状态及
非法请求拒绝。并非关闭控制器，不修改眼形/虹膜/妆容。

证据`main_pose_capture_acceptance_v3/`与`main_pose_capture_acceptance_v4_native_no_look/`；
当前部署DLL SHA为d8a3d6fcfec97a5dec0a96551167d9f61a6675bcb262e2b6bcb2a1d20ae13304，
完整IL审计仍仅两个捕获类型改变，其余117类型保持。
此通过只支撑本次记录条件下继续量测，既不认证所有私有缓存也不认证辨识度。
SSS像素仍不完全相等，未虚构阈值。原失败全部保留。
新的v16固定姿态量测在`v16_fixed_pose_proportions/`，实际完整源状态在
`v16_fixed_native_geometry.json`；导出用于恢复源映射，不是游戏参数扫掠。
下巴仍在处理中，必须比较完整轮廓；最终卡/常态表现恢复原TARGET(index1)。
