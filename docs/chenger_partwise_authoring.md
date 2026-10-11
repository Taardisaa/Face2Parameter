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

## v17实际下巴候选与量测修正

V4虽然原门槛通过，但实际闭眼；其FAN眼距对比不能作为作者改形依据。
V5先原生eyes_blink=false并经过正常LateUpdate展开眼睑，再捕获相位零。
八个表面相等；全actor精确旋转恢复门槛再次失败，原证据保留，未降阈值。
固定睁眼画面人工全轮廓表明v16的25/50/75%宽差约+4/-1/+1%，
90/95%末端差约+23/+60%，唇下长度-3%。旧姿态的中段偏窄判断撤回，
不再据此扩大两侧。末端百分数分母很小，结合贴边图及6px人工误差解释，
不作为精确宽度或改善幅度。FAN自动颏点明显偏高，必须保留人工皮肤边界。

仅选择一套源支持候选，不进行游戏参数扫掠：Tip Y -.13→-.18，Low Y0→+.025。
二骨Length=1使用安装ABMX `_posBaseline+PositionModifier`分支；其余所有
scale/rotation/length、59原生参数、材质、眼鼻唇眉保持。静态实际混合权重
用于区分中央与侧部支持，不以平均权重拟合参数增益或假装精确逆解。
实际main camera正/侧捕获的head pose相同，两次即时全TRS恢复检查均通过。
不是把V5失败改判通过，也不是私有状态普适认证。

v17全轮廓25/50/75%宽差约+3/-4/-6%，90/95%约+16/+46%，
唇下长度约+4%，嘴线至颏约+3%。底部圆度有改善但仍偏宽，长度略偏长，
人工误差及参考2D残余姿态保留，**未通过、未锁定**。侧面头发遮挡颏轮廓，
不能编造侧面3D目标或宣称验收；下一步需可恢复原生隐藏头发以检查实际轮廓。
配方及哈希绑定全轮廓：`chenger_v17_chin_recipe.json`、`chenger_v17_chin_trace.json`。
修正后的v16基线trace：`chenger_v16_fixed_chin_trace.json`。
输出：v16_fixed_open_proportions、v16_fixed_chin_review、v17_fixed_open_proportions、
v17_fixed_chin_capture、v17_fixed_chin_review。闭眼V4输出仍保留但不用于作者结论。

保存前恢复TARGET1/blink=true。v17卡SHA
4763faf9dd7d426932c6b3457e909eece96df6157091fbd0dc532bcbaca97ecd。
保存重载snapshot/renderstyle/expression完全相同；ABMX列表42→30不是列表
精确相等。逐项核对删除12个身体modifier及全部coordinate副本均identity，
保留30个modifier无增加/无变更，细节在v17_abmx_reload_identity_pruning.json；
原列表不等结果在v17_state_acceptance.json保留，不用规范化覆盖失败。
当前可见是重载v17普通原生头2，Hanmen/SSS保持，旧卡均留存，目标active。
诊断代码里程碑HS2Mod b6d170a / Face2Parameter 043f273均已推送。

## v17侧面与源平移作用范围

本轮未再改下巴数值。最初原生离屏hide_hair正/侧输出有异常小身体投影叠影，
保留`v17_native_unobstructed*`失败证据，不用于形状通过，也不冒充主相机SSS图。
随后通过原生hair接口临时使用back0长直发（**不是无发**；front/side/option0为无），
真实main camera的正/侧图清楚露出颏尖和唇下折线。诊断发型没有保存为人物输出。
原普通v17全卡已重新载入，并显式恢复TARGET1/blink=true。
不能假定load card必然恢复runtime gaze：本次重载后index3仍保留，已发现并修正。
最终snapshot/renderstyle/expression相等；原30个modifier完全相同，正常运行又
出现12个identity身体modifier，原始精确列表不等如实保留。

前后主相机的head pose和o_head BakeMesh hash相等，六个其它面部表面相等；
左右o_eyebase hash不同而blendshape权重相等。NO_LOOK采用各次加载的fixAngle，
并不保证跨重载眼球缓存相同。不能泛称完整脸表面不变或把侧图当眼睛验收。
当前侧图可读颏尖/唇下深度，后方下颌仍有头发；无合格真人/美术侧面目标时，
不编造侧面3D误差或宣布通过。证据`v17_main_unobstructed/front.png`、`side.png`、
`v17_side_diagnostic_final_restore.json`；该次即时全TRS精确恢复仍失败，未掩盖。

新增`tools/character_authoring/chin_position_audit.py`读取本次实际几何、所有
祖先矩阵、原始权重/bindpose、blendshape帧、安装ABMX私有缓存；只处理二骨
Length1的位置分支。平移作用包含每个实际后代骨的权重，由parent世界线性
矩阵转成head坐标；不是从游戏截图拟合gain，不以骨名假定只影响颏尖。
非单100帧的活动blendshape、缺失baseline或非Length1明确拒绝，不静默近似。
源顶点LBS与实际BakeMesh完整4439顶点比较：完整renderer矩阵转换最大误差
.126896游戏单位，scale_free_trs候选最大4.119e-6、均值1.006e-6。保留二者，
不以自行设置阈值宣称所有模型/蒙皮质量兼容；该实际数据支持当前坐标解释。
两骨actual position与captured baseline+offset分量差最多1e-8。

相同明确源区域（中央40点、侧部24点）真实解析平移响应：Tip local Y对应
head Y中央约.285、侧部.152；Low中央.199、侧部.200。该值来自矩阵与权重，
不是学习/拟合变形斜率，也不是解剖目标。逐点矩阵/完整缓存及误差保存于
`v17_chin_position_audit.json`，摘要`v17_chin_translation_regions.json`。
下一步可分开处理中央圆度和整体长度；不能把该模型审计当人物外观验收。
配方、人物卡不变；当前可见是原发型v17，Hanmen/SSS保持，目标active。

```powershell
.venv/Scripts/python.exe tools/character_authoring/chin_position_audit.py --geometry outputs/chenger_stylized_arcface_20261011/v17_current_geometry.json --out outputs/chenger_stylized_arcface_20261011/v17_chin_position_audit_repeat.json
```

## v18已拒绝，完整顶点与投影作用范围修正

本轮只作者修改Tip Y-.18→-.215、Low Y.025→.125，保留所有宽度/旋转/长度、
原生形状、妆、发、材质。原生Length1缓存分支负责实际应用。
候选依据中央/侧部诊断区域平均选择上收同时保留中央圆度；这次作者选择失败。
**区域平均不能替代投影轮廓。**不得把这一作者近似提升为交付的反演实现。

源平移响应与实际完整4439个raw BakeMesh顶点变化，转actual head坐标比较：
最大残差9.871869e-7、均值4.920167e-7游戏单位，完整残差保留而不自行设阈值。
说明二骨位置应用机制在此对数据中吻合，但不能证明外观正确。
新增可重现`chin_translation_check.py`只支持二骨Length1位置差，scale/rotation/
length变化明确拒绝，源mesh不一致拒绝；其它driver/父链变化不扣除，保留于残差。
并非全动画/所有分支兼容认证，也不是拟合游戏数据。

源最低1184并非当前颏底投影点：1206实际投影Y836.12698→839.91334，
向下约3.78636px；1184反而Y823.73635→820.49504。1206的Tip/Low真实
权重.450688/.047155，1184则.312873/.198383。两点已在实际main camera
原图上以几何诊断标出，末端长度变长，候选拒绝，不伪造改善百分数。
头部source hash仍b2c3831da3e025b23a8dc6231a8cd73ed3540536f81b3aea1aa4b9459e360ece。

新`chin_projected_audit.py`保留完整原vertex IDs、全部投影坐标、来源骨权重、
几何/捕获/图像SHA及两极值，并可产生标注图。使用实际BakeMesh scale-free
转换与head matrix回head坐标，再按记录的frozen head TRS/FOV/yaw/target做
该主相机源透视诊断。当前导出head matrix与TRS表示最大分量差约6.1e-8；
FOV23及768x1024是记录输入，旧capture响应未报告这两项；投影不认证CTAA
jitter/shader visibility、任意shear或全场景遮挡。源Y<-.30/Z>.50只是明确
诊断选择，不把选点自动宣称解剖/可见标签。原图辅助确认当前颏底位置。

失败v18卡独立保存：
`E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_stylized_reconstruction_v18_chin_rejected.png`，
SHA ca4be825a68bee935d42af285ea3c1ffa9e22c4a761bf11f799ca80dae3191d6，
配方`chenger_v18_rejected_recipe.json`。不以失败卡保存当成功。
已恢复原生v17及原TARGET1/blink=true，snapshot/style/expression与完整ABMX
响应均与v18_before_state精确相等，证据v18_rejected_restoration.json。
当前没有新锁定项；下一步依据实际投影完整末端来源顶点而非区域平均值选参数。

```powershell
.venv/Scripts/python.exe tools/character_authoring/chin_translation_check.py --before outputs/chenger_stylized_arcface_20261011/v17_current_geometry.json --after outputs/chenger_stylized_arcface_20261011/v18_current_geometry.json --out outputs/chenger_stylized_arcface_20261011/v18_position_prediction_check_repeat.json
.venv/Scripts/python.exe tools/character_authoring/chin_projected_audit.py --geometry outputs/chenger_stylized_arcface_20261011/v18_current_geometry.json --capture outputs/chenger_stylized_arcface_20261011/v18_fixed_chin_capture/front.json --out outputs/chenger_stylized_arcface_20261011/v18_projected_vertex_audit_repeat.json --figure outputs/chenger_stylized_arcface_20261011/v18_projected_vertex_figure_repeat.png
```

证据还包括v18_source_translation_prediction.json、v18_fixed_chin_capture/、
v18_current_geometry.json、v18_fixed_open_proportions/、v17_projected_vertex_audit.json。
FAN自动颏点仍不作原图末端，未用检测点虚构完整轮廓通过。
前里程碑已推送：HS2Mod36ee9d3 / Face2Parameter4e87387；本轮保持goal active。

## 完整三角边界source author，v19/v21失败与v22未应用

新`chin_source_candidate.py`不拟合游戏变形gain：使用captured完整source vertices、
原生权重/bindpose/活动单100帧blendshape、实际父链local/world矩阵、安装ABMX
baseline Scale乘modifier及Length1位置分支。只作者缩放/位置，保留原生59参数、
Z/旋转/长度及其余modifier。Chin_rs原Length1.11仍保持，不冒用Length1位置分支。
修改scale按actual local affine列的比例变化、位置按实际baseline+offset差；
完整后代世界矩阵重算后按真实4权重LBS，非凸包/固定区域/经验斜率。
source forward基线对4439实际BakeMesh最大残差2.557e-6游戏单位（保留观测，
不自行设通过阈值）。实际两次作者后检查V19 max4.082e-6，V21 max5.368e-6。
这仅证明此实际数据的forward，不能证明外观或所有引擎/动画分支兼容。

原图hash绑定手工reference全trace、固定V17眼轮廓距离/嘴线对齐不重新检测，
避免FAN随下巴形变漂移。真实triangle union与每个scanline相交取最外边界，
不使用凸包或头骨origin代替皮肤轮廓。投影TRS/FOV/shear/CTAA/shader visibility
限定沿用前投影诊断，不能将正面拟合声称3D侧面目标已证明。

最初V19只做10–97%水平截面，虽然source与actual一致，但画面仍平底；
遗漏终端3%违反此处完整颏底要求，版本拒绝，并保留source输出、实际图和失败卡。
该作者目标已纠正：10–85%30个上中段双侧截面，加颏尖X±.30眼距的41个
完整bottom Y-at-X剖面（含真正最低点），末端高度为明确几何等式约束，
不能用多截面误差交换更短的平底。边界不存在直接失败，没有合成/补常数轮廓。
V20为未强制等式的离线诊断解，从未放进游戏；完整输入/残差仍保留。

V21四标量SLSQP达到200迭代上限，success=false与消息保留，不标收敛。
feasible候选仍仅作一次实际候选核对：actual最低点height.698208眼距，reference
.698202，长度接近；25/50/75%宽约-1/-5/-2%，90/95%却+26/+67%，
99/99.9%也如实保留（百分数分母极小，不能作精确改善量）。**外观仍平底，拒绝。**
height满足/模型吻合不作为外观通过。原主相机即时全TRS恢复也仍失败，记录保留。
数据：v19_source_candidate.json、v19_fixed_chin_capture/、v19_forward_actual_check.json、
v20_full_terminal_source_candidate.json、v21_full_terminal_constrained_source_candidate.json、
v21_current_geometry.json、v21_actual_full_contour_check.json、v21_actual_terminal_widths.json。

V19失败卡SHA08828a725f971535763145de55807fbef3839d94e0b04c6c761ad0823e893778；
V21失败卡SHA12225d94ea5cf826c950ab3bbdc0d51908557c6402b91da535c2347dbbc9ace4。
分别为`Chenger_stylized_reconstruction_v19_chin_rejected.png`和
`Chenger_stylized_reconstruction_v21_chin_rejected.png`，配方保留在tracked tools。
当前已恢复普通原生V17、原发型、TARGET1/blink=true；snapshot/style/expression
与before相等；按骨名42个modifier全部相等，但完整ABMX响应不等原结果保留。
无需重启游戏/换桥接DLL，无已锁定部位。

为解决缺少颏底圆度控制，source forward扩成8标量：rs XY scale、Tip XY scale+
Y position、Low XY scale+Y position。沿用真实源scale branch和位置Length1分支，
不引入近似形变公式；复核既有V21 pair仍max5.368e-6，不采新游戏扫描。
新scale通道尚无改变它们后的游戏验收，不能泛称八通道已全验收。
V22离线解success=true但rsY/TipXY/LowY/Low位置等触及作者bounds，95%宽仍
约+51%，99%约+212%；不足以通过，**没有应用游戏**。bounds只是作者搜索范围，
不是原生引擎限制/接受阈值。继续覆盖剩余末端形状与参考视角/不对称残差，
不得把当前支持子集当成已复现程儿或改小成功定义。

```powershell
.venv/Scripts/python.exe tools/character_authoring/chin_source_candidate.py --geometry outputs/chenger_stylized_arcface_20261011/v17_current_geometry.json --capture outputs/chenger_stylized_arcface_20261011/v17_fixed_chin_capture/front.json --measurements outputs/chenger_stylized_arcface_20261011/v17_fixed_open_proportions/measurements.json --trace tools/character_authoring/chenger_v17_chin_trace.json --out outputs/chenger_stylized_arcface_20261011/full_terminal_extended_source_candidate_repeat.json
```

前提交HS2Mod89d8517 / Face2Parameter9e6919d均已推送；完整目标仍active。


### Reference boundary correction; v23/v24 remain rejected (2026-10-11)

Full-size original-image review found a concrete annotation error: the old sparse
left terminal trace passed (501,980)/(528,992), cutting inside the actual curved
light/shadow boundary, and interpolation created a pointed end. The accepted
portrait itself has a rounded terminal. Preserve previous traces, cards and
failures; their extreme terminal-width percentages are observations against that
old annotation, not reliable proof that the portrait demands a sharp chin.

`reference_boundary_annotation.py` observes the actual Rec709 luminance gradient
inside the previously reviewed +/-15px contour band (Gaussian sigma2), fails on
band-edge picks and mismatched image hashes, and retains the existing 6px edge
uncertainty. The corrected 102-point reference was inspected on the original
image; bottom plateau is around (544,999). It does not recover hidden 3D or claim
that a lighting boundary is exact geometry. Both original and corrected source
image hashes remain bound. Reproduction matched the committed annotation exactly:

```powershell
.venv/Scripts/python.exe tools/character_authoring/reference_boundary_annotation.py --image outputs/chenger_stylized_arcface_20261011/accepted_style_target.png --prior tools/character_authoring/chenger_v17_chin_trace.json --out outputs/chenger_stylized_arcface_20261011/reference_boundary_reproduce_fresh.json
```

`chin_source_candidate.py` now records configurable author bounds and actual
source-topology normal-sign diagnostics. Broader v23 (scale span .75, position
span .30, scale range .20..2) converged but produced two reversed normal signs
and extreme parameters; never applied. These diagnostics are not universal
collision/animation certification. V24 used corrected reference and ordinary
bounds, converged, no reversed normals. Terminal RMS .029256 -> .019298 eye
distances; upper/mid .070059 -> .066377, so the complete jaw still differs.
Actual 4439-point game BakeMesh versus source forward max 1.411716e-6, mean
5.171829e-7 game units. Correct deformation does not establish likeness.

V24 actual front/side/card are retained under outputs/.../v24_actual_chin;
side remains hair-occluded, so it cannot certify depth. Rejected card SHA
43dcb5ec1c41700f853f095e0f221f9ca89bd3585465b9132c21195e974dc180.
Recipe is chenger_v24_rejected_recipe.json. Reloaded ordinary native v17 with
original hair and explicit TARGET1/blink=true; snapshot, full raw ABMX response,
style and expression all matched pre-candidate responses. No locked parts.
Continue the complete jaw shape using actual source-supported controls; do not
keep compressing three central bones to compensate for the whole jaw/pose.


### v28 native jaw/chin historical milestone: previous lock revoked by user feedback

Recovered lower-cheek controls from actual head2: CheekLow L/R each has237
weighted source vertices, max weight .396344. Expanded source author profiles
chin8 -> jaw14 (tip/low X offsets and cheek XY scales) -> jaw18 (cheek Y/Z
rotations) -> jaw22 (cheek XY offsets). Rotations use installed cache quaternion
multiplied by Unity qY*qX*qZ Euler via the existing pure math helper, retaining
actual affine columns and all descendants. Recorded local rotations agree with
cached computation within 1.92e-8 matrix components in this baseline. Existing
old full ABMX/deformation implementation is not used for its position/length path.
Position channels require actual length1 and _lenModForceUpdate=false; excluded
rotations or absent caches fail. Output includes required length1 in its replay
patch. Installed BoneModifierData.HasPosition/HasLength are exact comparisons.

Author baseline v17b explicitly changed only CheekLow lengths to1. Their actual
positions still retained the earlier non-unit-length result with no offset and
_changedPosition=false; deviations from _posBaseline ~.000377 X are preserved.
New nonzero position uses _posBaseline+modifier, not currentPosition+delta. This
source branch correction is implemented; v27 predates it and was never applied.
V25/26 likewise remained offline. Baseline jaw22 versus actual BakeMesh max
1.995411e-6 units; source mesh unchanged. No game sweep or fitted deformation gain.

V28 source candidate converged, no source-topology normal-sign reversals.
Actual game4439-point forward error max1.453767e-6, mean6.153858e-7 units.
Using fixed eye/mouth alignment, **actual** upper/mid complete30 dual sections
reference-pixel RMS1.546862, max3.142567; complete41 terminal Y-at-X values
RMS.656863, max1.957089. Both are below retained6px reference boundary uncertainty.
These are geometric observations against the inspected rounded portrait contour,
not exact shader silhouettes. Figure v28_actual_jaw/actual_contour_review.png
includes prior baseline, complete target and actual current mesh. Native front,
90-degree profile and45-degree view were visually reviewed. Temporary vanilla
back0 long hair exposes chin/lip relief; saved/reloaded card retains original bob.
Real ordinary photos img-002/img-020 were revisited for rounded taper and angled
jaw cues; no false same-camera side metric or hidden art3D geometry is claimed.

**Historical lock claim revoked:** the user rejects the actual v28/v30 broad, flat, angular chin. Numerical projection evidence below remains historical; it is not acceptance of the visible jaw/chin. Not complete whole-character identity.
Self-contained lock recipe is tools/character_authoring/chenger_v28_jaw_lock.json:
full5 controlled modifiers, native head2/shape0..18 and all59 values as context,
actual shared ancestry, card/reference/geometry hashes, actual contour residuals,
profile evidence and globals. Skin20/HanmenSSS retained. Keep vertex geometry and
shared deformation dependencies under review in subsequent layout/nose/lip/eye
work; do not claim preserved jaw from unchanged5 parameters if parents or other
weighted bones changed. Reopen explicitly on an actual dependency/appearance gap.

Saved card SHA c5a7b4c0e9971352f3a986b800296eb9e7e7f0efa92527e2f3a0871c26752cb1.
Reload snapshot/style/expression exactly equal. Raw ABMX list42->30 mismatch is
retained:12 identity body modifiers were pruned; all30 remaining records exact,
including the5 jaw bones. This is settings evidence, not full private eye-cache
restoration certification. Normal visible result is native v28, original bob,
TARGET1/blink=true. Next part: quantify layout from existing v28 image before
choosing source-supported local controls; preserve/revalidate the jaw lock.

### 用户反馈后重新打开下颌；生成侧面设计参照 v1

用户提供实际游戏图和原美术图，指出下巴两侧折角及宽平底部越来越明显。
撤销v28部位完成/锁定，保留旧卡、参数和数值证据。当前未修改游戏人物。
旧轮廓图为源网格投影、并非实际图像可见轮廓；两者外观矛盾的具体原因仍需
定位，不能把源形变正确或误差小继续当成外观通过。

按用户建议，以其Downloads原美术图为唯一输入，通过内置image_gen分别生成
向左侧面、向右侧面及向右四分之三侧面。三张已在对话显示，文件保存在
outputs/chenger_stylized_arcface_20261011/generated_side_references_v1/。
完整三个提示词、输入/输出SHA及原生成文件身份记录在
[生成记录](../tools/character_authoring/chenger_generated_side_references_v1.json)。
这些是候选设计参照，尚未获用户接受；左右颏尖突出程度与颏下曲线有差异，
角度亦未经标定，不能冒充同一精确三维头或同时作为硬性几何真值。
原正面仍为脸型基准。下一步先解决轮廓投影/实际外观矛盾，采用一致的侧面
设计约束处理颏部体积，避免只逼近正面投影而造成局部畸变。

HS2主相机配对几何补丁仍为之前遗留的未提交/未部署任务修改；本次未安装、
未重启游戏，未声称修好下巴。完整Codex goal active。
