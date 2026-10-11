# 程儿v28五官布局：当前观察与源控制

下巴/下颌已经作为独立部位锁定。对应已推送里程碑：
HS2Mod `f6c1863` / Face2Parameter `ca4a3d9`。
完整goal仍active，接下来只处理五官纵横布局；尚未修改布局参数。
可见游戏人物为原生v28、原bob、Hanmen/SSS、TARGET1/blink=true。

## 当前量化不能混用自动检测结果

复用现有真实v28正面图，不采新游戏参数样本：

```powershell
.venv/Scripts/python.exe tools/character_authoring/portrait_compare.py --reference outputs/chenger_stylized_arcface_20261011/accepted_style_target.png --candidate outputs/chenger_stylized_arcface_20261011/v28_actual_jaw/front.png --out outputs/chenger_stylized_arcface_20261011/v28_layout_observation_fresh
```

新FAN68把目标颏底标在y992，实际修正参考边界底端约y999；游戏颏底
标在y821，也偏离真实源三角边界。不能以新的eye_to_chin -3.05%重开
已经由真实完整边界验收的下巴。原始检测/贴图/失败保留。

更重要的是，改变下颌形状影响了检测器的其他标点：nose33由638→632，
嘴内点66由(380,685)→(374,687)，眼轮廓若干点漂移7–9px。
对同一native source triangle的透视重心表面附件作静态对照，nose31/33/35
在两次真实捕获的投影变化仅约.0003–.0004px，mouth48约.011px，其余
51/54/57约.0003–.0004px。它们是此前图像观察绑定的相同皮肤表面点，
不是私有eye/gaze恢复或解剖标记完整认证。source_coupling_check.json保留
绑定与全部差值。故不能按新检测器的6px鼻点变化决定新的鼻位置。

固定v17观察在相同眼距下给出：目标与游戏鼻底高度差约3.30游戏像素，
嘴线约8.37像素。鼻底小差值可能处于标点误差中，暂不直接改。嘴线差值
需结合独立眼轮廓与实际唇缝确认。原始检测及固定观察的鼻嘴间距方向
互相矛盾，禁止取更方便的一组。source实际眼相关权重>0的913个点中，
最大位移.023325游戏单位，不能笼统声称所有眼周皮肤完全没受下颌影响。

## 已查到的局部源控制与下一步

实际父链：FaceUp_ty是FaceBase子节点；FaceUp_tz在其下。NoseBase_s
在NoseBase_trs下，EyePos_rz_L/R在各Eye_s下，MouthBase_s在MouthBase_tr下。
FaceUp_ty、FaceUp_tz、NoseBase_s已有有效ABMX cached baseline，length1。
FaceUp_ty positionY当前-.0003605336、NoseBase_s Y-.00238890946。
MouthBase_s与EyePos_rz没有现存modifier/cache，不能凭空假定新modifier基线。
FaceUp_ty不是已锁定下颌骨的祖先，但自身蒙皮权重及后代会耦合其他表面，
仍必须在完整源码模型中检查下颌边界。原生0–18锁定继续保持；不能为了
布局静默改全脸父骨，或用参数没变替代形状没变。

下一步先核对现有图的眼轮廓与唇缝标点，再复用实际缓存/父链/权重模型
计算局部上脸与鼻位置控制，检查v28 jaw lock和共享骨耦合后才进入游戏。
布局尚未验收或锁定；鼻/唇/眼眉/发型/整体渲染仍排队。


## 本轮实际制作：v30工作卡，布局仍未锁定

图像像素网格复核并保存独立眼裂6点、鼻底和唇缝观察，分别保留3/4/2px
标点不确定性。chenger_v28_layout_landmarks.json与local8_layout_marks.json
绑定原图SHA；继续沿用v28下颌的固定v17尺度/滚转/嘴线注册，未逐候选
重新检测、拉伸或选择更方便的FAN结果。复核图在v28_reviewed_layout_marks.png。

扩展SourceForward为明确parameter_fields及mesh_name，读取实际完整source、
weights、bindposes、active blendframe、全部父链矩阵和 installed ABMX cache。
分别覆盖真实o_head4439点、o_eyebase_L/R各148点。未注册root不假定基线：
核查安装HS2ABMX BoneController.AddModifier调用FillInTransforms+CollectBaseline，
通过该原生流程注册identity Eye_s_L/R、NoseBase_trs、MouthBase_tr，再读取
实际_hasBaseline、_posBaseline和length1/_lenModForceUpdate=false。临时identity
不会当作已制作的结果；真正非零4项根节点位置保存在v30卡中。

先制作v29上脸+鼻根候选，实际head源预测一致但眼网格误差约.00283游戏单位，
保留该失败，不靠参数补偿。安装IL.dll EyeLookCalc.EyeUpdateCalc源码确认：
NO_LOOK直接写fixAngle；TARGET等分支在每次更新结束重新写fixAngle。
基线采集后恢复TARGET1再切回NO_LOOK，改变了源缓存！实际cf_J_look_L/R
local quaternion及driver fixed_angles_xyzw确认该变化。v29已保存为未锁定候选，
卡SHAfab1beebb3d3046c5d6b0da2a16125cb6dd5b4cb1aa8b1532a312d76bfaf2fb7。

修正协议：从新源基线到实际候选验证持续保留当前角色NO_LOOK3/blink=false，
不关闭全局EyeLook。新增gaze_contract绑定actor、bridge/game MVID、实际模式表
与双眼fixAngle；物理写入前独立export读取确认相等，实际后再确认相等。
正常TARGET1/blink在完成该窗口后恢复。该协议处理本次缓存生命周期问题，
不认证全部私有gaze历史或所有渲染回调。

v30使用layout_source_candidate.py的local8配置，独立求解左右Eye_s、NoseBase_trs、
MouthBase_tr的XY位置，共8标量；原上脸FaceUp_ty恢复v28值。全部控制来自上述
实际缓存长度1位置分支，完整后代/权重计算，无截图拟合形变gain。
实际源4735点对预测：head max4.406261e-6、mean2.451966e-6游戏单位；
left eyebase max2.397258e-6、right2.247064e-6。前/后fixAngle合同相等。
下颌真实完整轮廓仍约上中段max3.1433px、颏底max1.9583px，源预测的完整下颌
位移为0；5项锁定骨和native0–18重载后完全相同。

实际已改善鼻底—唇缝间距、左右眼/鼻/唇的源点布局。保留真实front/quarter图。
但**不将表面附件四点几乎零残差视为可见眼裂中心精确对应**。附件在眼网格上，
眼裂还受头部/材质遮挡；新自动FAN也有漂移。可见眼裂6点须独立检查，不能
因为眼网格点移动就声称整个可见眼裂已按同样像素移动。
拓扑读取发现head有7组boundary：两眼各10点，其他为口/颈/UV边界；眼部边界
投影和可见白眼轮廓不同，不把拓扑hole直接等同可见开口。

另外，main capture face_surfaces的实际raw BakeMesh hash与后续geometry export
raw hash三网格均不相同，保留为未建立严格配对的证据。Source数学对export
通过不等于该export一定是冻结渲染当刻的完整几何；头矩阵/配置相同也不能替代。
下一步恢复实际主相机冻结窗口内的几何与投影来源，再完成可见眼裂对应；
不放宽现有门槛，也不反复采样去猜可恢复逻辑。布局保持进行中。

已保存/重载工作卡：Chenger_stylized_reconstruction_v30_layout_working.png，SHA
427404a17b11a6afc485a0f23a20f8ebe4155c168380b5b07b259d93da9f359c。
正常可见原生v30、原bob、Hanmen/SSS、TARGET1/blink。snapshot、完整raw ABMX、
style、expression全部重载相等，无缺失identity行；下颌锁定仍有效。
配方chenger_v30_layout_working_recipe.json；代码、观察和失败都保留，完整goal active。
