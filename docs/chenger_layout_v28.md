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
