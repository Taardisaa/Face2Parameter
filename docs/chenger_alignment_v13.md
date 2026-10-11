# 程儿：纵向比例、嘴角与妆容对齐 v13

2026-10-11，承接 [人物制作记录](chenger_stylized_reconstruction.md)。用户目的仍是
在游戏内复现人物；测量仅指导本次造型，不代表身份目标完成。

## 比较与测量纠错

参照为接受美术图 `accepted_style_target.png`，基线为实际 Hanmen/SSS v10 正面，
最终为保存、切换 v10、重载 v13 后的实际主相机正面。证据目录为
`outputs/chenger_stylized_arcface_20261011/`，游戏图在 HS2Mod
`artifacts/chenger_style_20261011/`；不改变原图纵横比。

同一 FAN68 的眼轮廓质心统一尺度，消除平面 roll。最终 FAN 下巴8落在可见
轮廓内部约37px；v12还出现鼻宽检测波动，因此不能把自动值视为可靠真值并
据此继续拉长。全部原始失败估计保留，不改阈值或删除观测。
`portrait_review.py` 从缓存观测读取，明确只替换人工复核的下巴8；
标注绑定图片 SHA，保留原点，以红空心圈显示原 FAN 点。标注 JSON 在
`tools/character_authoring/chenger_v10_chin_review.json` 与 `chenger_v13_chin_review.json`。
下巴标签为参照[539,1000]、v10[414,775]、v13[393,831]，人工判断界限分别
约±5/4/4px；这不是检测器的统计误差认证。复核图已检查点位。

| 按眼距归一化的二维指标 | v10 相对美术图 | v13 相对美术图 |
|---|---:|---:|
| 眼线到可见下巴 | -6.9% | +1.3% |
| 眼线到鼻底 | -9.4% | -3.6% |
| 鼻底到嘴线 | +14.0% | +3.8% |
| 嘴线到可见下巴 | -11.5% | +5.7% |

几百分点不能认作精确差异：残余姿态、透视、人工边缘及其余 FAN 标点均有误差。
最终嘴宽自动估计仍约短10%，且角点落于内侧、纹理/表情边界不一致，未据此
宣称嘴完全对齐。鼻宽最终自动值约-0.6%。眉/虹膜/嘴角深度未给出虚构认证。

复现复核命令（项目 venv）：

```powershell
.venv/Scripts/python.exe tools/character_authoring/portrait_review.py --measurements outputs/chenger_stylized_arcface_20261011/v13_reloaded_proportion_review/measurements.json --annotations tools/character_authoring/chenger_v13_chin_review.json --out outputs/chenger_stylized_arcface_20261011/v13_reviewed_chin_repeat
```

## 实际源码机制与造型

重新反编译当前安装 HS2ABMX.dll，4.4.6.0，SHA256
`9224876CE931EB3C40A347BA57ABEBCCA515BDE58AF2A6B9802B367A5B3563AF`。
`BoneModifier.CollectBaseline` 已有长度基线时不重置 `_lenBaseline`；
`Apply` 非单位 Length 用当前 localPosition 的方向乘缓存长度和 Length。
因此原生 Chin_ty 的位置修改并非完全保留其位移幅度。第一轮 native6=.92、
11=.70未得到预期下脸长度，保留失败图片与量测；最终二者恢复 v10的.68/.49，
直接制作已有骨骼。长度修正骨 Chin_rs 保持 position=0，避免非单位长度加
非零位置的重复叠加分支。ChinTip/ChinLow保持 Length=1，直接修改 local Y，
保留其余 scale/rotation；不是另造近似游戏形变公式。

最终相对 v10 的配方在 `tools/character_authoring/chenger_v13_recipe.json`：
原生鼻32=.12、鼻整体宽35=.50、小鼻宽39=.59、嘴位置47=.36、嘴宽48=.70。
Chin_rs Length1.16；ChinTip local Y=-.16；ChinLow Y=-.08；
Mouth_L/R local Z=-.02111438（比v10内收.007），X改为±.015保留镜像。
只改已有5个 modifier；全部42项其余字段保留。该骨骼单位不是毫米。

眉13改7，外侧纹理本体上扬，最终 Tilt=.64；颜色/透明度和布局见配方。
眉 Shader UV布局/旋转依据 ChaControl 与已反汇编 AIT路径；没有把滑条当角度。
虹膜23改0，使用完整放射纹理，灰褐色(.42,.39,.37)，WH=.61；黑瞳0的WH=.28，
保留原高光、眼白，不以发光代替颜色。唇3改为(.62,.28,.40,.74)，Gloss=.45。
实际正面看得到虹膜与黑瞳分离、外扬眉尾和粉色唇；嘴角凹陷仍轻微，
与参照轮廓感/唇形有剩余差异。侧面下巴与唇下起伏可见，未认作真人三维还原。

## 保存与状态验证

最终卡 `E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_stylized_reconstruction_v13_alignment.png`。
v10/v11/v12与更早卡保留；v11/v12为制作过程候选，不是最终。v13切换v10再重载后
all snapshot与42项ABMX数组等值。Face/Body的Hanmen shader、Gloss/SecondaryGloss/
VeinsOpacity及液体AIT/Liquid均等值，SSS=true；SSS是全局效果，卡本身不保证开启。
正面/35度/侧面实际主相机捕获保留完整DHH/CTAA链，capture前后camera/lights等值，
临时关闭blink后恢复true。当前可见人物为v13，无诊断头壳或隐藏发型。
`v13_saved_state.json`、`v13_reloaded_state.json`、`v13_capture_acceptance.json`与
`v13_roundtrip_acceptance.json`绑定验证；`v13_reviewed_chin/`保留最终复核图。
v12只读几何导出保留，因与主相机非同帧，不用它认证图片的精确像素投影。

本轮测量工具里程碑：Face2Parameter b29373b、HS2Mod cf0f9fb，均已推送。
后续交付提交见本次 Git 历史；两仓库无关的模型桥接/颈部改动不纳入本任务提交。
