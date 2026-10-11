# 程儿 v14：修正窄长下巴

2026-10-11，用户明确要求修复 [v13 下巴异常](chenger_alignment_v13.md)。
本次实际修改游戏并保存新卡，未把量测图当作修复交付。

## 源码与资产对应修改

沿用当前安装ABMX 4.4.6.0的已反编译Apply/CollectBaseline及实际头2资产。
原生4是FaceLow_sx，关键帧X Scale .92→1.08；5是Chin_sx，X .93→1.07。
两个字段都是X轴控制。原生10虽然面板写下巴尖宽度，但ChinTip_sx实际同时
缩放XYZ .5→1.5；直接用它补横向宽度会联动长度/深度，因此保持原值，使用
已有ABMX的单独X Scale。上述端点说明控制范围，不替代原生25关键帧和映射。

实际骨骼父子链：FaceLow_s包含两侧CheekLow/Up、ChinLow、Chin_rs；
Chin_rs再包含ChinTip_s。已有头2源网格权重和父骨结构指导补回两侧体积，
不按图片拟合经验变形函数。新配方在
`tools/character_authoring/chenger_v14_recipe.json`，以完整v13卡为基线：

- 原生4 .30→.45；5 .3079933→.78，补回下颌宽度。
- Chin_rs Length1.16→1.11，X Scale .9998406→1.16；保持position=0，减少前伸与下移。
- ChinTip localY -.16→-.09，X Scale1.0542556→1.10；其Length保持1，保留YZ Scale/Z位置/旋转。
- ChinLow localY -.08→-.03，X Scale1.03095818→1.16；Length保持1，保留其余字段。

使用实际原生batch重建，再由安装ABMX LateUpdate应用。只是人物造型制作，
没有修改或宣称新的离线形变兼容实现；未扫掠参数或从截图推导可恢复逻辑。

## 小范围最终确认

最终保存后切换v13再重载v14，all snapshot和42个ABMX数组等值。
与v13相比只有上述3个modifier和2个native滑条变动，meta改版本名；
眉毛、唇色、虹膜、发型/衣物、其他身体/脸面妆容区域均等值。
Hanmen Face/Body及shader浮点属性、AIT/Liquid保持，SSS=true。
主相机正面/侧面捕获前后camera/lights等值，临时blink结束恢复true。
当前人物是v14，所有旧卡和失败图保留，没有诊断壳/隐藏头发。

采用与v13相同人工作业：实际可见完整双侧轮廓，图SHA绑定，眼距统一尺度，
嘴线平移对齐，不单独拉伸图像。新人工标注在
`tools/character_authoring/chenger_v14_jaw_trace.json`。原图坐标网格和贴边折线
已检查。FAN只提供眼/嘴对齐；没有靠FAN内部下巴点评价轮廓。

| 相对接受图的局部二维差异 | v13失败 | v14修正 |
|---|---:|---:|
| 下颌宽度，高度25% | 窄20% | 窄3% |
| 下颌宽度，高度50% | 窄27% | 窄7% |
| 下颌宽度，高度75% | 窄30% | 窄5% |
| 下唇下缘到下巴 | 长12% | 短3% |

中段二维割线夹角从84度到95度，接受图94度；不等于解剖下颌角。
嘴线到下巴现约短5%，不以这些差异宣称人物完全一致。人工边界误差约
参照±6px/游戏±4px，唇下缘±5/4px，加上残余姿态/透视；3%-7%量级不能当
无误差尺寸。当前主要窄长V形已改善，右侧曲线与接受图仍非完全重合。
没有接受图侧面，未虚构侧面三维前突差值；实际游戏侧面确认下巴前伸减弱。

证据根目录 `outputs/chenger_stylized_arcface_20261011/`：
`v14_before.json`、`v14_saved_state.json`、`v14_reloaded_state.json`、
`v14_roundtrip_acceptance.json`、`v14_capture_acceptance.json`、
`v14_jaw_shape_review/jaw_comparison.json`及`jaw_contour_comparison.png`。
原生图在HS2Mod `artifacts/chenger_style_20261011/chenger_v14_reloaded_front.png`
与`chenger_v14_reloaded_side.png`。身份目标未宣称完成。

最终卡：
`E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_stylized_reconstruction_v14_jaw_corrected.png`。
SSS是全局设置，卡重载本身不保证开启。新配方以v13为基线，和原v13配方串联
可从v10完整卡重建；原图/游戏资产/输出均保留在忽略目录，提交小型配方和标注。

```powershell
.venv/Scripts/python.exe tools/character_authoring/jaw_trace_compare.py --measurements outputs/chenger_stylized_arcface_20261011/v14_reloaded_proportion_review/measurements.json --trace tools/character_authoring/chenger_v14_jaw_trace.json --out outputs/chenger_stylized_arcface_20261011/v14_jaw_shape_review_repeat
```
