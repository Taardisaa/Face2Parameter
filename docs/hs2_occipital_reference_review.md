# 原生后脑与当前程儿轮廓对照

2026-10-09。用户指出高位后脑太靠后，同时后脑显得薄；要求先对照原版。
本轮只做来源和形状审查，没有进一步修改候选头形或安装包。

当前人物原生保存为 `pre_occipital_comparison_20261009.png`，暂时只把
head/skin 改为实际供体 `headId=2, skinId=20`，用同一相机捕获侧面／后斜／
背面及同帧几何，随后正常加载预运行人物卡，回到程儿 v5。没有重启游戏。
59 个头部驱动值在原生与当前捕获中均为 0.5。

## 发现

原生后部在侧面上是连续圆弧，下部圆度与后颈回收相连。当前后部突出
位置较高且更靠后，往耳后偏下则较快进入一段长斜面。这使高位突出和
中下部欠圆同时可见，不能再靠整体上移／向前推后脑解决。

完整共同坐标剖面还表明，当前后部总体其实比原生更靠后。因此这里的
“薄”不能被报告成整个后脑前后尺寸小于原生；应区分总体尺寸和中下部
枕形的饱满度／曲率分布。上一版局部前收使后凸减少，但没有重建合理的
上中下曲率分布。下一次几何调整应同时处理高位后凸与中下部圆弧。

对照图使用真实原生、v4、v5 捕获。源网格先应用实际单帧表情，再用实际
骨骼矩阵和 bindpose 重放蒙皮，转换到同一个 `cf_J_Head_s` 坐标系。
没有把带缩放的 BakeMesh 任意假设为 renderer-local 后重复施加缩放，
也没有归一化耳高、整体头大小或位置来掩盖差别。

## 重现与证据

```powershell
.venv/Scripts/python.exe -m tools.native_head.occipital_reference_review --native ../HS2Mod/artifacts/native_mother_20261008/original_head2_comparison.geometry.json --before ../HS2Mod/artifacts/native_mother_20261008/neck_v4_final.geometry.json --current ../HS2Mod/artifacts/native_mother_20261008/occipital_v5_final.geometry.json --out outputs/native_mother_template_20261008/occipital_reference_review_v1
```

蓝线为原生头 02，灰线为 v4，橙线为当前 v5。完整中央剖面、中线后部放大
和侧向截面位于 `occipital_reference_sections.png`；原生游戏三视图为
HS2Mod ignored `artifacts/native_mother_20261008/original_head2_comparison_0_sheet.png`。
许可网格、完整捕获和图形继续 ignored，来源 SHA 记录在输出 receipt。

此对照证明曲面分布的差别，不证明程儿真实颅骨应与原生模板完全一致。
