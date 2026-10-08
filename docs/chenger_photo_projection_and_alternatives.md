# 照片投影贴脸与 FLAME 替代路线

本轮按用户建议把 img-002 的照片面部投影到该照片的原始 MICA 头模，
没有修改任何顶点、形状系数或原始拓扑，没有操作游戏。这里只做一个
可查看的例子和公开算法调研，不扩大成完整离线渲染器项目。

## 实际贴图示例

工具 `tools/model_bridge/photo_projection_review.py` 读取上一轮原始导出，
使用原检测器五点和源 FLAME 的 68 点来估计刚性相机姿态、平移和焦距。
对应关系为两侧眼环顶点均值、鼻尖 30、唇角 48/54；眼中心的对应是近似，
不能当成真实标定，也没有独立验证五点准确性。只调整相机，不调整头模。

相机估计的焦距到达预设上界，说明这五点无法充分确定真实镜头；没有
把相机解当成可信内参。叠图只能作为视觉观察，不能据此量化骨相误差。

原照片生成投影 UV；标准 PyTorch3D 相机转换和光栅化进行遮挡判断。
仅对原视角观测到、且属于源资产 face/eyeball 区域的三角形贴图，其余
用灰色。源 FLAME mask 只限定贴图范围，不删面；它不是照片皮肤分割，
所以照片上的部分头发仍可能投到额头区域。纹理保留照片光照，未声称是
去光照 albedo；没有用图像生成来补不可见部分。

输出为四列：原图脸部区域、固定网格与照片叠图、原相机贴图结果、同一
贴图转到固定 35° 相机。青线是整个头/颈网格的投影外缘，不是提取的
照片脸部轮廓；没有把头发外缘与颅骨轮廓直接当作同一目标。

```bash
cd /mnt/c/Users/13666/Workspace/Face2Parameter
/home/taardis/envs/smirk/bin/python -m tools.model_bridge.photo_projection_review \
  --manifest outputs/chenger_mica_review_20261007/raw/manifest.json \
  --image-index 1 \
  --face-mask /mnt/c/Users/13666/Workspace/smirk/assets/FLAME_masks/FLAME_masks.pkl \
  --out outputs/chenger_mica_review_20261007/projection_face_v3
```

原始图、网格、相机和贴图范围的出处见该目录 `receipt.json`；PNG 为
`projection_board.png`。v1 全头投影、v2 不贴眼球的版本保留，不覆盖。
照片、模型、原始资产、渲染图和 receipt 都在 ignored 输出路径。

**更新视觉判断：贴图后正面辨识度提高；在这张相机估计叠图里，整体脸部
轮廓比上一轮固定正面灰模给人的观感更接近照片。** 因此，上一轮“不选
可靠候选”的保守选择保留，但仅凭灰模把脸型几何判定为明显失败，依据
过强。没有由此宣告骨相正确；眼周、下颌等仍需在可靠的相机配准和其他
视角下判断。源视角投影贴图能够让错误的几何也显示出照片特征，这是
构造性质，不是独立三维验证。转 35° 是查看同一几何/贴图，不是另一张
照片或另一视角的独立真值。

## 公开算法调研（2026-10-07）

### HRN / MV-HRN：优先作为另一条模板和局部形变路线研究

- 官方代码：https://github.com/youngLBW/HRN
- 论文：https://arxiv.org/abs/2302.14434
- 官方 `models/bfm.py` 直接读取 BFM 基模与身份/表情基底，非 FLAME；
  HRN 加入分层几何表示，官方 demo 有 single_view 和 multi_view 两条入口。
- 现有实现提醒高频位移在导出网格与二维渲染中的效果可能不同；必须
  看实际导出几何，不能仅用漂亮的渲染证明游戏底模可用。
- 本轮没有下载或运行它，未证明比 MICA 更准确。完整头部扩展在官方
  README 链接的 ModelScope 中另行提供；不能把 face demo 当作完整头模。

### H3D-Net：绕开固定线性形状空间的完整头部路线

- 作者项目页：https://crisalixsa.github.io/h3d-net/
- 论文：https://arxiv.org/abs/2107.12512
- 使用从扫描学习的隐式 SDF 头部先验，再对几张照片优化几何与表面外观，
  后期允许先验本身微调；不是 FLAME 的 300 维线性系数空间。
- 输入需要照片、遮罩和相机姿态。项目页展示三视角方案；相机与 mask
  的获得仍是工作，不是任意三张随拍直接给出准确头模的保证。
- 我的适用性判断：现有照片跨场次、光照和表情不同，需要先确认是否
  足够符合共享静态头部重建的条件。本轮没有部署或验证该路线。

### Pixel3DMM：更强的观测约束，但仍然是 FLAME

- 官方代码：https://github.com/SimonGiebenhain/pixel3dmm
- 论文：https://arxiv.org/abs/2505.00615
- 官方流程从图像预测密集 UV/法线，再拟合 FLAME；支持多张不连续照片，
  也能忽略 MICA 初值并使用 FLAME2023。源码 README 明确这些开关。
- 可以作为判断“原 MICA 系数预测是否不足”的现成实现候选，但不能把
  它说成独立于 FLAME 的几何真值，也不能用它排除 FLAME 空间的限制。

另查到 MoGe（像素三维点图）和 TRELLIS.2（通用图像到三维资产）。前者
提供可见表面的预测，后者生成任意拓扑网格和材质；官方材料没有提供
足以保证程儿面部身份几何忠实度的依据，因此本轮不优先推荐它们。
来源：https://github.com/microsoft/MoGe 和 https://github.com/microsoft/TRELLIS.2 。

## 当前选择

先交付投影图帮助理解当前模型。若继续评估替代管线，优先查 HRN/MV-HRN
的真实导出和现成依赖，再考虑 H3D-Net 的相机/输入条件。两者都是候选，
本轮不承诺交付准确三维真值，不安装一大批新模型，不重新改造头颈。

只提交本轮工具与此记录；此前未完成的头颈/attachment 草稿继续保持
分离。HS2Mod 本轮无源文件、DLL、目标记录或游戏状态修改。
