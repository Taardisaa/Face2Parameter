# 程儿：三张照片的原始 MICA 输出审查

本次只回答一个问题：换照片能否得到明显更像程儿的原始头模？
不修改游戏状态，不处理脖子，不进行形状系数拟合。所有结论是直接视觉
判断，不是三维真值测量，也不是 FLAME 表达能力上限的证明。

## 输入和实现

选择 `img-001.jpg`（原输入）、`img-002.jpg`（较正面、脸部遮挡少）、
`img-020.jpg`（另一张斜侧面）。原图位于
`C:/Users/13666/Workspace/NewWriting/references/raw/characters/程儿/photos/`。
只把这三张照片按原字节复制到本次 ignored 输出目录，来源与 SHA256 记录
于 `outputs/chenger_mica_review_20261007/selection.json`。

复用已提交的 `scripts/mica_export_geometry.py`：官方 antelopev2 检测、
五点对齐、112×112 ArcFace 输入、归一化身份向量、官方 Generator 和
FLAME 解码；checkpoint 完整严格加载。没有额外学习模型、身份混合或
网格局部变形。模型版本 `af22e7a5810d474bc28a1433db533723d6bd2b07`；
checkpoint SHA256
`4542a467d9e8f7521474a1d00eac89552bebef0b331b72bf7fbd6f065ff64d7b`。
完整依赖、输入、输出和解码资产出处保存在 raw manifest。

原始推断保存于 `outputs/chenger_mica_review_20261007/raw/`：

- `pred_shape_code`：300 个原始形状系数。
- `faceid`：512 维原始归一化身份特征。
- `pred_canonical_shape_vertices`：原始 5023 个顶点。
- 同时保留实际输入裁图、输入张量、检测框及五点坐标。

## 执行

在 WSL Ubuntu 的既有 SMIRK 环境中运行，不下载模型、不安装新环境：

```bash
cd /mnt/c/Users/13666/Workspace/Face2Parameter
/home/taardis/envs/smirk/bin/python scripts/mica_export_geometry.py \
  --mica-dir /mnt/c/Users/13666/Workspace/HS2Mod/tools/parameter_audit/source_mica_20261007 \
  --assets outputs/mica_assets_20261007 \
  --flame-model /mnt/c/Users/13666/Workspace/smirk/assets/FLAME2020/generic_model.pkl \
  --landmark-embedding /mnt/c/Users/13666/Workspace/smirk/assets/landmark_embedding.npy \
  --in outputs/chenger_mica_review_20261007/inputs \
  --out outputs/chenger_mica_review_20261007/raw --device cuda
/home/taardis/envs/smirk/bin/python -m tools.model_bridge.mica_photo_review \
  --manifest outputs/chenger_mica_review_20261007/raw/manifest.json \
  --selection outputs/chenger_mica_review_20261007/selection.json \
  --out outputs/chenger_mica_review_20261007/review
```

重跑必须指定新的输出目录；命令不覆盖已有证据。

## 实际结果和判断

三份推断成功，身份特征和形状系数两两不同，并非循环误用了同一个结果。
img-001 的系数和原始顶点与此前已导入的那份原始输出完全一致；此证据
记录在 `review/original_output_check.json`。三份输出使用同一模板、形状
基底与拓扑；零姿态、零表情下由原始系数直接解码，与保存的原始网格相符。

`review/raw_models.png` 使用 PyTorch3D 的标准平滑法线着色，相同的正面、
35° 和 90° 相机、同一尺度和固定光照，左列为实际零系数模板。只调整
画面取景，没有裁切网格或修改顶点，不尝试模拟游戏材质。
`review/photo_inputs.png` 显示原图脸部区域及模型实际收到的 112×112 裁图；
没有明显选错脸或对齐错位。较低分辨率是这条官方输入路径的规格，不能
仅凭裁图的像素感判定推断错误。

审查者：本次 Codex agent，方法：直接查看原照片、实际裁图和几何视图。
不是独立人类复核；相机未按照片配准，也没有真实头部扫描作为依据。

**结论：换这三张照片没有带来足够明显的脸型改善，不选出可靠的新候选。**
img-002 的下颌略收，但不足以认定相似度有实质优势。三份网格仍保留
接近默认模板的宽钝下巴和颧颊过渡，未充分表现照片中下半脸向下巴收窄
的形态。没有以缺少化妆、贴图或眼球外观解释这些几何差异。

下一选择：停止仅靠更换照片期待 MICA 自动解决脸型；保留这三份原始输出
作初值，另立小目标探索照片约束的 FLAME 系数拟合。共享身份系数，各图
相机和表情单独处理，首先针对下半脸轮廓与下巴。尚未实现或执行这一步，
没有证明它必然能还原真实骨相，也没有证明 FLAME 表达不了目标。

## Git 和工作范围

提交本次 review 工具、结论，以及此前已实际运行但未提交的
`tools/native_head/template_view.py`。生成网格、照片、系数、PNG 和原始
模型输出继续按既有规则忽略。没有修改既有原始推断或先前失败证据。
此前头颈、attachment 和桥接文档的未完成草稿仍保留在工作区，不混入
此次提交。HS2Mod 只更新目标记录和 roadmap；没有 DLL 或游戏操作。
