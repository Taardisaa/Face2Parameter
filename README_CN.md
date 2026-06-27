# Face2Parameter：从人脸图像预测游戏角色捏脸参数

[English](README.md) | **中文**

输入一张人脸图像，即可预测对应的**游戏角色捏脸参数**（已在 HoneySelect2 上测试）。模型由**冻结的预训练骨干网络**和 MLP 回归头组成，输出的参数向量可以直接写入游戏角色卡。

```
                       第一阶段：冻结骨干网络              第二阶段：MLP 回归头
                      ┌───────────────────────┐         ┌──────────────────┐
   人脸图像   ─────►  │  DINOv2 ViT-S/14（默认）│  ────►  │   4 层 MLP        │  ────►  205 维
  224×224 RGB         │  或 ArcFace（512 维）   │ 特征向量 │  （回归器）        │         参数向量
                      │  参数冻结，特征缓存至磁盘 │         └──────────────────┘            │
                      └───────────────────────┘                                          ▼
                                                                          写入 HS2 角色卡
                                                                      （54 维基础参数 + 骨骼参数）
```

> 当前版本已不再使用从零训练的 **VAE** 提取特征，而是改用**冻结的预训练骨干网络**：默认使用 DINOv2，同时提供对表情变化更不敏感的 ArcFace 方案。回归头以及数据、角色卡和标签相关的处理逻辑均保持不变。原有的 VAE 代码已移至 [`legacy/`](legacy/) 目录。

## 0. 项目简介

模型分为两个阶段：

1. **第一阶段——冻结的骨干网络。** 预训练骨干网络将 224×224 的 RGB 人脸图像编码为特征向量。由于网络参数全程冻结，因此每张图像只需提取一次特征，并将结果缓存到磁盘即可。
   - **DINOv2 ViT-S/14**（默认，384 维）：通用、自监督的视觉骨干网络，特征提取能力较强。
   - **ArcFace**（`w600k_r50`，512 维）：用于人脸识别的特征模型，对表情变化更加鲁棒，可缓解“输入笑脸照片时，预测脸型过宽”的问题。详见 [docs/expression-invariance.md](docs/expression-invariance.md)。
2. **第二阶段——MLP 回归头。** 一个轻量级 MLP 根据缓存的图像特征，回归得到 **205 维捏脸参数向量**，其中包括 54 维基础 `shapeValueFace` 参数和经过掩码筛选的骨骼参数。

与从零训练 VAE 相比，使用成熟的预训练骨干网络不仅能以更少的训练成本获得更好的图像特征，也无需额外调整重建损失。骨干网络采用可插拔接口（[src/models/backbone.py](src/models/backbone.py)），因此可以自由切换 DINOv2 和 ArcFace，而无需修改回归头、训练流程或角色卡写入逻辑。

## 1. 安装

克隆项目：

```bash
git clone https://github.com/Taardisaa/Face2Parameter.git
cd Face2Parameter
```

创建虚拟环境并安装指定版本的依赖。`torch` 和 `torchvision` 使用 CUDA 版本，需要从 PyTorch 软件源安装：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

运行以下命令检查依赖导入和 CUDA 环境；添加 `--with-dinov2` 参数后，还会测试能否通过 PyTorch Hub 加载 DINOv2：

```bash
python tools/check_env.py
```

> 本项目已在 RTX 4080、Python 3.13、PyTorch 2.11（CUDA 13）环境下测试通过。请根据本机显卡驱动选择合适的 CUDA 标记（如 `cu130`）。`mtcnn_ort` 人脸检测器、ArcFace ONNX 模型和角色卡序列化工具 `HS2ABMX.exe` 仅用于图像对齐、推理或 ArcFace 相关流程；单独训练 DINOv2 回归头时不需要这些组件。依赖文件中的 `onnxruntime-directml` 针对 Windows 环境固定了版本；如果使用 Linux 或纯 CPU 环境，请改用 `onnxruntime`，也可以在 CUDA 版本受支持时使用 `onnxruntime-gpu`。

## 2. 配置

项目中的所有选项都由 [`config.py`](config.py) 统一管理，可通过 `--config` 选择预设配置：

- `smoke`：离线 smoke test 配置。使用低维度的 `DummyBackbone` 和合成数据，无需下载模型或准备数据集，适合快速检查整套流程能否正常运行。
- `dinov2_vits14`：默认训练配置，使用冻结的 DINOv2 ViT-S/14，特征维度为 384。
- `dinov2_vitb14`：体量更大的 ViT-B/14，特征维度为 768。
- `arcface`：使用对表情变化更鲁棒的 ArcFace 骨干网络，特征维度为 512；训练和验证均使用拟真图像目录 `aug_images/`。详见 [docs/expression-invariance.md](docs/expression-invariance.md)。

## 3. 数据目录

真实数据应存放在 `data/` 目录下（该目录已加入 `.gitignore`），结构如下：

```
data/
  cards/         # HS2 角色卡 PNG，标签数据的来源
  images/        # 对齐并裁剪后的 224×224 游戏内人脸图像
  aug_images/    # 同一批人脸的拟真（Stable Diffusion）图像，文件名与 images/ 一致
  labels.json    # 名称 -> 205 维向量，由 cards/ 生成
```

运行 `extract_features.py` 后，从 `images/` 提取的特征会缓存至 `features<tag>/`，从 `aug_images/` 提取的特征则会缓存至 `aug_features<tag>/`。训练时通过 `aug_prob` 按样本混合两个图像域，使回归头既能学习清晰的游戏内画面，也能适应实际推理时可能输入的拟真人脸图像。

## 4. 训练

```bash
# 准备好数据集后依次运行
python tools/gen_labels.py --cards-dir data/cards --out data/labels.json   # 从角色卡生成 205 维标签
python tools/make_splits.py --config dinov2_vits14 --val 1000 --test 200    # 生成 train/val/test 索引
python extract_features.py --config dinov2_vits14 --variant both            # 第一阶段：缓存两个图像域的特征
python train_head.py      --config dinov2_vits14                            # 第二阶段：训练 MLP 回归头
```

训练程序会自动读取 `exp/<exp_name>/ckpts/` 中最新的 checkpoint 并继续训练。可以使用 `tensorboard --logdir exp` 查看训练指标。通常训练 15～30 个 epoch 即可获得较好的回归效果。

如需训练对表情变化更鲁棒的版本，将配置改为 `--config arcface` 即可。ArcFace 特征会保存在 `*_arcface/` 目录中，不会覆盖 DINOv2 的特征缓存。

在预留的测试集上评估已训练的回归头，并分别计算两个特征域的 MSE、L2 距离和余弦相似度：

```bash
python tools/eval_head.py --config dinov2_vits14 --split test
```

**Smoke test（无需数据集或额外下载）：**

```bash
python tools/make_synthetic_data.py
python train_head.py      --config smoke
python extract_features.py --config smoke
```

## 5. 推理

项目提供两个推理入口，二者共用同一套骨干网络和回归头。

**图像 → 205 维参数向量**

此方式较为轻量，只需要骨干网络和回归头，不需要角色卡序列化工具或模板角色卡：

```bash
python predict.py --config dinov2_vits14 --image test/my.png   # 输出 outputs/my_out.json 和 .npy 文件
```

**图像 → HS2 角色卡**

此方式会将预测参数写入一张真实的 HS2 角色卡，需要 `HS2ABMX.exe`、属性 JSON 文件，以及用于人脸对齐的 `mtcnn_ort`：

```bash
python infer.py --config dinov2_vits14 \
    --head exp/dinov2_vits14_head/weights/head_epoch_30_step_XXXX.pth \
    --image test/my.png --out outputs/
```

如果没有指定 `--head`，程序会优先读取当前配置对应的 `exp` 目录中最新的权重。若目录中尚无权重（例如刚刚克隆项目），则会自动使用 [`release/`](release/) 中随项目提供的回归头。因此，无需自行训练也可以直接运行 `predict.py` 或 `infer.py`。

`--template` 默认使用项目自带的 [`assets/default_template.png`](assets/default_template.png)。模型只负责预测脸型参数，身体、发型和服装等信息均由模板角色卡提供。可以传入 `--no-detector` 跳过 mtcnn 人脸对齐，改用保持宽高比的居中裁剪。

**同一个人的多张照片 → 更稳定的向量。** 给 `--image` 传入一个**目录**，程序会把每张图各自预测的结果聚合成一个向量（默认在 embedding 空间用逐维中位数聚合，可剔除个别糟糕的帧）。这能平均掉姿态、光照、检测噪声，对 DINOv2 还能跨不同表情抵消表情泄漏。建议放 ~5–10 张多样化的照片，**多样性比数量更重要**。详见 [docs/multi-image-aggregation.md](docs/multi-image-aggregation.md)。

```bash
python predict.py --config arcface --image dir_of_photos/          # 一个平均后的向量
python predict.py --config arcface --image dir_of_photos/ --aggregate trimmed --save-per-image
# 可选的集成：同时跑两套 backbone，在 param 空间合并
python predict.py --ensemble dinov2_vits14,arcface --image dir_of_photos/
```

相关参数：`--aggregate {median,mean,trimmed}`、`--aggregate-space {embedding,param}`、`--ensemble <configs>`、`--save-per-image`。`infer.py` 同样支持目录与上述参数，并把合并后的结果写入角色卡（其中最具代表性的那张照片会作为卡片缩略图）。

将完整模型导出为 ONNX；如只需导出回归头，可添加 `--head-only`：

```bash
python export_onnx.py --config dinov2_vits14 --head <weights.pth> --out outputs/face2param.onnx
```

## 已知局限

1. **笑脸输入可能导致表情信息泄漏。** DINOv2 特征不仅包含身份信息，也会编码表情。因此，明显的笑容可能影响预测出的几何形状，通常表现为嘴部或下巴略有偏差。ArcFace 骨干网络对表情变化更不敏感，可以在很大程度上缓解这一问题。详见 [docs/expression-invariance.md](docs/expression-invariance.md)。
2. **不适合处理 2D 动漫图像。** 模型主要面向 HS2 角色截图和真实人脸照片。虽然也能将二维插画转换成看似合理的脸部参数，但最终效果往往不够自然。对于 2D 到 3D 的转换，关键通常不只是拟合面部几何形状，还需要还原发型、服装和配饰等角色风格，而这些并不在本项目的建模范围内。详见 [docs/stylization-and-anime-inputs.md](docs/stylization-and-anime-inputs.md)。
3. **仅预测脸型参数。** 模型只输出 54 维 `shapeValueFace` 参数和骨骼参数。头身比例、发型、肤色、睫毛和眉毛细节等信息均来自模板角色卡，仍需手动调整。

## 关于 HS_FACE 数据集

[HS_FACE](https://pan.baidu.com/s/1yPftN5rmtY5QDF7G2RjN4A?pwd=p8qd) 数据集包含约 14 万张游戏角色人脸图像，主要由以下三部分组成：

1. 直接从游戏中采集的角色人脸图像（对应 `images/`）；
2. 以第一部分图像为条件，通过 Stable Diffusion 生成的拟真人脸图像（对应 `aug_images/`）；
3. 第一部分游戏角色对应的脸部参数（由 `cards/` 生成 `labels.json`）。

## 许可证

本项目采用 MIT 许可证，并基于原项目 [ChasonJiang/Face2Parameter](https://github.com/ChasonJiang/Face2Parameter) 开发。
