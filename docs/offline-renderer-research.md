# Offline renderer research — 逆向游戏渲染 recipe，做离线（理想可导）人脸渲染

> **状态:调研中 / 未定案** (dumped 2026-07-19)。独立议题——专门研究"写一个 offline 渲染器"这一块。
> 与 beauty 优化管线相关但独立;那条线见 [beauty-guided-generation.md](beauty-guided-generation.md)。

## 动机

beauty 优化的三条路线(代理 reward / 黑盒 RL / 图空间美化)全都卡在同一处:**没有一个能吃
`param` → 吐"游戏脸图"的渲染器**。现在只有游戏引擎自己在渲染,是个不可导黑盒。

如果能**逆向出游戏的渲染 recipe、做成 offline 渲染器**,尤其是**可导**的,就能:
- 给代理 `S(param)→beauty` 白嫖训练数据;
- 直接把"渲染"嵌进 pipeline;
- **可导版**:`param → 渲染 → 图 → beauty` 全程可导 → 直接对 param 梯度优化,溶解核心难点。

**现状(已核实 2026-07-19,详见下方"现有资产"):** 白模比想象成熟——已是完整 `params → (verts, faces)`
几何管线,甚至带个玩具灰度软渲染器,但 **appearance 为零**:无贴图 / UV / 颜色 / 材质 / 平滑法线,只有
单光灰度。逆向细节另见 [hs2-renderer-and-mesh.md](hs2-renderer-and-mesh.md)。

## 两种策略(其实是一条谱系:跑真引擎 ↔ 把 recipe 移植进可导框架)

### 策略 1:游戏引擎当渲染器(faithful,不可导)
Illusion 这几个(AI少女 / KKS / HS2)都是 **Unity + BepInEx 可插件化**,KK/HS2 逆向社区非常成熟。
写插件/自动化:喂 card data → 无头/后台场景加载角色 → 固定相机+打光 → 截脸。本质是把游戏自己做的事
**自动化+离线化**(项目已在调 `HS2ABMX.exe` 写回卡,有先例)。
- ✅ 像素级忠实,不用重写 shader。
- ❌ 重、慢(秒级/张)、需 Unity 运行时、**不可导**。
- 用途:代理 S 的数据源 / 路线 B 的 reward / ground-truth 验证器。

### 策略 2:参考/移植 Illusion 的渲染 recipe,做风格贴近的(可导)渲染器 （PREFERRED）
**不是"凭空自研"**——而是把游戏那套 shader / 材质 / 光照 recipe **逆向出来、忠实移植**进可导光栅化器
(nvdiffrast / PyTorch3D / Mitsuba3)。这和现有 **mesh constructor 是同一套 pattern**:白模已经把
"解析 game data → 忠实重建**几何**"走通了,策略 2 只是把同样的 tracing 从几何**延伸到 shading /
material / lighting**。保真度是"移植完整度"的函数,奔着游戏风格去,而非天生近似。

**为什么这个 case 可导(已核实几何管线,修正之前的说法):**
- `param → 几何`:**注意——HS2 的 54 个 `shapeValueFace` 其实是 bone-driven,不是 blendshape morph**
  (头上 58 个 blendshape 全是表情、恒为 0)。真实链路是
  `shapeValueFace → 关键帧插值 → 硬编码 Update 方程 → ABMX → FK 世界矩阵 → LBS 蒙皮`。
  这是"插值 + 光滑变换 + 线性蒙皮"的复合,**处处可导**(不是因为 blendshape 线性,而是整条复合光滑)。
- `几何 → 图`:可导光栅化器把梯度从像素倒回顶点 / 贴图 / 光照。
- `图 → beauty`:reward 本就可导。
- ⇒ **`param → 可导渲染 → 图 → beauty` 全程可导。** 可直接对 param 梯度上升提 beauty,或训 amortized head。
- **落地要点:** 现有 deform 是**纯 numpy(无 autograd)**;要拿解析梯度需把 deform **移植到 torch**
  (纯算术 + LBS,可行但要照搬关键帧采样和 Update 方程)。或先用**有限差分 Jacobian**——比例编辑器
  (`src/face_metrics/edit.py`)已对 `param→ratios` 这么干,同法可用于小规模 `param→render→beauty` 优化。

- ✅ 可导、快(GPU)、干净;而且是 trace-first 的自然延伸(照 recipe 移植,不凭空造)。
- ⚠️ **保真度 = 移植完整度的函数**(不是天生近似)。残余 gap 只来自:①还没移植 / 难移植的 shader 特性
  (Illusion 皮肤次表面、ramp、动漫眼),②**可导光栅化器本身**的近似(可见性 / 抗锯齿的梯度)。靠增量
  补齐 + 用策略 1 做 ground-truth 验证来收敛,而不是接受一个笼统的"近似脸"。

## 现有资产(已核实 2026-07-19)

白模已是完整 `params → (verts, faces)` 几何管线,连玩具渲染器都有。逆向细节见
[hs2-renderer-and-mesh.md](hs2-renderer-and-mesh.md)。

- **几何 deform(纯 numpy)**:`src/hs2_mesh_deform.py`(`HeadRig` / `_fk_world` / `build_mesh`,V=3254、
  skin 到 43 骨)、`src/hs2_mesh.py`(`card_to_mesh` / `fd_to_inputs` / `save_obj`)。吃 **54 slider + 30
  ABMX 骨**,产出顶点 + 三角面 + `.obj`。
- **软渲染器(玩具)**:`scripts/hs2_render_mesh.py::render()` —— 纯 numpy z-buffer,着色仅
  `0.25+0.75·max(n·L,0)` 单向光 Lambert、逐面法线、**单通道灰度**。无贴图 / UV / 颜色 / 平滑法线 / AA。
- **比例度量 + 反解编辑器**:`src/face_metrics/`(landmark = 骨骼世界坐标;25 个尺度无关比例;Gauss-Newton
  把"比例目标"反解回 param 写卡)。
- **资产提取(关键基建)**:`scripts/hs2_extract_head.py` 用 **UnityPy** 从游戏 bundle 拔 `o_head` 的
  verts/faces/skin 权重/bindpose → `data/hs2_head/*.npz`(**gitignore,需装游戏跑**)。
  → 这就是"trace 游戏"的既有基建:**加贴图 / 材质 = 给它扩几行**(拔 `m_UV`、texture、材质参数)。

## 缺的"调料"在哪 & 难度(已核实)

| 调料 | 现状 / 在哪 | 工作量 |
| --- | --- | --- |
| 几何(bone + LBS) | **已有**(纯 numpy;可导,但要 torch 移植才有解析梯度) | ✅ |
| **UV 坐标** | 提取器没拔 `m_UV`;`o_head_mesh.npz` 只有 verts/faces/权重 | 小(给 `hs2_extract_head.py` 加 `m_UV`) |
| 皮肤/眼/眉 贴图 | 未提取;游戏 bundle 里有 | 中(扩提取器拔 texture) |
| 肤色/腮红/唇/眼影 颜色 | card 有字段,但**没接进 mesh**(`Card.py` 里大多注释掉) | 小-中(读卡颜色 → 喂材质) |
| 顶点法线 + 真实着色 | 只有逐面 flat 法线、单光、灰度;无平滑法线 / Phong / PBR / 次表面 | 中 |
| **shader(皮肤/眼睛)** | 完全没有;Illusion 自定义 shader | **难**(移植;MaterialEditor 有参数文档可 trace) |
| 光照 | 单个硬编码方向光 | 易(补打光台:几盏灯 + 环境) |
| 眼/眉/睫/齿/发 子网格 | 只提取了 `o_head` 一个头网 | 中(逐个提取) |
| 彩色 / 真渲染器 | 玩具灰度光栅器;仓库无 3D 库依赖 | 中(换 nvdiffrast/PyTorch3D → 同时拿到**可导**) |

## 判断 & 建议

1. **先 trace,别先造**(合 [prefer-tracing-and-principled-methods] 原则):KK/HS2 的**资产提取 +
   截图/渲染 mod + MaterialEditor** 生态已把一大半做完;动手前先摸清社区有多少 shader/asset 文档,
   省掉最痛的 shader 重实现。
2. 像素忠实 → **策略 1**;要可导直接优化 → **策略 2**。
3. **大概率最优 = 混合**:策略 2 近似可导渲染器当**优化内循环**(快、可导),策略 1 忠实游戏渲染当
   **验证器 + 代理数据源**。"便宜可导代理上优化、真渲染器上验证"是标准打法。
4. **诚实提醒:** 只要 recipe 没移植完整,残余渲染差 + 之前的 OOD 会**叠加**。移植越全渲染差越小,但
   "优化结果人眼真更好看"仍取决于 ①reward 在游戏脸域准不准 ②移植保真度。所以策略 1 当验证器是必须的。

## 开放问题(Q1 已答,见"现有资产")

1. ~~现有白模吐了什么~~ → **已答**:纯几何 `params→(verts,faces)` + 玩具灰度渲染器,零 appearance。
2. 策略 1 的自动化:能否无头运行?还是必须开游戏后台截图?单张渲染耗时?
3. 渲染器选型:nvdiffrast vs PyTorch3D vs Mitsuba3;先做到什么保真度就够喂 beauty reward?(换它同时拿到可导)
4. shader:社区(MaterialEditor 等)已有多少可复用的皮肤/眼睛实现或参数文档?值不值得重写,还是近似 PBR 够。
5. 保真度指标:怎么量化"近似渲染 vs 真游戏渲染"的差距(像素?ArcFace 身份?beauty 分一致性?)。
6. 几何可导化:把 numpy deform 移植到 torch 的成本?还是有限差分 Jacobian 就够(小规模优化)?
7. 提取器扩展:`hs2_extract_head.py` 加 `m_UV` + texture + 材质参数 + 眼/眉/齿子网格,一次能拔多少?
