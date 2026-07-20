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

- **几何 deform(纯 numpy)**:`src/hs2_mesh_deform.py`(`HeadRig(head_id)` / `_fk_world` / `build_mesh`,
  headId=2 → V=4439、skin 到 43 骨)、`src/hs2_mesh.py`(`card_to_mesh` / `fd_to_inputs` / `save_obj`)。吃
  **54 slider + 30 ABMX 骨**,产出顶点 + 三角面 + `.obj`。
  > **勘误(2026-07-19):** 之前写的 V=3254 是**碰撞网格**(`p_cf_head_NN_hit`)——旧提取器按"顶点数最大的
  > `o_head`"选,选中了碰撞体且是错的 head 类型。现在按 list 解析出的**渲染** prefab `p_cf_head_02` 选,
  > UV 才对得上。详见 [hs2-renderer-and-mesh.md](hs2-renderer-and-mesh.md) 第 0/1 条。
- **软渲染器(玩具)**:`scripts/hs2_render_mesh.py::render()` —— 纯 numpy z-buffer,着色仅
  `0.25+0.75·max(n·L,0)` 单向光 Lambert、逐面法线、**单通道灰度**。无贴图 / UV / 颜色 / 平滑法线 / AA。
- **比例度量 + 反解编辑器**:`src/face_metrics/`(landmark = 骨骼世界坐标;25 个尺度无关比例;Gauss-Newton
  把"比例目标"反解回 param 写卡)。
- **资产提取(关键基建)**:`scripts/hs2_extract_head.py` 用 **UnityPy** 从游戏 bundle 拔 `o_head` 的
  verts/faces/skin 权重/bindpose → `data/hs2_head/*.npz`(**gitignore,需装游戏跑**)。
  → 这就是"trace 游戏"的既有基建:**加贴图 / 材质 = 给它扩几行**(拔 `m_UV`、texture、材质参数)。

## 面部资产实测清单(直接扫 fo_head_00 + st_eyebrow,2026-07-19)

不再靠猜。整张脸 = 以下 skinned 子网格(都绑在同一套 head 骨架上),每个 = mesh + UV + material +
textures + shader:

| 部件 | mesh | material | shader | 关键贴图 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 皮肤 | `o_head` | cf_m_skin_head_*_create | `AIT/Skin True Face` (+ `Skin Translucency`) | detail✅ / mask✅;`_MainTex`+occlusion **外部** | 几何+UV✅;皮肤 diffuse+shader ❌ |
| 眼球 L/R | `o_eyebase_L/R` | c_m_eye(_01) | `AIT/Eye Translucency` | 巩膜 c_t_eye_white_01;**虹膜/瞳孔 外部(卡选)**;normal c_t_eye_n;overlay c_t_eye_o_01 | ❌(视觉差最大) |
| 睫毛 | `o_eyelashes` | c_m_eyelashes | `AIT/eyelashes` | `_MainTex` 外部(卡选) | ❌(需 alpha) |
| 泪膜 | `o_namida` | c_m_eye_namida | `AIT/main namida` | namida_1_t | ❌(透明,可后置) |
| 眼影 | `o_eyeshadow` | c_m_eyekage(_cf2) | `AIT/main eyeshadow lambert` | c_t_eyeshadow_00/01 | ❌(妆容 overlay,卡选色) |
| 牙齿 | `o_tooth` | c_m_tooth | (matcap) | c_tooth_t/n + occlusion | ❌(仅张嘴可见) |
| 舌头 | `o_tang` | c_m_tang | (matcap) | c_tang_t/n/o | ❌(仅张嘴可见) |
| **眉毛** | `st_eyebrow_00.unity3d`(**独立 bundle**) | ? | ? | 卡选眉型 | ❌ 需从该 bundle 提取 |

跨部件的实现工作(即"还得 implement 什么"):
1. **提取器泛化** — 一次拔 fo_head_00 里全部子网格(verts/faces/uv/bone_idx/bone_w/bindpose + 各自
   material + 馆内贴图),外加 st_eyebrow_00。
2. **外部/卡选贴图** — 皮肤 `_MainTex`、虹膜(fid2-4)、睫毛(fid5)、occlusion 等是**卡驱动**、在别的
   bundle。需 ①从 FaceData 读卡的选择(skinId / 眼睛 id / 眉型…),②跨 bundle 解析这些贴图 = "运行时合成"。
3. **子网格形变** — 每个子网格绑同一套 head 骨架 → 复用现有 FK/LBS,只需各自 skin 权重/bindpose(眼球还有
   注视旋转,中性可略)。
4. **渲染器升级** — 多网格 + **alpha 混合 + 绘制顺序**(睫毛/泪膜/眼影是半透明 overlay);逐 material 着色。
5. **Shader** — 复刻 `AIT/Skin True Face`(次表面)、`AIT/Eye Translucency` 等(保真度)。名字已知 → 可
   trace(dnSpy / MaterialEditor / 社区实现)。

**已完成**:几何 deform、UV、in-bundle 头部贴图、**nvdiffrast 可导渲染器 + 平滑法线 + 近似皮肤**、game 对比。

**2026-07-19 新增(Phase 0,见 `.claude/plans/read-docs-offline-renderer-research-md-*.md`):**
1. **离线 ChaListControl**(`src/hs2_assets.py`)—— 合并全部 `list/characustom/*.unity3d`,
   `category → id → row`。上表"外部/卡选贴图"这一项**已解决**:皮肤/虹膜/睫毛/眉/眼影/唇彩全部按卡上的 id
   解析到具体 bundle + asset。
2. **card-driven 提取器**(`scripts/hs2_extract_head.py --card <卡>`)—— 按 list 选**渲染** prefab(不是
   碰撞体)、提 normals/tangents/uv/uv1/colors、全部子网格(眼球/睫毛/泪膜/眼影/牙/舌)+ 各自 material
   与 shader 名、卡选贴图,并写 `data/hs2_head/cards/<卡>.json` 清单。
3. 渲染器已接上**该卡真实的皮肤 diffuse**(`cf_head_02_00_t`),UV 对位正确(唇/耳/鼻翼各就各位)。

## 判断 & 建议

1. **先 trace,别先造**(合 [prefer-tracing-and-principled-methods] 原则):KK/HS2 的**资产提取 +
   截图/渲染 mod + MaterialEditor** 生态已把一大半做完;动手前先摸清社区有多少 shader/asset 文档,
   省掉最痛的 shader 重实现。
2. 像素忠实 → **策略 1**;要可导直接优化 → **策略 2**。
3. **大概率最优 = 混合**:策略 2 近似可导渲染器当**优化内循环**(快、可导),策略 1 忠实游戏渲染当
   **验证器 + 代理数据源**。"便宜可导代理上优化、真渲染器上验证"是标准打法。
4. **诚实提醒:** 只要 recipe 没移植完整,残余渲染差 + 之前的 OOD 会**叠加**。移植越全渲染差越小,但
   "优化结果人眼真更好看"仍取决于 ①reward 在游戏脸域准不准 ②移植保真度。所以策略 1 当验证器是必须的。

## 开放问题(Q1/Q4/Q7 已答)

1. ~~现有白模吐了什么~~ → **已答**:纯几何 `params→(verts,faces)` + 玩具灰度渲染器,零 appearance。
2. 策略 1 的自动化:能否无头运行?还是必须开游戏后台截图?单张渲染耗时?
   → 计划:给 `HS2_McpBridge` 加 camera/screenshot 端点(游戏须运行 + Maker 打开)。
3. 渲染器选型:~~nvdiffrast vs PyTorch3D vs Mitsuba3~~ → **定 nvdiffrast**(已在用)。
   保真度门槛待 Phase 4 用真渲染标定后定量。
4. shader → **已答(2026-07-19)**:`AIT/*` 全是 **Amplify Shader Editor 生成的 Unity Standard *surface*
   shader**(`m_CustomEditorName = ASEMaterialInspector`),forward-only(ASE 的 translucency 端口强制
   `exclude_path:deferred`),光照 = stock `BRDF1_Unity_PBS` + 加性 translucency
   (Barré-Brisebois/DICE 快速 SSS)。**社区没有 HS2 `AIT/Skin*` 的开源反编译**(Koikatsu 的
   `Shader Forge/*` 是另一套 toon,不通用;Hanmen 的 skin/eye 是闭源付费)→ 必须自己反编译(已获批准)。
   另:皮肤 `_MainTex` 是运行时 `Graphics.Blit` 合成的,合成 shader = `chara/mm_base.unity3d` 里的
   `Create/skin color`(唇/腮红/眼影/痣/纹身各一层)+ `Create/skin detail`。
5. 保真度指标:怎么量化"近似渲染 vs 真游戏渲染"的差距(像素?ArcFace 身份?beauty 分一致性?)。
   → 计划三者都报:掩膜内 L1/PSNR + ArcFace 余弦 + beauty 分差。
6. 几何可导化:把 numpy deform 移植到 torch 的成本?还是有限差分 Jacobian 就够(小规模优化)?
   → 决定移植(Phase 1),因为要给 amortized head 反传,不只是小规模优化。
7. ~~提取器扩展~~ → **已答**:一次全拔。见上面"2026-07-19 新增"。
