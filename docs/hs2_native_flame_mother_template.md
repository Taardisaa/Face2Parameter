# 原生 HS2 头制作 FLAME 母版：可行性探索

2026-10-08。用户提出：先把完整原生头变形成接近 FLAME 基模的头，
以后直接用 MICA 系数驱动，避免每个身份重复导入 FLAME 网格及适配资产。
下文保留最初的源码／现有资产研究；最新开工进展见末尾。当前已生成
临时头壳配准候选，已将全部部件／表情帧及中性绑定写入独立开发 bundle。
完整曲面检查不通过，尚无可交付母版或迁移形状基，游戏未改动。

## 结论

这是一条值得优先制作原型的路线。输出仍用原生顶点顺序、三角形连接、
UV、口内结构及配套部件；FLAME 提供参考外形与身份变化数据。
母版制作时完成一次部位对应及形状数据转换，人物生成时只做系数加权。
不必先训练 MLP，也不必每张照片重新求网格对应。

但“外形相似”不能让两套顶点自动一一对应。不能把原始 FLAME 数组直接
加到原生头上，也不能把前面若干行裁出来当原生顶点的位移。
一次性部位对应可以封装在母版制作中，不能声称已经消除了这个问题。

## 本机实现依据

读取 `outputs/chenger_mica_review_20261007/raw/manifest.json` 的原始 MICA
状态和 `tools/model_bridge/artifact.py`，并核对 manifest 指向的官方
`models/flame.py` / `models/lbs.py`：

- FLAME 参考头有 5023 个顶点；保存的形状基是 `[5023, 3, 400]`。
  本次 MICA 输出使用前 300 个身份分量；后 100 个是表达分量。
- `lbs` 先执行 `v_template + blend_shapes(betas, shapedirs)`，随后才是
  关节回归、姿态修正和蒙皮。形状基的每一行对应指定的 FLAME 顶点，
  它不是按空间位置任意查询的变形函数。
- 当前导出的 MICA 默认眼／颈姿态均为零。此处讨论的是中性身份形状，
  不将完整关节运动或游戏表情等同于线性身份加权。

读取 `outputs/oral_integration_20261008/source_v2/o_head_donor.npz` 与
同目录 `audit.json`：原生女性 02 号头有 4439 个顶点、8516 个三角形，
UV／法线／切线按同一原生顶点顺序存储；原生头使用 43 个蒙皮骨骼，
而这份 FLAME 使用五关节。顶点数量、连接与骨架均不是同一数据布局。
供体为安装版 `abdata/chara/38/fo_head_38.unity3d` 的 `p_cf_head_02`，
SHA256 `66ba2ef7f95fa4d79ad87b3d1e11e90e8f1f78ed79e0dc7cb1e4e2969f100edf`。

安装版 `ChaControl.ChangeHeadAsync` 复制同名骨骼变换、重新绑定蒙皮，
随后连接原生 FaceBlendShape 控制器。没有读取 FLAME 系数的分支。
此来源见 HS2Mod 的 `tools/parameter_audit/source_model_bridge_20261007/`；
程序集与完整驱动链已记录于 [口腔研究](hs2_oral_integration.md)。

## 最少的制作步骤

1. **从完整原生供体开始。** 复制原生 02 号的真正头 prefab 和完整网格，
   包括原来的牙齿、舌头、泪液、眼部及表情表。当前私有 0.1.6 导入包
   已清空多种部件及表情数据，不能作为“完整原生母版”的起点。
2. **一次性改造参考头。** 以真正零身份系数的 FLAME 参考头为目标，
   将原生外部皮肤曲面变形过去。记录坐标转换，分别约束上下眼睑、内外
   唇缘、鼻翼、耳根；口内表面不向外部脸皮做最近点吸附。
   保持原生顶点编号、面连接及 UV，不删面，不将 UV 缝的重复顶点焊掉。
   重复顶点的几何运动必须保持一致，但其 UV／切线／表达数据各自保留。
3. **保留实际身体接口。** 原生颈圈位置及蒙皮约束是硬约束，参考颈部
   只在连接带内适配。接口法线以实际所用身体为依据；目前供体 head2 与
   BP 身体不能直接假定一致。脸部目标不能为了压向接口而整体高度压缩。
4. **将身份变化存入母版。** 给对应到同一外部部位的原生顶点保存身份
   位移，形成原生顶点顺序下的 300 组形状数据。对应固定后，可在制作时
   将坐标转换与采样合并进去；运行时执行“母版位置＋系数加权位移”。
   原始 MICA 系数不重定义，也不拟合成 HS2 滑杆。
5. **配套数据一并适配。** 原生 UV 无需重新画一套地图，但拉伸后的贴图
   仍可能失真；法线／切线要对应最终几何。眼球、口内部件、骨骼枢轴和
   原有表情帧要适配母版及身份。保留数组并不证明动画与外观自然。

若母版的每个外部顶点正好落在其对应的 FLAME 表面点上，同样的系数
会使这些顶点跟随相应表面点。不同三角形连接仍可能给出不同的连续曲面，
不能声称完整曲面严格相同；口内、颈部和配套部件也没有自动的 FLAME
对应。保形误差应明确记录，不能用保留原生脸型偏差冒充目标身份。

## 哪种现成方法能用于制作母版

带部位锚点的非刚性网格配准与本需求相符：移动原生顶点来贴近参考
曲面，不更换其连接。FLAME 作者的 [TemplateFitting](https://github.com/TimoBolkart/TemplateFitting)
提供头部模板拟合实现，但依赖较老的 ITK／Clapack／ANN；本次只检查
其发布说明，没有在本机部署或运行。

[Trimesh 的 NRICP 文档](https://github.com/mikedh/trimesh/blob/main/docs/content/nricp.md)
及 [registration.py](https://github.com/mikedh/trimesh/blob/main/trimesh/registration.py)
提供 Amberg／Sumner 配准实现，支持顶点或表面锚点，返回原生顺序下的新
位置。它们的锚点是加权项，不等于本任务的颈圈硬锁定；也未提供本任务
所需的上下唇分区、UV 缝副本运动约束和自动语义认证。不能直接调用
默认全表面最近点配准后就宣布完成。本机 Face2Parameter 的 Windows
`.venv` 当前没有安装 `trimesh`；本次没有安装新依赖。

[Sumner／Popović 的形变迁移](https://people.csail.mit.edu/sumner/research/deftransfer/)
证明不同拓扑间可以通过一次对应迁移形变，不过它仍需要对应与求解。
对当前中性线性身份形状，先制作母版并存原生形状基，比先引入完整
姿态梯度迁移或训练 MLP 更直接。这里是路线选择，不是已实现性能结论。

## 下一个具体原型及停止条件

先交付一份 **原生 02 号拓扑的 FLAME 零身份母版候选**，附完整外部
曲面和保留部件；仍不改变目前已接受的程儿游戏资产。
实现前需要从供体拓扑／既有区域资料确定嘴、眼、耳和颈圈的边界。
现有二维底色映射和宽泛 UV 矩形不能代替这些三维对应。

候选须明确区分原始 bind 网格和游戏默认闭嘴表面。默认关闭图案包含
非零头部形变；不能将它烘焙进基模又重复应用同一帧，也不能把全零
表情权重当原生闭嘴参考。

母版候选阶段用原始供体逐项核对顶点／索引／UV／部件／表情表未丢失，
实际接口未移动，上下唇眼睑没错配，没有翻面／折叠。只有这一份母版
路线成立，才制作身份形状基并接入程儿系数；完整表情和材质仍需完成。
这是母版路线的第一道门槛，不把它重新定义为整个原目标已经完成。

## 本次状态

完成源码与现有输入的可行性研究，尚无改造后的原生母版或身份形状基。
没有调用游戏采集、Computer Use、参数扫描或训练。当前游戏包、人物卡
和原有开放目标保持。本次对应的 HS2Mod 入口为
[母版探索](../../HS2Mod/docs/hs2_native_flame_mother_template.md)。

## 用户授权开工后的输入准备

用户已要求开始，并设置第一份母版候选为当前原生目标，读回 active。
旧头颈接口目标保存为暂停状态，未声称完成。

已实现 `tools/native_head/mother_template_inputs.py`：校验安装程序集、
反编译来源、供体 bundle 和原始 MICA decoder，复制完整供体 bundle，
导出实际 prefab 下全部八个 renderer 的顶点／面／UV／颜色／法线／
切线／权重／bindpose、完整稀疏表情帧及原始 renderer／controller 数据。
保存后逐项读取数组核对；不重新生成或清空原生数据。bundle 中的外部
依赖引用保持，但这一步不声称已导出全部外部材质／纹理依赖闭包。

逻辑图只将 bind 位置和完整有序蒙皮字节一致的 UV 副本视为同一位置；
原始顶点编号和网格不焊接、不删面。由闭合边界与骨骼支持确定原生
FaceRoot 全权重颈圈及左右眼开口。由独立 index component 中的真实
MouthCavity 完全权重点定位嘴内组件，保留整个组件而不是按权重切面。
嘴内组件边界不自动等于可见唇缝；上下眼睑、内外唇缘、耳根的完整
语义对应仍待制作。不能用这一输入步骤声称已经得到母版。

参考头保存 decoder 实际 `v_template`，附完整状态和原始表面 landmark
embedding；不用 OBJ 的模板位置，不使用程儿预测头作为零身份参考。
原生 bind 网格仍未烘焙默认闭嘴帧，原始控制表及对应帧全部保存。

重现命令（Face2Parameter 根目录）：

```powershell
.venv/Scripts/python.exe -m tools.native_head.mother_template_inputs `
  --reference outputs/model_bridge_20261007/oral_asset_audit_v2.json `
  --source-manifest outputs/chenger_mica_review_20261007/raw/manifest.json `
  --out outputs/native_mother_template_20261008/inputs_fresh
```

输出 `outputs/native_mother_template_20261008/inputs_v2/` 已完成输入核对；
早先 v1 的外部依赖字段措辞不准确，原结果保留，v2 明确仅保持引用、
没有导出依赖闭包。重复运行必须使用新目录。收据、完整供体和模型数据均留在 ignored
输出中。当前游戏未改动，母版配准、部件适配、实际 BP 接口匹配、身份
基迁移尚未实现，完整目标保持 active。

## 原生参考状态与头壳制作工具：2026-10-08 后续进展

输入准备后已实现两个可复用步骤，当前母版目标仍 active。

`mother_default_pose.py` 从原始 prefab 的三个控制器和全部稀疏表情帧
恢复一份明确的静态参考：pattern 0、过渡结束、blink open rate 1、
中性视线修正、voice 0。按照安装版 `FaceBlendShape.OnLateUpdate`
的眉毛→眼睛→嘴顺序及 `FBSBase.CalculateBlendShape` 的字典写入求值；
后控制器覆盖同一通道，不能将它们累加。浮点 Lerp、乘 100 和转 int
保持 float32；否则供体的眼睛 OpenMax 会被错误截成 89 而不是 90。
新补充的 Eyes/Eyebrow 控制类直接从同一安装版 IL.dll 反编译。

所有八个部件都导出参考位置以及原有法线／切线与帧增量之和，另存
`default_reference_v1/`；原始 bind 数组未改。当前供体通道均为单帧
且权重 100，若这份真实资产合同变化则拒绝而不推测其他插值分支。
这不是运行时蒙皮采集，不模拟眨眼／视线轨迹，也不把参考状态烘焙成
bind 网格或称全零表情为默认状态。

头壳制作工具 `mother_shell_candidate.py` 已支持：

- 保持两份资产的 X 横向／Y 高度／Z 前后坐标方向；用眼角确定统一
  大小与位置，不再用原生眼角的斜率把整个 FLAME 头旋转成低头姿态。
- 只在逻辑图合并位置／蒙皮相同的 UV 副本，输出仍用原始顶点编号、
  面连接和 UV；颈圈所有真实副本硬固定，逐一校验未移动。
- 物理眼开口按原始闭环区分上下路径；鼻和外唇二维候选注记落在实际
  原生三角形上，使用精确重心约束，不用最近顶点替代。
- [ARAP 局部旋转／全局位置求解](https://libigl.github.io/tutorial/#as-rigid-as-possible)
  保留局部结构，替代早期单纯位移平滑。这里是新资产制作算法；使用
  明确的正逆边长权重，不冒充 libigl 默认余切权重或 HS2 游戏计算。
  嘴内不吸附到外部皮肤；嘴部骨骼支持区由外唇约束和结构能量带动。
- 原始 FLAME 耳朵 mask 与原生耳骨主导区域分别对应，按实际坐标侧
  核对，避免耳朵贴到脸颊。该支持区不是已经认证的耳根解剖边界。

`mother_candidate_review.py` 输出完整头壳正交图和全高度中央剖面，
包括嘴内交线，保持共同尺度；没有隐藏面来美化结果。图片仅是实际
网格的科学示意，不是游戏截图，也未包括已适配的完整部件。

早期结果全部保留：`shell_v1` 的自由旋转配准错误；`shell_v2` 固定
方向后仍使嘴内鼓起；`shell_v3/v4` 通过 ARAP 改善完整头形和嘴内，
但唇、眼睑仍有可见折皱，尚不能作为交付。几何收据均明确
`deliverable=false`、`installed=false`、`game_mutated=false`。
`shell_v5` 加入完整原生闭嘴／睁眼参考，并让整个嘴部骨骼支持区由
锚点及结构能量带动，唇部的明显折叠得到改善。完整图保存于
`shell_review_v5/`；它仍是已变形的参考状态，不是可直接安装的 bind
资产，不能重复烘焙再施加原生表情。眼睑、耳根与配套部件仍须处理。
鼻唇注记、耳根／内唇语义、表面自交、完整部件和表情适配、最终法线／
切线、实际 BP 接口仍有待完成；不降低母版的原定验收要求。

重现（输出须用新目录）：

```powershell
.venv/Scripts/python.exe -m tools.native_head.mother_default_pose `
  --inputs outputs/native_mother_template_20261008/inputs_v2 `
  --out outputs/native_mother_template_20261008/default_reference_fresh
.venv/Scripts/python.exe -m tools.native_head.mother_shell_candidate `
  --inputs outputs/native_mother_template_20261008/inputs_v2 `
  --default-reference outputs/native_mother_template_20261008/default_reference_fresh `
  --out outputs/native_mother_template_20261008/shell_fresh
.venv/Scripts/python.exe -m tools.native_head.mother_candidate_review `
  --candidate outputs/native_mother_template_20261008/shell_fresh `
  --out outputs/native_mother_template_20261008/shell_review_fresh
.venv/Scripts/python.exe -m unittest tools.native_head.test_mother_authoring
```

五项解析检查已通过：刚体运动应是 ARAP 的驻点、UV 重心坐标及歧义
拒绝、中央剖面不漏同平面边、原生 float32 权重和共享通道、保留仅有
法线增量的表情帧。它们核对制作工具的数学和来源合同，不代替完整
母版验收。工具依赖的素材与产生的候选保持在 ignored 目录，未安装
任何新资产、未覆盖程儿人物卡或原有包。

## 全部部件、绑定及原资产写回：后续里程碑

完成可重现的部件数据适配／原资产写回步骤，但 **shell_v5 的几何不通过**，
不能安装为正常人物或宣称母版交付。新产物为独立开发候选，不覆盖程儿。

`mother_surface_warp.py` 用源／目标三角形局部边框构造梯度，法向列长度
由三角形面积确定；按真实最近三角形重心位置及法向偏移迁移配套部件。
这是新资产的制作算法，不是对 HS2 驱动的近似替代；正局部行列式不证明
体积映射全局一一对应。物理 UV 副本共享梯度，各自的原始着色属性保留。
法线使用逆转置，切线投影到新法线平面，原始切线手性保留。

`mother_component_adaptation.py` 适配全部八部件及全部稀疏表情行：
眼球及其阴影层使用原生眼枢轴到真实 FLAME 眼关节的相似变换；牙齿和
舌头仅查原生嘴内组件，睫毛／泪液查原头表面。保留顶点、连接、UV、
颜色、权重、通道名、帧表和稀疏索引。先迁移原生默认帧，再从设计参考
扣除这些默认增量制作 bind 数据；再次施加控制器默认权重还原到设计面，
不把闭嘴帧烘焙两次。这一步不认证全表情轨迹、部件碰撞或滑杆范围。

`mother_reference_bindings.py` 校验安装版源码／缓存来源、实际 ShapeAnime
和 customhead 表，经恢复的实际 Update/FK 在 59 个面部值 0.5、无 ABMX、
孤立头部祖先参考下重建 bindpose。原 HeadRig 缓存只有蒙皮骨骼与祖先，
先核对这份子集，再接入原始 prefab 完整层级；renderer 和饰品节点不丢。
只适配没有 ShapeHeadInfoFemale 位置写入的两个 `cf_J_look_L/R` 注视父节点，
并按眼球尺度同步子级距离。原生表情控制器和驱动表保持。所有蒙皮矩阵
在该参考下为单位，不能据此称全部面部枢轴、注视或 ABMX 已兼容。

`mother_bundle_candidate.py` 从完整供体原字节包改造，未使用会清空表情／
重新生成 UV 和蒙皮通道的旧 `write_mesh`。仅在原顶点流布局内改位置／
法线／切线，保留其他所有顶点字节与索引；更新完整帧、bindpose 和包围盒。
控制器／renderer 原字节、材质和纹理不改。CAB／资源 URI 与 prefab 使用
独立名称；资源 payload 保持字节一致。序列化后重新读回所有数组／帧和
所有改写字段，未触碰的对象原字节也逐一核对。该 bundle 尚未注册列表
或皮肤项，**原 UV 分布 metric 尚未按新几何更新**，需按引擎路径处理。
每个单帧包围盒不等于全部骨骼／组合表情的动态覆盖；原加载器仍有实际
AssignedWeightsAndSetBounds 路径，需在最终集成核对。

产物（均 ignored）：`components_v1/`、`bindings_v1/`、`bundle_v2/`。
`bundle_v1/` 是加强逐字段序列化核对前的版本，仍保留。三维说明图在
`component_review_v1/`：头壳加原生眼球共同比例；另将八部件全部线框展开，
每项单独缩放且不隐藏背面。透明层没有用假材质装成完整游戏效果。

完整头壳检查 `mother_surface_quality.py` 调用既有三角形几何谓词，新增
显式模式检查共顶点的三角形，以免邻接关系掩盖正面积折叠。真实穿插、
共面重叠与接触分开保存，保留源资产已有交叉和候选新增交叉；结果为
`surface_quality_v5/receipt.json` 的 `no_new_crossings=false`。图中眼睑等
折皱并非只有显示阴影；当前少量轮廓约束和区域最近点配准未防止曲面
贴错侧／折叠。下一步先修这份几何对应，不能将原始交叉作为豁免理由，
不能靠改法线、藏部件或禁用表情掩盖它。

重现后续步骤（Face2Parameter 根目录，新输出名）：

```powershell
.venv/Scripts/python.exe -m tools.native_head.mother_component_adaptation `
  --inputs outputs/native_mother_template_20261008/inputs_v2 `
  --candidate outputs/native_mother_template_20261008/shell_v5 `
  --out outputs/native_mother_template_20261008/components_fresh
.venv/Scripts/python.exe -m tools.native_head.mother_reference_bindings `
  --inputs outputs/native_mother_template_20261008/inputs_v2 `
  --components outputs/native_mother_template_20261008/components_fresh `
  --out outputs/native_mother_template_20261008/bindings_fresh
.venv/Scripts/python.exe -m tools.native_head.mother_bundle_candidate `
  --inputs outputs/native_mother_template_20261008/inputs_v2 `
  --bindings outputs/native_mother_template_20261008/bindings_fresh `
  --out outputs/native_mother_template_20261008/bundle_fresh
.venv/Scripts/python.exe -m tools.native_head.mother_component_review `
  --bindings outputs/native_mother_template_20261008/bindings_fresh `
  --out outputs/native_mother_template_20261008/component_review_fresh
.venv/Scripts/python.exe -m tools.native_head.mother_surface_quality `
  --candidate outputs/native_mother_template_20261008/shell_v5 `
  --out outputs/native_mother_template_20261008/surface_quality_fresh
```

八项母版数学／数据检查及十九项几何谓词检查通过；原包写回也通过。
这些是工具和来源数据的逻辑核对，**几何母版本身未通过**。实际 BP
接口受光、完整语义、UV metric、所有部件间相交、独立注册和有限原生
验收仍在原目标内；完整表情／滑杆／ABMX 的范围认证另留后续里程碑。
全部工作未调用游戏、Computer Use、参数扫描、训练或增加新依赖。

## 保留整体头形的局部修复（2026-10-08）

用户明确要求：整体结构已经接近，只修小错误，不再整头重拟合。
当前固定输入为 `shell_v6`；此前整头 RBF 的 `kernel_v1` 未通过几何／
Jacobian 检查，源码只归档，未安装，不纳入后续路径。v6 的同侧法线
最近点限制也未解决薄曲面的全部折叠，不能作为成功的对应证明。

局部制作工具 `mother_local_repair.py` 从已记录的新增穿插面及两圈拓扑
邻域建立修复区域，合并实际 UV 副本；区域外所有顶点、真实颈圈及两侧
眼睛切口点精确固定。耳／眼等区域使用原生默认参考的局部细节和固定
边界位移；上下唇使用共同的原生细节摆放，避免两片很近的曲面分别移动。
这是新资产的制作算法，不是游戏形变逻辑的替代或完整解剖对应。

`mother_local_contacts.py` 只在上述范围内解决剩余交叉。源三角形提供
初始分离方向；每次移动后，受影响三角形与**整个头**重新分类，未变化
的面沿用经校验的结果，不能漏掉新产生的邻接穿插。最后右内眼角的一小
段折返使用固定周围点的正权重平滑；最终四个逻辑顶点的联立接触约束
使用当前实际三角形平面。原生模型的全局半平面次序不能作为新曲面的
解剖不变量，收据明确记录这两对的分离方向来源；没有改相交谓词或
降低最终的 `no_new_crossings` 要求。

较差的眼角前移、切口／可见边缘候选对应和整头位移场实验全部保留。
最终选择 **`local_contacts_v16`**，没有采用这些会增加穿插的实验。
独立完整检查 `local_contacts_quality_v16/receipt.json` 返回
`no_new_crossings=true`。`locality_acceptance.json` 另核对：区域外、
颈圈、全部眼切口、UV 副本保持；没有删面，全部其他原始数组保持，
所有三角形有限且不塌陷。源资产原有交叉单独保留，并非宣布源资产
绝对无相交。工具逻辑检查 14 项、既有几何谓词检查 19 项通过。

同一比例的完整正面／斜侧／侧面修正前后，以及修改区域图在
`outputs/native_mother_template_20261008/local_contacts_review_v16/before_after_regions.png`。
这是无贴图网格示意，**不是游戏截图**。肤色、绑定、表情及所有配套
部件还需随这个最终参考重新适配；本次完成的是头壳局部静态几何修复，
不是整套可安装母版的交付。游戏、程儿资产及人物卡未改变。

从冻结的前一局部候选重现最终步骤（模型／网格保持 ignored，输入收据
及数组 SHA 校验是命令的前置合同）：

```powershell
.venv/Scripts/python.exe -m tools.native_head.mother_local_contacts `
  --candidate outputs/native_mother_template_20261008/local_contacts_v7 `
  --quality outputs/native_mother_template_20261008/local_contacts_quality_v7 `
  --out outputs/native_mother_template_20261008/local_contacts_fresh `
  --smooth-residual --joint
.venv/Scripts/python.exe -m tools.native_head.mother_surface_quality `
  --candidate outputs/native_mother_template_20261008/local_contacts_fresh `
  --out outputs/native_mother_template_20261008/local_quality_fresh `
  --source-quality outputs/native_mother_template_20261008/surface_quality_v6
.venv/Scripts/python.exe -m tools.native_head.mother_local_review `
  --base outputs/native_mother_template_20261008/shell_v6 `
  --candidate outputs/native_mother_template_20261008/local_contacts_fresh `
  --out outputs/native_mother_template_20261008/local_review_fresh
```

下一步仅将通过的参考带入已有部件／法线／表情适配步骤，继续保护
整体轮廓；不重新开展整头配准、照片采样或新模型训练。
