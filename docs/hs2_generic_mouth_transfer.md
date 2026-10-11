# 通用完整嘴唇与口内身份迁移：进行中

2026-10-11。用户要求修复现有原生拓扑母版的通用规则：每个身份重新计算，
不逐人手工设计，不通过过滤非仿射唇形、退回共同仿射或关闭表情隐藏穿插。
当前阶段**局部静态迁移已通过，八个原生部件/帧已生成；未安装新资产，
尚未完成程儿资产合并和游戏验收，不是完整交付**。

## 2026-10-11：固定表面对应与独立闭嘴动作

新版不再靠闭嘴状态的空间最近面选择薄层。实际 native skin UV 是无折叠
展开，口腔组件的共享物理边界提供内轮廓，现有供体色素嵌入提供外轮廓。
`mouth_chart_correspondence` 把它们接到原始 FLAME UV/68地标上；原生三角形
上的虚拟外唇嵌入和完整内轮廓作为精确约束，母版对应只计算一次。
色素边界仍是现有解剖候选，不宣称所有供体已经逐个解剖认证。

仅要求正面积的最小位移 chart 会把少数单元压得近乎扁平，已保留其失败
探索。当前使用实际 native UV 的表面度量与精确 log-area 能量，约束全部
解剖嵌入，防止对应单元塌缩。求解器在试探反转状态有明确的数值延拓，
返回结果必须全部处于精确能量分支，不能交付该诊断延拓。
该算法是**资产制作规则，不是游戏原有变形公式**；不修改源模型或形状系数。

`mouth_chart_authoring` 在固定 source 三角形/重心坐标上采样原始完整300列
身份位移，再逐身份重算完整 native 口腔。不会退回共同仿射、平均唇形或
过滤高频系数。不同拓扑的连续表面完全一致仍不被宣称。

`mouth_contact_contract` 从实际原生 Close/Open 帧和原始索引恢复闭口虚拟
接触点。原接触既有上下唇，也有同一片唇的内外薄层，不能统称上下唇；
原骨权重在嘴角混合，作为来源记录，**不用于过滤这些接触**。
`mouth_close_pose` 以完整身份为独立输入，精确保持母版接触及外唇嵌入，
在局部 native 拓扑上计算平滑的 Close 动作；局部之外严格不变。

`mother_component_adaptation` 把动作写入该 renderer 的原生 Close 索引，
包括 vertex/normal/tangent，补足原稀疏帧需要的新行，完整保留原通道及行。
然后按真实默认权重反扣 bind，验证身份 bind 没吃到动作。牙、舌通过实际
口腔场获得同一动作，仍使用原始各自索引；全部八个部件保留。
同时修复了引用共享导致目标参考被原地修改、后续组件重复使用 Close
位移的问题。闭口校正不是身份数据，Open 时由原控制器权重自然退去。

当前结果：同一 chart/接触合同复用于 img-002、img-001 的两份真实 MICA
输出，各自重算后完整头部均无新增穿插；这是同一人的两个输入，不冒充
广泛不同人群验证。img-002 的八组件/原帧已完成作者数据生成。21项检查
通过，包含 Close 不烘焙身份 bind、不污染 Open、稀疏新增行、精确接触及
非塌缩对应。静态通过不代表口内组件互碰或全部动画已认证。

当前默认 pipeline 已接入这条路径，旧推面修复只保留作失败重放。
失败输出仍阻止发布，保留完整源参数、chart、动作、几何及局部修复输入。
下一步把局部规则并入已接受程儿资产制作链，保持既有颈部/头部位置及
外观配置，完成绑定、独立包/卡和小量游戏收尾。其他 Closed pattern 原始
帧完整迁移，只有 pattern0 接触动作在此阶段另行重建；其他 pattern 的
接触重建/动态表现尚未认证，不能宣称全面表情兼容。

当前证据：ignored `outputs/generic_mouth_20261011/lip_chart_v9/`、
`pipeline_v3/`、`pipeline_other_identity_v2/`、`chenger_components_v3/`。
下文保留早期源审查与失败历史，不代表当前默认实现。

## 已确认的原生机制

核查安装版 `Assembly-CSharp.dll`、`IL.dll` 与既有反编译来源 SHA；供体
来源是 `native_mother_template_20261008/inputs_v2`，母版是 `local_contacts_v16`。
`ChaControl.ChangeHeadAsync` 接回 `FaceBlendShape`；其 LateUpdate 按眉、眼、嘴
顺序调用原控制器。`FBSCtrlMouth.CalcBlend` 使用 `FBSBase.CalculateBlendShape`
为每个目标写 Close/Open 权重，并另有嘴宽调整；舌头状态由
`ChaControl.ChangeMouthPtn` 的实际 pattern 分支控制。

供体 MouthCtrl 有四个目标、每个26个 pattern；各目标 Close/Open 索引不同。
默认 pattern0/voice0 是实际非零闭口帧，不能把全零权重视为闭嘴，也不能
只套一个自制下颌角度。后续仍使用原始目标、索引、稀疏帧及写入链。

原生 `o_head` 的口内索引组件具有真实边界；通过已审计的同位置/同蒙皮
UV副本关系，可以确定它与皮肤共用的物理接口。口内、牙齿、舌头、原始
表情数据都保留。原生唇色支撑区同时覆盖外部唇面和闭唇薄层，**不是
已完成的内外唇解剖分割**。旧完整身份候选的新增穿插位于唇薄层，不能
笼统归因于独立口腔，也不能据此宣称只修口内便解决全部嘴部问题。

源码/资产审查入口 `tools.native_head.mouth_source_contract`，完整来源和原生
开口帧支撑图在 ignored `outputs/generic_mouth_20261010/source_contract_v1/`。
原 FLAME 零身份/目标身份的接触分类另保留在 `flame_original_contact_v1/`。
这些浮点分类包含原有接触/穿插，不等于源模型准确性或解剖性质的认证。

## 早期候选与失败处理历史

- `mouth_identity_transfer` 保留源 lips mask 的全部唯一顶点和300维身份
  字段，使用零平滑的完整控制点插值，未缩减为嘴部地标或共同仿射。
  母版到目标身份的口内 ARAP 逐身份重算，共享物理口内/皮肤接口保持。
  此处是新资产制作规则，**不是游戏原有算法**；控制点精确再现也不证明
  两个不同拓扑的连续表面完全一致。当前解剖对应仍需修正。
- `mother_identity --generic-mouth` 是显式未完成入口；旧完整字段/旧诊断
  重放路径保留。它与 `--mouth-detail-preview` 互斥，通用 pipeline 不调用
  仿射诊断分支。新的嘴部候选不是原线性身份 basis 单独能够重建的结果；
  必须保留逐身份口内求解及修正数据，不能静默省略。
- `mouth_contact_repair` 保存完整源对应、穿插面、固定/可移动顶点及修正
  位移。固定外部细节后执行局部接触修复；不重置成原生平均嘴、不改系数。
  当前正面几何遮挡的固定区只是保守候选，不冒充已完成的解剖分割。
- `generic_mouth_pipeline` 串行执行完整候选、静态几何门槛、接触修复及
  修复后的完整几何门槛。未通过返回 `requires_correspondence_repair`、退出2，
  保留可重放的修复请求；成功的静态头才进入 `accepted_static_head`。
  `mother_component_adaptation` 对通用候选验证接收状态、实际数组和几何
  证据哈希；失败候选不能继续制作八部件/帧或发布游戏资产。

早期程儿完整控制点候选的口内求解收敛，但仍有闭唇穿插；固定区局部修复
没有解决它们，且出现额外接触，全部失败保留。**程儿本身是未通过的普通
输入，不能把它改称极端脸型便宣称通用规则已完成。** 下一步明确原生上下
唇接触层/外唇/内唇对应，区分身份形状与原生闭口帧的约束，然后修正迁移
和闭口帧适配。不要再单纯更换全头位移插值或反复推面补偿。

此前游戏人物卡、已接受的颈部、后期02卡以及任何已安装 zipmod 均未改变。
当时17项身份/母版/嘴部合同检查通过；实际失败候选的组件入口被拒绝且没有
生成组件目录。这些检查不代替未通过的几何、口内配套或动态表情验收。

## 重现入口

```powershell
.venv/Scripts/python.exe -m tools.native_head.mouth_source_contract --out outputs/generic_mouth_20261010/source_contract_fresh
.venv/Scripts/python.exe -m unittest tools.native_head.test_mouth_identity_transfer tools.native_head.test_mother_identity tools.native_head.test_mother_authoring
.venv/Scripts/python.exe -m tools.native_head.generic_mouth_pipeline --inputs outputs/native_mother_template_20261008/inputs_v2 --mother outputs/native_mother_template_20261008/local_contacts_v16 --manifest outputs/chenger_mica_review_20261007/raw/manifest.json --image-index 1 --source-quality outputs/native_mother_template_20261008/local_contacts_quality_v16 --out outputs/generic_mouth_20261010/pipeline_fresh
```

审查脚本和 pipeline 都要求新目录，拒绝覆盖原证据。候选静态头
通过后仍需已有八部件、原生默认帧重建、绑定和少量实际闭嘴/张嘴收尾，
再制作程儿独立人物卡。全面滑杆/ABMX兼容不在此小目标内。
