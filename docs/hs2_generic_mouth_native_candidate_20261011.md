# 通用嘴部规则：程儿独立原生候选 v7

2026-10-11。**源实现/静态制作完成，游戏加载与闭嘴/张嘴收尾未完成。**
不是角色交付或全动态兼容证明。平台目标继续 active。

通用规则见 [实现记录](hs2_generic_mouth_transfer.md)。固定表面 chart 和实际
Close/Open 接触供每个身份复用；完整300列身份和口内/闭嘴动作逐身份重算。
两份真实 MICA 输出已通过完整静态头检查，是同一人的两个输入，不冒充
不同人群验证。完整源数据保留，未采用共同仿射/高频过滤或禁用表情。

## 现有资产合并与原生包

`mouth_identity_merge` 把完整嘴部合入已接受 `chenger_balance_v6`。源身份
必须相同，改变范围由源方法的明确 core/transition/唇面/口内/Close 支撑
确定；区域外及既有颈带/底端逐顶点保持原样。实际局部必须匹配记录中的
刚性摆放，额外美术变形不能静默覆盖。算术一致性界来自300列累加的浮点
误差界，含原始项绝对和，不是拟合的距离门槛。

合并后完整静态头无新增穿插；全八组件及原生帧重建、原生0.5参考骨路径
绑定、完整 bundle 写入读回通过。包装器允许必要的 sparse frame 新行，但
逐帧检验所有原行的顺序/索引、全部原通道/权重及属性标志，没有放松成
“只保留几个表情”。原材质、纹理/resource 字节及未编辑对象严格保持。

独立 native head/skin CSV210/211 和 `faceSkinInfo` 注册仍采用已验证供体
契约。新 GUID `codex.native.flame_native_chenger_mouth_v7`，原 head/skin
local slot 各1。不是 SourceHead 外壳或显示替换。现有v6、后期卡及其他
任务正在制作的原生02风格版均保留。

包文件已按 SHA 核对复制至：
`E:/HoneySelect2_ArcticFox/mods/CodexNativeMother/flame_native_chenger_mouth_v7.zipmod`。
**文件安装不代表运行中 loader 已注册或人物卡已完成。** 未修改可见 actor；
只读观察为 head2/skin20。真实动态ID、新独立卡、native闭嘴/张嘴表现及
口内部件互碰仍待收尾；其他 Closed pattern 接触重建尚未认证。

Ignored证据根：`outputs/generic_mouth_20261011/`，含 `lip_chart_v9`、
`pipeline_v3`、`pipeline_other_identity_v2`、`chenger_v6_mouth_merge_v1`、
`chenger_v6_mouth_quality_v1`、`chenger_v6_mouth_components_v1`、
`chenger_v6_mouth_bindings_v1`、`chenger_v6_mouth_bundle_v1`、
`chenger_v6_mouth_package_v1/installation_receipt.json`。

## 可重现入口

```powershell
.venv/Scripts/python.exe -m tools.native_head.generic_mouth_pipeline --inputs outputs/native_mother_template_20261008/inputs_v2 --mother outputs/native_mother_template_20261008/local_contacts_v16 --manifest outputs/chenger_mica_review_20261007/raw/manifest.json --image-index 1 --source-quality outputs/native_mother_template_20261008/local_contacts_quality_v16 --out outputs/generic_mouth_fresh
.venv/Scripts/python.exe -m tools.native_head.mouth_identity_merge --accepted outputs/native_mother_template_20261008/chenger_balance_v6 --identity-reference outputs/native_mother_template_20261008/chenger_identity_v3 --mouth outputs/generic_mouth_fresh/closed_pose --out outputs/generic_mouth_merged_fresh
```

后续依次调用既有 `mother_surface_quality`，通过后写入匹配数组 SHA 的
`mouth_acceptance`，再走 `mother_component_adaptation` →
`mother_reference_bindings` → `mother_bundle_candidate`（独立asset key）→
`mother_registration`（独立GUID）。不得绕过失败几何门槛发布。
失败 pipeline 保存结构化 `repair_request.json`，含具体阶段、原因、完整源
身份、原Close动作、已知穿插面及可重放输入；未知具体区域明确保留未知。
之后通过 `--lip-chart/--close-contract` 接回修复后的通用合同，不删源细节。

检查入口：
`.venv/Scripts/python.exe -m unittest tools.native_head.test_mouth_surface_transfer tools.native_head.test_mouth_identity_transfer tools.native_head.test_mother_identity tools.native_head.test_mother_authoring`。
完成 milestone 时及时同步两仓库；许可资产/输出继续 ignored。
