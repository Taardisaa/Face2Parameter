# 原生拓扑母版的完整颈部过渡

2026-10-09。用户指出当前程儿 v3 的耳后和下巴下方呈头套式直角台阶，
授权复用已有曲面处理，并允许小幅整体上提头部。

## 原因与适配

旧 `placement_review.rebuilt_baseline` 根据完整源头和原生资产的曲面
方向生成 Hermite 连接带；`posterior_neck.fair_posterior` 用平方拉普拉斯
铺顺后部网格。两者依赖旧 FLAME 裁切网格及新增连接排的布局。
当前母版使用完整原生拓扑，原流程只固定颈圈位置，身份位移的
`neck_pin_basis` 也只有位置约束，并没有接入这两侧的曲面过渡。

`mother_neck_transition` 在身份候选生成后、八部件适配前增加一个步骤。
复用两端曲面方向约束和局部曲面铺顺原则，改为在现有原生顶点上求解
夹持双调和**位移**，而不新增或删除面：

- 原生接口圈和第一邻接圈恢复供体原来的位置，保持原生下端曲面方向。
- 外端保留现有身份头形，并做同一个向上平移；两排端部约束防止突变。
- 内部过渡带在耳后、两侧和下巴下方共同求解，覆盖完整一圈。
- 原生耳／眼／嘴／鼻／眉骨支持区及独立口内硬锁为同一平移，不参与局部整形。
- 物理 UV 副本统一求解后映回原索引，UV、面、权重和其他原始数组保留。
- FLAME 参考和 placement 同步上移，后续眼球、牙舌、睫毛、泪液和全部
  原始表情帧继续经过完整八部件适配及原生骨架绑定，不能留在原来位置。

带宽为七个拓扑邻接环，上提量为当前耳骨主要支持区域高度的四分之一。
这是显式的新资产创作设置，不是游戏算法、拟合出的滑杆增益，也不声称
任意拓扑或任意身体兼容。接口来源须由实际头／BP 身体捕获核对。
本次前后两侧用同一夹持曲面求解；没有直接调用旧裁切布局的贝塞尔生成器。

## 重现

在 Face2Parameter 根目录执行，输出目录必须全新：

```powershell
.venv/Scripts/python.exe -m tools.native_head.mother_neck_transition --candidate outputs/native_mother_template_20261008/chenger_identity_v3 --capture ../HS2Mod/artifacts/native_mother_20261008/seam_head_body_01.geometry.json --out outputs/native_mother_template_20261008/chenger_neck_v4
.venv/Scripts/python.exe -m tools.native_head.mother_surface_quality --candidate outputs/native_mother_template_20261008/chenger_neck_v4 --source-quality outputs/native_mother_template_20261008/chenger_quality_v3 --out outputs/native_mother_template_20261008/chenger_neck_quality_v4
.venv/Scripts/python.exe -m tools.native_head.mother_component_adaptation --inputs outputs/native_mother_template_20261008/inputs_v2 --candidate outputs/native_mother_template_20261008/chenger_neck_v4 --out outputs/native_mother_template_20261008/chenger_components_v4
.venv/Scripts/python.exe -m tools.native_head.mother_reference_bindings --inputs outputs/native_mother_template_20261008/inputs_v2 --components outputs/native_mother_template_20261008/chenger_components_v4 --out outputs/native_mother_template_20261008/chenger_bindings_v4
.venv/Scripts/python.exe -m tools.native_head.mother_bundle_candidate --inputs outputs/native_mother_template_20261008/inputs_v2 --bindings outputs/native_mother_template_20261008/chenger_bindings_v4 --asset-key flame_native_chenger_v4 --out outputs/native_mother_template_20261008/chenger_bundle_v4
.venv/Scripts/python.exe -m tools.native_head.mother_registration --bundle outputs/native_mother_template_20261008/chenger_bundle_v4 --title 'Chenger native neck v4' --out outputs/native_mother_template_20261008/package_chenger_v4
.venv/Scripts/python.exe -m unittest tools.native_head.test_mother_neck_transition
```

静态完整头壳检查没有新增穿插。独立检查保护下端位置／方向、外部刚性
平移、无位移时保持原网格，以及受保护特征与接口冲突时拒绝执行。
打包重读核对全部八部件、全部原始表情帧和未修改纹理资源。

原生游戏加载和最终画面结果见配套 HS2Mod 文档。此步骤只处理颈部几何；
不会宣称原来未完成的非仿射唇形、材质外观、全部表情／滑杆／ABMX已完成。
提取／许可网格、候选和完整证据继续位于 ignored outputs 与 artifacts。
