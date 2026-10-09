# 程儿头部摆放与后脑体积分布 v6

2026-10-09。用户指出高位后凸过于靠后、中下后脑显得薄，并要求考虑
整个头的位置。本步骤是原生底模资产创作，不是医学头位推断。

## 参考与决定

原生女性02号与 v4/v5 的同骨架完整剖面已在
[后脑参考对照](hs2_occipital_reference_review.md)记录。当前头整体的
后侧范围比原生更靠后，但圆度集中在较高位置；下部过早变成长斜面。
所以不能继续只向上推局部后脑。

[Ferrario 等的头位研究](https://pubmed.ncbi.nlm.nih.gov/8074090/)在其
年轻正畸样本中发现自然头位有较大个体差异，软组织耳眼参考线与骨性
Frankfort平面也不重合。该结果不能为本模型推出一个通用“正确角度”。
网格耳部与眼睑点也不是骨性 porion/orbitale。本版保留现有正视朝向，
不强行旋转或缩放以对齐原生风格。

## 实际修改

- 从已接好颈部的 `chenger_neck_v4` 出发，明确拒绝叠加 v5 局部修改。
- 整头参考及 FLAME 眼关节参考共同小幅前移；高度和倾角不增加变化。
- 后脑凸起向前并向下调整，使圆弧重心下降，中下部圆度延续更低。
  前后及高低范围采用原生耳骨支持区、已记录颈端和颅顶确定；三方向
  五次 C2 衰减。不会局部改变脸、耳或口内。
- 原生底端两排保持供体位置；用已有完整颈带约束求解重建新摆放与
  身体接口之间的过渡。下巴下部随整体摆放重新接颈，没有单独缩放下巴。
- 不增删面，不变更顶点顺序、UV、蒙皮、颜色、原资源或控制器。全部
  八部件、原生稀疏表情帧、绑定和法线/切线经过完整现有适配链。

显式创作设置以当前耳部高度为单位：整头前移0.10，后脑最大下移0.24、
前收0.22。它们是可配置的设计幅度，不是从游戏反推的计算参数，也不是
解剖标准。底端、保护区、UV副本和 SurfaceWarp 正定合同在生成时检查。

## 重现

```powershell
.venv/Scripts/python.exe -m tools.native_head.mother_head_balance --candidate outputs/native_mother_template_20261008/chenger_neck_v4 --out outputs/native_mother_template_20261008/chenger_balance_v6
.venv/Scripts/python.exe -m tools.native_head.mother_surface_quality --candidate outputs/native_mother_template_20261008/chenger_balance_v6 --source-quality outputs/native_mother_template_20261008/chenger_occipital_quality_v5 --out outputs/native_mother_template_20261008/chenger_balance_quality_v6
.venv/Scripts/python.exe -m tools.native_head.mother_component_adaptation --inputs outputs/native_mother_template_20261008/inputs_v2 --candidate outputs/native_mother_template_20261008/chenger_balance_v6 --out outputs/native_mother_template_20261008/chenger_components_v6
.venv/Scripts/python.exe -m tools.native_head.mother_reference_bindings --inputs outputs/native_mother_template_20261008/inputs_v2 --components outputs/native_mother_template_20261008/chenger_components_v6 --out outputs/native_mother_template_20261008/chenger_bindings_v6
.venv/Scripts/python.exe -m tools.native_head.mother_bundle_candidate --inputs outputs/native_mother_template_20261008/inputs_v2 --bindings outputs/native_mother_template_20261008/chenger_bindings_v6 --asset-key flame_native_chenger_v6 --out outputs/native_mother_template_20261008/chenger_bundle_v6
.venv/Scripts/python.exe -m tools.native_head.mother_registration --bundle outputs/native_mother_template_20261008/chenger_bundle_v6 --out outputs/native_mother_template_20261008/package_chenger_v6 --title 'Chenger native head balance v6'
```

完整静态头壳无新增穿插，全部原生部件/表情帧及未改资源写入后读回
吻合。候选、场、完整剖面、质量结果与全部中间产物留在上述 ignored
outputs 路径。独立GUID `codex.native.flame_native_chenger_v6`，旧版保留。

原来非仿射唇形细节、颈部着色分界以及全动态兼容的缺口没有因本次
形状调整而完成。实际游戏结果另行记录，不把静态候选当成游戏验收。
