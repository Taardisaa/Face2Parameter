# 可观测特征与材料点的类型合同

`tools/observable_features/` 是新增的离线诊断工具。它区分数学可定义的曲面特征、固定材料候选、图像外观线索与不可观测项，防止用相同名字把它们合并。没有修改现有骨骼代理、FAN、形变、拟合、质量检查或冻结标注；没有连接游戏。

本轮 pitched train 的九张原图、54 条记录仍是 `ambiguous/null`，SHA `9899dd0375088291275eceaeea966a430c49c9beb2913a08cf38ce8ad38b5663`。本工具既不填这些空值，也不改变固定 2px、深度或唯一材料点门限。

## 类型和能够支持的比较

| 类型 | 可执行内容 | 尚不能声称 |
|---|---|---|
| `formal_surface` | 完整资产 submesh 的方向跨度、support 集合、精确三角形面积、平面与实际三角形的线段交集 | 鼻尖、脸宽、面长、唇缘、唯一解剖点、图像可见材料点 |
| `fixed_material_candidate` | 绑定源三角形、有序顶点和固定 barycentric，在该实际 BakeMesh 上求位置 | 图像可见、shader 深度、已标定的解剖点或跨底模语义 |
| `appearance_pixel` | 原 PNG 坐标和真实定位不确定度，或 occluded/ambiguous 的 null；绑定自己的相机、pose 和 geometry 文件 | 皮肤深度、唯一材料身份、纹理与皮肤边界等价 |
| `unobservable` | 明确记录要观察什么、为什么不能观察，并保留 null | 零坐标、零误差、默认点或正面证据 |

方向跨度是全 submesh 的 `max(v·axis)-min(v·axis)`。三角形上的线性方向 support 在顶点取得；工具报告容差内的所有 source vertex IDs，不把 tie 随便挑成唯一点。每次形变重求 extrema 的合同，与跟随固定 source triangle/barycentric 的材料点合同不同。一个 support 以前恰好只有一个顶点，也不会生成解剖或跨视角认证。

平面截线输出逐区域三角形的交线段、切点与共面三角形。工具不焊接位置、不将 UV/index 边界认作唇/眼/颈，不把多段或共面区域宣布成唯一有序曲线。共面项保持 `ambiguous_coplanar_surface`。

这些数学量能描述完整头部资产的宽/高/深方向跨度、表面积和分层外形，可供同底模的形变诊断。`o_head` 完整 submesh 的范围可能包含头皮、颈部、内口腔和 seam，**全资产 y 跨度不能自动称为面长**。要比较真人或不同底模的脸宽、面长、局部骨相，仍需要明确的脸部区域、经过独立证明的共同轴/单位/姿态和跨底模区域等价合同。当前 `compare_spans` 会拒绝 head/source/renderer/raw-frame policy 不同的比较；同源结果也只称 raw-frame diagnostic，不能掩盖姿态或框架误差。

## 文件与数值绑定

Pydantic schema 禁止额外字段，强制区分上述类型。输入绑定绝对 geometry 路径和字节 SHA、head ID、renderer 完整路径、source geometry SHA、frame、pose signature、原生参数数组及 expression 配置的 canonical JSON SHA。要求 geometry 导出结束 frame 相同。ABMX 参数证据明确标为 `unverified_not_bound_by_this_contract`；实际姿态与文件绑定不能冒充 ABMX 行为验收。

只读取 `renderer_baked_raw` 和 `game_units_unscaled`。没有把 raw 游戏单位称为毫米，也没有借用一个未经验证的世界矩阵。完整/分 submesh 索引流必须实际一致，Triangles topology 与已应用 base vertex 明确成立；非法、分数、越界或改变顺序的 triangle/index 被拒绝。点权重保持固定，不能通过重找 extrema 修复 material correspondence。

图像项另外校验 PNG 字节、实际 PNG 编码与完整性、实际尺寸、single-capture payload 字节和配对 geometry SHA/pose/frame，要求实际 camera matrices 与同帧 pose pairing。`Camera.from_capture(..., diagnostic=True)` 只检查相机及配对结构，不绕过当前 pixel/LBS certificate 去宣称测量认证。PNG pixel、世界/LBS、shader visibility 的科学证明仍交给既有独立验收器；该工具所有有关认证字段保持 false。

Manifest 合同及 source byte 验证通过，与解剖/图像可见性通过是不同结论。所有五种 claim（anatomy、image visibility、cross-view fixed material point、cross-base semantics、likeness）只能为 false。`certified_anatomical_label` 必须 null。任意将 formal span 叫作 certified nose/face point、塞入 shader/depth true 字段、放宽 2px gate 的输入都被拒绝；schema 通过不等于整个基础设施完成。

## 实际执行

在 Face2Parameter 仓库使用项目 Python：

```powershell
.\.venv\Scripts\python.exe -m unittest tools.observable_features.test_contract -v
.\.venv\Scripts\python.exe -m tools.observable_features.measure --manifest outputs/observable_features_20261005/head2_old_raw_example_v2/manifest.json --out outputs/observable_features_20261005/<new-report.json>
```

`example` 命令需要确切 renderer path，不按同名 mesh 回退。它为一个旧 actual geometry 构造八条 formal 量和六条 unobservable 项；没有拿旧 seed 或新留出图创建材料候选。可生成 IDE 使用的 `manifest.schema.json`。所有 CLI 输出路径必须是新的，旧报告不覆盖。

本轮真实旧 head2 raw 示例的 x/y/z 跨度约为 `1.57679106 / 1.931299155 / 1.630554055` raw 游戏单位。这些是 literal 全 submesh 量，没有人体毫米或脸宽/面长认证。16 个 synthetic tests 验证数学行为和拒绝合同：不准升级语义/可见性、空值不填零、coplanar 保持歧义、源字节/state/pose/triangle order 改动拒绝、跨 base/frame 比较拒绝。Synthetic tests 不认证真实人物、shader 或解剖。

## 下一步仍需的证据

完整脸型和 3D 形状可继续用既有全曲面面积加权距离、法线和网格质量检查；这些不需要伪造鼻尖坐标。但要把那些比较解释为特定脸部比例或真人对应，需要可观察且唯一的定义和经审查的 region/material 证据。RGB 中平滑隆起没有唯一点、鼻翼 margin 是一段曲线、闭唇暗线可能是外观纹理，这些事实不会因多了一个数学 extrema 而消失。可以另行预注册曲线/区域任务或几何定义，与现有六个失败命名候选保持区别，再测其适用范围；不能反过来改动冻结 null 审查以凑通过。

