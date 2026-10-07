# 多骨骼ABMX实际状态回放与完整头部验收

新增工具只读分析actual trace／geometry／PNG，未操作游戏、改core／adapter／fitter／缓存或旧报告。游戏采集和恢复由root完成。本次把单ChinTip条件回放扩展为四个**实际观察到的local状态**共同驱动完整头部FK/LBS；不是无状态任意候选模型，也不是恢复程儿人物拟合。

## 输入与可执行接口

```powershell
.venv/Scripts/python.exe -m tools.abmx_multibone.run ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_multibone_v1/live_cases.json --contract outputs/abmx_replay_20261005/installed_contract_v2.json --out outputs/abmx_multibone_20261005/<new-audit>
.venv/Scripts/python.exe -m tools.abmx_multibone.summarize outputs/abmx_multibone_20261005/<new-audit>/summary.json --contract outputs/abmx_replay_20261005/installed_contract_v2.json --out outputs/abmx_multibone_20261005/<new-call-summary>.json
.venv/Scripts/python.exe -m unittest tools.abmx_multibone.test_multibone -v
```

`run.py`消费root finalized manifest：四head case，各自trace路径＋producer SHA和early／late window的geometry路径＋SHA、capture三视图、head_id、名字。manifest必须有state／expression／bones完成恢复；predeclared protocol路径／SHA必须匹配，包含四名称、四底模、完整native59和固定ABMX patches。sampling profile显式记录，默认实际安装 `slider_unlocker_18_2`；本次native全.5，两种profile在区间内一致。不是依赖尚未加载的新DLL／capture_state layer。

trace用现有独立NumPy `abmx_replay.validate_trace.validate`重新验收，严格installed Apply IL／MVID／版本／observer policy、零drop／pending／errors、raw local/cache合同和真实events连续sequence。实际安装DLL／Unity core／反编译源码也对独立contract SHA复核。不能只读producer或旧报告passed字段。

`geometry.select_cursor_predictions(snapshot, trace, contract, required_names)`为**每根骨骼**从全局cursor之前的events选择最后一条completed call。不是让四骨都取全局最后sequence，也不是拿trace停止时状态代替截图时刻。每名必须在observer filter中，prefix至少两条调用、modifier／transform identity稳定、最后call至snapshot最多一frame；否则缺失／稀疏／stale明确拒绝。这些是固定保守覆盖合同，不是从目标曲面拟合调用次数。snapshot可能先于当前frame LateUpdate，因此一frame年龄允许，但local/privatecache另与预测数值逐项一致才可使用。

snapshot current bone必须同ID／名字／路径，private cache必须同boneID／frame／安装MVID／modifier类型，全部12字段齐，所有bool精确一致。独立预测来源优先propagated chain prediction；actual after／snapshot locals只用于比较，不复制为重建输入。若事件间发生local/cache变化，保留 `external_boundaries_at_or_before_cursor` 和未识别writer声明；观察回放可从真实before作新segment anchor，但未来候选不能免费跨未知writer边界。

## 多骨骼FK/LBS合同

输入骨骼是 `cf_J_Chin_rs`、其子 `cf_J_ChinTip_s`、`cf_J_CheekUp_L/R`。先由完整actual native59与真实head cached tables算原生local，再把四个独立预测local TRS覆写到对应bone，按**parent-first拓扑**重新计算所有world matrices。父骨rotation／scale自然传递给子骨，不分别修改最终world矩阵。

源geometry与缓存要求全o_head顶点、ordered triangle、skin palette／indices／weights／bindposes一致；检查每个skin骨的所有祖先local和内部parent映射，不只检查skin endpoints。所有非override祖先仍由native59预测，若出现其它未建模局部形变则拒绝。cached asset root `p_cf_head_NN`与live bone-prefab root `p_cf_head_bone`作为显式资产frame边界记录；其内部child parent关系仍检查，外部ancestor uniform scale由actual renderer matrix导出，最后完整头部残差独立验证该边界。没有任意affine补偿。

actual blendshape frame delta是独立表达式nuisance输入；FK＋source bindpose＋四权重LBS重建完整o_head。实际bone LBS先独立认证Unity BakeMesh world conversion，renderer full path／source SHA绑定，disabled／inactive对象由既有certified-head loader明示跳过，不按重复 `o_tang`名字覆盖。单位固定1，仅导出的ancestor uniform与proper rigid pose；禁止从顶点拟合scale／affine，禁止skin-quality override。完整头部gate保持 `max_normalized = max vertex L2 / actual bbox diagonal <=1e-5`；这也比同量纲component最大值gate保守。

三视图还逐项对geometry文件SHA／路径、export与render frame、pose signature、sampled visibility signature交叉绑定。新增负例证明 `paired_visibility_sampled_unchanged=true`也不能掩盖签名值不一致；完整bone/FK/LBS数值验收是额外证据，不靠pair booleans单独宣布通过。签名仍覆盖导出的sample scope，未证明完整shader alpha／depth/material图。

## 真实结果

最终输入 `HS2Mod/artifacts/infrastructure_live_20261005/abmx_multibone_v1/live_cases.json`恢复flags全部true；predeclared patches在测量前已保存。实际四trace分别116／116／124／116条，**472/472 unique calls通过**。每window都重算完整trace，统计去重后不把八window计成944个独立调用。max position component `5.9604645e-8`、scale误差0、quaternion符号等价component `2.3283064e-10`、rotation角 `3.7731464e-8°`；所有private numeric残差0、bool mismatch0、missing fields0。

当前真实branch覆盖是28次identity CanApply false，444次nonempty，后者同时scale baseline、Euler rotation、length current-direction normalization和position offset。本次没有runtime覆盖HScene／short-length／zero-direction／neutral restore／additional modifier／rotation exclusion等四骨联合分支；旧单骨反例／解析tests不能自动升级为四骨联合runtime覆盖。

截图cursor逐骨绑定的32组local／cache都通过；全部latest call年龄是一frame。early实际prefix每骨9／8／7／8 calls，late全28；不能以HTTP请求的等待数字直接代替这些实际调用数。

| Head | 全o_head顶点 | Early global cursor | Late cursor | Early max normalized | Late max normalized |
|---|---:|---:|---:|---:|---:|
| 0 | 4418 | 36 | 112 | 9.746e-8 | 1.036e-7 |
| 1 | 4438 | 32 | 112 | 9.454e-8 | 9.671e-8 |
| 2 | 4439 | 28 | 112 | 9.631e-8 | 1.058e-7 |
| 3 | 3650 | 32 | 112 | 9.683e-8 | 9.071e-8 |

**8/8完整头部窗口通过**。最大world L2 `3.1464518e-7`，最大RMS约 `1.1603e-7`。early排除future80／84／96／84 calls，late排除4／4／12／4；四骨每根取各自last completed sequence（例如head2 early25／26／27／28），而非都用28或124。全部观察segment无检测到的数值external boundary；这不证明不存在无数值变化的外部writer调用。

四底模初始实际persistent length history均保留：Chin_rs `0.347958982`、ChinTip_s `0.120832346`、CheekUp_L/R `0.543772757`。未换成candidate native length或各head cached rest；当前local方向沿每条真实调用链传播。

最终报告位于 `outputs/abmx_multibone_20261005/actual_v2_signature_binding/summary.json` 和八个详细JSON；初轮 `actual_v1/`保留，v2增加signature值交叉绑定后再次验收，数值结论不变。`call_summary_final.json`是去重且重新调用NumPy verifier的汇总。该工具验证early／late24个raw PNG身份与metadata；root提供的另外12baseline PNG未在此工具独立做baseline geometry拟合，不能把36张都称此轮像素／几何认证。

十三项新解析／拒绝tests通过，涵盖每骨单独cursor、future／sparse／stale／drop／filter拒绝、actualafter篡改不能当预测输入、snapshot local／flags／ID错配、外部boundary仍未识别，以及父子90°rotation＋scale＋两bone local覆写后的手算LBS、负scale保留、不合法parent-first／unknown bone拒绝、green pairflag但signature值不同拒绝。只有这些tests自身的验收逻辑经过验证；runtime结论来自独立真实数据。

## 仍未实现／未验证

本次native59全.5、四个温和predeclared modifiers、四base、两个观测时刻。没有证明任意mixed native59＋四骨参数候选、其它bones、不同持久cache历史或未知writer边界都可预测。shared stateless模型／stateful fitting adapters没有被此工具修改，也不能据此宣称自动修复。真实before/cache是条件输入；候选native如何生成它们、何时CollectBaseline／Reset、如何保证调用schedule仍需单独验收。

整头parity不是形变安全质量／self-intersection／真人语义对应／程儿相似度通过，也不是底模表达力全局结论。未知scope会保留unsupported／failed，不改变fixed数值gate，不拟合调用次数，不把实际after注入预测结果。
