# 明确截图姿态阶段的消费者 V2

新文件 `src/hs2_controlled_radial_v2.py` 继承冻结 V1 的参数、driver、资产、trace、whole8/temporal 及恢复合同，保留 V1 `d0d584…` 字节与实际失败证据。它只把原始姿态的比较阶段改为**显式指定且重复观测通过的截图姿态**，没有放宽 source→guard→window→recovery 的 local/cache 门槛。

## V1 实际拒绝的原因

root 的 `controlled_consumer_v1` 实测在第一筆 expression/native/ABMX 写入前拒绝，0 scoring；公共与私有字段保持一致，但 live GET geometry 与配对截图有26项 outside actual local 分量不同，包含 Neck、ShoulderIK 和身体骨。证据为 [root_boundary_diff.json](../../HS2Mod/artifacts/infrastructure_live_20261006/controlled_consumer_v1/provider/b2d28b8361c647a8a709d04854d38e83/root_boundary_diff.json)。

`HS2Mod/plugins/HS2_McpBridge/MakerRenderService.cs` 的 `PoseFreeze.Apply`（约771–830行）对每个有 runtime controller 的 Animator 保存 layer0 state/time/speed，设置 speed0，并 `Play(current_state,0,0)` / `Update(0)`；截图结束恢复 saved time/speed。截图也临时关闭 blinking。这是声明的截图阶段操作，live animated TRS 不能直接要求等于 rewind TRS。`froze_animators>0` 仅确认有 Animator 被处理，不能认证所有 layer、driver 或动态身体都规范化。每项 live→canonical 差异仍保存且不单独推断成动画因果，不能隐藏 Neck delta 或改称全身恢复。

## 显式 API

```python
from src.hs2_controlled_radial_v2 import ControlledRadialProvider

provider = ControlledRadialProvider(
    original_pose_policy='canonical_rewind_snapshot_v1',
    canonical_settle=60,
    work_dir=..., contract_path=..., driver_contract_path=..., asset_plan=...,
    strict_backend='identity_source_assets_strict_v2',
    outside_state_policy='source_bound_identity_nuisance_v1',
    resolution=512,
)
```

其他 provider/evaluator 接口与 V1 相同。policy 必须显式传入；unknown policy、bool/其他等待数拒绝。`canonical_settle=60` 固定绑定到 original、original_confirm、restored 三个配对截图，在请求文件中先声明；不是共同 Apply N。已有 source5/guard2/early5/late18/far60 与 consumer capture 仍取实际 trace cursor、实际 local 差异，并保留原严格比较。

## 写入前的两次独立观测

1. 再读 live context；公共配置、native59、head/actor/game/profile/private cache 仍严格绑定，不随姿态 policy 忽略。
2. 得到第一个 canonical original capture（freeze_pose=true、same-frame全 actor transform export、settle60），将其作为声明的**局部恢复参照**。live→canonical 必须 actor/head/native/game、全部 actual transform ID/name/path/parent、renderer/component/object/mesh instance IDs 和有序 skin bone IDs 一致；outside 公共与完整私有 flags/fields 除 frame wrapper 全相同。
3. 保存 `live_to_canonical_observation.json`：全部 live→canonical local 检查和差异（包含 Neck）、两个实际阶段标签、live geometry JSON hash、配对几何 descriptor，以及本地 PoseFreeze source SHA。没有 live pose 相同或已恢复的结论。
4. 第一筆 owned 写入前取得独立 `original_confirm`（同 camera/phase/settle60）。再次核对原始公共表达/config/ABMX、私有 cache、component/palette provenance 和**每个导出的 actual local TRS**。同 canonical 阶段的未知 local/cache/public 漂移仍拒绝，不能视为 rewind 的允许差异。
5. 第二次通过后才进行最终 native59、native drivers、absent30 创建与全套 fresh source-only 编译/guard/实际 trace/capture。没有用 candidate assets/after 生成上述原始参照。

两次确认保存 `canonical_original_guard.json`。新 self、冻结 V1、已有 helper/contract/asset plan，以及本地 `MakerRenderService.cs` SHA 在 source_files 中绑定；源文件改变要求重建 provider。编译器与严格资产 provider 的原有依赖冻结继续保留。

## 恢复与接受的具体范围

恢复顺序仍是 stop owned trace→identity owned30→remove owned30 **先于** original native59→restore original expression，逐步失败仍尝试其余步骤。V2 restored capture 也用预声明 settle60。原始 public/private 与**第一个 canonical paired snapshot** 的完整 actual local 骨架须通过原容差，Neck 或任何非 ALL30 漂移都不能豁免。

`canonical_restoration_report.json` 引用不可变 base restoration report、第一次 canonical reference、独立 confirm guard 和 live→canonical observation。即便数值与此恢复全部通过，`original_live_actor_pose_restoration_certified=false`、`full_actor_restoration_certified=false`；不宣称正在播放的 live actor 骨架、所有 animator layer、私有动画时序、无限期动态姿态或 rendered RGB 恢复。若 canonical repeat/restore 中 gaze、Neck、body 或私有状态不稳定，必须保留真实失败并禁止 scoring。

## 离线验证

9个 NEW fake phase tests覆盖：显式 policy 下保存并接受 live↔rewind 差异且 canonical repeat 通过；canonical repeat local 漂移在owned写入前拒绝；同 GO 的 mesh instance 替换拒绝；actor/path provenance变化拒绝；canonical私有 cache或公共表达改变拒绝；候选 source 后非受控 body local仍拒绝；固定等待 schedule和phase/self/native source SHA冻结。Ruff通过。fake transport/math不等于实机接受，本轮agent未操作游戏；root新的V2实际报告单独决定接受或拒绝。
