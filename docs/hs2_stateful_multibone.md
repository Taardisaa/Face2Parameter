# 四骨显式状态候选协议

实现位于 [adapter.py](../tools/stateful_multibone/adapter.py)。支持且仅支持 `cf_J_Chin_rs`、`cf_J_ChinTip_s`、`cf_J_CheekUp_L`、`cf_J_CheekUp_R`，四者均须存在于当前 head 的缓存骨架及 `BONE_NAME_LIST`。这扩展了单骨协议，不会修改旧 `stateful_fit` 或静态 ABMX 核心。

## 输入与拒绝条件

`MultiNativeBaselineProtocol.from_manifest(manifest, contract, head_id, window='late', common_apply_count=None)` 接受原有四骨 `live_cases.json` 格式：manifest 必须确认参数、表情和骨 modifier 已恢复，SHA 绑定预声明 `protocol_path`、各 case trace 和各 window geometry。预声明协议须含完整 `native59` 和四个具名 `patches`（scale xyz、length、position xyz、Euler rotation xyz），与快照及 cursor 对应的有效 modifier 数值匹配。

安装分支由独立 contract 的 ABMX DLL、Unity Core、反编译 source SHA/MVID/Apply IL 和 patch-owner 合同约束。每次构造会重新执行整个 stopped trace 的 NumPy replay，并用 [多骨 selector](../tools/abmx_multibone/geometry.py) 校验所有骨在精确 global cursor 之前各自最后完成的调用、实际 snapshot local/cache、骨实例 ID、观察覆盖数、同帧和最多一帧年龄。不能只读取已有绿色报告。

候选协议进一步要求：

- 整个 trace 无外部 local/cache 写入边界；各骨实例在完整 trace 内稳定，没有额外 modifier、H-scene 或 rotation-excluded 分支。
- 每骨首个 observed modifier 为 identity；首个 before 的 `_hasBaseline=true`，其他六个 flags 均为实际 `false`。任何骨不干净，整组拒绝，不能补 flags 或强制重置。
- cursor prefix 中，每骨 identity 阶段之后只有一个恒定非 identity patch。观测 early/late 各自截断，不能取整个 trace 的最后状态。
- 缓存 native59 计算出的每骨 local position/quaternion/scale，分别与首个 before 及实际 `_posBaseline/_rotBaseline/_sclBaseline` 通过现有固定 TRS 容差。失败不能用 actual before/after 替代计算结果。

`_lenBaseline` 和 `_positionBaseline` 是实测持久历史。它们不会因为候选 native59 改变而按长度、默认姿势或 bone center 重建；整个 transition 也不隐式清理历史。后者是历史位置向量，不能直接声称是当前表面法线或固定单位方向。

## 两种 Apply 计数模式

默认 `common_apply_count=None` 是**原记录验证模式**。每骨从该 cursor prefix 内第一个非 identity patch 开始独立计算调用次数；本次旧数据 early 为 6/7 次、late 为 26/27 次，而非拍摄请求中的 settle_frames。模型仅接受该记录的 native59 与四 patch 数值，输入改变便要求显式 candidate count。

新候选必须传 `common_apply_count=N`，其中 N 为 1..10000 的整数且不能是 bool。所有四骨应用同一预声明 N；禁止拟合 N 来缩小 target 或 snapshot 误差，也不允许用不同次数字典伪装共同协议。游戏验证时应按 observer **完成的各骨实际次数**截取精确 cursor；只等待 N 帧不能证明每骨调用恰好 N 次。

```python
import torch
from src.hs2_mesh_deform import HeadRig
from tools.stateful_multibone.adapter import MultiNativeBaselineProtocol, MultiStatefulTorchHeadRig

protocol = MultiNativeBaselineProtocol.from_manifest(
    manifest_path, contract_path, head_id=2, window='late', common_apply_count=8)
model = MultiStatefulTorchHeadRig(
    HeadRig(2, sampling_profile='slider_unlocker_18_2'),
    protocol, device='cuda', dtype=torch.float64)
native = torch.as_tensor(candidate_native59, dtype=torch.float64, device='cuda')[None]
ab = model.ab_tensor(candidate_four_patch_dict)  # 必须恰好四骨，字段齐全
vertices = model(native, ab)
local_trs, per_bone_local, per_bone_cache = model.local_states(native, ab)
```

native 和 FK/LBS 使用 float64；每骨 local 与 cache transition 使用 installed float32 算术，经独立 `apply_sequence` 传播，再将四骨结果写入 local table；继承 `TorchHeadRig.bone_world` 的 parent-first FK 后统一 skinning。没有先计算 world 再只替换父骨 world 的捷径，Chin 子骨会继承父骨变化。所有其他 ABMX slot 必须 identity；未知/缺少字典名、非有限参数、其他骨非 identity 均拒绝，不会回落静态 `pos*=length`。

输出 metadata 始终保留 `new_candidate_runtime_certified=false`。梯度可计算、记录回归通过，都不能自动认证任意候选，也不能证明几何质量或程儿相似度。

## 实际记录复验

```powershell
& ./.venv/Scripts/python.exe -B -m unittest discover -s tools/stateful_multibone -p test_protocol.py -v
& ./.venv/Scripts/python.exe -B tools/stateful_multibone/validate_recorded.py `
  ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_multibone_v1/live_cases.json `
  --contract outputs/abmx_replay_20261005/installed_contract_v2.json `
  --out outputs/stateful_multibone_20261005/new_cpu --device cpu
# --device cuda 使用另一个未存在的 --out 目录
```

验收逐 window 重新认证实际 o_head LBS/world candidate、缓存 mesh/bindpose/weight 对应、所有非覆盖 skin ancestors、paired geometry/PNG/frame/pose/visibility sampled signature；四骨 final local 与 private cache 使用现有固定容差；整头最大 L2 / actual bbox diagonal ≤1e-5。比较仅允许 proper rigid pose，以及**来自实际祖先矩阵测量**的 uniform scale，禁用拟合 scale/affine。实际 blendshape 与祖先 frame 是声明的 nuisance 输入，不是候选五官控制。

报告保存输入/实现/缓存 SHA；SHA 负责证据绑定，数值门槛负责证明具体状态对应。旧四 head × early/late 的复验不扩展到所有 30 骨、动态表情、未知写入、不同插件分支或任意长度分支。

## Fresh varied-native 最小采集协议

1. 预先写 immutable protocol JSON：具体 head、完整 varied native59、四 patches、预声明 common count N、早晚观察窗口及上述固定门槛；记录 sampling profile 与 bridge/ABMX 实际版本及 source contract。不要以真实 PNG 误差反推 N。
2. 游戏先恢复四骨 identity，再设置目标 native59，按安装插件实际生命周期完成 baseline 刷新；不私写 length/history/flags。开始四骨 observer，至少留一轮 identity call。核对四骨首个 before clean flags，保留其所有实际 cache 字段及 native59。若任一不干净，应拒绝这次证据并修正生命周期，不能在离线改 flag。
3. 一次统一下发四 patches，不穿插 native、表情或其他 modifier 写入。捕获包含 exact completed-call cursor 的 paired geometry 与三视图，停 observer 后导出完整 trace。每骨最后调用年龄 ≤1 帧、没有 pending/dropped/errors，trace 无外部边界。
4. 若测试“新候选 N 次”预测，必须实际 snapshot 的四骨 candidate prefix counts 都等于预声明 N，且初始 persistent history 与预测使用的 source-bound protocol 一致。现有 early/late自然帧捕获可验证实测次数预测，但不会因此证明预声明 N 的调度已经实现。
5. 离线运行上述 native baseline guard + CPU/CUDA full-head fixed gates。保存旧 protocol → new candidate 参数及 fresh trace/geometry 哈希关联；新 native、length<.1/zero-position 分支、其他骨需要各自新证据。完成后恢复游戏原始状态并写 manifest restored flags。

当前未完成的泛化门槛：fresh varied-native 四骨组实测、新候选共同 N 的实际调度对应、额外骨协议及受控长度分支、逐候选 mesh quality/cross-mesh/intersection 检查和目标人物相似度验收。文档与 adapter 不会自行解除人物制作暂停状态。
