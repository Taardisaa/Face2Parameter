# 程儿照片与当前原生底模脸部审查

## 2026-10-09 后续：用户授权 MCP 后期

用户选择先用游戏原生参数进行后期，而暂缓多照片表情分离重建。
已经从独立 v6 卡制作第一版后期：恢复原来 alpha=0 的眉毛/口红，
深棕虹膜、较细原生睫毛、偏分短发、蓝色挂颈裙，以及原生滑杆
5/10/48/49 的少量调整。完整配方和源码/资产依据在
[HS2 后期记录](../../HS2Mod/docs/hs2_chenger_postproduction.md)及其 JSON。

当前卡为 `E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_native_v6_postproduction_02.png`；
原 v6 卡保留，底模 bundle 与本仓库几何迁移代码没有修改。保存后原生
重载通过，全部 snapshot 区域一致。未将化妆或滑杆调整冒充完整唇形
迁移修复；非仿射唇形缺口仍存在，人物相似度和全动态兼容尚未完成。
头发侧面可见少量头皮穿出，需适配当前母版；颈部外观保持用户接受状态。

下文是后期前的审查依据，保留原判断和未完成问题。

2026-10-09。用户确认颈部明暗分界已在可接受范围；不再把它列作本轮
待修问题。当前 v6 的头部摆放及颈部保持，先对照真实照片，不改游戏。

## 所看材料与展示

查看原照片 img-001、img-002、img-020，三份原始 MICA 无贴图视图、
img-002 的原始网格投影板，以及实际游戏 v6 正面原生截图。
生成 ignored `outputs/chenger_native_photo_review_20261009/front_v1/photo_native_board.png`：
左原照片 img-002、右现有 v6。只以两眼参考点统一显示尺度、平面倾斜
与位置，不单独缩放横纵方向，不调整照片或模型局部形状。
照片点来自原始检测器；游戏截图点是明确记录的人工视觉估计。
不是相机标定，也没有消除照片的俯仰、偏航、透视、微笑和化妆差异。

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.chenger_native_photo_review --photo C:/Users/13666/Workspace/NewWriting/references/raw/characters/程儿/photos/img-002.jpg --mica-output outputs/chenger_mica_review_20261007/raw/face_0001_74014fe1504e.npz --game-image ../HS2Mod/artifacts/native_mother_20261008/balance_v6_final_0.png --game-eye-centers 201 333 353 333 --out outputs/chenger_native_photo_review_20261009/front_v1
```

输入字节 SHA、显示变换、锚点及局限记录在同目录 receipt。没有新推断、
新游戏采样、参数修改或模型修改；也没有开启离线渲染器项目。

## 当前可见差异与下一小步

1. 游戏嘴唇显得较薄，嘴角更平，嘴部横向范围偏小。照片的微笑与口红
   影响比较，但现有迁移确实有已知缺口：`mother_identity.coherent_mouth`
   用共同仿射位移保留原生闭唇/口内顺序，明确丢弃源 FLAME 非仿射唇部
   身份细节。v6 继承 `chenger_identity_v3` 的这个未完成预览分支，未因
   头颈修复而恢复细节。不能把所有嘴部差异单独归因于该分支或化妆。
2. 游戏眼睑轮廓和眼部开口与照片有差异；眉毛过浅，当前睫毛外观也
   不同。先区分实际眼睑开口、默认表情权重及材质/眉形的影响，不把
   所有可见差异直接判成三维骨相误差。
3. 下颌到下巴的收窄方式仍可疑，游戏的下巴较宽钝。img-001/020 提供
   额外可见轮廓参考，但照片头位、笑容与镜头未标定，暂不根据这个
   正面板直接削脸或宣称恢复模型/FLAME空间失败。

建议下一小目标：先完整对照源 MICA/FLAME 的唇缘与当前原生唇缘，查清
丢失的形状，并修复这段迁移，同时保留闭合口内层和完整原生表情。
然后再判断眼睑与下颌；眉形/唇色等外观也应忠实配置，不能成为遮盖
几何错误的补丁。三条比较链分别为照片→恢复网格、恢复网格→原生网格、
原生网格→游戏呈现；不能从最终不像就直接断定某一阶段必然失败。
