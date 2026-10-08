# 原生底色迁移与固定表面区域

2026-10-08。把安装版 head2 的皮肤底色适配到当前 FLAME 头部，保持已经接受的
头形、脖子接口、三角形位置、几何法线及蒙皮。入口为
`tools.native_head.build --retarget-skin --native-eyes`；zipmod 版本 0.1.5。
这次迁移的是游戏皮肤底图，人物肤色仍走原生 `CreateFaceTexture` 与共享
`fileBody.skinColor`。它不是程儿照片的 albedo。

## 区域从哪里来，能否复用

`surface_regions.py` 使用原解码器的固定面编号、`FLAME_masks.pkl`、与解码器
对应的 `head_template.obj` UV，以及已经绑定 SHA 的 68 点 landmark embedding。
OBJ 的顶点索引和三角形顺序必须与实际解码器完全一致，不能用模板顶点替换
人物形状。区域只依赖这些固定参考数据，不依赖人物身份系数或变形后的坐标。

- 原生 mask 提供眼周、鼻、额头、头皮、左右耳、颈部、脸部等宽泛范围。
  不同范围可以重叠；面与 mask 的关联来自原始顶点成员比例。
- 宽泛的 `lips` 包含嘴周皮肤，不能当作唇红。本实现用固定参考 UV 上的
  外唇缘 48–59 与内唇缘 60–67，把它进一步分成唇红、嘴内投影区域及嘴周皮肤。
  标签按三角形中心确定；跨轮廓的三角形仍是边界，**不声称逐像素或逐子面
  精确分割，也不声称嘴内投影标签等于完整口腔的解剖分割**。
- 裁切和细分后的面沿 `crop_parent_faces → source_parent_faces` 追溯原始面。
  新建过渡带单列 `authored_neck_transition`，不假装具有原模型面标签。

每次构建输出 `surface_regions.json`：原始区域面编号、实际 `o_head` 区域面编号、
逐面来源以及拓扑、UV、mask、embedding 的指纹。之后可以直接查询嘴唇的面，
不再按当前坐标或肉眼临时猜测。

**相同 FLAME 面布局的另一个人物可以直接复用这套区域规则，已接入构建流程。**
裁切后的面依赖相应来源记录。任意网格、重排面编号或不同拓扑不能直接复用，
会拒绝不匹配。纹理对应配置目前适用于原生 head2 的这一套 atlas；其他原生
皮肤布局需要自己的对应配置，不宣称全部底模通用。

## 贴图怎样适配

`atlas_retarget.py` 保留源模型逐面 UV，以固定唇缘、眼部范围、耳部范围与
`native_atlas_profile.json` 的对应点，把它注册到原生底图。鼻／下颌对应点
属于明确记录的贴图制作选择，不是游戏隐藏计算或模型推断的真实解剖点。
原生唇色轮廓从该原生底图的声明区域提取；原图像素保持。

实际原生眼开口的 UV 在眼角折返，不能把其有序轮廓直接套成连续 FLAME chart。
这里使用它的 UV 边界矩形内的连续凸包轮廓作为制作对应。三角化的对应单元
如果发生折叠或退化，构建拒绝。原生口腔独立 tile 从外表 UV 中排除，头皮
不会再次采到顶端的口腔图案。

FLAME 的 U／游戏 X 方向与原生 head2 相反，注册时显式匹配方向。
后颈过渡按每条边取得相邻源面的 UV，再连到原生 collar 的 UV；不把 U=0、
U=1 的接缝副本取平均成 U=0.5，否则后颈会错误采到正面唇色。

UV 接缝通过 gather 复制原顶点，三角形顺序和每个三角形角的几何／法线／
骨权重保持不变。相应 tangent 按新 UV 生成。原生眼球 UV、材质和网格保持。

本版只迁移 `cf_head_02_00_t` 底色。直接带上原生 normal／AO 的失败 v5 已保留：
游戏里出现头侧条纹，说明底色注册不能顺带认证细节图迁移。本版恢复平坦
normal 和中性 AO；沿用均匀 `_NailMask`，关闭旧 atlas 的眉毛／nipple 高度门控
和微细节。身体资产未改，完整头身材质匹配仍是独立的未完成项。
改回平坦 normal 后头侧仍有既存分面阴影，未定位全部成因；不把该现象
全部归因于 normal 迁移。

## 重现与收尾

```powershell
.venv/Scripts/python.exe -m tools.native_head.build --source outputs/native_head_20261008/source_img002.json --audit outputs/head_base_audit_20261007/prefab_contract_v1.json --scale 9.851049 --translation 0.00339057157 0.458031476 0.603983462 --authored-neck outputs/native_head_20261007/placement_review_final_v1/chenger/size_and_pitch/geometry.npz --retarget-skin --native-eyes --out outputs/appearance_20261008/retarget_skin_fresh
.venv/Scripts/python.exe -m tools.native_head.verify_appearance_assets --previous outputs/appearance_20261008/authored_skin_v6 --built outputs/appearance_20261008/retarget_skin_fresh --out outputs/appearance_20261008/retarget_skin_fresh/appearance_static_checks.json
.venv/Scripts/python.exe -m tools.native_head.review_atlas --built outputs/appearance_20261008/retarget_skin_fresh
```

实际输出为 `outputs/appearance_20261008/retarget_skin_v6`。静态检查确认三角形角的
位置、法线、骨权重、骨编号和 bindpose 与接受版相同，底色原像素一致，正常
streamed texture URI 可解析，对应单元无折叠。`atlas_regions.png` 展示实际
贴图上的区域位置；`atlas_static_review.json` 保留原图像素指纹和分区数量。
安装资产、模型数据及生成图片保持 ignored；源代码、配置和证据元数据入 Git。
实际游戏加载及最终可见结果见 `hs2_native_skin_retarget_receipt.json`。

仍未完成：眉毛／睫毛／眼膜／牙齿与舌头配套、法线/AO 迁移、后侧分面阴影、
完整头身光照连续性、表情／所有滑杆兼容，以及照片纹理与人物相似度验证。
