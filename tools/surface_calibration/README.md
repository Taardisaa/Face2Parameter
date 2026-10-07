# Surface calibration scaffolds

core/calibrate/probe 处理本地 JSON 与 NumPy，pixel_certificate 另读取实际 saved PNG 作独立测量，禁止从声明自动推断 PNG 纵轴、骨骼点语义或解剖对应。显式 `detect_cached.py` 可使用已经缓存的 FAN68/S3FD；缺权重则拒绝，禁用下载和 torch.compile。没有游戏连接或参数写入。

在 Face2Parameter 根目录执行：

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s tools\surface_calibration -p test_core.py -v
& .\.venv\Scripts\python.exe -B tools\surface_calibration\calibrate.py --help
```

`core.py` 提供：

- `Camera.from_capture(view, certification=..., pixel_certificate=..., diagnostic=False)`：解析实际 `capture_camera` 的 CPU view/projection、row-major/column-vector、CPU NDC near/far 和 pixelRect；不使用 Camera.main 或 GPU P。pixel_certificate 必须由 `pixel_certificate.certify_pixel_contract(report_path, view)` 独立验证后取得。
- `Camera.ray(xy)` / `project(world)`：top-left，整数 xy 为像素中心；viewport 使用 `(u+.5)/W` 与 `1-(v+.5)/H`，支持 offset pixelRect、正交、透视与 roll。
- `meshes_from_geometry(snapshot, candidate=..., certification=..., diagnostic=False)`：解析真实 BakeMesh 拓扑和 world candidates；默认拒绝未认证 policy。世界变换候选名称默认值只是明确选择，不能当作认证。
- `observe(camera, meshes, detector_point)`：对所有已启用/active mesh 计算 clip 内最近三角交点、cull 和几何遮挡。可通过 allowed_renderer_paths 限定目标；其他已显示 mesh 仍作为 occluder。返回 renderer path、submesh/triangle ID、原 vertex IDs、barycentric、world、几何法线与 occlusion。相同距离多交点标记 ambiguous。
- `follow(surface, deformed_meshes)`：保持 triangle+barycentric，拒绝 source hash、renderer 或拓扑顺序改变；这只追踪 material point。
- `validate_reprojection(surface, heldout_camera, meshes, observed_xy=...)`：重投影到未参与标定的 view，可比较独立 detector xy；不可见/遮挡时不给假误差。没有独立观察时仅报告投影，永不认证 anatomy。

`silhouette` 保留 view observation，不生成 surface/triangle/barycentric。该点类随视角变化，应使用轮廓约束。

## 输入与 CLI

capture 输入接受真实 bridge JSON，或 contact-sheet manifest 内的 `bridge_result`，使用 `--view-index` 选择原图。paired_geometry 的 SHA 若存在必须与 geometry 文件一致；姿态签名与 frame 对应由 bridge 提供，PNG flip 的 false 状态不会自动升级。

points JSON 必须先把 detector crop/resize/镜像变换逆回原图，使用如下合同：

```json
{
  "coordinate_convention": "top_left_pixel_centers_integer",
  "image_path": "C:/path/to/raw_view.png",
  "points": [
    {"id": "eye_corner_candidate", "xy": [123.5, 234.5], "kind": "surface", "allowed_renderer_paths": ["exact/renderer/path"]},
    {"id": "jaw_outline", "xy": [200.0, 300.0], "kind": "silhouette"}
  ]
}
```

首阶段真实 PNG flip/world-conversion 未认证时只能显式运行 diagnostic：

```powershell
& .\.venv\Scripts\python.exe -B tools\surface_calibration\calibrate.py --capture C:\path\capture.json --geometry C:\path\geometry.json --points C:\path\points.json --out C:\path\candidate_points.json --view-index 3 --world-candidate scale_free_trs --diagnostic-unvalidated
```

省略该开关时要求像素合同、pose pairing 和 world policy 各自满足合同。`--certification` 是几何证书：`evidence_id`、`capture_file_sha256`、`geometry_file_sha256` 必填且匹配文件，可包含 `world_policy_validated=true`、`world_candidate`、`mesh_source_hashes={renderer_path:source_hash}`。pixel_contract_validated=true 或 capture_camera.image_y_flip_validated=true 都不能认证 .5 像素中心；旧工具可显式 diagnostic 输出未认证候选。pose pairing 必须由 bridge 的 paired_pose_unchanged、before/after/geometry 非空签名和三个一致整数 frame 共同成立，任何外部 pose_pairing_validated 布尔值均不能绕过结构检查。

独立像素管线证据通过 `--pixel-report C:\path\report.json` 传入。逐 case 检查独立 top-left 方向与 .5 中心均 validated，绑定实际 metadata/PNG 的 SHA，重新读取 saved PNG 并运行独立 marker 支撑边缘/质心测量。报告 overall uncertain 可以包含可用的 AA1 validated case；AA4/8 的声明不会升级为实际 MSAA。hash 只绑定来源，重测才提供像素证据。

普通 maker_character 的坐标迁移目前仅允许已审阅 MVID `f7becf90-9a8d-4997-8188-d9d39f0f9bcc` 的共享路径：`MakerRenderService.RenderHead -> Camera.Render -> SaveRenderTexture -> Texture2D.ReadPixels -> EncodeToPNG`。Marker 分支只改变 cullingMask/clearFlags/background 并加入六个 unlit quad；它与普通分支共享相机、ARGB32 RT 和 PNG readback。源审阅文件 hash 与说明随证书输出，见 `pixel_certificate.py` 的 REVIEWED_PIPELINES；未来 MVID 必须重新审阅。

迁移严格匹配 MVID、D3D API、UV/reversed-Z、color space、RT format、HDR、AA、allowMSAA、requested/actual rendering path、quality AA、矩阵布局、CPU depth range、pixelRect、PNG尺寸、正交/透视、near/far、aspect、roll 与 CPU projection。camera world pose 可以随 view 变化；每个实际 view 的结构 pairing 独立检查。普通迁移仅支持实际测量 AA1 且 allowMSAA=false；不认证材质透明/深度、可见性、蒙皮或解剖。证书绑定目标 view payload 与目标 PNG，并在 Camera 使用时重查。

**证书属于外部实验的声明，不是这个脚本自动生成的证明。** 不能为了通过解析把未知状态改为 true。输出永远保留 `anatomical_correspondence_validated=false`，并记录输入 hash、证据 ID、各项认证状态。

现在的 geometry 文件未完整描述 camera layers 与材质 alpha/depth，因此 CLI 导入的 visibility 标记为 geometry-only；cull 未提供时为 unknown，并保留 uncertainties。由几何最近交点、单视图重投影或骨名不能推出解剖真值。未来应接入已验证的 mask/ID/depth 和独立多视角人工/算法证据。

当前测试使用合成相机、平面和 occluder，不证明真实游戏坐标或实际 detector 语义。完整计划见 HS2Mod 的 `docs/hs2_surface_calibration_plan.md`。

## 实际 paired 数据的诊断

`probe_capture.py --capture ... --geometry ... --out-dir ... --parity-report ...` 生成真实 mesh centroid 的投影/ray 可用性和无独立观察的持出 view 投影。parity 报告的 snapshot SHA 必须匹配输入；只有报告明确通过且 matching_candidates 包含所选项的 renderer 才有 world certificate。该证据只认证此 snapshot 的 skinning world policy。

`detect_cached.py --capture ... --geometry ... --certification ... --out ...` 在 raw PNG 上输出完整 68 点、真实 FAN heatmap peak、bbox，并对鼻尖/嘴角/眼角索引生成头表面候选及 front→heldout 误差/world drift。不同 triangle 的 barycentric 不能直接相减；只有同 renderer/triangle 才报告 barycentric L2 drift。FAN heatmap peak 不是校准的可见性概率，侧脸被遮挡点可能是预测补全。没有任何解析或模型成功率会认证解剖点。

早期实际结果保存于 `outputs/surface_calibration_20261004/`：七 view 的 pose pairing 一致，独立 LBS 报告支持该 snapshot 的 scale_free_trs；该次 capture 的 AA4 范围仍不满足像素中心认证。后续 `outputs/pixel_calibration_20261004/report_v3.json` 有四个 AA1 case 各自验证 top-left/.5；`HS2Mod/artifacts/infrastructure_live_20261004/paired_capture_aa1/response.json` 的七个普通 ortho512² view 均匹配其中 ortho_square_aa1 的严格 runtime/camera scope。此证据只解除该范围的 PNG 坐标未知，FAN crop/resize 的逆变换和解剖对应仍须独立验证。鼻/嘴候选的跨view漂移与眼角的几何遮挡不能升级成固定表面真值。
