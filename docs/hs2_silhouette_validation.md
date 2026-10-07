# 实际 PNG 与同步头部曲面轮廓

`tools/surface_calibration/silhouette.py` 把同次冻结导出的实际三角形，使用实际离屏相机 CPU 矩阵投影到原图像素中心。它同时输出 `o_head` 和全部启用面部 renderer 的不透明、双面三角形并集，两种假设分别报告，不挑误差较小者当成正确模型。没有按 view 拟合比例、移动相机、平移或重新裁剪。

```powershell
& .venv/Scripts/python.exe tools/surface_calibration/silhouette.py `
  --capture C:/Users/13666/Workspace/HS2Mod/artifacts/infrastructure_live_20261004/paired_headonly_aa1/response.json `
  --geometry C:/Users/13666/Workspace/HS2Mod/artifacts/infrastructure_live_20261004/paired_headonly_aa1/geometry.json `
  --parity-report outputs/unity_parity_20261004/paired_headonly_aa1.json `
  --pixel-report outputs/pixel_calibration_20261004/report_v3.json `
  --out outputs/silhouette_headonly_aa1_20261004
```

默认严格要求匹配真实 marker PNG 重测证书、运行模块 MVID、相机／readback scope、同帧 pose pairing，以及 snapshot SHA 绑定的逐 renderer 独立 LBS world-policy 报告。`--diagnostic-unvalidated` 只用于明确标记的未认证投影探索。已记录在报告中的 `pixel_contract_validated` 和 `pose_pairing_validated` 不由原图 metadata 的单个布尔值决定。

PNG 前景使用整条顶边实际一致的 RGB 作为背景，另要求至少 25% 图片精确符合此 RGB。前景阈值默认每通道差异大于 2；这是明确的图像分割假设，不是语义真值。填充内部孔洞后比较外轮廓，输出像素面积、IoU、双向边界 RMS／p95／max，overlay 中洋红为仅图像存在、青色为仅几何存在。三角形跨近平面／远平面时拒绝，不能静默漏掉；viewport 边界仅裁剪采样范围。

## 2026-10-04 实测

第一次普通采集包含颈部，导出只包含面部 renderer，因此 IoU 约 0.886–0.920，p95 约 57–87 px。检查 overlay 后，主要差异是未导出的身体／颈部。保留报告 `outputs/silhouette_aa1_20261004/report.json`，不把这个差异当作头部投影误差，也不以自动裁剪掩盖。

通过原有 capture `scene.hide_meshes=["o_body_cf"]` 临时隐藏实际存在的身体网格，再拍七个角度，生成第二份独立 paired geometry。报告 `outputs/silhouette_headonly_aa1_20261004/report.json`：

| yaw | head-only IoU | 双向边界 p95 px | max px |
|---|---:|---:|---:|
| -90 | 0.996061 | 2.828 | 3.606 |
| -60 | 0.998524 | 0 | 8 |
| -30 | 0.999991 | 0 | 1 |
| 0 | 0.999991 | 0 | 1 |
| 30 | 1 | 0 | 0 |
| 60 | 0.998545 | 0 | 7.071 |
| 90 | 0.999992 | 0 | 1 |

斜侧视图的小面积残差在远侧眼睛／睫毛处可见；纯 head union 没有完整材质可见性，全部 face-renderer union 也未实现睫毛 alpha。p95 为零不代表最大差异为零，报告保留两者。左侧 profile 有约 3 px 的局部轮廓差异，未通过改变相机或像素偏移消除。

这是一项真实 PNG 与三维曲面外轮廓对照，不是目标人物相似度、固定解剖表面点、材质 shader、连续曲面 Hausdorff 距离或完整底模表达力的认证。5 项解析测试覆盖已知矩形、双面绕序、可测平移、背景污染和 clipping／非法拓扑拒绝。
