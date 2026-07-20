# HS2 face shading recipe — 逆向出来的"游戏怎么把卡片画成脸"

> 目标:让 `src/render/` 忠实复刻,而不是"调参调到像"。
> **来源分级:** ✅ = 本机二进制/资产直接核实(ILSpy `Assembly-CSharp.dll` / UnityPy 读 bundle);
> ⏳ = 还没做的 DXBC 反编译;📚 = 外部资料(已标注出处)。
>
> 反编译命令(`ilspycmd` 已装):
> ```bash
> ASM="E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/Assembly-CSharp.dll"
> ilspycmd "$ASM" -t AIChara.ChaControl > ChaControl.cs      # 命名空间是 AIChara(HS2 沿用 AI 少女代码库)
> ilspycmd "$ASM" -t AIChara.CustomTextureCreate
> ilspycmd "$ASM" -t AIChara.ChaShader                       # 属性名映射表
> ```

## 0. 全景:两个材质,不是一个 ✅

脸的像素来自**两级**,新手最容易在这里搞错:

| | 作用 | 资产 | shader |
| --- | --- | --- | --- |
| **`matCreate`** | 把卡片选择**合成**成一张 2048² 贴图 | `chara/mm_base.unity3d` 的 `create_skin_face` / `create_skin detail_face` | `Create/skin color` / `Create/skin detail` |
| **`matDraw`** | 真正渲染头部网格 | 头部 bundle(`chara/38/fo_head_38.unity3d`)的 `cf_m_skin_head_NN` | `AIT/Skin True Face` |

`CustomTextureCreate.RebuildTextureAndSetMaterial()`(逐字核实):

```csharp
GL.sRGBWrite = true;                                  // ← 合成在 sRGB write 下做
Graphics.SetRenderTarget(createTex);                  // RenderTexture(2048,2048,0,ARGB32), useMipMap
GL.Clear(false, true, Color.clear);
Graphics.SetRenderTarget(null);
Graphics.Blit(texMain, createTex, matCreate, 0);      // 一趟 blit,pass 0
GL.sRGBWrite = sRGBWrite;
```

结果通过 `CustomTextureControl.SetNewCreateTexture(i, propId)` 塞回 `matDraw`:
`0 → _MainTex`(`ChaShader.SkinTex`),`1 → _DetailMainTex`(`SkinCreateDetailTex`)。

**一句话:`AIT/Skin True Face` 的 `_MainTex` 不是磁盘上任何一张贴图,而是运行时合成的 RT。**
所以 `src/render/composite.py` 必须先复刻 `Create/skin color`,`shading.py` 才有正确的 albedo 可用。

## 1. `create_skin_face` 的输入(= `Create/skin color` 的 uniform) ✅

`ChaControl.CreateFaceTexture()` 逐条核实;属性名来自 `ChaShader`。
**顺序即 C# 里的赋值顺序**;真正的混合次序在 shader 里(⏳ 待反编译)。

| 层 | 贴图 uniform | 颜色/参数 uniform | 数据来源(卡) |
| --- | --- | --- | --- |
| 基础皮肤 | `_MainTex` ← `ft_skin_f.MainTex` | `_Color` ← **`fileBody.skinColor`** | `skinId`(注意肤色在 **body** 段) |
| 眼影 | `_Texture11` ← `st_eyeshadow.AddTex` | `_Color11` ← `makeup.eyeshadowColor` | `makeup.eyeshadowId` |
| 腮红 | `_Texture10` ← `st_cheek.AddTex` | `_Color10` ← `makeup.cheekColor` | `makeup.cheekId` |
| 唇彩 | `_Texture9` ← `st_lip.AddTex` | `_Color9` ← `makeup.lipColor` | `makeup.lipId` |
| 痣 | `_Texture12` ← `st_mole.AddTex` | `_Color12`,`_Texture12UV`(布局) | `moleId` / `moleColor` / `moleLayout` |
| 纹身 ①② | `_Texture5` / `_Texture6` ← `st_paint.AddTex` | `_Color5/6`,`_Texture5UV/6UV`,`_Texture5Rotator/6Rotator` | `makeup.paintInfo[i]` |

**布局参数是 `Mathf.Lerp` 映射,必须照抄**(卡里存 0..1,shader 吃的是这些实数):

```csharp
// 痣 (MoleLayout -> _Texture12UV)
v.x = Lerp(5f, 1f, moleLayout.x);      v.y = Lerp(5f, 1f, moleLayout.y);      // 缩放(反向)
v.z = Lerp(0.3f, -0.3f, moleLayout.z); v.w = Lerp(0.3f, -0.3f, moleLayout.w); // 平移
// 纹身 (Paint0NLayout -> _Texture5UV/_Texture6UV, Paint0NRot -> _TextureNRotator)
v.x = Lerp(10f, 1f, layout.x);         v.y = Lerp(10f, 1f, layout.y);
v.z = Lerp(0.28f, -0.3f, layout.z);    v.w = Lerp(0.28f, -0.3f, layout.w);
rot = Lerp(1f, -1f, rotation);
```

detail 通道(`create_skin detail_face` → `_DetailMainTex`)吃的是同样的层,但取各 list 的
**`GlossTex`** 而不是 `AddTex`,主贴图是 `chara/etc.unity3d` 里的 **`black2048`**;
另加 `_Gloss9/10/11`(唇/腮红/眼影光泽)、`_Gloss5/6` + `_Metallic5/6`(纹身)。

## 1b. `Create/skin color` 的**实际混合数学** ✅(DXBC 反编译,2026-07-19)

反编译产物在 `data/hs2_head/shaders/Create/skin color.shader`(gitignore)。译回可读形式:

```glsl
// ---- ① 基础皮肤:HSV 空间重着色,不是简单的 albedo * _Color ----
base   = tex2D(_MainTex, uv);                       // 卡选肤色贴图
skinC  = lerp(_Color.rgb, _Color7.rgb, tex2D(_Texture7,uv).r * _Color7.a);   // 晒痕
skinC  = lerp(skinC, _Color13.rgb, tex2D(_NailMask,uv).r);                   // 指甲区(脸上恒 0)
hm,sm,vm = RGBtoHSV(base);       ht,st,vt = RGBtoHSV(skinC);
rgb    = HSVtoRGB( h = |hm| + |ht|,  s = sm + st - 0.5,  v = (vm - 0.1) * vt );
dst    = float4(rgb, 0);                            // 注意 alpha 起始为 0

// ---- ② 叠层,严格按此顺序(shader 说了算,不是 C# 的赋值顺序)----
// 纹身 ①②(带旋转):
uv' = rot2D(uv + _TextureNUV.zw - 0.5, _TextureNRotator * PI) + 0.5;
uv' = uv' * _TextureNUV.xy + (1 - _TextureNUV.xy) * 0.5;      // 绕中心缩放
t   = tex2D(_TextureN, uv');
col = lerp(t.r, t.r * _ColorN, t.b);                          // t.b = 上色遮罩
a   = t.a * _ColorN.a * tex2D(_PaintMask, uv).r;
dst = lerp(dst, col, a);
// 腮红 _Texture10 → 唇彩 _Texture9 → 眼影 _Texture11 → 痣 _Texture12(都不带旋转):
uv' = (uv + _TextureNUV.zw) * _TextureNUV.xy + (1 - _TextureNUV.xy) * 0.5;
t   = tex2D(_TextureN, uv');
dst = lerp(dst, t.r * _ColorN, t.a * _ColorN.a);
```

**只有反编译才能知道的事实:**
1. 基础皮肤是 **HSV 加性重着色**(色相相加、饱和度相加减 0.5、明度相乘),照"albedo × _Color"写会错。
2. **混合顺序 = 纹身 → 腮红 → 唇彩 → 眼影 → 痣**,与 `CreateFaceTexture()` 里的赋值顺序
   (眼影→纹身→腮红→唇彩→痣)**不同**。眼影盖在唇彩之上,痣在最后。
3. 每层只用贴图的 **`.r` 当颜色、`.a` 当遮罩**(纹身多用 `.b` 当上色遮罩)。

### ⚠️ 必须在**线性空间**里算(实测踩过的坑)

`RebuildTextureAndSetMaterial` 里那句 `GL.sRGBWrite = true` 在 gamma 色彩空间下是 no-op ——
它出现在这里,说明**游戏跑的是 Linear 色彩空间**。于是 sRGB 标记的贴图采样出来是**线性值**。

直接拿 sRGB 字节套上面的公式会**把脸洗成灰白**:实测皮肤贴图 `S=0.215`、`skinColor S=0.200`,
`S = 0.215 + 0.200 − 0.5 = −0.085` → 负饱和度。转到线性空间后 `S = 0.416 + 0.393 − 0.5 = 0.309`,
是正常肤色饱和度;而且色相 `0.050 + 0.057 − 0.05(HSVtoRGB 内建偏移) = 0.057` 几乎精确回到贴图原色相
—— 这个巧合度足以佐证解读正确。

**色彩空间是逐贴图的,不能一刀切**:`Texture2D.m_ColorSpace`(1=sRGB, 0=Linear)实测
皮肤 `cf_head_02_00_t`=**1**、唇彩 `c_t_lip_00_03`=**0**、`paint_mask_face`=**0**。
提取器现在把这个标记写进 `textures/pool.json` 与卡清单,`composite.py` 逐张遵守。
材质颜色(`SetColor`)由 Unity 做 sRGB→linear,也要转。

### ✅ 已用游戏 ground truth 判决(2026-07-19)

`/maker/facetex` dump 出游戏自己合成的 2048² RT,与 `composite.py` 逐像素比对
(`scripts/hs2_capture_gt.py --compare-facetex`):

```
L1 = 0.0490   p95 = 0.0547   PSNR = 26.16 dB
逐通道线性回归: game = 0.457*ours + 0.295 (R) / 0.457*ours + 0.263 (G) / 0.458*ours + 0.246 (B)
相关系数        R 0.9975   G 0.9956   B 0.9985
```

**结论:结构完全正确**——三通道斜率一致(不是色偏)、相关系数 0.996+(不是位置或分层错误)。
HSV 重着色、各层 UV 布局、混合顺序、线性空间假设**全部得到验证**。

**剩余唯一偏差 = 一个全局标量。** 线性空间下 `game / ours` 的中位数:

| 区域 | 比值 |
| --- | --- |
| 纯皮肤 | 0.8257 (IQR 0.8236–0.8287) |
| 妆容覆盖区(2.0% texel) | 0.8239 |
| 左上 / 右上 / 左下 / 右下象限 | 0.8256 / 0.8256 / 0.8258 / 0.8258 |

四个象限一致到小数点后三位,妆容区与皮肤区一致 ⇒ **不是任何一层的公式问题,是一个乘性常数**
(我们偏亮约 1/0.826 = 1.21 倍)。来源待查(候选:`AIT/Skin True Face` 片元里那句
`tmp9.xyw * float3(0.8,0.8,0.8)`;或 RT 回读路径的一次转换)。
在查清之前**不要**把它硬编码进 `composite.py` —— 它是一个可测量的标量,
放进 Phase 4 的标定环节比手调更诚实。

### 另一条被 ground truth 确认的

`/maker/lights` 报告 **`color_space = Linear`** —— 印证了 §1b 里"必须在线性空间算"的推断
(此前只是从 `GL.sRGBWrite = true` 反推)。
同时 `/maker/facetex` 报告 `draw_material = cf_m_skin_head_02`、`draw_shader = AIT/Skin True Face`,
印证了"运行时用 list 的 MatData 替换 prefab 自带材质"这一条。

`Create/skin detail` → `_DetailMainTex`,通道含义 **`.r`=metallic `.g`=法线强度 `.b`=gloss**:

```glsl
p1 = tex2D(_Texture5, uv_rot5);  p2 = tex2D(_Texture6, uv_rot6);   // 同上的 UV 变换
v1 = float4(min(p1.a*_Color5.a, _Metallic5), p1.g*_BumpScale5, min(p1.a*_Color5.a, _Gloss5), p1.a*_Color5.a);
v2 = ... 同理 ...
dst = max(max(v1, v2), tex2D(_MainTex, uv)) * tex2D(_PaintMask, uv).r;   // 脸的 _MainTex = black2048
gloss = max(tex2D(_Texture10,uv').a*_Gloss10,
        max(tex2D(_Texture9, uv').a*_Gloss9,
            tex2D(_Texture11,uv').a*_Gloss11));                   // 腮红/唇/眼影光泽
o = max(dst, float4(0, 0, gloss, 0));
```

## 2. 直接写在 `matDraw`(渲染材质)上的东西 ✅

**不经过合成**,是渲染 shader 自己的层——这解释了为什么 `Create/skin color` 的属性表里没有眉毛:

| 项 | uniform | 来源 |
| --- | --- | --- |
| **眉毛** | `_Texture3` ← `st_eyebrow.AddTex`;`_Color3` ← `eyebrowColor` | `ChangeEyebrowKind/Color` |
| 眉毛布局 | `_Texture3UV` | `ChangeEyebrowLayout`(见下) |
| 眉毛倾斜 | `_Texture3Rotator` ← `Lerp(-0.15f, 0.15f, eyebrowTilt)` | `ChangeEyebrowTilt` |
| 遮蔽 | `_OcclusionMap` ← `ft_skin_f.OcclusionMapTex` | `CreateFaceTexture` |
| 法线 | `_BumpMap` ← `ft_skin_f.NormalMapTex` | 同上 |
| 皮肤 detail | `_BumpMap2` ← `ft_detail_f.AddTex`,强度 `_BumpScale2` | `ChangeFaceDetailKind/Power` |
| 光泽 | `_Gloss` ← `Lerp(0f, 0.8f, skinGlossPower) + 0.2f * skinTuyaRate` | `ChangeFaceGlossPower` |
| 金属度 | `_Metallic` ← `fileBody.skinMetallicPower` | `ChangeFaceMetallicPower` |

```csharp
// 眉毛布局 (EyebrowLayout -> _Texture3UV)
v.x = Lerp(-0.2f, 0.2f, eyebrowLayout.x);  v.y = Lerp(0.16f, 0f, eyebrowLayout.y);
v.z = Lerp(2f, 0.5f, eyebrowLayout.z);     v.w = Lerp(2f, 0.5f, eyebrowLayout.w);
```

## 2b. ⚠️ 通用陷阱:遮罩几乎都**不在 alpha 里**

移植时连踩三次的同一个坑 —— HS2 的图层贴图 **alpha 常年恒为 1**,形状信息藏在某个颜色通道。
拿 alpha 当遮罩 = 整个区域被刷成该层的颜色(实测:眉毛把整张脸刷黑、瞳孔把整个眼球刷黑)。
实测各贴图的通道统计:

| 层 | 贴图 | R | G | B | A | 遮罩通道 | 依据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 眉毛 | `c_t_eyebrow_00` | .016 | .016 | .016 | **1.000** | **`.b`** | ✅ 反编译 `tmp9.z * _Color3.w` |
| 虹膜 | `c_t_eye_08` | .109 | — | .227 | **1.000** | **`.b`** | ✅ 反编译 `tmp5.xxxz * _Color2` |
| 瞳孔 | `c_t_eyeblack_00` | **1.000** | — | **1.000** | .432 | `.a` | ⏳ 推断(RGB 全白,形状只在 alpha) |
| 高光 | `c_t_eyehigh_06` | .008 | — | .008 | **1.000** | `.r` | ⏳ 推断(alpha 恒 1) |
| 睫毛 | `c_t_eyelash_01` | .187 | **.812** | .314 | **1.000** | ? | ⏳ 未定,当前用 alpha → **会糊成一片** |

**另一半:顶点色门控。** 眉毛的最终 alpha 是 `brow.b * _Color3.a * (1 - vertexColor.b)` ——
`o_head` 的顶点色 `.b` 通道标出了眉毛能出现的区域(实测 4439 顶点里 363 个被标记)。
所以眉毛**不能预先烘进合成贴图**,必须逐像素、拿插值后的顶点色去算。
提取器现在会把顶点色从 0-255 归一化到 0-1(Unity 存的是字节)。

## 3. 眼睛 / 睫毛 ✅(属性映射)

`o_eyebase_L/R` 用 `AIT/Eye Translucency`,一个材质里叠三层:

| 层 | 贴图 | 颜色 | 布局 |
| --- | --- | --- | --- |
| 巩膜(眼白) | `_MainTex`(`c_t_eye_white_01`,in-bundle) | `_Color` ← `pupil[].whiteColor` | — |
| 虹膜 | `_Texture2` ← `st_eye.AddTex` | `_Color2` ← `pupilColor`;`_Emission` | `_texture2uv` |
| 瞳孔 | `_Texture3` ← `st_eyeblack.AddTex` | `_Color3` | `_texture3uv` |
| 高光 | `_Texture4` ← `st_eye_hl.AddTex` | `_Color4` ← `hlColor` | `_Texture4UV`,`_Texture4Rotator` |

高光开关复用 `_Smoothness`(`EyesHighlightOnOff`),眼影范围是 `_ShadowScale`(`whiteShadowScale`)。
睫毛(`AIT/eyelashes`)最简单:`_MainTex` ← `st_eyelash.AddTex`,`_Color` ← `eyelashesColor`,
外加 `_Cutoff` / `_CutoutScale`。

## 4. 渲染状态 ✅(UnityPy 读 `m_ParsedForm`,本机 `fo_head_38.unity3d`)

全是 **Amplify Shader Editor 生成的 Unity Standard surface shader**
(`m_CustomEditorName = ASEMaterialInspector`),d3d11 单平台。

| shader | queue / RenderType | Blend | ZWrite | Cull | pass |
| --- | --- | --- | --- | --- | --- |
| `AIT/Skin True Face` | AlphaTest-40 / TransparentCutout | One Zero | On | Back | FORWARDBASE + FORWARDADD + ShadowCaster |
| `AIT/Eye Translucency` | Geometry+0 / Opaque | One Zero | On | Back | 同上 |
| `AIT/eyelashes` | Transparent+0 / TransparentCutout | **One OneMinusSrcAlpha**(预乘) | On | **Off** | FORWARDBASE(+Shadow) |
| `AIT/main eyeshadow lambert` | AlphaTest+0 / Opaque | **SrcAlpha Zero** | On | Back | FORWARDBASE |
| `AIT/main namida` | — | 带 refraction(GrabPass) | — | — | — |

⇒ 绘制顺序:不透明(skin / eye / tooth / tang)→ 睫毛 → 眼影 / 泪膜。

## 5. 光照模型 📚(ASE 模板,⏳ 待 DXBC 反编译逐项核对)

基础 = stock `LightingStandard`(`BRDF1_Unity_PBS`:GGX + Smith-Joint 可见性 + Schlick Fresnel,
metallic workflow),**外加一项 translucency**(Barré-Brisebois / DICE 快速次表面近似)。
ASE `StandardSurface.cs` 生成的原文:

```hlsl
#if !DIRECTIONAL
  float3 lightAtten = gi.light.color;
#else
  float3 lightAtten = lerp( _LightColor0.rgb, gi.light.color, _TransShadow );
#endif
half3 lightDir     = gi.light.dir + s.Normal * _TransNormalDistortion;
half  transVdotL   = pow( saturate( dot( viewDir, -lightDir ) ), _TransScattering );
half3 translucency = lightAtten * ( transVdotL * _TransDirect
                                  + gi.indirect.diffuse * _TransAmbient ) * s.Translucency;
half4 c = half4( s.Albedo * translucency * _Translucency, 0 );
return LightingStandard( r, viewDir, gi ) + c;      // 加性,叠在标准 BRDF 之上
```

- `_Trans*` 六个属性是 ASE 连上 Translucency 端口时**自动生成**的,取值范围与本机 dump 完全一致
  (`_Translucency` 0–50、`_TransNormalDistortion` 0–1、`_TransScattering` 1–50、其余 0–1)。
- 这个端口会强制 `exclude_path:deferred` ⇒ **皮肤/眼睛恒走 forward**,与 §4 的 pass 表吻合。
- 每个材质的真实数值已在 `data/hs2_head/head_<id>/materials/*.json`(如 `cf_m_skin_head_02.json`)。
- 社区没有 HS2 `AIT/Skin*` 的开源反编译(Koikatsu 的 `Shader Forge/*` 是另一套 toon,不通用;
  Hanmen 的 skin/eye 替代 shader 闭源付费)。可读的同族参考只有
  [Hanmen-lab/HS2-AI-ASE-Shaders](https://github.com/Hanmen-lab/HS2-AI-ASE-Shaders)(仅衣服/道具)。

## 6. `AIT/Skin True Face` 片元的结构 ✅(已反编译,细节待 Phase 3 逐段移植)

`data/hs2_head/shaders/AIT/Skin True Face.shader`,15099 行 = 3 个 pass
(FORWARDBASE / FORWARDADD / ShadowCaster)× 各 8–13 个 keyword 变体
(`DIRECTIONAL` × `INSTANCING_ON` / `LIGHTPROBE_SH` / `SHADOWS_SCREEN` / `VERTEXLIGHT_ON`)。
离线渲染只需 **FORWARDBASE + DIRECTIONAL + LIGHTPROBE_SH**(无阴影、无顶点光)这一支。

片元的前半段(≈150 行)全在**堆法线**,这是肉眼绝对猜不出来的一点:

- `_DetailGlossMap`、`_Texture2`、**`_Texture3`(眉毛)**、`_Texture5` 四层都不是直接读法线贴图,
  而是**对绿通道做有限差分求高度梯度**(UV 偏移 `0.0015625 = 1/640`,眉毛那层用 `0.0054872`),
  再 `normalize(float3(0,0,1) - gradient)` 得到切线空间法线;
- 然后依次与 `_BumpMap`(× `_BumpScale`)做 **whiteout 式法线叠加**(逐层 `xy` 相加、`z` 相乘再归一化);
- 眉毛层的 UV 用的是 **`texcoord.zw`(UV1)**,绕 **(0.3, 0.4)** 旋转 `_Texture3Rotator*PI`,
  再 `uv*_Texture3UV.zw + (1-_Texture3UV.zw)*(-0.1,-0.43)` —— 与 §1b 合成层的"绕 0.5 缩放"**不同**,
  照抄合成层的约定会让眉毛错位。
- 之后是 `_WeatheringMask` / `_WeatheringMap`(带 `1-mask.a` 驱动的旋转)——脸上基本不用,可先略。

光照部分是 §5 的 stock `BRDF1_Unity_PBS` + 加性 translucency,与 ASE 模板一致。

## 7. 还缺什么(⏳)

1. **Maker 灯光 rig**:不在 `Assembly-CSharp` 里(`CustomControl` 无 `Light`/`RenderSettings` 引用),
   在 CharaCustom 场景资产中 → 用 `HS2_McpBridge` 的 `/maker/lights` 端点运行时读(Phase 4)。
2. **验证捷径**:hook `CustomTextureCreate.RebuildTextureAndSetMaterial` 把合成好的 2048² RT 存盘,
   就能对 `composite.py` 做**逐像素**比对,而不是靠肉眼。
3. `AIT/main namida`(泪膜)带 GrabPass 折射,未反编译;先按普通半透明近似,记为已知偏差。

## 附:如何复现这些反编译产物

```bash
# 1) 托管 C#(ilspycmd 已装;命名空间 AIChara)
ASM="E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/Assembly-CSharp.dll"
ilspycmd "$ASM" -t AIChara.ChaControl > ChaControl.cs

# 2) DXBC -> HLSL(USCSandbox,需 dotnet SDK;classdata.tpk 要放在 exe 同目录)
git clone --depth 1 https://github.com/nesrak1/USCSandbox.git && cd USCSandbox
dotnet build -c Release && cd USCSandbox/bin/Release/net8.0 && cp ../../../../Files/classdata.tpk .
./USCSandbox.exe <bundle.unity3d>                      # 列出 bundle 内的 CAB 名
./USCSandbox.exe <bundle.unity3d> <CAB-...>            # 列出 shader 及 path id
./USCSandbox.exe <bundle.unity3d> <CAB-...> <pathid> --platform d3d11   # 输出到 ./out/
```

产物已缓存到 `data/hs2_head/shaders/`(gitignore,2.3 MB)。

---

## 2026-07-20 · 消融验证工具,以及被它推翻的四个结论

### 新增能力(bridge v0.15.0)

反编译能告诉你 shader **写了什么**,但不能告诉你某一项在最终图像里**贡献了多少**。
后者才是移植正确性的判据。两个端点补上这一环:

- `POST /maker/material {mesh, property, value|color, restore}` —— 在游戏里把某个 shader 项归零,
  与正常截图相减 = **该项的精确贡献**。
- `GET /maker/render?hide_meshes=o_eyelashes` —— 按网格名隐藏 renderer,两边用**同样方式**做
  有/无差分来隔离图层。

Python 侧:`scripts/hs2_capture_gt.py::aligned_framing()`(相机对齐,必须先用)。

### 已用它定案的事实

| 项 | 游戏的真实贡献(消融测得) | 结论 |
| --- | --- | --- |
| `_Translucency` | 40% 像素,+[21.4, 14.1, 11.3],覆盖全脸 | **暖色主来源**;我们缺它 → 偏冷偏灰 |
| `_Gloss`(镜面) | 18% 像素,平均仅 **[−1.05, −0.77, −0.61]** | 游戏镜面近乎可忽略;我们强 **16×**,已关停 |
| 睫毛图层 | ink@40 = 0.320 | 我们 ×1 采样时 1.168、**×2 超采样 0.943** |

### 睫毛:shader 是对的,问题是抗锯齿

`AIT/eyelashes` 完整数学(逐行转录,已验证):

```
albedo = t.R × _Color.rgb                         // R 通道,不是 alpha
alpha  = min(t.R^k, 1),  k = (1−_Color.a)×2 + _CutoutScale
a2     = alpha × saturate(2×_Color.a)
if (max(Bayer4x4抖动, a2) − _Cutoff < 0) discard  // 我们原先完全没有这个测试
out.rgb = albedo × (atten × _LightColor0 × max(dot(N,L),0) + ambient)   // 普通 Lambert,有光照
out.a   = alpha
```

(recipe 上文那条"睫毛遮罩通道 ⏳ 未定"据此**关闭**:是 **R**。)

游戏通过 **4× MSAA** 的 RenderTexture 截图。我们渲走样图,把睫毛压成"过深的核心 + 缺失的柔和过渡" ——
这就是肉眼看到的"浓"。`src/render/scene.py::render(ss=2)` 默认 2× 超采样,成本约 0(36.3 vs 35.9 ms)。

### ⚠️ 被推翻的四个结论(记下来避免重犯)

1. ~~"我们的睫毛多 44% 墨水"~~ —— 指标用**各自图像**的局部皮肤亮度作基准,而两边皮肤差 13%。
2. ~~"超采样没用"~~ —— 用的就是上面那个坏指标。
3. ~~"我们覆盖面积小 8 倍"~~ —— 游戏侧差分图里混着**整张脸的残差**(鼻/唇/耳都可见),
   低阈值量到的大部分不是睫毛;只有 |Δ|>40 才在比睫毛。
4. ~~"眉毛和睫毛都暗 24–38%"~~ —— 阈值指标把"强度"和"扩散"混为一谈。

**教训:数字反直觉时,先验证指标,再解释结果。** 在一个本身错误的指标上测掉了四个假设。

### ⚠️ 第三次踩同一个坑:提取的资产 ≠ 运行时真值

`_Smoothness` 在提取的材质 JSON 里存在,但**运行时材质没有这个属性**(`/maker/material` 返回 404);
真正的是 `_Gloss`。前两次是①眉毛布局(材质存默认值,运行时由卡片经 Lerp 算出)、
②皮肤材质运行时被替换。**移植前先用 `/maker/material` 问游戏,不要信 JSON。**

### 未完成:皮肤 translucency

`AIT/Skin True Face` 的 translucency 项结构(已从反编译读出):

```
mask      = _NailMask.a                    // tmp5 = tex2D(_NailMask, uv),tmp5.w 存活到最后
transColor= RGB→HSV→RGB 往返,源色由 _Color / _Color13 / _Color4 混合
                                           // _Color4 = [0.5, 0, 0, 0.66] 是血红色
trans     = diffColor × transColor × mask
            × lerp(Lcol, Lcol×atten, _TransShadow)
            × (VdotL^_TransScattering × _TransDirect + indirect × _TransAmbient)
            × _Translucency(=30)
```

**还差**:HSV 链里 `tmp1.w`(明度修正)与源色混合权重的最后几跳。
`_NailMask.a` 均值 0.58 × 30 = 17.4,量级明显不对,说明链条里还有衰减未读到。
**验收标准已就位**:用消融隔离出我们的 translucency 贡献,与游戏的 +[21.4, 14.1, 11.3] / 40% 对比。
