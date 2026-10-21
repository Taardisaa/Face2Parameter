// Full render-scene state export for the frame-pair probe v5 (read-only).
//
// Reflects the public Unity API surface of cameras, lights, renderers, probes,
// RenderSettings/QualitySettings/GraphicsSettings/Screen/Time/ReflectionProbe/GL,
// ALL instance fields of every MonoBehaviour on cameras and lights plus every
// effect/volume component (DHH, CTAA, PostProcessing v2, liquid), the static
// fields of the DHH, CTAA and PPv2 assemblies, and the plugin member paths that
// the generators name explicitly. Native state the managed API does not expose
// (shader property sheets with their raw GPU-bound bytes, global keyword bits,
// the temporary RenderTexture pool, camera previous VP) comes from UnityNative.
//
// v5 changes against v4 (each was a reported gap or a v4 defect):
//  * RenderTexture.colorBuffer/depthBuffer are no longer invoked: their native
//    RenderTextureScripting::GetColorBuffer (UnityPlayer 0x705390) calls
//    RenderTexture::Create, so v4's reflection could create never-created RTs.
//  * Shader.IsKeywordEnabled is no longer called: it runs keywords::Create
//    (0x712040 -> 0x82bac0), which registers unknown names in the 256-entry
//    keyword table. Global keywords are read natively instead.
//  * Managed material/global getters are only called for names already in the
//    native FastPropertyName table (PropertyToID would register new names).
//  * Reflection depth 12 (v4: 6) with a cycle guard along the reflection path, and an
//    AnimationCurve case (keys + wrap modes) so PPv2 Spline curves, Spline.cachedData
//    and MultiScaleVO.m_AmbientOnlyAO are reached. Output formats are v4's.
//  * Light/camera transforms, Behaviour.enabled of every effect component, all
//    RenderTexture objects (pooled temporaries and camera targets included).
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace HS2Codex.FramePairProbe
{
    internal sealed class ShaderManifest
    {
        // shader name -> list of (property, type, arraySize); type in float|vector|color|matrix|texture|int
        public readonly Dictionary<string, List<string[]>> Properties = new Dictionary<string, List<string[]>>();
        public readonly List<string> Keywords = new List<string>();
        public readonly List<string[]> Globals = new List<string[]>();
        public string Path, Sha256;

        // Plain line format (no JSON dependency); '#' lines are provenance comments:
        //   P<TAB>shader<TAB>name<TAB>type<TAB>arraySize
        //   K<TAB>keyword
        //   G<TAB>name<TAB>type<TAB>arraySize
        public static ShaderManifest Load(string path)
        {
            var m = new ShaderManifest { Path = path };
            var bytes = File.ReadAllBytes(path);
            using (var h = System.Security.Cryptography.SHA256.Create()) m.Sha256 = Mem.Hex(h.ComputeHash(bytes));
            foreach (var raw in System.Text.Encoding.UTF8.GetString(bytes).Split('\n'))
            {
                var line = raw.TrimEnd('\r');
                if (line.Length == 0 || line[0] == '#') continue;
                var f = line.Split('\t');
                if (f[0] == "P" && f.Length == 5)
                {
                    List<string[]> list;
                    if (!m.Properties.TryGetValue(f[1], out list)) m.Properties[f[1]] = list = new List<string[]>();
                    list.Add(new[] { f[2], f[3], f[4] });
                }
                else if (f[0] == "K" && f.Length == 2) m.Keywords.Add(f[1]);
                else if (f[0] == "G" && f.Length == 4) m.Globals.Add(new[] { f[1], f[2], f[3] });
                else throw new FormatException("Bad manifest line: " + line);
            }
            if (m.Properties.Count == 0) throw new FormatException("Shader manifest has no properties: " + path);
            return m;
        }
    }

    internal sealed class RefEq : IEqualityComparer<object>
    {
        public static readonly RefEq Instance = new RefEq();
        public new bool Equals(object a, object b) { return ReferenceEquals(a, b); }
        public int GetHashCode(object o) { return RuntimeHelpers.GetHashCode(o); }
    }

    internal sealed class ExportContext
    {
        public readonly Dictionary<int, Texture> Textures = new Dictionary<int, Texture>();
        public readonly Dictionary<int, Material> Materials = new Dictionary<int, Material>();
        public readonly HashSet<object> Path = new HashSet<object>(RefEq.Instance);   // reflection ancestors (cycle guard)
        public UnityNative Native;
        public ShaderManifest Manifest;
        public HashSet<string> RegisteredNames = new HashSet<string>(StringComparer.Ordinal);
        public readonly HashSet<int> SheetTextureIds = new HashSet<int>();   // native TextureIDs bound in material sheets
        public readonly HashSet<int> GlobalTextureIds = new HashSet<int>();  // native TextureIDs bound in the global sheet (any callback)

        public SheetSnapshot GlobalSheet()
        {
            var sheet = Native.GlobalSheet();
            foreach (var e in sheet.Entries)
                if (e.Type == "texture") { int id = BitConverter.ToInt32(e.Value, 0); if (id != 0) GlobalTextureIds.Add(id); }
            return sheet;
        }

        public ExportContext(UnityNative native, ShaderManifest manifest)
        {
            Native = native; Manifest = manifest;
            RefreshNames();
        }

        // Native FastPropertyName and keyword tables grow when shaders/scripts register
        // names; re-read once per exported frame.
        public void RefreshNames()
        {
            Native.RefreshNames();
            RegisteredNames.Clear();
            foreach (var n in Native.PropertyNames) if (n != null) RegisteredNames.Add(n);
        }

        public void Add(Texture t) { if (t != null) Textures[t.GetInstanceID()] = t; }
        public void Add(Material m) { if (m != null) Materials[m.GetInstanceID()] = m; }

        // Materials/textures are collected across frames (renderers, effect fields, the
        // recorder). Some are runtime instances that plugins destroy and recreate (session 2
        // window 574f36ad: a material referenced in frames 0-1 was destroyed before frame 2;
        // PPv2 held two 'PostProcess - CopyStd' instances). A destroyed object must never be
        // dereferenced (Object.name throws) and is no longer part of the scene: drop it and
        // report its id. Keyed by instance id, so a dropped id is never re-added.
        public readonly List<int> DroppedDestroyed = new List<int>();
        public List<int> PruneDestroyed()
        {
            var dropped = new List<int>();
            foreach (var kv in Materials.ToArray()) if (kv.Value == null) { Materials.Remove(kv.Key); dropped.Add(kv.Key); }
            foreach (var kv in Textures.ToArray()) if (kv.Value == null) { Textures.Remove(kv.Key); dropped.Add(kv.Key); }
            DroppedDestroyed.AddRange(dropped);
            return dropped;
        }
    }

    internal static class SceneState
    {
        static readonly Dictionary<string, string> SkippedGetters = new Dictionary<string, string>
        {
            { "UnityEngine.Renderer.material", "instantiating getter" }, { "UnityEngine.Renderer.materials", "instantiating getter" },
            { "UnityEngine.MeshFilter.mesh", "instantiating getter" }, { "UnityEngine.Collider.material", "instantiating getter" },
            { "UnityEngine.SkinnedMeshRenderer.material", "instantiating getter" }, { "UnityEngine.SkinnedMeshRenderer.materials", "instantiating getter" },
            { "UnityEngine.MeshRenderer.material", "instantiating getter" }, { "UnityEngine.MeshRenderer.materials", "instantiating getter" },
            { "UnityEngine.ParticleSystemRenderer.material", "instantiating getter" }, { "UnityEngine.ParticleSystemRenderer.materials", "instantiating getter" },
            { "UnityEngine.TrailRenderer.material", "instantiating getter" }, { "UnityEngine.TrailRenderer.materials", "instantiating getter" },
            { "UnityEngine.LineRenderer.material", "instantiating getter" }, { "UnityEngine.LineRenderer.materials", "instantiating getter" },
            { "UnityEngine.RenderTexture.colorBuffer", "RenderTextureScripting::GetColorBuffer (UnityPlayer 0x705390) calls RenderTexture::Create" },
            { "UnityEngine.RenderTexture.depthBuffer", "RenderTextureScripting::GetDepthBuffer (UnityPlayer 0x705400) calls RenderTexture::Create" },
        };

        const int MaxDepth = 12;
        const int MaxElements = 1 << 16;

        // Types whose static constructor has necessarily run: their statics are used by the
        // per-frame code of live instances (PPv2 layer/volumes, DHH plugin, CTAA component).
        static readonly HashSet<string> InitializedStaticTypes = new HashSet<string>(StringComparer.Ordinal)
        {
            "UnityEngine.Rendering.PostProcessing.PostProcessManager", "UnityEngine.Rendering.PostProcessing.RuntimeUtilities",
            "UnityEngine.Rendering.PostProcessing.ShaderIDs", "UnityEngine.Rendering.PostProcessing.ColorUtilities",
            "UnityEngine.Rendering.PostProcessing.TextureLerper",
            // BetterAA plugin class: its Awake reads the static ConfigEntries and _instances.
            // (DHH_Base, DHH_Main, SSSManager have static fields without initializers, so no list entry is needed.)
            "AI_stuff.BetterAA",
        };

        public static string[] PluginAssemblyAnchors = { "DHH.SSS", "CTAA_PC", "UnityEngine.Rendering.PostProcessing.PostProcessLayer" };

        public static void Write(string path, ExportContext ctx, int windowIndex)
        {
            ctx.Path.Clear();
            ctx.RefreshNames();
            var dropped = ctx.PruneDestroyed();
            var w = new JsonOut(1 << 22);
            w.Begin('{');
            w.Key("schema_version"); w.Raw("5");
            w.Key("window_index"); w.Int(windowIndex);
            w.Key("frame_count"); w.Int(Time.frameCount);
            w.Key("rendered_frame_count"); w.Int(Time.renderedFrameCount);
            w.Key("statics"); w.Begin('{');
            foreach (var t in new[] { typeof(Time), typeof(QualitySettings), typeof(RenderSettings), typeof(GraphicsSettings),
                                      typeof(Screen), typeof(SystemInfo), typeof(Shader), typeof(LightmapSettings), typeof(ReflectionProbe),
                                      typeof(RenderTexture), typeof(GL), typeof(Camera) })
            {
                w.Key(t.FullName); StaticMembers(w, t, ctx);
            }
            w.Key("RenderSettings.ambientProbe"); SH(w, RenderSettings.ambientProbe);
            w.Key("ReflectionProbe.defaultTexture"); TextureRowOrNull(w, ReflectionProbe.defaultTexture, ctx);
            w.Key("ReflectionProbe.defaultTextureHDRDecodeValues"); Value(w, ReflectionProbe.defaultTextureHDRDecodeValues, 0, ctx);
            w.End('}');

            w.Guard("native", n => NativeFrame(n, ctx));

            w.Key("global_keywords_manifest"); w.Begin('{');
            w.Key("source"); w.Str("native keyword table + g_SharedPassContext bits; Shader.IsKeywordEnabled is not called (keywords::Create side effect)");
            w.Guard("keywords", k =>
            {
                var bits = ctx.Native.GlobalKeywordBits();
                var byName = ctx.Native.KeywordNames.ToDictionary(p => p.Value, p => p.Key, StringComparer.Ordinal);
                k.Begin('{');
                foreach (var name in ctx.Manifest.Keywords.Distinct())
                {
                    int idx;
                    k.Key(name);
                    if (!byName.TryGetValue(name, out idx)) k.Str("unregistered");
                    else k.Bool(((bits[idx >> 6] >> (idx & 63)) & 1) != 0);
                }
                k.End('}');
            });
            w.End('}');

            w.Key("global_uniforms_managed"); w.Begin('{');
            w.Key("source"); w.Str("Shader.GetGlobal* for manifest names already registered natively; the complete global sheet is native.global_sheet");
            foreach (var g in ctx.Manifest.Globals)
            {
                if (!ctx.RegisteredNames.Contains(g[0])) continue;
                w.Key(g[0]); GlobalUniform(w, g[0], g[1], int.Parse(g[2], CultureInfo.InvariantCulture), ctx);
            }
            w.End('}');

            var cameras = Resources.FindObjectsOfTypeAll<Camera>().Where(c => c.gameObject.scene.IsValid()).ToArray();
            w.Key("cameras"); w.Begin('[');
            foreach (var cam in cameras.OrderBy(c => c.depth)) CameraRow(w, cam, ctx);
            w.End(']');

            var lights = Resources.FindObjectsOfTypeAll<Light>().Where(l => l.gameObject.scene.IsValid()).ToArray();
            w.Key("lights"); w.Begin('[');
            foreach (var l in lights)
            {
                w.Begin('{'); Ident(w, l);
                w.Key("enabled"); w.Bool(l.enabled);
                w.Key("transform"); TransformBlock(w, l.transform, ctx);
                w.Key("properties"); Members(w, l, 0, ctx, false);
                w.Key("command_buffers"); w.Begin('{');
                foreach (LightEvent e in Enum.GetValues(typeof(LightEvent)))
                { w.Key(e.ToString()); Names(w, l.GetCommandBuffers(e).Select(b => b.name + "|size=" + b.sizeInBytes)); }
                w.End('}');
                w.Key("components"); Components(w, l.gameObject, ctx);
                w.End('}');
            }
            w.End(']');

            var renderers = Resources.FindObjectsOfTypeAll<Renderer>().Where(r => r.gameObject.scene.IsValid()).ToArray();
            w.Key("renderers"); w.Begin('[');
            foreach (var r in renderers) RendererRow(w, r, ctx);
            w.End(']');

            w.Key("reflection_probes"); w.Begin('[');
            foreach (var p in Resources.FindObjectsOfTypeAll<ReflectionProbe>().Where(p => p.gameObject.scene.IsValid()))
            {
                w.Begin('{'); Ident(w, p); w.Key("enabled"); w.Bool(p.enabled);
                w.Key("transform"); TransformBlock(w, p.transform, ctx);
                w.Key("properties"); Members(w, p, 0, ctx, false); w.End('}');
            }
            w.End(']');

            w.Key("light_probes"); LightProbesRow(w);

            w.Key("effect_components"); w.Begin('[');
            foreach (var mb in Resources.FindObjectsOfTypeAll<MonoBehaviour>().Where(m => m != null && m.gameObject.scene.IsValid() && IsEffect(m)))
            {
                w.Begin('{'); Ident(w, mb); w.Key("type"); w.Str(mb.GetType().AssemblyQualifiedName);
                w.Key("enabled"); w.Bool(mb.enabled);
                w.Key("component_index"); w.Int(Array.IndexOf(mb.gameObject.GetComponents<Component>(), mb));
                w.Key("fields"); AllFields(w, mb, 0, ctx); w.End('}');
            }
            w.End(']');

            w.Guard("plugin_member_paths", p => PluginPaths.Write(p, ctx));
            w.Key("plugin_static_fields"); PluginStatics(w, ctx);
            w.Guard("chara_custom_textures", c => CharaTextures(c, ctx));

            // All RenderTexture objects: camera targets ("TempBuffer N WxH"), pooled
            // temporaries, CTAA/PPv2 persistent RTs. Rows are emitted in "textures".
            w.Key("render_texture_ids"); w.Begin('[');
            foreach (var rt in Resources.FindObjectsOfTypeAll<RenderTexture>()) { ctx.Add(rt); w.Int(rt.GetInstanceID()); }
            w.End(']');

            w.Key("destroyed_references_dropped"); w.Begin('['); foreach (var id in dropped) w.Int(id); w.End(']');
            w.Key("materials"); w.Begin('[');
            var done = new HashSet<int>();
            while (true)
            {
                var pending = ctx.Materials.Where(kv => !done.Contains(kv.Key)).ToArray();
                if (pending.Length == 0) break;
                foreach (var kv in pending) { done.Add(kv.Key); if (kv.Value != null) MaterialRow(w, kv.Value, ctx); }
            }
            w.End(']');
            w.Key("textures"); w.Begin('[');
            var texDone = new HashSet<int>();
            while (true)
            {
                var pending = ctx.Textures.Where(kv => !texDone.Contains(kv.Key)).ToArray();
                if (pending.Length == 0) break;
                foreach (var kv in pending) { texDone.Add(kv.Key); if (kv.Value != null) TextureRow(w, kv.Value, ctx); }
            }
            w.End(']');
            w.End('}');
            File.WriteAllText(path, w.ToString());
        }

        static void NativeFrame(JsonOut n, ExportContext ctx)
        {
            n.Begin('{');
            n.Key("identity"); ctx.Native.WriteIdentity(n);
            n.Guard("settings", s => ctx.Native.WriteSettings(s));
            n.Guard("global_keywords", k => ctx.Native.WriteKeywords(k));
            n.Guard("global_sheet", g => SheetReader.Write(g, ctx.GlobalSheet(), ctx.Native.Name));
            n.Guard("render_buffer_pool", p => PoolWalker.Write(p, ctx.Native.Pool()));
            n.Guard("builtin_params", b => ctx.Native.WriteBuiltins(b));
            n.End('}');
        }

        public static void WriteObject(string path, object value, ExportContext ctx)
        {
            ctx.Path.Clear();
            var w = new JsonOut();
            Value(w, value, 0, ctx);
            File.WriteAllText(path, w.ToString());
        }

        static bool IsEffect(MonoBehaviour m)
        {
            var n = m.GetType().FullName ?? "";
            return n.StartsWith("DHH.") || n.Contains("CTAA") || n.Contains("BetterAA") || n.StartsWith("UnityEngine.Rendering.PostProcessing.")
                   || n.IndexOf("Liquid", StringComparison.OrdinalIgnoreCase) >= 0 || n.IndexOf("Siru", StringComparison.OrdinalIgnoreCase) >= 0
                   || n == "CharaCustom.CustomRender"
                   || m.GetComponent<Camera>() != null || m.GetComponent<Light>() != null;
        }

        public static void TransformBlock(JsonOut w, Transform t, ExportContext ctx)
        {
            w.Begin('{');
            w.Key("position"); Value(w, t.position, 0, ctx);
            w.Key("rotation"); Value(w, t.rotation, 0, ctx);
            w.Key("lossy_scale"); Value(w, t.lossyScale, 0, ctx);
            w.Key("local_position"); Value(w, t.localPosition, 0, ctx);
            w.Key("local_rotation"); Value(w, t.localRotation, 0, ctx);
            w.Key("local_scale"); Value(w, t.localScale, 0, ctx);
            w.Key("local_to_world_rowmajor"); Value(w, t.localToWorldMatrix, 0, ctx);
            w.Key("world_to_local_rowmajor"); Value(w, t.worldToLocalMatrix, 0, ctx);
            w.End('}');
        }

        static void CameraRow(JsonOut w, Camera cam, ExportContext ctx)
        {
            w.Begin('{'); Ident(w, cam);
            w.Key("enabled"); w.Bool(cam.enabled);
            w.Key("is_main"); w.Bool(cam == Camera.main);
            w.Key("transform"); TransformBlock(w, cam.transform, ctx);
            w.Guard("native", n => ctx.Native.WriteCamera(n, cam));
            w.Key("properties"); Members(w, cam, 0, ctx, false);
            w.Key("matrices"); CameraMatrices(w, cam, ctx);
            w.Key("target_texture"); TextureRowOrNull(w, cam.targetTexture, ctx);
            w.Key("command_buffers"); w.Begin('{');
            foreach (CameraEvent e in Enum.GetValues(typeof(CameraEvent)))
            { w.Key(e.ToString()); Names(w, cam.GetCommandBuffers(e).Select(b => b.name + "|size=" + b.sizeInBytes)); }
            w.End('}');
            w.Key("components"); Components(w, cam.gameObject, ctx);
            w.End('}');
        }

        public static void CameraMatrices(JsonOut w, Camera cam, ExportContext ctx)
        {
            w.Begin('{');
            w.Key("order"); w.Str("row-major m00,m01,m02,m03,m10,...");
            w.Key("projectionMatrix"); Value(w, cam.projectionMatrix, 0, ctx);
            w.Key("nonJitteredProjectionMatrix"); Value(w, cam.nonJitteredProjectionMatrix, 0, ctx);
            w.Key("worldToCameraMatrix"); Value(w, cam.worldToCameraMatrix, 0, ctx);
            w.Key("cameraToWorldMatrix"); Value(w, cam.cameraToWorldMatrix, 0, ctx);
            w.Key("cullingMatrix"); Value(w, cam.cullingMatrix, 0, ctx);
            w.Key("GL.GetGPUProjectionMatrix(proj,true)"); Value(w, GL.GetGPUProjectionMatrix(cam.projectionMatrix, true), 0, ctx);
            w.Key("GL.GetGPUProjectionMatrix(proj,false)"); Value(w, GL.GetGPUProjectionMatrix(cam.projectionMatrix, false), 0, ctx);
            w.End('}');
        }

        static void RendererRow(JsonOut w, Renderer r, ExportContext ctx)
        {
            w.Begin('{'); Ident(w, r);
            w.Key("type"); w.Str(r.GetType().FullName);
            w.Key("enabled"); w.Bool(r.enabled);
            w.Guard("native", n => ctx.Native.WriteRenderer(n, r));
            w.Key("transform"); TransformBlock(w, r.transform, ctx);
            w.Key("properties"); Members(w, r, 0, ctx, false);
            var shared = r.sharedMaterials;
            w.Key("shared_materials"); w.Begin('[');
            foreach (var m in shared) { Ref(w, m); ctx.Add(m); }
            w.End(']');
            var block = new MaterialPropertyBlock();
            w.Key("property_blocks_per_slot"); w.Begin('[');
            for (int i = 0; i < shared.Length; i++)
            {
                r.GetPropertyBlock(block, i);
                w.Begin('{'); w.Key("slot"); w.Int(i); w.Key("is_empty"); w.Bool(block.isEmpty);
                w.Guard("native_sheet", s => SheetReader.Write(s, ctx.Native.BlockSheet(block), ctx.Native.Name));
                w.End('}');
            }
            w.End(']');
            block.Clear();
            r.GetPropertyBlock(block);
            w.Key("property_block"); w.Begin('{'); w.Key("is_empty"); w.Bool(block.isEmpty);
            w.Guard("native_sheet", s => SheetReader.Write(s, ctx.Native.BlockSheet(block), ctx.Native.Name));
            w.End('}');
            var anchor = r.probeAnchor != null ? r.probeAnchor.position : r.bounds.center;
            w.Key("probe_position"); Value(w, anchor, 0, ctx);
            SphericalHarmonicsL2 sh;
            try { LightProbes.GetInterpolatedProbe(anchor, r, out sh); w.Key("interpolated_sh"); SH(w, sh); }
            catch (Exception e) { w.Key("interpolated_sh_error"); w.Str(e.ToString()); }
            var blends = new List<ReflectionProbeBlendInfo>();
            try
            {
                r.GetClosestReflectionProbes(blends);
                w.Key("reflection_probe_blends"); w.Begin('[');
                foreach (var b in blends) { w.Begin('{'); w.Key("weight"); w.Num(b.weight); w.Key("probe"); Ref(w, b.probe); w.End('}'); }
                w.End(']');
            }
            catch (Exception e) { w.Key("reflection_probe_blends_error"); w.Str(e.ToString()); }
            var smr = r as SkinnedMeshRenderer;
            Mesh mesh = null;
            if (smr != null)
            {
                mesh = smr.sharedMesh;
                if (mesh != null)
                {
                    w.Key("blendshape_weights"); w.Begin('[');
                    for (int i = 0; i < mesh.blendShapeCount; i++) w.Num(smr.GetBlendShapeWeight(i));
                    w.End(']');
                }
                w.Key("bones"); w.Begin('['); foreach (var b in smr.bones) Ref(w, b); w.End(']');
                w.Key("root_bone"); Ref(w, smr.rootBone);
            }
            var mf = r.GetComponent<MeshFilter>();
            if (mf != null) { mesh = mf.sharedMesh; w.Key("mesh_filter_shared_mesh"); Ref(w, mesh); }
            if (mesh != null) w.Guard("submeshes", s => Submeshes(s, mesh));
            w.Key("components"); Components(w, r.gameObject, ctx);
            w.End('}');
        }

        static void Submeshes(JsonOut w, Mesh mesh)
        {
            w.Begin('{');
            w.Key("mesh"); Ref(w, mesh);
            w.Key("index_format"); w.Str(mesh.indexFormat.ToString());
            w.Key("list"); w.Begin('[');
            for (int i = 0; i < mesh.subMeshCount; i++)
            {
                w.Begin('{'); w.Key("index"); w.Int(i);
                w.Key("index_start"); w.Int(mesh.GetIndexStart(i)); w.Key("index_count"); w.Int(mesh.GetIndexCount(i));
                w.Key("base_vertex"); w.Int(mesh.GetBaseVertex(i)); w.Key("topology"); w.Str(mesh.GetTopology(i).ToString());
                w.End('}');
            }
            w.End(']');
            w.End('}');
        }

        public static void Components(JsonOut w, GameObject go, ExportContext ctx)
        {
            w.Begin('[');
            foreach (var c in go.GetComponents<Component>())
            {
                w.Begin('{');
                if (c == null) { w.Key("missing_script"); w.Raw("true"); w.End('}'); continue; }
                Ident(w, c); w.Key("type"); w.Str(c.GetType().AssemblyQualifiedName);
                var b = c as Behaviour; if (b != null) { w.Key("enabled"); w.Bool(b.enabled); }
                if (c is MonoBehaviour) { w.Key("fields"); AllFields(w, c, 0, ctx); }
                w.End('}');
            }
            w.End(']');
        }

        static void LightProbesRow(JsonOut w)
        {
            var lp = LightmapSettings.lightProbes;
            w.Begin('{');
            if (lp == null) { w.Key("present"); w.Raw("false"); w.End('}'); return; }
            w.Key("present"); w.Raw("true"); w.Key("count"); w.Int(lp.count);
            w.Key("positions"); w.Begin('[');
            foreach (var p in lp.positions) { w.Num(p.x); w.Num(p.y); w.Num(p.z); }
            w.End(']');
            w.Key("baked_probes"); w.Begin('[');
            foreach (var sh in lp.bakedProbes) SH(w, sh);
            w.End(']');
            w.End('}');
        }

        public static void SH(JsonOut w, SphericalHarmonicsL2 sh)
        {
            w.Begin('[');
            for (int c = 0; c < 3; c++) for (int k = 0; k < 9; k++) w.Num(sh[c, k]);
            w.End(']');
        }

        static void MaterialRow(JsonOut w, Material m, ExportContext ctx)
        {
            w.Begin('{'); Ident(w, m);
            w.Key("shader"); w.Str(m.shader == null ? null : m.shader.name);
            w.Key("shader_id"); w.Raw(m.shader == null ? "null" : m.shader.GetInstanceID().ToString(CultureInfo.InvariantCulture));
            w.Key("render_queue"); w.Int(m.renderQueue);
            w.Key("shader_keywords"); Names(w, m.shaderKeywords);
            w.Key("pass_count"); w.Int(m.passCount);
            w.Key("passes"); w.Begin('[');
            for (int i = 0; i < m.passCount; i++)
            {
                var name = m.GetPassName(i);
                w.Begin('{'); w.Key("index"); w.Int(i); w.Key("name"); w.Str(name);
                w.Key("enabled"); w.Bool(name == null || m.GetShaderPassEnabled(name)); w.End('}');
            }
            w.End(']');
            w.Key("global_illumination_flags"); w.Str(m.globalIlluminationFlags.ToString());
            w.Key("enable_instancing"); w.Bool(m.enableInstancing);
            w.Key("double_sided_gi"); w.Bool(m.doubleSidedGI);
            w.Key("tags"); w.Begin('{');
            foreach (var tag in new[] { "RenderType", "Queue", "LightMode", "IgnoreProjector", "DisableBatching", "ForceNoShadowCasting", "PreviewType", "CanUseSpriteAtlas" })
            { w.Key(tag); w.Str(m.GetTag(tag, false, null)); }
            w.End('}');
            SheetSnapshot sheet = null;
            w.Guard("native_sheet", s =>
            {
                sheet = ctx.Native.MaterialSheet(m);
                if (sheet == null) s.Str("not_built (Material::BuildProperties not yet run; the probe does not force it)");
                else SheetReader.Write(s, sheet, ctx.Native.Name);
            });
            if (sheet != null)
            {
                foreach (var e in sheet.Entries)
                    if (e.Type == "texture")
                    {
                        int id = BitConverter.ToInt32(e.Value, 0);
                        if (id != 0) ctx.SheetTextureIds.Add(id);
                    }
            }
            w.Key("texture_properties"); w.Begin('{');
            foreach (var name in m.GetTexturePropertyNames())
            {
                w.Key(name); w.Begin('{');
                var t = m.GetTexture(name); w.Key("texture"); Ref(w, t); ctx.Add(t);
                w.Key("scale"); Value(w, m.GetTextureScale(name), 0, ctx);
                w.Key("offset"); Value(w, m.GetTextureOffset(name), 0, ctx);
                w.End('}');
            }
            w.End('}');
            List<string[]> props;
            w.Key("typed_properties_managed");
            if (sheet == null) w.Str("skipped: native sheet not built (managed getters would run Material::BuildProperties)");
            else if (m.shader == null || !ctx.Manifest.Properties.TryGetValue(m.shader.name, out props)) w.Str("shader not in library manifest; native_sheet is complete");
            else
            {
                w.Begin('{');
                foreach (var p in props)
                {
                    w.Key(p[0]); w.Begin('{'); w.Key("type"); w.Str(p[1]);
                    if (!ctx.RegisteredNames.Contains(p[0])) { w.Key("registered"); w.Bool(false); w.End('}'); continue; }
                    w.Key("has"); w.Bool(m.HasProperty(p[0]));
                    if (m.HasProperty(p[0]))
                    {
                        try
                        {
                            int n = int.Parse(p[2], CultureInfo.InvariantCulture);
                            w.Key("value");
                            switch (p[1])
                            {
                                case "float": if (n > 0) Value(w, m.GetFloatArray(p[0]), 0, ctx); else w.Num(m.GetFloat(p[0])); break;
                                case "int": w.Int(m.GetInt(p[0])); break;
                                case "vector": if (n > 0) Value(w, m.GetVectorArray(p[0]), 0, ctx); else Value(w, m.GetVector(p[0]), 0, ctx); break;
                                case "color": Value(w, m.GetColor(p[0]), 0, ctx); break;
                                case "matrix": if (n > 0) Value(w, m.GetMatrixArray(p[0]), 0, ctx); else Value(w, m.GetMatrix(p[0]), 0, ctx); break;
                                case "texture": var t = m.GetTexture(p[0]); Ref(w, t); ctx.Add(t); break;
                                default: throw new FormatException("Unknown manifest type " + p[1]);
                            }
                        }
                        catch (Exception e) { w.Key("error"); w.Str(e.GetType().Name + ": " + e.Message); }
                    }
                    w.End('}');
                }
                w.End('}');
            }
            w.End('}');
        }

        public static void GlobalUniform(JsonOut w, string name, string type, int n, ExportContext ctx)
        {
            w.Begin('{');
            try
            {
                w.Key("value");
                switch (type)
                {
                    case "float": if (n > 0) Value(w, Shader.GetGlobalFloatArray(name), 0, ctx); else w.Num(Shader.GetGlobalFloat(name)); break;
                    case "int": w.Int(Shader.GetGlobalInt(name)); break;
                    case "vector": case "color": if (n > 0) Value(w, Shader.GetGlobalVectorArray(name), 0, ctx); else Value(w, Shader.GetGlobalVector(name), 0, ctx); break;
                    case "matrix": if (n > 0) Value(w, Shader.GetGlobalMatrixArray(name), 0, ctx); else Value(w, Shader.GetGlobalMatrix(name), 0, ctx); break;
                    case "texture": var t = Shader.GetGlobalTexture(name); Ref(w, t); ctx.Add(t); break;
                    default: throw new FormatException("Unknown global type " + type);
                }
            }
            catch (Exception e) { w.Key("error"); w.Str(e.GetType().Name + ": " + e.Message); }
            w.End('}');
        }

        public static void TextureRowOrNull(JsonOut w, Texture t, ExportContext ctx) { if (t == null) w.Raw("null"); else { ctx.Add(t); Ref(w, t); } }

        static void TextureRow(JsonOut w, Texture t, ExportContext ctx)
        {
            w.Begin('{'); Ident(w, t); w.Key("type"); w.Str(t.GetType().FullName);
            var rt = t as RenderTexture;
            bool created = rt == null || rt.IsCreated();
            if (rt != null)
            {
                w.Key("is_created"); w.Bool(created);
                w.Key("descriptor"); Value(w, rt.descriptor, 0, ctx);
            }
            w.Guard("native", n => ctx.Native.WriteTexture(n, t));
            w.Key("properties"); Members(w, t, 0, ctx, false);
            if (created)
            {
                try { w.Key("native_ptr"); w.Str("0x" + t.GetNativeTexturePtr().ToInt64().ToString("x")); }
                catch (Exception e) { w.Key("native_ptr_error"); w.Str(e.Message); }
            }
            else { w.Key("native_ptr"); w.Str("not queried: RenderTexture not created"); }
            w.End('}');
        }

        static void CharaTextures(JsonOut w, ExportContext ctx)
        {
            w.Begin('[');
            foreach (var cha in UnityEngine.Object.FindObjectsOfType<AIChara.ChaControl>())
                foreach (var prop in new[] { "customTexCtrlFace", "customTexCtrlBody" })
                {
                    w.Begin('{'); w.Key("chara"); Ref(w, cha); w.Key("control"); w.Str(prop);
                    var pi = cha.GetType().GetProperty(prop, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    var control = pi == null ? null : pi.GetValue(cha, null);
                    if (control == null) { w.Key("value"); w.Raw("null"); w.End('}'); continue; }
                    w.Key("fields"); AllFields(w, control, 0, ctx);
                    var f = control.GetType().GetField("createCustomTex", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    var arr = f == null ? null : f.GetValue(control) as Array;
                    w.Key("create_textures"); w.Begin('[');
                    if (arr != null)
                        foreach (var creator in arr)
                        {
                            if (creator == null) { w.Raw("null"); continue; }
                            var get = creator.GetType().GetMethod("GetCreateTexture", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, Type.EmptyTypes, null);
                            var tex = get == null ? null : get.Invoke(creator, null) as Texture;
                            Ref(w, tex); ctx.Add(tex);
                        }
                    w.End(']');
                    w.End('}');
                }
            w.End(']');
        }

        static void PluginStatics(JsonOut w, ExportContext ctx)
        {
            w.Begin('{');
            var assemblies = new List<Assembly>();
            foreach (var anchor in PluginAssemblyAnchors)
            {
                var t = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType(anchor, false)).FirstOrDefault(x => x != null);
                if (t == null) { w.Key(anchor + "_error"); w.Str("type not loaded"); continue; }
                if (!assemblies.Contains(t.Assembly)) assemblies.Add(t.Assembly);
            }
            foreach (var asm in assemblies)
            {
                w.Key(asm.GetName().Name); w.Begin('{');
                Type[] types;
                try { types = asm.GetTypes(); }
                catch (ReflectionTypeLoadException e) { types = e.Types.Where(x => x != null).ToArray(); }
                foreach (var t in types.Where(x => !x.ContainsGenericParameters).OrderBy(x => x.FullName, StringComparer.Ordinal))
                {
                    var fields = t.GetFields(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                                  .Where(f => !f.IsLiteral).OrderBy(f => f.Name, StringComparer.Ordinal).ToArray();
                    if (fields.Length == 0) continue;
                    // Reading a static field runs the type's static constructor if it has not run
                    // yet; that would be a probe side effect. Only types without one, or types whose
                    // instances/statics are demonstrably in use every rendered frame, are read.
                    if (t.TypeInitializer != null && !InitializedStaticTypes.Contains(t.FullName))
                    { w.Key(t.FullName); w.Str("<skipped: has a static constructor that may not have run; reading would run it>"); continue; }
                    w.Key(t.FullName); w.Begin('{');
                    foreach (var f in fields)
                    {
                        w.Key(f.Name);
                        object v;
                        try { v = f.GetValue(null); } catch (Exception e) { Err(w, e); continue; }
                        Value(w, v, 1, ctx);
                    }
                    w.End('}');
                }
                w.End('}');
            }
            w.End('}');
        }

        // ---- reflection helpers ----

        static void StaticMembers(JsonOut w, Type t, ExportContext ctx)
        {
            w.Begin('{');
            foreach (var p in t.GetProperties(BindingFlags.Public | BindingFlags.Static).OrderBy(p => p.Name))
            {
                if (!p.CanRead || p.GetIndexParameters().Length > 0 || Obsolete(p)) continue;
                w.Key(p.Name);
                object v;
                try { v = p.GetValue(null, null); }
                catch (Exception e) { Err(w, e); continue; }
                Value(w, v, 1, ctx);
            }
            foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.Static).OrderBy(f => f.Name))
            {
                if (f.IsLiteral) continue;
                w.Key("field:" + f.Name);
                object v;
                try { v = f.GetValue(null); } catch (Exception e) { Err(w, e); continue; }
                Value(w, v, 1, ctx);
            }
            w.End('}');
        }

        public static void Members(JsonOut w, object o, int depth, ExportContext ctx, bool nonPublicFields)
        {
            w.Begin('{');
            var t = o.GetType();
            foreach (var p in t.GetProperties(BindingFlags.Public | BindingFlags.Instance).OrderBy(p => p.Name))
            {
                if (!p.CanRead || p.GetIndexParameters().Length > 0 || Obsolete(p)) continue;
                string why;
                if (SkippedGetters.TryGetValue(t.FullName + "." + p.Name, out why) || SkippedGetters.TryGetValue(p.DeclaringType.FullName + "." + p.Name, out why))
                { w.Key(p.Name); w.Str("<skipped: " + why + ">"); continue; }
                if (p.Name == "gameObject" || p.Name == "transform") { w.Key(p.Name); try { Ref(w, (UnityEngine.Object)p.GetValue(o, null)); } catch (Exception e) { Err(w, e); } continue; }
                w.Key(p.Name);
                object v;
                try { v = p.GetValue(o, null); }
                catch (Exception e) { Err(w, e); continue; }
                Value(w, v, depth + 1, ctx);
            }
            foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.Instance | (nonPublicFields ? BindingFlags.NonPublic : 0)).OrderBy(f => f.Name))
            {
                w.Key("field:" + f.Name);
                object v;
                try { v = f.GetValue(o); }
                catch (Exception e) { Err(w, e); continue; }
                Value(w, v, depth + 1, ctx);
            }
            w.End('}');
        }

        public static void AllFields(JsonOut w, object o, int depth, ExportContext ctx)
        {
            w.Begin('{');
            for (var t = o.GetType(); t != null && t != typeof(MonoBehaviour) && t != typeof(ScriptableObject) && t != typeof(object); t = t.BaseType)
            {
                foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly).OrderBy(f => f.Name))
                {
                    w.Key(t.Name + "." + f.Name);
                    object v;
                    try { v = f.GetValue(o); }
                    catch (Exception e) { Err(w, e); continue; }
                    Value(w, v, depth + 1, ctx);
                }
            }
            w.End('}');
        }

        static bool Obsolete(MemberInfo m) { return m.GetCustomAttributes(typeof(ObsoleteAttribute), true).Length > 0; }

        public static void Err(JsonOut w, Exception e)
        {
            var inner = e is TargetInvocationException && e.InnerException != null ? e.InnerException : e;
            w.Begin('{'); w.Key("error"); w.Str(inner.GetType().Name + ": " + inner.Message); w.End('}');
        }

        public static void Ident(JsonOut w, UnityEngine.Object o)
        {
            w.Key("id"); w.Int(o.GetInstanceID());
            w.Key("name"); w.Str(o.name);
            var c = o as Component;
            if (c != null) { w.Key("path"); w.Str(Path(c.transform)); w.Key("active_in_hierarchy"); w.Bool(c.gameObject.activeInHierarchy); w.Key("layer"); w.Int(c.gameObject.layer); }
        }

        public static string Path(Transform t)
        {
            var parts = new List<string>();
            for (; t != null; t = t.parent) parts.Add(t.name + "[" + t.GetSiblingIndex() + "]");
            parts.Reverse();
            return "/" + string.Join("/", parts.ToArray());
        }

        public static void Ref(JsonOut w, UnityEngine.Object o)
        {
            if (o == null) { w.Raw("null"); return; }
            w.Begin('{'); w.Key("ref_id"); w.Int(o.GetInstanceID());
            w.Key("ref_type"); w.Str(o.GetType().FullName); w.Key("ref_name"); w.Str(o.name); w.End('}');
        }

        public static void Names(JsonOut w, IEnumerable<string> names) { w.Begin('['); foreach (var n in names) w.Str(n); w.End(']'); }

        public static void Value(JsonOut w, object v, int depth, ExportContext ctx)
        {
            if (v == null) { w.Raw("null"); return; }
            if (v is UnityEngine.Object)
            {
                var uo = (UnityEngine.Object)v;
                if (uo == null) { w.Raw("null"); return; }   // destroyed native object
                var t = uo as Texture; if (t != null) ctx.Add(t);
                var m = uo as Material; if (m != null) ctx.Add(m);
                if (uo is ScriptableObject && depth < MaxDepth && !ctx.Path.Contains(uo))
                {
                    // ScriptableObject settings (PPv2 profiles and effect settings) are expanded as in v4.
                    ctx.Path.Add(uo);
                    try { w.Begin('{'); w.Key("ref"); Ref(w, uo); w.Key("fields"); AllFields(w, uo, depth, ctx); w.End('}'); }
                    finally { ctx.Path.Remove(uo); }
                }
                else Ref(w, uo);
                return;
            }
            switch (v)
            {
                case bool b: w.Bool(b); return;
                case float f: w.Num(f); return;
                case double d: w.Str(d.ToString("R", CultureInfo.InvariantCulture)); return;
                case int i: w.Int(i); return;
                case uint ui: w.Int(ui); return;
                case long l: w.Int(l); return;
                case ulong ul: w.Raw(ul.ToString(CultureInfo.InvariantCulture)); return;
                case short s: w.Int(s); return;
                case ushort us: w.Int(us); return;
                case byte by: w.Int(by); return;
                case sbyte sb: w.Int(sb); return;
                case char ch: w.Int(ch); return;
                case string str: w.Str(str); return;
                case Enum e: w.Str(e.GetType().Name + "." + e.ToString() + "=" + Convert.ToInt64(e).ToString(CultureInfo.InvariantCulture)); return;
                case Vector2 a: Floats(w, a.x, a.y); return;
                case Vector3 a: Floats(w, a.x, a.y, a.z); return;
                case Vector4 a: Floats(w, a.x, a.y, a.z, a.w); return;
                case Quaternion a: Floats(w, a.x, a.y, a.z, a.w); return;
                case Color a: Floats(w, a.r, a.g, a.b, a.a); return;
                case Color32 a: w.Begin('['); w.Int(a.r); w.Int(a.g); w.Int(a.b); w.Int(a.a); w.End(']'); return;
                case Rect a: Floats(w, a.x, a.y, a.width, a.height); return;
                case Bounds a: Floats(w, a.center.x, a.center.y, a.center.z, a.extents.x, a.extents.y, a.extents.z); return;
                case Matrix4x4 a:
                    w.Begin('['); for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) w.Num(a[r, c]); w.End(']'); return;
                case SphericalHarmonicsL2 sh: SH(w, sh); return;
                case LayerMask lm: w.Begin('{'); w.Key("m_Mask"); w.Int(lm.value); w.End('}'); return;
                case IntPtr p: w.Str("ptr:0x" + p.ToInt64().ToString("x")); return;
                case Delegate dl:
                    w.Begin('{'); w.Key("delegate"); w.Str(dl.GetType().FullName); w.Key("invocation_list"); w.Begin('[');
                    foreach (var inv in dl.GetInvocationList()) w.Str((inv.Method.DeclaringType == null ? "?" : inv.Method.DeclaringType.FullName) + "." + inv.Method.Name);
                    w.End(']'); w.End('}'); return;
            }
            var curve = v as AnimationCurve;
            if (curve != null)
            {
                w.Begin('{'); w.Key("$type"); w.Str("UnityEngine.AnimationCurve");
                w.Key("preWrapMode"); Value(w, curve.preWrapMode, depth + 1, ctx);
                w.Key("postWrapMode"); Value(w, curve.postWrapMode, depth + 1, ctx);
                w.Key("keys"); w.Begin('[');
                foreach (var key in curve.keys) Value(w, key, depth + 1, ctx);   // Keyframe private fields m_Time..m_OutWeight
                w.End(']'); w.End('}');
                return;
            }
            var type = v.GetType();
            if (depth >= MaxDepth) { w.Str("<depth limit " + MaxDepth + ": " + type.FullName + ">"); return; }
            if (type.IsPrimitive) { w.Str(Convert.ToString(v, CultureInfo.InvariantCulture)); return; }
            // Cycle protection along the current reflection path only; shared (DAG) objects
            // are expanded at every occurrence exactly as v4 did, so v4 consumers keep working.
            bool tracked = !type.IsValueType;
            if (tracked)
            {
                if (ctx.Path.Contains(v)) { w.Begin('{'); w.Key("$cycle"); w.Str(type.FullName); w.End('}'); return; }
                ctx.Path.Add(v);
            }
            try
            {
                var dict = v as IDictionary;
                if (dict != null)
                {
                    w.Begin('{'); int n = 0;
                    foreach (DictionaryEntry kv in dict)
                    {
                        if (n++ >= MaxElements) { w.Key("<truncated>"); w.Int(dict.Count); break; }
                        w.Key(Convert.ToString(kv.Key, CultureInfo.InvariantCulture)); Value(w, kv.Value, depth + 1, ctx);
                    }
                    w.End('}'); return;
                }
                var en = v as IEnumerable;
                if (en != null)
                {
                    w.Begin('['); int n = 0;
                    foreach (var x in en) { if (n++ >= MaxElements) { w.Str("<truncated>"); break; } Value(w, x, depth + 1, ctx); }
                    w.End(']'); return;
                }
                // Plain managed object/struct (plugin settings, PPv2 parameter overrides, splines...): all fields.
                w.Begin('{'); w.Key("$type"); w.Str(type.FullName);
                for (var t = type; t != null && t != typeof(object) && t != typeof(ValueType); t = t.BaseType)
                    foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly))
                    {
                        w.Key(t.Name + "." + f.Name);
                        object fv;
                        try { fv = f.GetValue(v); } catch (Exception e) { Err(w, e); continue; }
                        Value(w, fv, depth + 1, ctx);
                    }
                w.End('}');
            }
            finally { if (tracked) ctx.Path.Remove(v); }
        }

        static void Floats(JsonOut w, params float[] f) { w.Begin('['); foreach (var x in f) w.Num(x); w.End(']'); }
    }
}
