// Full render-scene state export for the frame-pair probe (read-only).
//
// Reflects the complete public Unity API surface of cameras, lights, renderers,
// reflection probes, light probes, RenderSettings/QualitySettings/GraphicsSettings/
// Screen/Time, and ALL (public and non-public) instance fields of every
// MonoBehaviour on cameras plus effect/volume components (DHH SSS, CTAA,
// PostProcessing v2 layer/volume/profile, liquid). Materials are exported by
// shader/keywords/queue/passes/textures and typed properties from a supplied
// shader manifest (Unity 2018.4 cannot enumerate material uniforms at runtime).
// Getters with known side effects (instantiation) are never invoked; every
// failing getter is recorded with its error instead of being dropped.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
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

        // Plain line format (no JSON dependency):
        //   P<TAB>shader<TAB>name<TAB>type<TAB>arraySize
        //   K<TAB>keyword
        //   G<TAB>name<TAB>type<TAB>arraySize
        public static ShaderManifest Load(string path)
        {
            var m = new ShaderManifest();
            if (string.IsNullOrEmpty(path)) return m;
            foreach (var raw in File.ReadAllLines(path))
            {
                var f = raw.Split('\t');
                if (f.Length == 0 || f[0].Length == 0) continue;
                if (f[0] == "P" && f.Length == 5)
                {
                    List<string[]> list;
                    if (!m.Properties.TryGetValue(f[1], out list)) m.Properties[f[1]] = list = new List<string[]>();
                    list.Add(new[] { f[2], f[3], f[4] });
                }
                else if (f[0] == "K" && f.Length == 2) m.Keywords.Add(f[1]);
                else if (f[0] == "G" && f.Length == 4) m.Globals.Add(new[] { f[1], f[2], f[3] });
                else throw new FormatException("Bad manifest line: " + raw);
            }
            return m;
        }
    }

    internal static class SceneState
    {
        static readonly HashSet<string> SideEffectGetters = new HashSet<string>
        {
            "UnityEngine.Renderer.material", "UnityEngine.Renderer.materials",
            "UnityEngine.MeshFilter.mesh", "UnityEngine.Collider.material",
            "UnityEngine.SkinnedMeshRenderer.material", "UnityEngine.SkinnedMeshRenderer.materials",
            "UnityEngine.MeshRenderer.material", "UnityEngine.MeshRenderer.materials",
            "UnityEngine.ParticleSystemRenderer.material", "UnityEngine.ParticleSystemRenderer.materials",
            "UnityEngine.TrailRenderer.material", "UnityEngine.TrailRenderer.materials",
            "UnityEngine.LineRenderer.material", "UnityEngine.LineRenderer.materials",
        };

        const int MaxDepth = 6;
        const int MaxElements = 1 << 16;

        public static void Write(string path, ShaderManifest manifest, int windowIndex)
        {
            var w = new JsonOut();
            var materials = new Dictionary<int, Material>();
            var textures = new Dictionary<int, Texture>();
            w.Begin('{');
            w.Key("schema_version"); w.Raw("1");
            w.Key("window_index"); w.Raw(windowIndex.ToString(CultureInfo.InvariantCulture));
            w.Key("frame_count"); w.Raw(Time.frameCount.ToString(CultureInfo.InvariantCulture));
            w.Key("statics"); w.Begin('{');
            foreach (var t in new[] { typeof(Time), typeof(QualitySettings), typeof(RenderSettings), typeof(GraphicsSettings),
                                      typeof(Screen), typeof(SystemInfo), typeof(Shader), typeof(LightmapSettings) })
            {
                w.Key(t.FullName); StaticMembers(w, t, textures, materials);
            }
            w.Key("RenderSettings.ambientProbe"); SH(w, RenderSettings.ambientProbe);
            w.End('}');
            w.Key("global_keywords"); w.Begin('{');
            foreach (var k in manifest.Keywords.Distinct()) { w.Key(k); w.Raw(Shader.IsKeywordEnabled(k) ? "true" : "false"); }
            w.End('}');
            w.Key("global_uniforms"); w.Begin('{');
            foreach (var g in manifest.Globals) { w.Key(g[0]); GlobalUniform(w, g[0], g[1], int.Parse(g[2], CultureInfo.InvariantCulture), textures); }
            w.End('}');

            var cameras = Resources.FindObjectsOfTypeAll<Camera>().Where(c => c.gameObject.scene.IsValid()).ToArray();
            w.Key("cameras"); w.Begin('[');
            foreach (var cam in cameras.OrderBy(c => c.depth)) { w.Next(); CameraRow(w, cam, textures, materials); }
            w.End(']');

            var lights = Resources.FindObjectsOfTypeAll<Light>().Where(l => l.gameObject.scene.IsValid()).ToArray();
            w.Key("lights"); w.Begin('[');
            foreach (var l in lights)
            {
                w.Next(); w.Begin('{'); Ident(w, l);
                w.Key("properties"); Members(w, l, 0, textures, materials, false);
                w.Key("command_buffers"); w.Begin('{');
                foreach (LightEvent e in Enum.GetValues(typeof(LightEvent)))
                { w.Key(e.ToString()); Names(w, l.GetCommandBuffers(e).Select(b => b.name)); }
                w.End('}');
                w.Key("components"); Components(w, l.gameObject, textures, materials);
                w.End('}');
            }
            w.End(']');

            var renderers = Resources.FindObjectsOfTypeAll<Renderer>().Where(r => r.gameObject.scene.IsValid()).ToArray();
            w.Key("renderers"); w.Begin('[');
            var blends = new List<ReflectionProbeBlendInfo>();
            foreach (var r in renderers)
            {
                w.Next(); w.Begin('{'); Ident(w, r);
                w.Key("type"); w.Str(r.GetType().FullName);
                w.Key("properties"); Members(w, r, 0, textures, materials, false);
                w.Key("shared_materials"); w.Begin('[');
                foreach (var m in r.sharedMaterials) { w.Next(); Ref(w, m); if (m != null) materials[m.GetInstanceID()] = m; }
                w.End(']');
                var block = new MaterialPropertyBlock();
                w.Key("property_block_empty_per_slot"); w.Begin('[');
                for (int i = 0; i < r.sharedMaterials.Length; i++) { r.GetPropertyBlock(block, i); w.Next(); w.Raw(block.isEmpty ? "true" : "false"); }
                w.End(']');
                r.GetPropertyBlock(block); w.Key("property_block_empty"); w.Raw(block.isEmpty ? "true" : "false");
                w.Key("probe_position"); Value(w, r.probeAnchor != null ? r.probeAnchor.position : r.bounds.center, 0, textures, materials);
                SphericalHarmonicsL2 sh;
                try { LightProbes.GetInterpolatedProbe(r.probeAnchor != null ? r.probeAnchor.position : r.bounds.center, r, out sh); w.Key("interpolated_sh"); SH(w, sh); }
                catch (Exception e) { w.Key("interpolated_sh_error"); w.Str(e.ToString()); }
                blends.Clear();
                try
                {
                    r.GetClosestReflectionProbes(blends);
                    w.Key("reflection_probe_blends"); w.Begin('[');
                    foreach (var b in blends) { w.Next(); w.Begin('{'); w.Key("weight"); Value(w, b.weight, 0, textures, materials); w.Key("probe"); Ref(w, b.probe); w.End('}'); }
                    w.End(']');
                }
                catch (Exception e) { w.Key("reflection_probe_blends_error"); w.Str(e.ToString()); }
                var smr = r as SkinnedMeshRenderer;
                if (smr != null && smr.sharedMesh != null)
                {
                    w.Key("blendshape_weights"); w.Begin('[');
                    for (int i = 0; i < smr.sharedMesh.blendShapeCount; i++) { w.Next(); Value(w, smr.GetBlendShapeWeight(i), 0, textures, materials); }
                    w.End(']');
                    w.Key("bones"); w.Begin('[');
                    foreach (var b in smr.bones) { w.Next(); Ref(w, b); }
                    w.End(']');
                }
                var mf = r.GetComponent<MeshFilter>();
                if (mf != null) { w.Key("mesh_filter_shared_mesh"); Ref(w, mf.sharedMesh); }
                w.Key("components"); Components(w, r.gameObject, textures, materials);
                w.End('}');
            }
            w.End(']');

            w.Key("reflection_probes"); w.Begin('[');
            foreach (var p in Resources.FindObjectsOfTypeAll<ReflectionProbe>().Where(p => p.gameObject.scene.IsValid()))
            { w.Next(); w.Begin('{'); Ident(w, p); w.Key("properties"); Members(w, p, 0, textures, materials, false); w.End('}'); }
            w.End(']');

            w.Key("light_probes"); LightProbesRow(w);

            w.Key("effect_components"); w.Begin('[');
            foreach (var mb in Resources.FindObjectsOfTypeAll<MonoBehaviour>().Where(m => m != null && m.gameObject.scene.IsValid() && IsEffect(m)))
            { w.Next(); w.Begin('{'); Ident(w, mb); w.Key("type"); w.Str(mb.GetType().AssemblyQualifiedName); w.Key("fields"); AllFields(w, mb, 0, textures, materials); w.End('}'); }
            w.End(']');

            w.Key("materials"); w.Begin('[');
            foreach (var m in materials.Values.ToArray()) { w.Next(); MaterialRow(w, m, manifest, textures); }
            w.End(']');
            w.Key("textures"); w.Begin('[');
            foreach (var t in textures.Values.ToArray()) { w.Next(); TextureRow(w, t); }
            w.End(']');
            w.End('}');
            File.WriteAllText(path, w.ToString());
        }

        // Serialize an arbitrary helper result (e.g. HS2_RenderAudit.Capture()) losslessly.
        public static void WriteObject(string path, object value)
        {
            var w = new JsonOut();
            Value(w, value, 0, null, null);
            File.WriteAllText(path, w.ToString());
        }

        static bool IsEffect(MonoBehaviour m)
        {
            var n = m.GetType().FullName ?? "";
            return n.StartsWith("DHH.") || n.Contains("CTAA") || n.Contains("BetterAA") || n.StartsWith("UnityEngine.Rendering.PostProcessing.")
                   || n.IndexOf("Liquid", StringComparison.OrdinalIgnoreCase) >= 0 || n.IndexOf("Siru", StringComparison.OrdinalIgnoreCase) >= 0
                   || m.GetComponent<Camera>() != null || m.GetComponent<Light>() != null;
        }

        static void CameraRow(JsonOut w, Camera cam, Dictionary<int, Texture> tex, Dictionary<int, Material> mats)
        {
            w.Begin('{'); Ident(w, cam);
            w.Key("properties"); Members(w, cam, 0, tex, mats, false);
            w.Key("matrices"); w.Begin('{');
            w.Key("projectionMatrix"); Value(w, cam.projectionMatrix, 0, tex, mats);
            w.Key("nonJitteredProjectionMatrix"); Value(w, cam.nonJitteredProjectionMatrix, 0, tex, mats);
            w.Key("worldToCameraMatrix"); Value(w, cam.worldToCameraMatrix, 0, tex, mats);
            w.Key("cameraToWorldMatrix"); Value(w, cam.cameraToWorldMatrix, 0, tex, mats);
            w.Key("GL.GetGPUProjectionMatrix(proj,true)"); Value(w, GL.GetGPUProjectionMatrix(cam.projectionMatrix, true), 0, tex, mats);
            w.Key("GL.GetGPUProjectionMatrix(proj,false)"); Value(w, GL.GetGPUProjectionMatrix(cam.projectionMatrix, false), 0, tex, mats);
            w.End('}');
            w.Key("target_texture"); TextureRowOrNull(w, cam.targetTexture);
            w.Key("command_buffers"); w.Begin('{');
            foreach (CameraEvent e in Enum.GetValues(typeof(CameraEvent)))
            { w.Key(e.ToString()); Names(w, cam.GetCommandBuffers(e).Select(b => b.name + "|size=" + b.sizeInBytes)); }
            w.End('}');
            w.Key("components"); Components(w, cam.gameObject, tex, mats);
            w.End('}');
        }

        static void Components(JsonOut w, GameObject go, Dictionary<int, Texture> tex, Dictionary<int, Material> mats)
        {
            w.Begin('[');
            foreach (var c in go.GetComponents<Component>())
            {
                w.Next(); w.Begin('{');
                if (c == null) { w.Key("missing_script"); w.Raw("true"); w.End('}'); continue; }
                Ident(w, c); w.Key("type"); w.Str(c.GetType().AssemblyQualifiedName);
                var b = c as Behaviour; if (b != null) { w.Key("enabled"); w.Raw(b.enabled ? "true" : "false"); }
                if (c is MonoBehaviour) { w.Key("fields"); AllFields(w, c, 0, tex, mats); }
                w.End('}');
            }
            w.End(']');
        }

        static void LightProbesRow(JsonOut w)
        {
            var lp = LightmapSettings.lightProbes;
            w.Begin('{');
            if (lp == null) { w.Key("present"); w.Raw("false"); w.End('}'); return; }
            w.Key("present"); w.Raw("true"); w.Key("count"); w.Raw(lp.count.ToString(CultureInfo.InvariantCulture));
            w.Key("positions"); w.Begin('[');
            foreach (var p in lp.positions) { w.Next(); w.Num(p.x); w.Next(); w.Num(p.y); w.Next(); w.Num(p.z); }
            w.End(']');
            w.Key("baked_probes"); w.Begin('[');
            foreach (var sh in lp.bakedProbes) { w.Next(); SH(w, sh); }
            w.End(']');
            w.End('}');
        }

        static void SH(JsonOut w, SphericalHarmonicsL2 sh)
        {
            w.Begin('[');
            for (int c = 0; c < 3; c++) for (int k = 0; k < 9; k++) { w.Next(); w.Num(sh[c, k]); }
            w.End(']');
        }

        static void MaterialRow(JsonOut w, Material m, ShaderManifest manifest, Dictionary<int, Texture> tex)
        {
            w.Begin('{'); Ident(w, m);
            w.Key("shader"); w.Str(m.shader == null ? null : m.shader.name);
            w.Key("shader_id"); w.Raw(m.shader == null ? "null" : m.shader.GetInstanceID().ToString(CultureInfo.InvariantCulture));
            w.Key("render_queue"); w.Raw(m.renderQueue.ToString(CultureInfo.InvariantCulture));
            w.Key("shader_keywords"); Names(w, m.shaderKeywords);
            w.Key("pass_count"); w.Raw(m.passCount.ToString(CultureInfo.InvariantCulture));
            w.Key("global_illumination_flags"); w.Str(m.globalIlluminationFlags.ToString());
            w.Key("enable_instancing"); w.Raw(m.enableInstancing ? "true" : "false");
            w.Key("double_sided_gi"); w.Raw(m.doubleSidedGI ? "true" : "false");
            w.Key("tags"); w.Begin('{');
            foreach (var tag in new[] { "RenderType", "Queue", "LightMode", "IgnoreProjector", "DisableBatching", "ForceNoShadowCasting", "PreviewType" })
            { w.Key(tag); w.Str(m.GetTag(tag, false, null)); }
            w.End('}');
            w.Key("texture_properties"); w.Begin('{');
            foreach (var name in m.GetTexturePropertyNames())
            {
                w.Key(name); w.Begin('{');
                var t = m.GetTexture(name); w.Key("texture"); Ref(w, t); if (t != null) tex[t.GetInstanceID()] = t;
                w.Key("scale"); Value(w, m.GetTextureScale(name), 0, tex, null);
                w.Key("offset"); Value(w, m.GetTextureOffset(name), 0, tex, null);
                w.End('}');
            }
            w.End('}');
            List<string[]> props;
            w.Key("typed_properties_from_manifest");
            if (m.shader == null || !manifest.Properties.TryGetValue(m.shader.name, out props)) { w.Raw("null"); }
            else
            {
                w.Begin('{');
                foreach (var p in props)
                {
                    w.Key(p[0]); w.Begin('{'); w.Key("type"); w.Str(p[1]); w.Key("has"); w.Raw(m.HasProperty(p[0]) ? "true" : "false");
                    try
                    {
                        int n = int.Parse(p[2], CultureInfo.InvariantCulture);
                        w.Key("value");
                        switch (p[1])
                        {
                            case "float": if (n > 0) Value(w, m.GetFloatArray(p[0]), 0, tex, null); else w.Num(m.GetFloat(p[0])); break;
                            case "int": w.Raw(m.GetInt(p[0]).ToString(CultureInfo.InvariantCulture)); break;
                            case "vector": if (n > 0) Value(w, m.GetVectorArray(p[0]), 0, tex, null); else Value(w, m.GetVector(p[0]), 0, tex, null); break;
                            case "color": Value(w, m.GetColor(p[0]), 0, tex, null); break;
                            case "matrix": if (n > 0) Value(w, m.GetMatrixArray(p[0]), 0, tex, null); else Value(w, m.GetMatrix(p[0]), 0, tex, null); break;
                            case "texture": var t = m.GetTexture(p[0]); Ref(w, t); if (t != null) tex[t.GetInstanceID()] = t; break;
                            default: throw new FormatException("Unknown manifest type " + p[1]);
                        }
                    }
                    catch (Exception e) { w.Key("error"); w.Str(e.GetType().Name + ": " + e.Message); }
                    w.End('}');
                }
                w.End('}');
            }
            w.End('}');
        }

        static void GlobalUniform(JsonOut w, string name, string type, int n, Dictionary<int, Texture> tex)
        {
            w.Begin('{');
            try
            {
                w.Key("value");
                switch (type)
                {
                    case "float": if (n > 0) Value(w, Shader.GetGlobalFloatArray(name), 0, tex, null); else w.Num(Shader.GetGlobalFloat(name)); break;
                    case "int": w.Raw(Shader.GetGlobalInt(name).ToString(CultureInfo.InvariantCulture)); break;
                    case "vector": case "color": if (n > 0) Value(w, Shader.GetGlobalVectorArray(name), 0, tex, null); else Value(w, Shader.GetGlobalVector(name), 0, tex, null); break;
                    case "matrix": if (n > 0) Value(w, Shader.GetGlobalMatrixArray(name), 0, tex, null); else Value(w, Shader.GetGlobalMatrix(name), 0, tex, null); break;
                    case "texture": var t = Shader.GetGlobalTexture(name); Ref(w, t); if (t != null) tex[t.GetInstanceID()] = t; break;
                    default: throw new FormatException("Unknown global type " + type);
                }
            }
            catch (Exception e) { w.Key("error"); w.Str(e.GetType().Name + ": " + e.Message); }
            w.End('}');
        }

        static void TextureRowOrNull(JsonOut w, Texture t) { if (t == null) w.Raw("null"); else TextureRow(w, t); }

        static void TextureRow(JsonOut w, Texture t)
        {
            w.Begin('{'); Ident(w, t); w.Key("type"); w.Str(t.GetType().FullName);
            w.Key("properties"); Members(w, t, 0, null, null, false);
            try { w.Key("native_ptr"); w.Str("0x" + t.GetNativeTexturePtr().ToInt64().ToString("x")); }
            catch (Exception e) { w.Key("native_ptr_error"); w.Str(e.Message); }
            w.End('}');
        }

        // ---- reflection helpers ----

        static void StaticMembers(JsonOut w, Type t, Dictionary<int, Texture> tex, Dictionary<int, Material> mats)
        {
            w.Begin('{');
            foreach (var p in t.GetProperties(BindingFlags.Public | BindingFlags.Static).OrderBy(p => p.Name))
            {
                if (!p.CanRead || p.GetIndexParameters().Length > 0 || Obsolete(p)) continue;
                w.Key(p.Name);
                object v;
                try { v = p.GetValue(null, null); }
                catch (Exception e) { Err(w, e); continue; }
                Value(w, v, 1, tex, mats);
            }
            w.End('}');
        }

        static void Members(JsonOut w, object o, int depth, Dictionary<int, Texture> tex, Dictionary<int, Material> mats, bool nonPublicFields)
        {
            w.Begin('{');
            var t = o.GetType();
            foreach (var p in t.GetProperties(BindingFlags.Public | BindingFlags.Instance).OrderBy(p => p.Name))
            {
                if (!p.CanRead || p.GetIndexParameters().Length > 0 || Obsolete(p)) continue;
                if (SideEffectGetters.Contains(t.FullName + "." + p.Name) || SideEffectGetters.Contains(p.DeclaringType.FullName + "." + p.Name)) { w.Key(p.Name); w.Str("<skipped: instantiating getter>"); continue; }
                if (p.Name == "gameObject" || p.Name == "transform") { w.Key(p.Name); try { Ref(w, (UnityEngine.Object)p.GetValue(o, null)); } catch (Exception e) { Err(w, e); } continue; }
                w.Key(p.Name);
                object v;
                try { v = p.GetValue(o, null); }
                catch (Exception e) { Err(w, e); continue; }
                Value(w, v, depth + 1, tex, mats);
            }
            foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.Instance | (nonPublicFields ? BindingFlags.NonPublic : 0)).OrderBy(f => f.Name))
            {
                w.Key("field:" + f.Name);
                object v;
                try { v = f.GetValue(o); }
                catch (Exception e) { Err(w, e); continue; }
                Value(w, v, depth + 1, tex, mats);
            }
            w.End('}');
        }

        static void AllFields(JsonOut w, object o, int depth, Dictionary<int, Texture> tex, Dictionary<int, Material> mats)
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
                    Value(w, v, depth + 1, tex, mats);
                }
            }
            w.End('}');
        }

        static bool Obsolete(MemberInfo m) { return m.GetCustomAttributes(typeof(ObsoleteAttribute), true).Length > 0; }

        static void Err(JsonOut w, Exception e)
        {
            var inner = e is TargetInvocationException && e.InnerException != null ? e.InnerException : e;
            w.Begin('{'); w.Key("error"); w.Str(inner.GetType().Name + ": " + inner.Message); w.End('}');
        }

        static void Ident(JsonOut w, UnityEngine.Object o)
        {
            w.Key("id"); w.Raw(o.GetInstanceID().ToString(CultureInfo.InvariantCulture));
            w.Key("name"); w.Str(o.name);
            var c = o as Component;
            if (c != null) { w.Key("path"); w.Str(Path(c.transform)); w.Key("active_in_hierarchy"); w.Raw(c.gameObject.activeInHierarchy ? "true" : "false"); w.Key("layer"); w.Raw(c.gameObject.layer.ToString(CultureInfo.InvariantCulture)); }
        }

        static string Path(Transform t)
        {
            var parts = new List<string>();
            for (; t != null; t = t.parent) parts.Add(t.name + "[" + t.GetSiblingIndex() + "]");
            parts.Reverse();
            return "/" + string.Join("/", parts.ToArray());
        }

        static void Ref(JsonOut w, UnityEngine.Object o)
        {
            if (o == null) { w.Raw("null"); return; }
            w.Begin('{'); w.Key("ref_id"); w.Raw(o.GetInstanceID().ToString(CultureInfo.InvariantCulture));
            w.Key("ref_type"); w.Str(o.GetType().FullName); w.Key("ref_name"); w.Str(o.name); w.End('}');
        }

        static void Names(JsonOut w, IEnumerable<string> names) { w.Begin('['); foreach (var n in names) { w.Next(); w.Str(n); } w.End(']'); }

        static void Value(JsonOut w, object v, int depth, Dictionary<int, Texture> tex, Dictionary<int, Material> mats)
        {
            if (v == null) { w.Raw("null"); return; }
            var uo = v as UnityEngine.Object;
            if (uo != null || (v is UnityEngine.Object))
            {
                if (uo == null) { w.Raw("null"); return; }
                var t = uo as Texture; if (t != null && tex != null) tex[t.GetInstanceID()] = t;
                var m = uo as Material; if (m != null && mats != null) mats[m.GetInstanceID()] = m;
                // ScriptableObject settings (e.g. PostProcessProfile and its effect settings) are expanded.
                if (uo is ScriptableObject && depth < MaxDepth) { w.Begin('{'); w.Key("ref"); Ref(w, uo); w.Key("fields"); AllFields(w, uo, depth, tex, mats); w.End('}'); }
                else Ref(w, uo);
                return;
            }
            switch (v)
            {
                case bool b: w.Raw(b ? "true" : "false"); return;
                case float f: w.Num(f); return;
                case double d: w.Str(d.ToString("R", CultureInfo.InvariantCulture)); return;
                case int i: w.Raw(i.ToString(CultureInfo.InvariantCulture)); return;
                case uint ui: w.Raw(ui.ToString(CultureInfo.InvariantCulture)); return;
                case long l: w.Raw(l.ToString(CultureInfo.InvariantCulture)); return;
                case ulong ul: w.Raw(ul.ToString(CultureInfo.InvariantCulture)); return;
                case short s: w.Raw(s.ToString(CultureInfo.InvariantCulture)); return;
                case ushort us: w.Raw(us.ToString(CultureInfo.InvariantCulture)); return;
                case byte by: w.Raw(by.ToString(CultureInfo.InvariantCulture)); return;
                case sbyte sb: w.Raw(sb.ToString(CultureInfo.InvariantCulture)); return;
                case string str: w.Str(str); return;
                case Enum e: w.Str(e.GetType().Name + "." + e.ToString() + "=" + Convert.ToInt64(e).ToString(CultureInfo.InvariantCulture)); return;
                case Vector2 a: Floats(w, a.x, a.y); return;
                case Vector3 a: Floats(w, a.x, a.y, a.z); return;
                case Vector4 a: Floats(w, a.x, a.y, a.z, a.w); return;
                case Quaternion a: Floats(w, a.x, a.y, a.z, a.w); return;
                case Color a: Floats(w, a.r, a.g, a.b, a.a); return;
                case Color32 a: w.Begin('['); foreach (var x in new[] { a.r, a.g, a.b, a.a }) { w.Next(); w.Raw(x.ToString(CultureInfo.InvariantCulture)); } w.End(']'); return;
                case Rect a: Floats(w, a.x, a.y, a.width, a.height); return;
                case Bounds a: Floats(w, a.center.x, a.center.y, a.center.z, a.extents.x, a.extents.y, a.extents.z); return;
                case Matrix4x4 a:
                    w.Begin('['); for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) { w.Next(); w.Num(a[r, c]); } w.End(']'); return;
                case SphericalHarmonicsL2 sh: SH(w, sh); return;
                case IntPtr p: w.Str("ptr:0x" + p.ToInt64().ToString("x")); return;
                case Delegate dl: w.Str("delegate:" + dl.GetType().FullName); return;
            }
            if (depth >= MaxDepth) { w.Str("<depth limit: " + v.GetType().FullName + ">"); return; }
            var dict = v as IDictionary;
            if (dict != null)
            {
                w.Begin('{'); int n = 0;
                foreach (DictionaryEntry kv in dict) { if (n++ >= MaxElements) { w.Key("<truncated>"); w.Raw(dict.Count.ToString(CultureInfo.InvariantCulture)); break; } w.Key(Convert.ToString(kv.Key, CultureInfo.InvariantCulture)); Value(w, kv.Value, depth + 1, tex, mats); }
                w.End('}'); return;
            }
            var en = v as IEnumerable;
            if (en != null)
            {
                w.Begin('['); int n = 0;
                foreach (var x in en) { if (n++ >= MaxElements) { w.Next(); w.Str("<truncated>"); break; } w.Next(); Value(w, x, depth + 1, tex, mats); }
                w.End(']'); return;
            }
            var type = v.GetType();
            if (type.IsPrimitive) { w.Str(Convert.ToString(v, CultureInfo.InvariantCulture)); return; }
            // Plain managed object/struct (plugin settings, PPv2 parameter overrides...): all fields.
            w.Begin('{'); w.Key("$type"); w.Str(type.FullName);
            for (var t = type; t != null && t != typeof(object) && t != typeof(ValueType); t = t.BaseType)
                foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly))
                {
                    w.Key(t.Name + "." + f.Name);
                    object fv;
                    try { fv = f.GetValue(v); } catch (Exception e) { Err(w, e); continue; }
                    Value(w, fv, depth + 1, tex, mats);
                }
            w.End('}');
        }

        static void Floats(JsonOut w, params float[] f) { w.Begin('['); foreach (var x in f) { w.Next(); w.Num(x); } w.End(']'); }
    }

    // Minimal streaming JSON writer; floats are written with round-trip format and
    // their exact IEEE bits when not finite.
    internal sealed class JsonOut
    {
        readonly StringBuilder _sb = new StringBuilder(1 << 22);
        readonly Stack<bool> _first = new Stack<bool>();
        bool _afterKey;

        public void Begin(char c) { Sep(); _sb.Append(c); _first.Push(true); }
        public void End(char c) { _sb.Append(c); _first.Pop(); }
        public void Next() { }
        public void Key(string k) { Sep(); _sb.Append(FramePairProbe.Json(k)).Append(':'); _afterKey = true; }
        public void Raw(string s) { Sep(); _sb.Append(s); }
        public void Str(string s) { Sep(); _sb.Append(FramePairProbe.Json(s)); }
        public void Num(float f)
        {
            Sep();
            if (float.IsNaN(f) || float.IsInfinity(f)) _sb.Append("{\"nonfinite_bits\":").Append(BitConverter.ToInt32(BitConverter.GetBytes(f), 0)).Append('}');
            else _sb.Append(f.ToString("R", CultureInfo.InvariantCulture));
        }

        void Sep()
        {
            if (_afterKey) { _afterKey = false; return; }
            if (_first.Count == 0) return;
            if (_first.Peek()) { _first.Pop(); _first.Push(false); }
            else _sb.Append(',');
        }

        public override string ToString() { return _sb.ToString(); }
    }
}
