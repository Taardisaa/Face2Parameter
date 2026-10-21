// Explicit plugin/engine member paths named by the generators
// (docs/hs2_offline_plugin_effects.md §4, docs/hs2_offline_ppv2.md §5). Every path
// is resolved by reflection on the installed types and written with its value or
// with the reason it could not be read; nothing is invoked except plain getters
// listed here. The same objects are also dumped completely in effect_components /
// plugin_static_fields; this section fixes the exact member names.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;

namespace HS2Codex.FramePairProbe
{
    internal static class PluginPaths
    {
        const BindingFlags Any = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

        static readonly string[] SssFields = {
            "<Enabled>k__BackingField", "<Dither>k__BackingField", "DitherEdgeTest", "FixPixelLeaks", "ProfilePerObject", "UseProfileTest", "DEBUG_DISTANCE",
            "<Downsampling>k__BackingField", "<ScatteringRadius>k__BackingField", "<DitherIntensity>k__BackingField", "<DitherScale>k__BackingField",
            "DepthTest", "NormalTest", "ProfileColorTest", "ProfileRadiusTest", "EdgeOffset",
            "<ScatteringIterations>k__BackingField", "<ShaderIterations>k__BackingField", "maxDistance",
            "sssColor", "SSS_Layer", "toggleTexture", "NoiseTexture", "LightingPassShader", "ProfileShader" };
        static readonly string[] CtaaFields = {
            "_enabled", "firstFrame", "swap", "PreEnhanceEnabled", "TemporalStability", "frameCounter", "HdrResponse", "AdaptiveEnhance",
            "TemporalJitterScale", "MicroShimmerReduction", "StaticStabilityPower", "jitterScale", "preEnhanceStrength", "preEnhanceClamp",
            "AdaptiveResolve", "rtAccum0", "rtAccum1", "txaaOut", "afterPreEnhace" };
        static readonly string[] ConvolutionFields = { "initFOV", "AllowMSAA", "_BlurMaterial" };

        public static void Write(JsonOut w, ExportContext ctx)
        {
            var main = Camera.main;
            w.Begin('{');
            w.Key("Camera.main"); if (main == null) w.Raw("null"); else SceneState.Ref(w, main);
            if (main != null)
            {
                Item(w, "Camera.main.fieldOfView", () => main.fieldOfView, ctx);
                Item(w, "Camera.main.nearClipPlane", () => main.nearClipPlane, ctx);
                Item(w, "Camera.main.farClipPlane", () => main.farClipPlane, ctx);
                Item(w, "Camera.main.allowHDR", () => main.allowHDR, ctx);
                Item(w, "Camera.main.allowMSAA", () => main.allowMSAA, ctx);
                Item(w, "Camera.main.usePhysicalProperties", () => main.usePhysicalProperties, ctx);
                Item(w, "Camera.main.stereoEnabled", () => main.stereoEnabled, ctx);
                Item(w, "Camera.main.pixelRect", () => main.pixelRect, ctx);
                Item(w, "Camera.main.transform.position", () => main.transform.position, ctx);
                Item(w, "Camera.main.transform.rotation", () => main.transform.rotation, ctx);
                w.Key("Camera.main.GetComponents<Behaviour>()"); w.Begin('[');
                foreach (var b in main.GetComponents<Behaviour>())
                { w.Begin('{'); w.Key("type"); w.Str(b == null ? null : b.GetType().FullName); w.Key("enabled"); w.Raw(b == null ? "null" : b.enabled ? "true" : "false"); w.End('}'); }
                w.End(']');

                var sss = Comp(main, "DHH.SSS");
                w.Key("SSS"); Obj(w, sss);
                foreach (var f in SssFields) Item(w, "SSS." + f, () => Field(sss, f), ctx);
                var ctaa = Comp(main, "CTAA_PC");
                w.Key("CTAA_PC"); Obj(w, ctaa);
                Item(w, "CTAA_PC.enabled", () => ((Behaviour)Need(ctaa)).enabled, ctx);
                foreach (var f in CtaaFields) Item(w, "CTAA_PC." + f, () => Field(ctaa, f), ctx);
                var custom = Comp(main, "CharaCustom.CustomRender");
                w.Key("CustomRender"); Obj(w, custom);
                Item(w, "CustomRender.update", () => Field(custom, "update"), ctx);
                Item(w, "CustomRender.<rtCamera>k__BackingField", () => Field(custom, "<rtCamera>k__BackingField"), ctx);
            }
            foreach (var name in new[] { "SSS profile Camera", "SSS Lighting Pass Camera" })
            {
                var all = Resources.FindObjectsOfTypeAll<GameObject>().Where(g => g.scene.IsValid() && g.name == name).ToArray();
                w.Key("GameObject:" + name); w.Begin('{');
                w.Key("GameObject.Find_found"); w.Raw(GameObject.Find(name) != null ? "true" : "false");
                w.Key("scene_objects"); w.Begin('[');
                foreach (var g in all)
                {
                    w.Begin('{'); w.Key("id"); w.Int(g.GetInstanceID()); w.Key("active_in_hierarchy"); w.Raw(g.activeInHierarchy ? "true" : "false");
                    var cam = g.GetComponent<Camera>();
                    w.Key("camera"); if (cam == null) w.Raw("null"); else SceneState.Ref(w, cam);
                    if (cam != null) Item(w, "Camera.allowHDR", () => cam.allowHDR, ctx);
                    var conv = Comp(g, "DHH.SSS_convolution");
                    w.Key("SSS_convolution"); Obj(w, conv);
                    if (conv != null) foreach (var f in ConvolutionFields) Item(w, "SSS_convolution." + f, () => Field(conv, f), ctx);
                    w.End('}');
                }
                w.End(']');
                w.End('}');
            }
            Item(w, "DHH.DHH_Base.ConfigSSS.Value", () =>
            {
                var cfg = StaticField("DHH.DHH_Base", "ConfigSSS");
                if (cfg == null) throw new InvalidOperationException("ConfigSSS is null");
                return cfg.GetType().GetProperty("Value", BindingFlags.Instance | BindingFlags.Public).GetValue(cfg, null);
            }, ctx);
            Item(w, "DHH_Main.depthOfFieldLayer.focusDistance.value", () =>
            {
                var t = FindType("DHH.DHH_Main");
                var main0 = UnityEngine.Object.FindObjectOfType(t);
                if (main0 == null) throw new InvalidOperationException("no active DHH.DHH_Main");
                return Field(Field(Field(main0, "depthOfFieldLayer"), "focusDistance"), "value");
            }, ctx);
            Item(w, "Screen.width", () => Screen.width, ctx);
            Item(w, "Screen.height", () => Screen.height, ctx);
            Item(w, "QualitySettings.antiAliasing", () => QualitySettings.antiAliasing, ctx);
            Item(w, "QualitySettings.pixelLightCount", () => QualitySettings.pixelLightCount, ctx);
            Item(w, "QualitySettings.shadows", () => QualitySettings.shadows, ctx);
            Item(w, "QualitySettings.anisotropicFiltering", () => QualitySettings.anisotropicFiltering, ctx);
            Item(w, "QualitySettings.activeColorSpace", () => QualitySettings.activeColorSpace, ctx);
            PpManager(w, ctx);
            foreach (var shader in new[] { "AIT/Liquid", "AIT/main namida" })
            {
                w.Key("renderers_with_shader:" + shader); w.Begin('[');
                foreach (var r in Resources.FindObjectsOfTypeAll<Renderer>().Where(r => r.gameObject.scene.IsValid()))
                {
                    var mats = r.sharedMaterials;
                    for (int i = 0; i < mats.Length; i++)
                        if (mats[i] != null && mats[i].shader != null && mats[i].shader.name == shader)
                        {
                            ctx.Add(mats[i]);
                            w.Begin('{'); w.Key("renderer"); SceneState.Ref(w, r); w.Key("slot"); w.Int(i); w.Key("material"); SceneState.Ref(w, mats[i]); w.End('}');
                        }
                }
                w.End(']');
            }
            w.End('}');
        }

        // PostProcessManager.instance volume registration (m_Volumes order, m_SortedVolumes per layer mask).
        static void PpManager(JsonOut w, ExportContext ctx)
        {
            w.Guard("PostProcessManager", p =>
            {
                var t = FindType("UnityEngine.Rendering.PostProcessing.PostProcessManager");
                var inst = t.GetProperty("instance", BindingFlags.Static | BindingFlags.Public).GetValue(null, null);
                p.Begin('{');
                p.Key("m_Volumes"); Volumes(p, Field(inst, "m_Volumes") as IEnumerable);
                p.Key("m_SortedVolumes"); p.Begin('[');
                var sorted = Field(inst, "m_SortedVolumes") as IDictionary;
                foreach (DictionaryEntry kv in sorted)
                { p.Begin('{'); p.Key("layer_mask"); p.Raw(Convert.ToString(kv.Key, System.Globalization.CultureInfo.InvariantCulture)); p.Key("volumes"); Volumes(p, kv.Value as IEnumerable); p.End('}'); }
                p.End(']');
                p.Key("m_SortNeeded"); SceneState.Value(p, Field(inst, "m_SortNeeded"), 1, ctx);
                p.End('}');
            });
        }

        static void Volumes(JsonOut w, IEnumerable list)
        {
            w.Begin('[');
            if (list != null)
                foreach (var o in list)
                {
                    var b = o as Behaviour;
                    if (b == null) { w.Raw("null"); continue; }
                    w.Begin('{'); w.Key("volume"); SceneState.Ref(w, b); w.Key("enabled"); w.Raw(b.enabled ? "true" : "false");
                    w.Key("active_in_hierarchy"); w.Raw(b.gameObject.activeInHierarchy ? "true" : "false");
                    w.Key("priority"); w.Raw(Convert.ToString(Field(b, "priority"), System.Globalization.CultureInfo.InvariantCulture));
                    w.Key("isGlobal"); w.Raw(((bool)Field(b, "isGlobal")) ? "true" : "false");
                    w.End('}');
                }
            w.End(']');
        }

        static void Item(JsonOut w, string key, Func<object> get, ExportContext ctx)
        {
            w.Guard(key, v => SceneState.Value(v, get(), 1, ctx));
        }

        static void Obj(JsonOut w, Component c) { if (c == null) w.Raw("null"); else SceneState.Ref(w, c); }
        static object Need(object o) { if (o == null) throw new InvalidOperationException("component absent"); return o; }

        static Type FindType(string name)
        {
            var t = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType(name, false)).FirstOrDefault(x => x != null);
            if (t == null) throw new InvalidOperationException("type not loaded: " + name);
            return t;
        }

        static Component Comp(Component c, string type) { return Comp(c.gameObject, type); }
        static Component Comp(GameObject g, string type)
        {
            return g.GetComponents<Component>().FirstOrDefault(x => x != null && x.GetType().FullName == type);
        }

        public static object Field(object o, string name)
        {
            if (o == null) throw new InvalidOperationException("owner is null");
            for (var t = o.GetType(); t != null; t = t.BaseType)
            {
                var f = t.GetField(name, Any | BindingFlags.DeclaredOnly);
                if (f != null) return f.GetValue(o);
            }
            throw new MissingFieldException(o.GetType().FullName, name);
        }

        static object StaticField(string type, string name)
        {
            var f = FindType(type).GetField(name, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
            if (f == null) throw new MissingFieldException(type, name);
            return f.GetValue(null);
        }
    }
}
