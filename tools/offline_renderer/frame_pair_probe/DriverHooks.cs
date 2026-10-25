// Per-frame driver hooks (probe v6, request DriverHooks=1), docs/hs2_offline_head_skinning.md §9.7.
//
// HarmonyX (BepInEx/core/0Harmony.dll 2.9.0) prefixes/postfixes, installed for the window only and
// removed afterwards. A Harmony patch replaces the patched method with a re-JITted copy; the
// drivers' float32 results depend on Mono JIT register allocation (head_skinning.md §9.1), so the
// hooks sit on the CALLERS whose bodies contain no driver arithmetic, never on the drivers:
//   eye   EyeLookController.LateUpdate        body: eyeLookScript.EyeUpdateCalc(target.position, ptnNo)
//   fbs   FaceBlendShape.<Start> lambda       body: OnLateUpdate()  (UniRx EveryLateUpdate subscription;
//                                             verified from IL that it calls OnLateUpdate)
//   chara ChaControl.UpdateForce / LateUpdateForce   bodies: calls only
//   abmx  KKABMX.Core.BoneController.LateUpdate      body: calls only (ApplyEffects etc.)
//   neck  NeckLookControllerVer2.LateUpdate  EXCEPTION: its body contains the target Lerp loop
//         (Mathf.Lerp over x,y) before NeckUpdateCalc; patching it re-JITs that loop. Recorded as
//         jit_risk in every neck record; the callees UpdateCall/NeckUpdateCalc are not patched.
// Prefixes run first (Priority.First), postfixes last, so other plugins' patches (Hooah/PovX on the
// neck controller) execute inside the bracket. Every record carries the global ProbeSequence value
// (shared with the camera callbacks) as the order certificate. Hooks never throw into the game.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using AIChara;
using HarmonyLib;
using UnityEngine;

namespace HS2Codex.FramePairProbe
{
    internal sealed class DriverHooks
    {
        public static DriverHooks Active;
        const BindingFlags Any = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

        readonly Harmony _harmony = new Harmony("codex.hs2.framepairprobe.v6.drivers");
        readonly ExportContext _ctx;
        readonly ChaControl _cha;
        readonly Transform[] _transforms;
        readonly SkinnedMeshRenderer[] _smrs;
        readonly List<string> _log;
        JsonOut _w;
        int _records;
        public int Errors;
        public readonly List<string> Patched = new List<string>();
        Type _eyeCalcType, _hooahType;

        DriverHooks(ExportContext ctx, ChaControl cha, Transform[] transforms, SkinnedMeshRenderer[] smrs, List<string> log)
        {
            _ctx = ctx; _cha = cha; _transforms = transforms; _smrs = smrs; _log = log;
            StartFrame();
        }

        static Type T(string name)
        {
            var t = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType(name, false)).FirstOrDefault(x => x != null);
            if (t == null) throw new InvalidOperationException("Driver hook target type not loaded: " + name);
            return t;
        }

        static MethodInfo M(Type t, string name)
        {
            var m = t.GetMethod(name, Any | BindingFlags.DeclaredOnly, null, Type.EmptyTypes, null);
            if (m == null) throw new InvalidOperationException("Driver hook target missing: " + t.FullName + "." + name + "()");
            return m;
        }

        // The UniRx subscription lambda of FaceBlendShape.Start: an instance method of
        // FaceBlendShape (it captures only 'this'), void(long), whose IL calls OnLateUpdate.
        static MethodInfo FbsLambda(Type fbs)
        {
            var onLate = M(fbs, "OnLateUpdate");
            var found = new List<MethodInfo>();
            foreach (var m in fbs.GetMethods(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (!m.Name.StartsWith("<Start>") || m.ReturnType != typeof(void) || m.GetParameters().Length != 1) continue;
                var il = m.GetMethodBody().GetILAsByteArray();
                for (int i = 0; i + 4 < il.Length; i++)
                    if (il[i] == OpCodes.Call.Value || il[i] == OpCodes.Callvirt.Value)
                    {
                        MethodBase callee = null;
                        try { callee = m.Module.ResolveMethod(BitConverter.ToInt32(il, i + 1)); } catch { }
                        if (callee != null && callee.MetadataToken == onLate.MetadataToken && callee.Module == onLate.Module) { found.Add(m); break; }
                    }
            }
            if (found.Count != 1) throw new InvalidOperationException("Expected exactly one FaceBlendShape <Start> lambda calling OnLateUpdate, found " + found.Count);
            return found[0];
        }

        public static DriverHooks Install(ExportContext ctx, ChaControl cha, Transform[] transforms, SkinnedMeshRenderer[] smrs, List<string> log)
        {
            if (Active != null) throw new InvalidOperationException("Driver hooks already installed");
            var h = new DriverHooks(ctx, cha, transforms, smrs, log);
            var targets = new List<KeyValuePair<string, MethodInfo>>
            {
                new KeyValuePair<string, MethodInfo>("eye", M(T("EyeLookController"), "LateUpdate")),
                new KeyValuePair<string, MethodInfo>("neck", M(T("NeckLookControllerVer2"), "LateUpdate")),
                new KeyValuePair<string, MethodInfo>("fbs", FbsLambda(T("FaceBlendShape"))),
                new KeyValuePair<string, MethodInfo>("chara_update_force", M(typeof(ChaControl), "UpdateForce")),
                new KeyValuePair<string, MethodInfo>("chara_late_update_force", M(typeof(ChaControl), "LateUpdateForce")),
                new KeyValuePair<string, MethodInfo>("abmx", M(T("KKABMX.Core.BoneController"), "LateUpdate")),
            };
            h._eyeCalcType = T("EyeLookCalc");
            // Hooah (BepInEx/plugins/HS2_Plugins/Hooh/HS2_Hooah.dll): global-namespace class HijackLook,
            // public static Dictionary<EyeLookController, Vector3> _updatePositions (its only static
            // initializer allocates that empty dictionary).
            var hooahAsm = AppDomain.CurrentDomain.GetAssemblies().FirstOrDefault(a => a.GetName().Name == "HS2_Hooah");
            h._hooahType = hooahAsm == null ? null : hooahAsm.GetType("HijackLook", false);
            Active = h;
            try
            {
                foreach (var kv in targets)
                {
                    var pre = new HarmonyMethod(typeof(DriverHooks).GetMethod("Pre_" + kv.Key, BindingFlags.Static | BindingFlags.NonPublic)) { priority = Priority.First };
                    var post = new HarmonyMethod(typeof(DriverHooks).GetMethod("Post_" + kv.Key, BindingFlags.Static | BindingFlags.NonPublic)) { priority = Priority.Last };
                    h._harmony.Patch(kv.Value, pre, post);
                    h.Patched.Add(kv.Key + " -> " + kv.Value.DeclaringType.FullName + "." + kv.Value.Name);
                }
            }
            catch
            {
                h.Uninstall();
                throw;
            }
            foreach (var p in h.Patched) log.Add("driver hook installed: " + p);
            return h;
        }

        public void Uninstall()
        {
            try { _harmony.UnpatchSelf(); }
            finally { Active = null; }
        }

        void StartFrame() { _w = new JsonOut(1 << 20); _w.Begin('['); _records = 0; }

        public void Flush(string path, int windowIndex)
        {
            _w.End(']');
            var doc = new JsonOut(_w.Length + 512);
            doc.Begin('{');
            doc.Key("schema_version"); doc.Raw("6");
            doc.Key("window_index"); doc.Int(windowIndex);
            doc.Key("frame_count"); doc.Int(Time.frameCount);
            doc.Key("record_count"); doc.Int(_records);
            doc.Key("hook_errors_total"); doc.Int(Errors);
            doc.Key("hooks"); SceneState.Names(doc, Patched);
            doc.Key("records"); doc.Raw(_w.ToString());
            doc.End('}');
            File.WriteAllText(path, doc.ToString());
            StartFrame();
        }

        // ---- Harmony entry points (never throw) ----
        static void Pre_eye(object __instance) { Run("eye", "prefix", __instance); }
        static void Post_eye(object __instance) { Run("eye", "postfix", __instance); }
        static void Pre_neck(object __instance) { Run("neck", "prefix", __instance); }
        static void Post_neck(object __instance) { Run("neck", "postfix", __instance); }
        static void Pre_fbs(object __instance) { Run("fbs", "prefix", __instance); }
        static void Post_fbs(object __instance) { Run("fbs", "postfix", __instance); }
        static void Pre_chara_update_force(object __instance) { Run("chara_update_force", "prefix", __instance); }
        static void Post_chara_update_force(object __instance) { Run("chara_update_force", "postfix", __instance); }
        static void Pre_chara_late_update_force(object __instance) { Run("chara_late_update_force", "prefix", __instance); }
        static void Post_chara_late_update_force(object __instance) { Run("chara_late_update_force", "postfix", __instance); }
        static void Pre_abmx(object __instance) { Run("abmx", "prefix", __instance); }
        static void Post_abmx(object __instance) { Run("abmx", "postfix", __instance); }

        static void Run(string hook, string phase, object instance)
        {
            var h = Active;
            if (h == null) return;
            try { h.Record(hook, phase, instance); }
            catch (Exception e)
            {
                h.Errors++;
                if (h.Errors <= 50) h._log.Add("driver hook " + hook + " " + phase + " error at frame " + Time.frameCount + ": " + e);
            }
        }

        void Record(string hook, string phase, object instance)
        {
            var body = new JsonOut(1 << 14);
            _ctx.Path.Clear();
            long seq = ProbeSequence.Next();
            body.Begin('{');
            body.Key("seq"); body.Int(seq);
            body.Key("hook"); body.Str(hook); body.Key("phase"); body.Str(phase);
            body.Key("frame_count"); body.Int(Time.frameCount);
            body.Key("time"); body.Num(Time.time); body.Key("delta_time"); body.Num(Time.deltaTime);
            var uo = instance as UnityEngine.Object;
            body.Key("instance"); if (uo == null) body.Raw("null"); else SceneState.Ref(body, uo);
            var comp = instance as Component;
            body.Key("instance_in_window_character"); body.Bool(comp != null && comp.transform.IsChildOf(_cha.transform));
            switch (hook)
            {
                case "eye": Eye(body, (Component)instance, phase); break;
                case "neck": Neck(body, (Component)instance); break;
                case "fbs": Fbs(body, (Component)instance, phase); break;
                case "chara_update_force":
                    if (phase == "postfix") Mouth(body, (ChaControl)instance);
                    break;
                case "chara_late_update_force":
                    body.Guard("skin_slots", s => SkinSlots(s));
                    break;
                case "abmx":
                    if (phase == "postfix") { body.Guard("mouth", m => MouthScales(m)); body.Guard("skin_slots", s => SkinSlots(s)); }
                    break;
            }
            body.End('}');
            _w.Raw(body.ToString());
            _records++;
        }

        // Field, else a plain property getter (data members of the game's component classes).
        static object Member(object o, string name)
        {
            for (var t = o.GetType(); t != null; t = t.BaseType)
            {
                var f = t.GetField(name, Any | BindingFlags.DeclaredOnly);
                if (f != null) return f.GetValue(o);
                var p = t.GetProperty(name, Any | BindingFlags.DeclaredOnly);
                if (p != null && p.GetIndexParameters().Length == 0) return p.GetValue(o, null);
            }
            throw new MissingMemberException(o.GetType().FullName, name);
        }

        void Fields(JsonOut w, string key, object o)
        {
            w.Key(key);
            if (o == null || (o is UnityEngine.Object && (UnityEngine.Object)o == null)) { w.Raw("null"); return; }
            _ctx.Path.Clear();
            SceneState.AllFields(w, o, 0, _ctx);
        }

        void Eye(JsonOut w, Component controller, string phase)
        {
            var calc = PluginPaths.Field(controller, "eyeLookScript");
            var target = PluginPaths.Field(controller, "target") as Transform;
            w.Key("jit_risk"); w.Str("none: hooked EyeLookController.LateUpdate forwards target.position only; EyeUpdateCalc not patched");
            w.Key("call_arguments"); w.Begin('{');
            w.Key("target_position"); if (target == null) w.Raw("null"); else SceneState.Value(w, target.position, 0, _ctx);
            w.Key("ptnNo"); SceneState.Value(w, PluginPaths.Field(controller, "ptnNo"), 0, _ctx);
            w.End('}');
            w.Key("target"); if (target == null) w.Raw("null"); else SceneState.Ref(w, target);
            Fields(w, "controller_fields", controller);
            Fields(w, "eye_look_calc_fields", calc);
            w.Key("EyeLookCalc.isEnabled"); SceneState.Value(w, _eyeCalcType.GetField("isEnabled", Any).GetValue(null), 0, _ctx);
            if (phase == "prefix")
            {
                w.Guard("hooah_hijack_look_update_positions", hw =>
                {
                    if (_hooahType == null) { hw.Str("HS2_Hooah HijackLook not loaded"); return; }
                    var f = _hooahType.GetField("_updatePositions", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
                    if (f == null) { hw.Str("static field _updatePositions not found"); return; }
                    var c = f.GetValue(null) as ICollection;
                    hw.Begin('{'); hw.Key("count"); if (c == null) hw.Raw("null"); else hw.Int(c.Count);
                    hw.Key("contains_this_controller"); hw.Bool(c is IDictionary && ((IDictionary)c).Contains(controller)); hw.End('}');
                });
            }
        }

        void Neck(JsonOut w, Component controller)
        {
            var calc = PluginPaths.Field(controller, "neckLookScript");
            w.Key("jit_risk"); w.Str("NeckLookControllerVer2.LateUpdate is patched and contains the target Mathf.Lerp loop (re-JITted); NeckLookCalcVer2.UpdateCall/NeckUpdateCalc are not patched");
            var target = PluginPaths.Field(controller, "target") as Transform;
            w.Key("controller_position"); SceneState.Value(w, controller.transform.position, 0, _ctx);
            w.Key("target"); if (target == null) w.Raw("null"); else { w.Begin('{'); w.Key("ref"); SceneState.Ref(w, target); w.Key("position"); SceneState.Value(w, target.position, 0, _ctx); w.End('}'); }
            Fields(w, "controller_fields", controller);
            Fields(w, "neck_look_calc_fields", calc);
            if (calc != null)
            {
                w.Guard("change_type_lerp_curve_eval", e =>
                {
                    var curve = (AnimationCurve)PluginPaths.Field(calc, "changeTypeLerpCurve");
                    float leap = (float)PluginPaths.Field(calc, "changeTypeLeapTime"), timer = (float)PluginPaths.Field(calc, "changeTypeTimer");
                    float t = Mathf.InverseLerp(0f, leap, timer);
                    e.Begin('{'); e.Key("inverse_lerp"); e.Num(t); e.Key("evaluate"); e.Num(curve.Evaluate(t)); e.End('}');
                });
                w.Guard("bone_transforms", b =>
                {
                    b.Begin('[');
                    foreach (var o in (IEnumerable)PluginPaths.Field(calc, "aBones"))
                    {
                        b.Begin('{');
                        foreach (var name in new[] { "referenceCalc", "neckBone", "controlBone" })
                        {
                            var t = PluginPaths.Field(o, name) as Transform;
                            b.Key(name); if (t == null) b.Raw("null"); else LocalTrs(b, t);
                        }
                        b.End('}');
                    }
                    b.End(']');
                });
            }
        }

        void Fbs(JsonOut w, Component fbs, string phase)
        {
            w.Key("jit_risk"); w.Str("none: hooked the UniRx subscription lambda that calls OnLateUpdate; OnLateUpdate not patched");
            Fields(w, "face_blend_shape_fields", fbs);
            w.Key("blink_ctrl_ex_present"); w.Bool(PluginPaths.Field(fbs, "BlinkCtrlEx") != null);
            w.Key("random_state"); SceneState.Value(w, UnityEngine.Random.state, 0, _ctx);
            if (phase == "prefix")
            {
                // First LateUpdate (order -16000), after the animation phase.
                w.Guard("animation", a => Animation(a));
                w.Guard("camera_and_ancestors", c => CameraChain(c));
                Mouth(w, _cha);
            }
            w.Guard("skin_slots", s => SkinSlots(s));
        }

        void Mouth(JsonOut w, ChaControl cha)
        {
            w.Guard("mouth", m =>
            {
                m.Begin('{');
                var voice = cha.asVoice;
                m.Key("as_voice_present"); m.Bool(voice != null);
                m.Key("as_voice_playing"); m.Bool(voice != null && voice.isPlaying);
                m.Key("file_status_mouth_adjust_width"); m.Bool(cha.fileStatus.mouthAdjustWidth);
                m.Key("scales"); MouthScales(m);
                m.End('}');
            });
        }

        void MouthScales(JsonOut w)
        {
            w.Begin('{');
            Transform adjust = null;
            try
            {
                var head = Member(_cha, "cmpBoneHead");
                var etc = head == null ? null : Member(head, "targetEtc");
                adjust = etc == null ? null : Member(etc, "trfMouthAdjustWidth") as Transform;
            }
            catch (Exception e) { w.Key("trf_mouth_adjust_width_error"); w.Str(e.Message); }
            w.Key("trf_mouth_adjust_width"); if (adjust == null) w.Raw("null"); else LocalTrs(w, adjust);
            var named = _transforms.Where(t => t != null && t.name == "cf_J_MouthMove").ToArray();
            w.Key("cf_J_MouthMove"); w.Begin('['); foreach (var t in named) LocalTrs(w, t); w.End(']');
            w.End('}');
        }

        void Animation(JsonOut w)
        {
            w.Begin('{');
            var f = new float[_transforms.Length * 10];
            for (int i = 0; i < _transforms.Length; i++)
            {
                var t = _transforms[i]; int o = i * 10;
                var p = t.localPosition; var q = t.localRotation; var s = t.localScale;
                f[o] = p.x; f[o + 1] = p.y; f[o + 2] = p.z; f[o + 3] = q.x; f[o + 4] = q.y; f[o + 5] = q.z; f[o + 6] = q.w; f[o + 7] = s.x; f[o + 8] = s.y; f[o + 9] = s.z;
            }
            var b = new byte[f.Length * 4]; Buffer.BlockCopy(f, 0, b, 0, b.Length);
            w.Key("transform_layout"); w.Str("static.json transform order; localPosition3 localRotation4 localScale3, float32 little endian");
            w.Key("local_trs_f32"); w.Str(Convert.ToBase64String(b));
            w.Key("animators"); w.Begin('[');
            foreach (var an in _cha.GetComponentsInChildren<Animator>(true))
            {
                w.Begin('{');
                w.Key("animator"); SceneState.Ref(w, an); w.Key("enabled"); w.Bool(an.enabled); w.Key("active"); w.Bool(an.gameObject.activeInHierarchy);
                w.Key("controller"); SceneState.Ref(w, an.runtimeAnimatorController);
                w.Key("avatar"); SceneState.Ref(w, an.avatar); w.Key("is_human"); w.Bool(an.isHuman);
                w.Key("speed"); w.Num(an.speed); w.Key("update_mode"); w.Str(an.updateMode.ToString());
                w.Key("culling_mode"); w.Str(an.cullingMode.ToString()); w.Key("apply_root_motion"); w.Bool(an.applyRootMotion);
                if (an.isActiveAndEnabled && an.runtimeAnimatorController != null && an.layerCount > 0)
                {
                    w.Key("layer0_in_transition"); w.Bool(an.IsInTransition(0));
                    w.Key("layer0_current"); State(w, an.GetCurrentAnimatorStateInfo(0));
                    w.Key("layer0_next"); State(w, an.GetNextAnimatorStateInfo(0));
                    var tr = an.GetAnimatorTransitionInfo(0);
                    w.Key("layer0_transition"); w.Begin('{'); w.Key("full_path_hash"); w.Int(tr.fullPathHash); w.Key("name_hash"); w.Int(tr.nameHash);
                    w.Key("normalized_time"); w.Num(tr.normalizedTime); w.Key("duration"); w.Num(tr.duration); w.Key("any_state"); w.Bool(tr.anyState); w.End('}');
                }
                w.End('}');
            }
            w.End(']');
            w.Guard("final_ik_full_body_biped_ik", k =>
            {
                var ik = _cha.GetType().GetProperty("fullBodyIK", Any);
                var comp = ik == null ? null : ik.GetValue(_cha, null) as Behaviour;
                if (comp == null) { k.Raw("null"); return; }
                k.Begin('{'); k.Key("component"); SceneState.Ref(k, comp); k.Key("enabled"); k.Bool(comp.enabled);
                _ctx.Path.Clear();
                k.Key("solver"); SceneState.Value(k, PluginPaths.Field(comp, "solver"), 4, _ctx);
                k.End('}');
            });
            w.Guard("maker_motion_ik", k =>
            {
                var t = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("CharaCustom.CustomBase", false)).FirstOrDefault(x => x != null);
                var baseObj = t == null ? null : UnityEngine.Object.FindObjectOfType(t);
                if (baseObj == null) { k.Str("CharaCustom.CustomBase not present"); return; }
                var mik = Member(baseObj, "customMotionIK");
                if (mik == null) { k.Raw("null"); return; }
                _ctx.Path.Clear();
                SceneState.Value(k, mik, 4, _ctx);
            });
            w.End('}');
        }

        static void State(JsonOut w, AnimatorStateInfo s)
        {
            w.Begin('{');
            w.Key("full_path_hash"); w.Int(s.fullPathHash); w.Key("short_name_hash"); w.Int(s.shortNameHash);
            w.Key("normalized_time"); w.Num(s.normalizedTime); w.Key("length"); w.Num(s.length);
            w.Key("speed"); w.Num(s.speed); w.Key("speed_multiplier"); w.Num(s.speedMultiplier); w.Key("loop"); w.Bool(s.loop);
            w.End('}');
        }

        void CameraChain(JsonOut w)
        {
            w.Begin('{');
            var cam = Camera.main;
            w.Key("camera_main"); if (cam == null) w.Raw("null"); else { SceneState.Ref(w, cam); }
            if (cam != null) { w.Key("camera_position"); SceneState.Value(w, cam.transform.position, 0, _ctx); w.Key("camera_ancestors_local_trs"); Chain(w, cam.transform); }
            w.Key("chara_ancestors_local_trs"); Chain(w, _cha.transform);
            w.End('}');
        }

        void Chain(JsonOut w, Transform t)
        {
            w.Begin('[');
            for (var x = t; x != null; x = x.parent) LocalTrs(w, x);
            w.End(']');
        }

        void LocalTrs(JsonOut w, Transform t)
        {
            w.Begin('{');
            w.Key("name"); w.Str(t.name); w.Key("id"); w.Int(t.GetInstanceID());
            w.Key("local_position"); SceneState.Value(w, t.localPosition, 0, _ctx);
            w.Key("local_rotation"); SceneState.Value(w, t.localRotation, 0, _ctx);
            w.Key("local_scale"); SceneState.Value(w, t.localScale, 0, _ctx);
            w.End('}');
        }

        void SkinSlots(JsonOut w)
        {
            w.Begin('[');
            foreach (var r in _smrs)
            {
                if (r == null) continue;
                w.Begin('{'); w.Key("renderer"); w.Int(r.GetInstanceID()); w.Key("name"); w.Str(r.name);
                w.Guard("native", n => { n.Begin('{'); _ctx.Native.WriteSkinOnly(n, r); n.End('}'); });
                w.End('}');
            }
            w.End(']');
        }
    }
}
