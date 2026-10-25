// Probe v7: the animation state that decides which transforms an Animator rewrites (also inside
// the bridge's PoseFreeze: speed 0, Play(fullPathHash, 0, 0), Animator.Update(0)).
//
// Mecanim binds every curve of the controller's clips (genericBindings, path = CRC32 of the
// transform path below the Animator's GameObject) to the hierarchy; the bindings are not reachable
// through the runtime managed API, so this exports the exact identity of what is bound:
//   * every Animator under the ChaControl with its transform, controller, avatar and layer state,
//     the controller's clips and the layer-0 current/next clip infos;
//   * the Maker pose source: CustomBase.poseNo -> chaListCtrl category custom_pose_m/f (500/501)
//     -> MainManifest/MainAB/MainData/Clip/IKAB/IKData (CustomBase.ChangeAnimation,
//     decompiled_acs/CharaCustom/CustomBase.cs 1464-1526), and the bundle file
//     AssetBundleManager.BaseDownloadingURL + MainAB that ChaControl.LoadAnimation loads.
// tools/offline_renderer/pose_clip_bindings.py parses that bundle offline (controller TOS, state
// blend-tree clip ids, clip genericBindings) and resolves the binding paths against static.json.
// Only side-effect-free getters are used; Singleton<T>.Instance is read only after IsInstance().
using System;
using System.IO;
using System.Linq;
using AIChara;
using UnityEngine;

namespace HS2Codex.FramePairProbe
{
    internal static class AnimationState
    {
        public static void Write(JsonOut w, ChaControl cha)
        {
            if (cha == null) throw new InvalidOperationException("no ChaControl bound to the export context");
            w.Begin('{');
            w.Key("animators"); w.Begin('[');
            foreach (var an in cha.GetComponentsInChildren<Animator>(true)) w.GuardValue(a => AnimatorRow(a, an));
            w.End(']');
            w.Guard("maker_pose_source", p => MakerPose(p, cha));
            w.End('}');
        }

        static void AnimatorRow(JsonOut w, Animator an)
        {
            w.Begin('{');
            w.Key("animator"); SceneState.Ref(w, an);
            w.Key("transform"); SceneState.Ref(w, an.transform);
            w.Key("enabled"); w.Bool(an.enabled); w.Key("active_in_hierarchy"); w.Bool(an.gameObject.activeInHierarchy);
            w.Key("has_transform_hierarchy"); w.Bool(an.hasTransformHierarchy);
            w.Key("avatar"); SceneState.Ref(w, an.avatar);
            w.Key("avatar_is_valid"); w.Bool(an.avatar != null && an.avatar.isValid);
            w.Key("is_human"); w.Bool(an.isHuman);
            w.Key("speed"); w.Num(an.speed); w.Key("update_mode"); w.Str(an.updateMode.ToString());
            w.Key("culling_mode"); w.Str(an.cullingMode.ToString()); w.Key("apply_root_motion"); w.Bool(an.applyRootMotion);
            var rac = an.runtimeAnimatorController;
            w.Key("controller"); SceneState.Ref(w, rac);
            w.Key("controller_clips"); w.Begin('[');
            if (rac != null) foreach (var c in rac.animationClips) Clip(w, c, null);
            w.End(']');
            if (an.isActiveAndEnabled && rac != null && an.layerCount > 0)
            {
                w.Key("layer_count"); w.Int(an.layerCount);
                w.Key("layer0_in_transition"); w.Bool(an.IsInTransition(0));
                w.Key("layer0_current"); State(w, an.GetCurrentAnimatorStateInfo(0));
                w.Key("layer0_next"); State(w, an.GetNextAnimatorStateInfo(0));
                w.Key("layer0_current_clips"); w.Begin('['); foreach (var ci in an.GetCurrentAnimatorClipInfo(0)) Clip(w, ci.clip, ci.weight); w.End(']');
                w.Key("layer0_next_clips"); w.Begin('['); foreach (var ci in an.GetNextAnimatorClipInfo(0)) Clip(w, ci.clip, ci.weight); w.End(']');
            }
            w.End('}');
        }

        static void Clip(JsonOut w, AnimationClip c, float? weight)
        {
            w.Begin('{');
            w.Key("clip"); SceneState.Ref(w, c);
            if (c != null)
            {
                w.Key("length"); w.Num(c.length); w.Key("frame_rate"); w.Num(c.frameRate);
                w.Key("wrap_mode"); w.Str(c.wrapMode.ToString()); w.Key("legacy"); w.Bool(c.legacy);
                w.Key("is_looping"); w.Bool(c.isLooping); w.Key("human_motion"); w.Bool(c.humanMotion);
                w.Key("local_bounds"); w.Begin('[');
                var b = c.localBounds; w.Num(b.center.x); w.Num(b.center.y); w.Num(b.center.z); w.Num(b.extents.x); w.Num(b.extents.y); w.Num(b.extents.z);
                w.End(']');
            }
            if (weight.HasValue) { w.Key("weight"); w.Num(weight.Value); }
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

        static void MakerPose(JsonOut w, ChaControl cha)
        {
            if (!Singleton<CharaCustom.CustomBase>.IsInstance()) { w.Str("CharaCustom.CustomBase not instantiated (not in the Maker)"); return; }
            if (!Singleton<Manager.Character>.IsInstance()) throw new InvalidOperationException("Manager.Character not instantiated");
            var cb = Singleton<CharaCustom.CustomBase>.Instance;
            var type = cha.sex == 0 ? ChaListDefine.CategoryNo.custom_pose_m : ChaListDefine.CategoryNo.custom_pose_f;
            var info = Singleton<Manager.Character>.Instance.chaListCtrl.GetCategoryInfo(type);
            var keys = info.Keys.ToArray();
            int pose = cb.poseNo;
            w.Begin('{');
            w.Key("category"); w.Str(type + "=" + (int)type);
            w.Key("pose_no"); w.Int(pose); w.Key("back_pose_no"); w.Int(cb.backPoseNo);
            var stateName = PluginPaths.Field(cb, "animeStateName") as string;
            w.Key("anime_state_name"); if (stateName == null) w.Raw("null"); else w.Str(stateName);
            w.Key("entry");
            ListInfoBase entry;
            if (pose < 0 || pose >= keys.Length || !info.TryGetValue(keys[pose], out entry)) w.Raw("null");
            else
            {
                w.Begin('{');
                w.Key("list_id"); w.Int(keys[pose]);
                foreach (var k in new[] { ChaListDefine.KeyType.MainManifest, ChaListDefine.KeyType.MainAB, ChaListDefine.KeyType.MainData,
                                          ChaListDefine.KeyType.Clip, ChaListDefine.KeyType.IKAB, ChaListDefine.KeyType.IKData })
                { w.Key(k.ToString()); w.Str(entry.GetInfo(k)); }
                string bundle = AssetBundleManager.BaseDownloadingURL + entry.GetInfo(ChaListDefine.KeyType.MainAB);
                w.Key("bundle_file"); w.Str(bundle);
                w.Key("bundle_file_exists"); w.Bool(File.Exists(bundle));
                w.End('}');
            }
            w.End('}');
        }
    }
}
