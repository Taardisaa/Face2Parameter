// Probe v7: local TRS of the transforms above Camera.main and the scene lights (orchestrator
// request 2026-10-25: CharaCamera, Main Camera's parent, and "Lights Custom" were assumed identity).
// Every chain is written root first with each transform's local and world TRS (float32 round-trip
// numbers), plus every scene transform named CharaCamera or Lights Custom with its path.
// Getters only.
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace HS2Codex.FramePairProbe
{
    internal static class TransformChains
    {
        public static readonly string[] Named = { "CharaCamera", "Lights Custom" };

        public static void Write(JsonOut w)
        {
            w.Begin('{');
            var cam = Camera.main;
            w.Key("camera_main"); if (cam == null) w.Raw("null"); else SceneState.Ref(w, cam);
            w.Key("camera_main_chain"); if (cam == null) w.Raw("null"); else Chain(w, cam.transform);
            var all = Resources.FindObjectsOfTypeAll<Transform>().Where(t => t.gameObject.scene.IsValid()).ToArray();
            w.Key("named"); w.Begin('{');
            foreach (var name in Named)
            {
                w.Key(name); w.Begin('[');
                foreach (var t in all.Where(t => t.name == name).OrderBy(t => t.GetInstanceID()))
                {
                    w.Begin('{');
                    w.Key("transform"); SceneState.Ref(w, t);
                    w.Key("path"); w.Str(Path(t));
                    w.Key("active_in_hierarchy"); w.Bool(t.gameObject.activeInHierarchy);
                    w.Key("chain"); Chain(w, t);
                    w.End('}');
                }
                w.End(']');
            }
            w.End('}');
            w.Key("lights"); w.Begin('[');
            foreach (var l in Resources.FindObjectsOfTypeAll<Light>().Where(l => l.gameObject.scene.IsValid()).OrderBy(l => l.GetInstanceID()))
            {
                w.Begin('{');
                w.Key("light"); SceneState.Ref(w, l);
                w.Key("enabled"); w.Bool(l.enabled); w.Key("active_in_hierarchy"); w.Bool(l.gameObject.activeInHierarchy);
                w.Key("path"); w.Str(Path(l.transform));
                w.Key("chain"); Chain(w, l.transform);
                w.End('}');
            }
            w.End(']');
            w.End('}');
        }

        static string Path(Transform t)
        {
            var parts = new List<string>();
            for (var c = t; c != null; c = c.parent) parts.Add(c.name);
            parts.Reverse();
            return "/" + string.Join("/", parts.ToArray());
        }

        static void Chain(JsonOut w, Transform t)
        {
            var chain = new List<Transform>();
            for (var c = t; c != null; c = c.parent) chain.Add(c);
            chain.Reverse();
            w.Begin('[');
            foreach (var c in chain)
            {
                w.Begin('{');
                w.Key("name"); w.Str(c.name); w.Key("id"); w.Int(c.GetInstanceID());
                w.Key("local_position"); V(w, c.localPosition);
                w.Key("local_rotation"); Q(w, c.localRotation);
                w.Key("local_scale"); V(w, c.localScale);
                w.Key("position"); V(w, c.position);
                w.Key("rotation"); Q(w, c.rotation);
                w.End('}');
            }
            w.End(']');
        }

        static void V(JsonOut w, Vector3 v) { w.Begin('['); w.Num(v.x); w.Num(v.y); w.Num(v.z); w.End(']'); }
        static void Q(JsonOut w, Quaternion q) { w.Begin('['); w.Num(q.x); w.Num(q.y); w.Num(q.z); w.Num(q.w); w.End(']'); }
    }
}
