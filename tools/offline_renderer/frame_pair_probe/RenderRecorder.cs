// Per-camera-render state recorder (read-only).
//
// Hooks the static Camera.onPreCull / onPreRender / onPostRender delegates. In
// UnityPlayer 2018.4.11f1 they fire AFTER the camera's component messages
// (Camera::CustomCull 0x660f70 sends kPreCull at 0x6610a8, then invokes the static
// callback at 0x6610c6; Camera::CustomRender 0x661400 sends kPreRender at 0x66157f
// then the static callback at 0x6615ae, kPostRender at 0x6617fc then 0x66181a).
// So a pre_cull record sees CTAA/PPv2 OnPreCull results (jittered projection),
// pre_render sees the camera after culling and after DHH SSS OnPreRender (whose
// nested profile/lighting camera renders appear as child records), post_render
// sees the state after the component OnPostRender.
//
// Not exposed by the 2018.4 managed API, hence explicit gaps: the per-camera
// CullResults visible-renderer list and each light's shadow-caster cull result
// (they live in the native CullResults of Camera::StandaloneCustomRender's stack
// frame). Recorded instead: every culling input per render and the renderer
// visibility bit (+0x128 & 0x1000, the isVisible source), which is frame-cumulative.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace HS2Codex.FramePairProbe
{
    internal sealed class BridgeRender
    {
        public int FrameCount, RenderId, Ordinal, TargetId;
    }

    internal sealed class RenderRecorder
    {
        readonly ExportContext _ctx;
        readonly List<string> _log;
        JsonOut _w;
        int _renderInFrame;
        readonly Stack<int> _open = new Stack<int>();
        readonly Dictionary<int, int> _renderOfCamera = new Dictionary<int, int>();
        string _lastGlobalSheet, _lastPool, _lastKeywords, _lastBuiltins, _lastRts;
        readonly Dictionary<int, string> _lastSmd = new Dictionary<int, string>();
        readonly Dictionary<int, string> _lastMaterialSheet = new Dictionary<int, string>();
        Renderer[] _renderers = new Renderer[0];
        Light[] _lights = new Light[0];

        // bridge arming (mode renderstyle)
        int _bridgeFrame = -1, _bridgeW, _bridgeH, _bridgeExpected;
        Camera _bridgeCamera;
        HashSet<int> _preexistingTargets;
        public readonly List<BridgeRender> BridgeRenders = new List<BridgeRender>();
        public Action<int, int> OnBridgeBegin, OnBridgeEnd;   // (frameCount, ordinal)
        public int Errors;
        // static.json transform order; exported at every bridge render's pre_cull (6.1.0).
        public Transform[] CharacterTransforms;

        public RenderRecorder(ExportContext ctx, List<string> log) { _ctx = ctx; _log = log; Refresh(); StartFrame(); }

        public void Attach()
        {
            Camera.onPreCull += PreCull;
            Camera.onPreRender += PreRender;
            Camera.onPostRender += PostRender;
        }

        public void Detach()
        {
            Camera.onPreCull -= PreCull;
            Camera.onPreRender -= PreRender;
            Camera.onPostRender -= PostRender;
        }

        public void Refresh()
        {
            _renderers = Resources.FindObjectsOfTypeAll<Renderer>().Where(r => r.gameObject.scene.IsValid()).ToArray();
            _lights = Resources.FindObjectsOfTypeAll<Light>().Where(l => l.gameObject.scene.IsValid()).ToArray();
            foreach (var r in _renderers) foreach (var m in r.sharedMaterials) _ctx.Add(m);
        }

        public void ArmBridge(int frameCount, Camera camera, int width, int height, int expectedRenders, HashSet<int> preexistingTargets)
        {
            _bridgeFrame = frameCount; _bridgeCamera = camera; _bridgeW = width; _bridgeH = height;
            _bridgeExpected = expectedRenders; _preexistingTargets = preexistingTargets;
        }

        void StartFrame()
        {
            _w = new JsonOut(1 << 22);
            _w.Begin('[');
            _renderInFrame = 0;
            _renderOfCamera.Clear();
        }

        // Writes the records of the frame that just finished rendering.
        public void Flush(string path, int windowIndex)
        {
            _w.End(']');
            var doc = new JsonOut(_w.Length + 512);
            doc.Begin('{');
            doc.Key("schema_version"); doc.Raw("5");
            doc.Key("window_index"); doc.Int(windowIndex);
            doc.Key("frame_count"); doc.Int(Time.frameCount);
            doc.Key("render_count"); doc.Int(_renderInFrame);
            doc.Key("callback_order"); doc.Str("component OnPreCull -> static onPreCull (pre_cull) -> cull -> component OnPreRender -> static onPreRender (pre_render) -> passes -> component OnPostRender -> static onPostRender (post_render)");
            doc.Key("records"); doc.Raw(_w.ToString());
            doc.End('}');
            File.WriteAllText(path, doc.ToString());
            StartFrame();
            if (_open.Count != 0) { _log.Add("render nesting not closed at frame end: " + _open.Count); _open.Clear(); }
        }

        bool IsBridge(Camera cam, out RenderTexture target)
        {
            target = cam.targetTexture;
            return _bridgeFrame == Time.frameCount && cam == _bridgeCamera && target != null && !_preexistingTargets.Contains(target.GetInstanceID())
                   && target.width == _bridgeW && target.height == _bridgeH;
        }

        void PreCull(Camera cam)
        {
            try
            {
                int id = ++_renderInFrame;
                _renderOfCamera[cam.GetInstanceID()] = id;
                RenderTexture target;
                BridgeRender bridge = null;
                if (IsBridge(cam, out target))
                {
                    bridge = new BridgeRender { FrameCount = Time.frameCount, RenderId = id, Ordinal = BridgeRenders.Count, TargetId = target.GetInstanceID() };
                    BridgeRenders.Add(bridge);
                    if (bridge.Ordinal == 0 && OnBridgeBegin != null) OnBridgeBegin(Time.frameCount, 0);
                }
                Record(cam, "pre_cull", id, bridge, w =>
                {
                    w.Key("camera_state"); CameraState(w, cam);
                    // The bridge's PoseFreeze runs before Camera.Render, so these are the transforms
                    // each bridge render culls and draws with (head_skinning.md §10.1).
                    if (bridge != null) w.Guard("character_transforms", t => CharacterTransformBlock(t));
                    w.Guard("render_buffer_pool", p => Dedup(p, ref _lastPool, () => { var j = new JsonOut(); PoolWalker.Write(j, _ctx.Native.Pool()); return j.ToString(); }));
                });
                _open.Push(id);
            }
            catch (Exception e) { Error("pre_cull", e); }
        }

        void PreRender(Camera cam)
        {
            try
            {
                int id; _renderOfCamera.TryGetValue(cam.GetInstanceID(), out id);
                Record(cam, "pre_render", id, BridgeOf(id), w =>
                {
                    w.Key("time"); TimeBlock(w);
                    w.Key("camera_state"); CameraState(w, cam);
                    w.Key("camera_properties"); SceneState.Members(w, cam, 0, _ctx, false);
                    w.Key("reflection_probe_default"); w.Begin('{');
                    w.Key("defaultTexture"); SceneState.TextureRowOrNull(w, ReflectionProbe.defaultTexture, _ctx);
                    w.Key("defaultTextureHDRDecodeValues"); SceneState.Value(w, ReflectionProbe.defaultTextureHDRDecodeValues, 0, _ctx);
                    w.End('}');
                    Globals(w);
                    w.Key("lights"); w.Begin('['); foreach (var l in _lights) if (l != null) LightState(w, l); w.End(']');
                    w.Key("renderers"); w.Begin('['); foreach (var r in _renderers) if (r != null) RendererState(w, r); w.End(']');
                    w.Key("material_sheets_changed"); w.Begin('[');
                    foreach (var m in _ctx.Materials.Values.ToArray())
                    {
                        if (m == null) continue;   // destroyed since it was collected (pruned at the next scene export)
                        w.Begin('{'); w.Key("material"); w.Int(m.GetInstanceID());
                        w.Guard("sheet", s =>
                        {
                            var sheet = _ctx.Native.MaterialSheet(m);
                            if (sheet == null) { s.Str("not_built"); return; }
                            string last;
                            if (_lastMaterialSheet.TryGetValue(m.GetInstanceID(), out last) && last == sheet.Hash) { s.Begin('{'); s.Key("unchanged_sha256"); s.Str(sheet.Hash); s.End('}'); return; }
                            _lastMaterialSheet[m.GetInstanceID()] = sheet.Hash;
                            SheetReader.Write(s, sheet, _ctx.Native.Name);
                        });
                        w.Guard("shared_material_data", s =>
                        {
                            var j = new JsonOut(); _ctx.Native.WriteSharedMaterialData(j, m);
                            var text = j.ToString(); string last;
                            if (_lastSmd.TryGetValue(m.GetInstanceID(), out last) && last == text) { s.Begin('{'); s.Key("unchanged"); s.Bool(true); s.End('}'); return; }
                            _lastSmd[m.GetInstanceID()] = text; s.Raw(text);
                        });
                        w.End('}');
                    }
                    w.End(']');
                });
            }
            catch (Exception e) { Error("pre_render", e); }
        }

        void PostRender(Camera cam)
        {
            try
            {
                int id; _renderOfCamera.TryGetValue(cam.GetInstanceID(), out id);
                var bridge = BridgeOf(id);
                Record(cam, "post_render", id, bridge, w =>
                {
                    Globals(w);
                    w.Guard("render_buffer_pool", p => Dedup(p, ref _lastPool, () => { var j = new JsonOut(); PoolWalker.Write(j, _ctx.Native.Pool()); return j.ToString(); }));
                });
                while (_open.Count > 0) { if (_open.Pop() == id) break; }
                if (bridge != null && bridge.Ordinal == _bridgeExpected - 1 && OnBridgeEnd != null) OnBridgeEnd(Time.frameCount, bridge.Ordinal);
            }
            catch (Exception e) { Error("post_render", e); }
        }

        BridgeRender BridgeOf(int id) { return BridgeRenders.FirstOrDefault(b => b.RenderId == id && b.FrameCount == Time.frameCount); }

        void CharacterTransformBlock(JsonOut w)
        {
            var ts = CharacterTransforms;
            if (ts == null) throw new InvalidOperationException("character transforms not set");
            foreach (var t in ts) if (t == null) throw new InvalidOperationException("a character transform was destroyed during the window");
            w.Begin('{');
            w.Key("order"); w.Str("static.json transforms");
            w.Key("count"); w.Int(ts.Length);
            w.Key("transform_layout"); w.Str(FramePairProbe.TransformLayout);
            w.Key("transforms_f32"); w.Str(FramePairProbe.B64(FramePairProbe.TransformRows(ts)));
            w.Key("active"); w.Str(Convert.ToBase64String(FramePairProbe.ActiveFlags(ts)));
            w.End('}');
        }

        void Error(string phase, Exception e)
        {
            Errors++;
            if (Errors <= 50) _log.Add("recorder " + phase + " error at frame " + Time.frameCount + ": " + e);
        }

        void Record(Camera cam, string phase, int id, BridgeRender bridge, Action<JsonOut> body)
        {
            var w = _w;
            w.Begin('{');
            w.Key("seq"); w.Int(ProbeSequence.Next());
            w.Key("phase"); w.Str(phase);
            w.Key("frame_count"); w.Int(Time.frameCount);
            w.Key("render_id"); w.Int(id);
            w.Key("depth"); w.Int(_open.Count - (phase == "pre_cull" ? 0 : 1));
            w.Key("parent_render_id"); w.Raw(phase == "pre_cull" ? (_open.Count > 0 ? _open.Peek().ToString(CultureInfo.InvariantCulture) : "null")
                                                               : (_open.Count > 1 ? _open.ToArray()[1].ToString(CultureInfo.InvariantCulture) : "null"));
            w.Key("camera"); w.Begin('{'); SceneState.Ident(w, cam); w.Key("is_main"); w.Bool(cam == Camera.main); w.End('}');
            w.Key("bridge_render");
            if (bridge == null) w.Raw("null");
            else { w.Begin('{'); w.Key("ordinal"); w.Int(bridge.Ordinal); w.Key("expected_count"); w.Int(_bridgeExpected); w.Key("target"); w.Int(bridge.TargetId); w.End('}'); }
            w.Key("camera_target"); SceneState.TextureRowOrNull(w, cam.targetTexture, _ctx);
            w.Key("render_texture_active"); SceneState.TextureRowOrNull(w, RenderTexture.active, _ctx);
            w.Guard("data", d => { d.Begin('{'); body(d); d.End('}'); });
            w.End('}');
        }

        static void Dedup(JsonOut w, ref string last, Func<string> make)
        {
            var text = make();
            if (text == last) { w.Begin('{'); w.Key("unchanged"); w.Bool(true); w.End('}'); return; }
            last = text;
            w.Raw(text);
        }

        // Every RenderTexture object (camera intermediates, motion-vector texture, GrabPass Temp,
        // CustomRender rtCamera, pooled temporaries): descriptor, filter/wrap/aniso and native
        // fields, deduplicated against the previous callback.
        void RenderTextures(JsonOut w)
        {
            w.Guard("render_textures", t => Dedup(t, ref _lastRts, () =>
            {
                var j = new JsonOut();
                j.Begin('[');
                foreach (var rt in Resources.FindObjectsOfTypeAll<RenderTexture>().OrderBy(x => x.GetInstanceID()))
                    if (rt != null) _ctx.Native.WriteRenderTextureCompact(j, rt);
                j.End(']');
                return j.ToString();
            }));
        }

        void Globals(JsonOut w)
        {
            RenderTextures(w);
            w.Guard("global_keywords", k => Dedup(k, ref _lastKeywords, () => { var j = new JsonOut(); _ctx.Native.WriteKeywords(j); return j.ToString(); }));
            w.Guard("global_sheet", g => Dedup(g, ref _lastGlobalSheet, () => { var j = new JsonOut(); SheetReader.Write(j, _ctx.GlobalSheet(), _ctx.Native.Name); return j.ToString(); }));
            w.Guard("builtin_params", b => Dedup(b, ref _lastBuiltins, () => { var j = new JsonOut(); _ctx.Native.WriteBuiltins(j); return j.ToString(); }));
        }

        static void TimeBlock(JsonOut w)
        {
            w.Begin('{');
            w.Key("time"); w.Num(Time.time); w.Key("timeSinceLevelLoad"); w.Num(Time.timeSinceLevelLoad);
            w.Key("deltaTime"); w.Num(Time.deltaTime); w.Key("smoothDeltaTime"); w.Num(Time.smoothDeltaTime);
            w.Key("unscaledTime"); w.Num(Time.unscaledTime); w.Key("unscaledDeltaTime"); w.Num(Time.unscaledDeltaTime);
            w.Key("timeScale"); w.Num(Time.timeScale); w.Key("realtimeSinceStartup"); w.Num(Time.realtimeSinceStartup);
            w.Key("frameCount"); w.Int(Time.frameCount); w.Key("renderedFrameCount"); w.Int(Time.renderedFrameCount);
            w.End('}');
        }

        void CameraState(JsonOut w, Camera cam)
        {
            w.Begin('{');
            w.Key("transform"); SceneState.TransformBlock(w, cam.transform, _ctx);
            w.Key("matrices"); SceneState.CameraMatrices(w, cam, _ctx);
            w.Guard("native", n => _ctx.Native.WriteCamera(n, cam));
            w.Key("pixel_rect"); SceneState.Value(w, cam.pixelRect, 0, _ctx);
            w.Key("aspect"); w.Num(cam.aspect); w.Key("field_of_view"); w.Num(cam.fieldOfView);
            w.Key("near_clip"); w.Num(cam.nearClipPlane); w.Key("far_clip"); w.Num(cam.farClipPlane);
            w.Key("orthographic"); w.Bool(cam.orthographic); w.Key("orthographic_size"); w.Num(cam.orthographicSize);
            w.Key("culling_mask"); w.Int(cam.cullingMask);
            w.Key("clear_flags"); w.Str(cam.clearFlags.ToString()); w.Key("background_color"); SceneState.Value(w, cam.backgroundColor, 0, _ctx);
            w.Key("depth_texture_mode"); w.Int((int)cam.depthTextureMode);
            w.Key("actual_rendering_path"); w.Str(cam.actualRenderingPath.ToString());
            w.Key("allow_hdr"); w.Bool(cam.allowHDR); w.Key("allow_msaa"); w.Bool(cam.allowMSAA);
            w.Key("command_buffers"); w.Begin('{');
            foreach (CameraEvent e in Enum.GetValues(typeof(CameraEvent)))
            {
                var bufs = cam.GetCommandBuffers(e);
                if (bufs.Length == 0) continue;
                w.Key(e.ToString()); SceneState.Names(w, bufs.Select(b => b.name + "|size=" + b.sizeInBytes));
            }
            w.End('}');
            w.End('}');
        }

        void LightState(JsonOut w, Light l)
        {
            w.Begin('{');
            w.Key("id"); w.Int(l.GetInstanceID()); w.Key("enabled"); w.Bool(l.enabled); w.Key("active_in_hierarchy"); w.Bool(l.gameObject.activeInHierarchy);
            w.Key("transform"); SceneState.TransformBlock(w, l.transform, _ctx);
            w.Key("type"); w.Str(l.type.ToString());
            w.Key("color"); SceneState.Value(w, l.color, 0, _ctx); w.Key("intensity"); w.Num(l.intensity);
            w.Key("bounce_intensity"); w.Num(l.bounceIntensity); w.Key("color_temperature"); w.Num(l.colorTemperature);
            w.Key("range"); w.Num(l.range); w.Key("spot_angle"); w.Num(l.spotAngle);
            w.Key("render_mode"); w.Str(l.renderMode.ToString()); w.Key("culling_mask"); w.Int(l.cullingMask);
            w.Key("shadows"); w.Str(l.shadows.ToString()); w.Key("shadow_strength"); w.Num(l.shadowStrength);
            w.Key("shadow_bias"); w.Num(l.shadowBias); w.Key("shadow_normal_bias"); w.Num(l.shadowNormalBias);
            w.Key("shadow_near_plane"); w.Num(l.shadowNearPlane);
            w.Key("shadow_resolution"); w.Str(l.shadowResolution.ToString()); w.Key("shadow_custom_resolution"); w.Int(l.shadowCustomResolution);
            w.Key("layer_shadow_cull_distances"); SceneState.Value(w, l.layerShadowCullDistances, 0, _ctx);
            w.Key("cookie"); SceneState.TextureRowOrNull(w, l.cookie, _ctx); w.Key("cookie_size"); w.Num(l.cookieSize);
            w.Key("baking_output"); SceneState.Value(w, l.bakingOutput, 0, _ctx);
            w.Key("is_sun"); w.Bool(RenderSettings.sun == l);
            w.End('}');
        }

        void RendererState(JsonOut w, Renderer r)
        {
            w.Begin('{');
            w.Key("id"); w.Int(r.GetInstanceID()); w.Key("type"); w.Str(r.GetType().FullName);
            w.Key("enabled"); w.Bool(r.enabled); w.Key("active_in_hierarchy"); w.Bool(r.gameObject.activeInHierarchy);
            w.Key("layer"); w.Int(r.gameObject.layer); w.Key("is_visible"); w.Bool(r.isVisible);
            w.Guard("native", n => _ctx.Native.WriteRenderer(n, r));
            w.Key("bounds"); SceneState.Value(w, r.bounds, 0, _ctx);
            w.Key("local_to_world_rowmajor"); SceneState.Value(w, r.localToWorldMatrix, 0, _ctx);
            w.Key("shadow_casting_mode"); w.Str(r.shadowCastingMode.ToString()); w.Key("receive_shadows"); w.Bool(r.receiveShadows);
            w.Key("motion_vector_mode"); w.Str(r.motionVectorGenerationMode.ToString());
            w.Key("light_probe_usage"); w.Str(r.lightProbeUsage.ToString()); w.Key("reflection_probe_usage"); w.Str(r.reflectionProbeUsage.ToString());
            w.Key("rendering_layer_mask"); w.Int(r.renderingLayerMask);
            w.Key("sorting_layer_id"); w.Int(r.sortingLayerID); w.Key("sorting_order"); w.Int(r.sortingOrder);
            w.Key("lightmap_index"); w.Int(r.lightmapIndex); w.Key("realtime_lightmap_index"); w.Int(r.realtimeLightmapIndex);
            w.Key("allow_occlusion_when_dynamic"); w.Bool(r.allowOcclusionWhenDynamic);
            w.Key("probe_anchor"); if (r.probeAnchor == null) w.Raw("null"); else SceneState.Value(w, r.probeAnchor.position, 0, _ctx);
            w.Key("shared_materials"); w.Begin('['); foreach (var m in r.sharedMaterials) w.Raw(m == null ? "null" : m.GetInstanceID().ToString(CultureInfo.InvariantCulture)); w.End(']');
            var smr = r as SkinnedMeshRenderer;
            if (smr != null)
            {
                w.Key("root_bone"); if (smr.rootBone == null) w.Raw("null"); else
                { w.Begin('{'); w.Key("id"); w.Int(smr.rootBone.GetInstanceID()); w.Key("position"); SceneState.Value(w, smr.rootBone.position, 0, _ctx); w.Key("rotation"); SceneState.Value(w, smr.rootBone.rotation, 0, _ctx); w.End('}'); }
                w.Key("local_bounds"); SceneState.Value(w, smr.localBounds, 0, _ctx);
                w.Key("update_when_offscreen"); w.Bool(smr.updateWhenOffscreen);
            }
            w.End('}');
        }
    }
}
