// Same-frame source-state / RDC pairing probe v5 (read-only with respect to the game,
// except mode "renderstyle", which issues the bridge's own capture request).
//
// On an explicit request file (HS2_FramePairProbe.request.ini beside this DLL) it runs
// one window of consecutive frames. At every end-of-frame point (WaitForEndOfFrame:
// after all Update/animation/LateUpdate writers, every camera render and the frame's
// CPU skinning, before PresentAfterDraw) it
//   * writes the frame's per-render records (RenderRecorder, static Camera callbacks),
//   * exports every transform of the single active ChaControl and every skinned
//     renderer's blendshape weights and bounds (v4 frame_*.json, unchanged layout),
//   * exports the full render-scene state (SceneState v5, scene_*.json),
//   * uploads the 64-byte v01 marker (magic + nonce + Time.frameCount, unchanged).
// Window plan (index t = TriggerIndex; RenderDoc 1.46 starts a Present-boundary capture
// at the Present after TriggerCapture; the first attempt fails with "Uncapped
// Map()/Unmap()" because Unity keeps maps open across Present, and RenderDoc's
// automatic retry (d3d11_device.cpp 2450..2516) captures the next frame):
//   end of t   : TriggerCapture            -> attempt covers frame t+1 (fails)
//   end of t+1 : raw texel export (native readback on the render thread, inside the
//                failed attempt) = contents at the start of frame t+2;
//                mode renderstyle: block until the export is written, then POST the
//                bridge capture and block until its action is queued in ThreadingHelper
//   frame t+2  : captured frame (retry); mode renderstyle: the bridge's 16 renders run
//                in its ThreadingHelper.Update, bracketed by HS2PAIRBRIDGEv05 markers.
// Static mesh data of the requested meshes is exported once. The probe sets nothing on
// the character, issues no render and touches no card.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using AIChara;
using BepInEx;
using UnityEngine;

namespace HS2Codex.FramePairProbe
{
    [BepInPlugin("codex.hs2.framepairprobe", "HS2 Frame Pair Probe", Version)]
    public class FramePairProbe : BaseUnityPlugin
    {
        public const string Version = "6.1.0";
        const string Magic = "HS2PAIRMARKERv01";
        const string BridgeMagic = "HS2PAIRBRIDGEv05";
        const string Tail = "HS2PAIRMARKEREND";
        public const int BridgeBegin = 1, BridgeEnd = 2;
        string _requestPath;
        bool _busy;
        Texture2D _marker, _bridgeBegin, _bridgeEnd;
        RenderDocApi _rd;
        DriverHooks _hooks;
        SkinnedMeshRenderer[] _smrsForSlots = new SkinnedMeshRenderer[0];
        int _expectedBridgeRenders;

        void Awake()
        {
            _requestPath = Path.Combine(Path.GetDirectoryName(Info.Location), "HS2_FramePairProbe.request.ini");
            Logger.LogInfo("FramePairProbe " + Version + " ready; request path " + _requestPath);
        }

        void Update()
        {
            if (_busy || Time.frameCount % 15 != 0 || !File.Exists(_requestPath)) return;
            Dictionary<string, string> req;
            try
            {
                req = ProbeRequest.ParseIni(File.ReadAllText(_requestPath));
                File.Delete(_requestPath);
            }
            catch (Exception e) { Logger.LogError("Request unreadable: " + e); return; }
            _busy = true;
            StartCoroutine(Run(req));
        }

        IEnumerator Run(Dictionary<string, string> raw)
        {
            var log = new List<string>();
            string outDir = null;
            Exception failure = null;
            IEnumerator inner = null;
            ProbeRequest quitRequest = null;
            try
            {
                var req = ProbeRequest.Parse(raw, p => Directory.Exists(p) || File.Exists(p));
                outDir = req.OutDir;
                if (req.Mode == "quit") quitRequest = req;
                if (quitRequest == null)
                {
                    Directory.CreateDirectory(Path.Combine(outDir, "frames"));
                    File.WriteAllText(Path.Combine(outDir, "request.ini"), string.Join("\n", raw.Select(kv => kv.Key + "=" + kv.Value).ToArray()));
                    inner = Window(req, log);
                }
            }
            catch (Exception e) { failure = e; }
            if (quitRequest != null && failure == null)
            {
                try { Quit(quitRequest, raw); }
                catch (Exception e) { failure = e; }
                if (failure != null) { log.Add("FAILURE " + failure); Logger.LogError(failure.ToString()); }
                _busy = false;
                yield break;
            }
            while (failure == null)
            {
                bool more;
                try { more = inner.MoveNext(); }
                catch (Exception e) { failure = e; break; }
                if (!more) break;
                yield return inner.Current;
            }
            if (failure != null) { log.Add("FAILURE " + failure); Logger.LogError(failure.ToString()); }
            try
            {
                if (outDir != null && Directory.Exists(outDir)) File.WriteAllText(Path.Combine(outDir, "probe_log.txt"), string.Join("\n", log.ToArray()));
                else Logger.LogError("Probe log (no OutDir): " + string.Join(" | ", log.ToArray()));
            }
            catch (Exception e) { Logger.LogError(e.ToString()); }
            foreach (var t in new[] { _marker, _bridgeBegin, _bridgeEnd }) if (t != null) UnityEngine.Object.Destroy(t);
            _marker = _bridgeBegin = _bridgeEnd = null;
            _rd = null;
            _busy = false;
        }

        // Clean exit through the game's own path: ExitDialog.GameEnd(isCheck: false) is what the
        // exit dialog's Yes button calls (decompiled ExitDialog.Initialize); it clears
        // _isGameEndCheck so the Application.wantsToQuit handler returns true, then calls
        // Application.Quit(). Game.OnApplicationQuit then saves the game progress data as on any
        // normal exit. The receipt is written before the call.
        void Quit(ProbeRequest req, Dictionary<string, string> raw)
        {
            var exitDialog = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("ExitDialog", false)).FirstOrDefault(t => t != null);
            if (exitDialog == null) throw new InvalidOperationException("ExitDialog type not loaded");
            var gameEnd = exitDialog.GetMethod("GameEnd", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static, null, new[] { typeof(bool) }, null);
            if (gameEnd == null) throw new InvalidOperationException("ExitDialog.GameEnd(bool) not found");
            var active = exitDialog.GetProperty("active", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            Directory.CreateDirectory(req.OutDir);
            var w = new JsonOut();
            w.Begin('{');
            w.Key("schema_version"); w.Raw("6"); w.Key("probe"); w.Str("HS2_FramePairProbe " + Version);
            w.Key("nonce"); w.Str(req.NonceHex);
            w.Key("frame_count"); w.Int(Time.frameCount);
            w.Key("utc"); w.Str(DateTime.UtcNow.ToString("o"));
            w.Key("exit_dialog_active_before"); w.Raw(active == null ? "null" : ((bool)active.GetValue(null, null) ? "true" : "false"));
            w.Key("call"); w.Str("ExitDialog.GameEnd(isCheck: false) -> Application.Quit()");
            w.End('}');
            File.WriteAllText(Path.Combine(req.OutDir, "quit_request_receipt.json"), w.ToString());
            Logger.LogInfo("FramePairProbe quit request " + req.NonceHex + ": ExitDialog.GameEnd(false)");
            gameEnd.Invoke(null, new object[] { false });
        }

        static Texture2D MarkerTexture(string name)
        {
            var t = new Texture2D(16, 1, TextureFormat.RGBA32, false, true);
            t.hideFlags = HideFlags.HideAndDontSave;
            t.name = name;
            return t;
        }

        IEnumerator Window(ProbeRequest req, List<string> log)
        {
            string outDir = req.OutDir;
            var charas = UnityEngine.Object.FindObjectsOfType<ChaControl>();
            if (charas.Length != 1) throw new InvalidOperationException("Exactly one active ChaControl required, found " + charas.Length);
            var cha = charas[0];
            var main = Camera.main;
            if (main == null) throw new InvalidOperationException("Camera.main is null");
            var transforms = cha.transform.GetComponentsInChildren<Transform>(true);
            var smrs = cha.transform.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            var index = new Dictionary<Transform, int>();
            for (int i = 0; i < transforms.Length; i++) index[transforms[i]] = i;

            // Everything that can refuse runs before TriggerCapture.
            var native = UnityNative.Open(req.ExpectedUnityPlayerSha256);
            native.ValidateWithScene();
            foreach (var v in native.Validation) log.Add("native " + v);
            if (!native.Valid) throw new InvalidOperationException("Native layout validation failed; no capture triggered");
            var raw = new RawTextures(req.ReadbackDll, req.ExpectedReadbackSha256);
            var manifest = ShaderManifest.Load(req.ShaderManifest);
            log.Add("shader manifest " + manifest.Path + " sha256=" + manifest.Sha256 + " shaders=" + manifest.Properties.Count);
            BridgeCapture bridge = req.Mode == "renderstyle" ? BridgeCapture.Prepare(req) : null;
            _expectedBridgeRenders = 0;
            _rd = new RenderDocApi(req.ExpectedRenderDocSha256, log);
            string oldTemplate = _rd.GetTemplate();
            uint capturesBefore = _rd.NumCaptures();
            _rd.SetTemplate(req.CapturePrefix);
            log.Add("renderdoc " + _rd.Describe() + " captures_before=" + capturesBefore + " old_template=" + oldTemplate);

            var ctx = new ExportContext(native, manifest);
            native.SkinSlots = new SkinSlotTracker();
            _smrsForSlots = smrs;
            WriteStatic(Path.Combine(outDir, "static.json"), cha, transforms, smrs, index, new HashSet<string>(req.Meshes.Split(',')), outDir, native);
            var recorder = new RenderRecorder(ctx, log);
            recorder.CharacterTransforms = transforms;
            _marker = MarkerTexture("HS2_FramePairProbe_marker");
            _bridgeBegin = MarkerTexture("HS2_FramePairProbe_bridge_begin");
            _bridgeEnd = MarkerTexture("HS2_FramePairProbe_bridge_end");
            var markersWritten = new List<string>();
            recorder.OnBridgeBegin = (fc, ord) => { WriteMarker(_bridgeBegin, BridgeMagic, req.Nonce, fc, BridgeBegin, ord); markersWritten.Add("bridge_begin frame " + fc + " before render ordinal " + ord); };
            recorder.OnBridgeEnd = (fc, ord) => { WriteMarker(_bridgeEnd, BridgeMagic, req.Nonce, fc, BridgeEnd, ord); markersWritten.Add("bridge_end frame " + fc + " after render ordinal " + ord); };
            var eof = new WaitForEndOfFrame();
            var frameCounts = new int[req.Frames];
            var capturing = new List<string>();
            int triggeredFrame = -1, rawFrame = -1;
            bool triggered = false, aborted = false;
            DriverHooks hooks = null;
            if (req.DriverHooks)
            {
                // Before the trigger: a missing hook target aborts the window (exception -> no capture).
                hooks = DriverHooks.Install(ctx, cha, transforms, smrs, log);
            }
            _hooks = hooks;
            recorder.Attach();
            try
            {
                for (int k = 0; k < req.Frames && !aborted; k++)
                {
                    yield return eof;
                    int fc = Time.frameCount;
                    frameCounts[k] = fc;
                    if (triggered) capturing.Add("end of frame " + fc + " IsFrameCapturing=" + _rd.IsFrameCapturing() + " captures=" + _rd.NumCaptures());
                    bool rawFailed = false;
                    foreach (var action in req.ActionsAt(k))
                    {
                        if (action == "bridge_post_until_queued" && rawFailed) { log.Add("bridge not posted: the raw export of this frame failed"); continue; }
                        string error = Step(action, k, fc, req, ctx, recorder, raw, bridge, main, transforms, smrs, outDir, log);
                        if (action.StartsWith("raw_export")) { rawFrame = fc; rawFailed = error != null; }
                        if (action == "renderdoc_trigger" && error == null) { triggered = true; triggeredFrame = fc; }
                        if (error == null) continue;
                        log.Add("step " + action + " failed at index " + k + " frame " + fc + ": " + error);
                        // Before the trigger any failure aborts: no capture is taken without its inputs.
                        // After it, the remaining exports and the marker still run so the capture stays paired.
                        if (!triggered) { aborted = true; log.Add("window aborted before TriggerCapture; nothing was captured"); break; }
                    }
                    recorder.Refresh();
                }
            }
            finally
            {
                recorder.Detach();
                if (hooks != null) { hooks.Uninstall(); log.Add("driver hooks removed; errors " + hooks.Errors); }
                _hooks = null;
            }

            float deadline = Time.realtimeSinceStartup + req.PollSeconds;
            uint after = _rd.NumCaptures();
            while (triggered && after == capturesBefore && Time.realtimeSinceStartup < deadline) { yield return null; after = _rd.NumCaptures(); }
            deadline = Time.realtimeSinceStartup + req.RawTimeoutSeconds;
            while (raw.Jobs.Count > 0 && raw.Pending() > 0 && Time.realtimeSinceStartup < deadline) yield return null;
            if (bridge != null)
            {
                deadline = Time.realtimeSinceStartup + req.BridgeTimeoutSeconds;
                while (bridge.PostFrame >= 0 && !bridge.Done && Time.realtimeSinceStartup < deadline) yield return null;
            }
            _rd.SetTemplate(oldTemplate);

            var w = new JsonOut();
            w.Begin('{');
            w.Key("schema_version"); w.Raw("5");
            w.Key("probe"); w.Str("HS2_FramePairProbe " + Version);
            w.Key("mode"); w.Str(req.Mode);
            w.Key("nonce"); w.Str(req.NonceHex);
            w.Key("frame_counts_by_index"); w.Begin('['); foreach (var f in frameCounts) w.Int(f); w.End(']');
            w.Key("trigger_index"); w.Int(req.TriggerIndex);
            w.Key("triggered_frame"); w.Int(triggeredFrame);
            w.Key("raw_export_frame"); w.Int(rawFrame);
            w.Key("expected_capture_index"); w.Int(req.ExpectedCaptureIndex);
            w.Key("expected_capture_frame"); w.Int(req.ExpectedCaptureIndex < frameCounts.Length && frameCounts[req.ExpectedCaptureIndex] != 0 ? frameCounts[req.ExpectedCaptureIndex] : -1);
            w.Key("expected_capture_rule"); w.Str("TriggerCapture at end of frame T -> attempt T+1 fails (Uncapped Map/Unmap) -> retry captures T+2; verify with the in-frame v01 marker");
            w.Key("aborted_before_trigger"); w.Bool(aborted);
            w.Key("captures_before"); w.Int(capturesBefore);
            w.Key("captures_after"); w.Int(after);
            w.Key("renderdoc"); w.Str(_rd.Describe());
            w.Key("captures"); w.Begin('[');
            for (uint i = capturesBefore; i < after; i++)
            {
                string path; ulong ts;
                _rd.GetCapture(i, out path, out ts);
                w.Begin('{'); w.Key("index"); w.Int(i); w.Key("path"); w.Str(path); w.Key("timestamp"); w.Raw(ts.ToString(CultureInfo.InvariantCulture)); w.End('}');
            }
            w.End(']');
            w.Key("is_frame_capturing_samples"); SceneState.Names(w, capturing);
            w.Key("markers_written"); SceneState.Names(w, markersWritten);
            w.Key("recorder_errors"); w.Int(recorder.Errors);
            w.Key("driver_hooks"); if (hooks == null) w.Raw("null"); else { w.Begin('{'); w.Key("installed"); SceneState.Names(w, hooks.Patched); w.Key("errors"); w.Int(hooks.Errors); w.End('}'); }
            w.Key("bridge_expected_renders"); w.Int(_expectedBridgeRenders);
            w.Key("bridge_expected_renders_rule"); w.Str("MakerRenderStyleService.Capture: 16 when an enabled CTAA_PC is on Camera.main, else 1; evaluated at the post");
            w.Key("bridge_renders"); w.Begin('[');
            foreach (var b in recorder.BridgeRenders)
            { w.Begin('{'); w.Key("frame_count"); w.Int(b.FrameCount); w.Key("render_id"); w.Int(b.RenderId); w.Key("ordinal"); w.Int(b.Ordinal); w.Key("target"); w.Int(b.TargetId); w.End('}'); }
            w.End(']');
            w.Key("bridge"); if (bridge == null) w.Raw("null"); else bridge.Write(w);
            w.Key("raw_textures_summary"); w.Begin('{');
            w.Key("out_dir"); w.Str(req.RawOutDir); w.Key("jobs"); w.Int(raw.Jobs.Count);
            foreach (var g in raw.Jobs.GroupBy(j => j.Status ?? "unknown")) { w.Key(g.Key); w.Int(g.Count()); }
            w.End('}');
            w.Key("end_frame"); w.Int(Time.frameCount);
            w.End('}');
            File.WriteAllText(Path.Combine(outDir, "capture_receipt.json"), w.ToString());
            var rw = new JsonOut(); raw.Write(rw);
            File.WriteAllText(Path.Combine(outDir, "raw_textures.json"), rw.ToString());
            log.Add("done; captures " + capturesBefore + " -> " + after);
        }

        // One end-of-frame action; returns null on success or the failure text.
        string Step(string action, int k, int fc, ProbeRequest req, ExportContext ctx, RenderRecorder recorder, RawTextures raw, BridgeCapture bridge,
                    Camera main, Transform[] transforms, SkinnedMeshRenderer[] smrs, string outDir, List<string> log)
        {
            try
            {
                switch (action)
                {
                    case "export_frame":
                        // Skin double buffer: this frame's skinning wrote the current slot-0 buffers.
                        foreach (var r in _smrsForSlots) if (r != null) ctx.Native.SkinSlots.EndOfFrame(fc, ctx.Native.SkinBuffer(r));
                        recorder.Flush(Path.Combine(outDir, "frames", "renders_" + k.ToString("D3") + ".json"), k);
                        if (_hooks != null) _hooks.Flush(Path.Combine(outDir, "frames", "drivers_" + k.ToString("D3") + ".json"), k);
                        WriteFrame(Path.Combine(outDir, "frames", "frame_" + k.ToString("D3") + ".json"), k, transforms, smrs);
                        break;
                    case "export_scene":
                        SceneState.Write(Path.Combine(outDir, "frames", "scene_" + k.ToString("D3") + ".json"), ctx, k);
                        break;
                    case "write_marker":
                        WriteMarker(_marker, Magic, req.Nonce, fc, k, Time.renderedFrameCount);
                        break;
                    case "raw_export":
                    case "raw_export_blocking":
                        raw.Queue(RawSelection(ctx), req.RawOutDir, req.RawMinFreeBytes, ctx.Native);
                        log.Add("raw export queued at end of frame " + fc + ": " + raw.Jobs.Count + " textures; " + raw.FreeSpaceCheck);
                        if (action == "raw_export_blocking")
                        {
                            if (!raw.WaitAll(req.RawTimeoutSeconds)) return "raw export not written within " + req.RawTimeoutSeconds + " s; bridge not posted";
                            log.Add("raw export written before the bridge post");
                        }
                        break;
                    case "bridge_post_until_queued":
                        var preexisting = new HashSet<int>(Resources.FindObjectsOfTypeAll<RenderTexture>().Select(t => t.GetInstanceID()));
                        _expectedBridgeRenders = BridgeCapture.ExpectedRenders(main);
                        // Armed before the post: the renders cannot start before the next frame's Update anyway.
                        recorder.ArmBridge(fc + 1, main, bridge.Body.Width, bridge.Body.Height, _expectedBridgeRenders, preexisting);
                        if (!bridge.PostAndWaitQueued(10f)) return "bridge action was not queued";
                        log.Add("bridge posted at end of frame " + fc + "; its renders are expected in frame " + (fc + 1));
                        break;
                    case "renderdoc_trigger":
                        _rd.Trigger();
                        log.Add("TriggerCapture called at end of frame " + fc + " (index " + k + ")");
                        break;
                    default: throw new InvalidOperationException("Unknown action " + action);
                }
                return null;
            }
            catch (Exception e) { return e.ToString(); }
        }

        // Every texture referenced by the scene export (materials, effect components,
        // probes, RenderSettings, the character's composed textures), every texture whose
        // native TextureID is bound in a material sheet or in the global sheet at any
        // recorded callback, and every RenderTexture object.
        static Dictionary<Texture, HashSet<string>> RawSelection(ExportContext ctx)
        {
            ctx.PruneDestroyed();
            var sel = new Dictionary<Texture, HashSet<string>>();
            Action<Texture, string> add = (t, why) =>
            {
                if (t == null) return;
                HashSet<string> set;
                if (!sel.TryGetValue(t, out set)) sel[t] = set = new HashSet<string>();
                set.Add(why);
            };
            foreach (var t in ctx.Textures.Values) add(t, "scene_export_reference");
            foreach (var t in Resources.FindObjectsOfTypeAll<RenderTexture>()) add(t, "render_texture_object");
            foreach (var t in Resources.FindObjectsOfTypeAll<Texture>())
            {
                int id;
                try { id = ctx.Native.TextureId(t); } catch { continue; }
                if (id == 0) continue;
                if (ctx.SheetTextureIds.Contains(id)) add(t, "material_sheet_texture_id");
                if (ctx.GlobalTextureIds.Contains(id)) add(t, "global_sheet_texture_id");
            }
            return sel;
        }

        static void WriteMarker(Texture2D tex, string magic, byte[] nonce, int frameCount, int word36, int word40)
        {
            var bytes = new byte[64];
            Encoding.ASCII.GetBytes(magic, 0, 16, bytes, 0);
            Buffer.BlockCopy(nonce, 0, bytes, 16, 16);
            Buffer.BlockCopy(BitConverter.GetBytes(frameCount), 0, bytes, 32, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(word36), 0, bytes, 36, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(word40), 0, bytes, 40, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(~frameCount), 0, bytes, 44, 4);
            Encoding.ASCII.GetBytes(Tail, 0, 16, bytes, 48);
            var px = new Color32[16];
            for (int i = 0; i < 16; i++) px[i] = new Color32(bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]);
            tex.SetPixels32(px);
            tex.Apply(false, false);
        }

        static void WriteFrame(string path, int k, Transform[] transforms, SkinnedMeshRenderer[] smrs)
        {
            var f = TransformRows(transforms);
            var sb = new StringBuilder(1 << 20);
            sb.Append("{\"index\":").Append(k).Append(",\"frame_count\":").Append(Time.frameCount)
              .Append(",\"rendered_frame_count\":").Append(Time.renderedFrameCount)
              .Append(",\"realtime\":").Append(Time.realtimeSinceStartup.ToString("R", CultureInfo.InvariantCulture))
              .Append(",\"time\":").Append(Time.time.ToString("R", CultureInfo.InvariantCulture))
              .Append(",\"delta_time\":").Append(Time.deltaTime.ToString("R", CultureInfo.InvariantCulture))
              .Append(",\"transform_layout\":\"").Append(TransformLayout).Append("\"")
              .Append(",\"transforms_f32\":\"").Append(B64(f)).Append("\",\"active\":\"");
            sb.Append(Convert.ToBase64String(ActiveFlags(transforms))).Append("\",\"renderers\":[");
            for (int s = 0; s < smrs.Length; s++)
            {
                var r = smrs[s];
                var mesh = r.sharedMesh;
                int n = mesh == null ? 0 : mesh.blendShapeCount;
                var w = new float[n];
                for (int i = 0; i < n; i++) w[i] = r.GetBlendShapeWeight(i);
                var b = r.bounds;
                if (s > 0) sb.Append(',');
                sb.Append("{\"id\":").Append(r.GetInstanceID()).Append(",\"enabled\":").Append(r.enabled ? "true" : "false")
                  .Append(",\"active\":").Append(r.gameObject.activeInHierarchy ? "true" : "false")
                  .Append(",\"visible\":").Append(r.isVisible ? "true" : "false")
                  .Append(",\"weights_f32\":\"").Append(B64(w)).Append("\",\"bounds_f32\":\"")
                  .Append(B64(new[] { b.center.x, b.center.y, b.center.z, b.extents.x, b.extents.y, b.extents.z })).Append("\"}");
            }
            sb.Append("]}");
            File.WriteAllText(path, sb.ToString());
        }

        static void WriteStatic(string path, ChaControl cha, Transform[] transforms, SkinnedMeshRenderer[] smrs,
            Dictionary<Transform, int> index, HashSet<string> meshNames, string outDir, UnityNative native)
        {
            var sb = new StringBuilder(1 << 20);
            var nw = new JsonOut();
            nw.Begin('{');
            nw.Key("identity"); native.WriteIdentity(nw);
            nw.Key("property_names"); nw.Begin('['); foreach (var n in native.PropertyNames) nw.Str(n); nw.End(']');
            nw.Key("keywords"); nw.Begin('{'); foreach (var kv in native.KeywordNames) { nw.Key(kv.Key.ToString(CultureInfo.InvariantCulture)); nw.Str(kv.Value); } nw.End('}');
            nw.End('}');
            sb.Append("{\"schema_version\":5,\"native_tables\":").Append(nw.ToString()).Append(",\"unity_version\":").Append(Json(Application.unityVersion))
              .Append(",\"frame_count\":").Append(Time.frameCount).Append(",\"chara_transform_id\":").Append(cha.transform.GetInstanceID())
              .Append(",\"chara_name\":").Append(Json(cha.name)).Append(",\"quality_blend_weights\":").Append(Json(QualitySettings.blendWeights.ToString()))
              .Append(",\"transforms\":[");
            for (int i = 0; i < transforms.Length; i++)
            {
                var t = transforms[i];
                if (i > 0) sb.Append(',');
                int parent = t.parent != null && index.ContainsKey(t.parent) ? index[t.parent] : -1;
                sb.Append("{\"i\":").Append(i).Append(",\"id\":").Append(t.GetInstanceID()).Append(",\"name\":").Append(Json(t.name))
                  .Append(",\"parent\":").Append(parent).Append(",\"sibling\":").Append(t.GetSiblingIndex()).Append('}');
            }
            sb.Append("],\"renderers\":[");
            int meshFile = 0;
            var exportedMeshes = new Dictionary<int, string>();
            for (int s = 0; s < smrs.Length; s++)
            {
                var r = smrs[s];
                var mesh = r.sharedMesh;
                if (s > 0) sb.Append(',');
                sb.Append("{\"id\":").Append(r.GetInstanceID()).Append(",\"name\":").Append(Json(r.name))
                  .Append(",\"transform\":").Append(index.ContainsKey(r.transform) ? index[r.transform] : -1)
                  .Append(",\"root_bone\":").Append(r.rootBone != null && index.ContainsKey(r.rootBone) ? index[r.rootBone] : -1)
                  .Append(",\"quality\":").Append(Json(r.quality.ToString()))
                  .Append(",\"update_when_offscreen\":").Append(r.updateWhenOffscreen ? "true" : "false")
                  .Append(",\"skinned_motion_vectors\":").Append(r.skinnedMotionVectors ? "true" : "false")
                  .Append(",\"motion_vector_mode\":").Append(Json(r.motionVectorGenerationMode.ToString()))
                  .Append(",\"local_bounds_f32\":\"").Append(B64(new[] { r.localBounds.center.x, r.localBounds.center.y, r.localBounds.center.z,
                      r.localBounds.extents.x, r.localBounds.extents.y, r.localBounds.extents.z })).Append('"')
                  .Append(",\"bones\":[");
                var bones = r.bones;
                for (int b = 0; b < bones.Length; b++)
                {
                    if (b > 0) sb.Append(',');
                    sb.Append(bones[b] != null && index.ContainsKey(bones[b]) ? index[bones[b]] : -1);
                }
                sb.Append(']');
                if (mesh != null)
                {
                    sb.Append(",\"mesh\":{\"id\":").Append(mesh.GetInstanceID()).Append(",\"name\":").Append(Json(mesh.name))
                      .Append(",\"vertex_count\":").Append(mesh.vertexCount).Append(",\"blendshape_count\":").Append(mesh.blendShapeCount)
                      .Append(",\"readable\":").Append(mesh.isReadable ? "true" : "false");
                    bool wanted = meshNames.Contains("ALL") || meshNames.Contains(mesh.name);
                    sb.Append(",\"export\":").Append(Json(!wanted ? "not_requested" : !mesh.isReadable ? "not_readable (Mesh.isReadable false: CPU data not retained)" : exportedMeshes.ContainsKey(mesh.GetInstanceID()) ? "same_mesh_as:" + exportedMeshes[mesh.GetInstanceID()] : "exported"));
                    if (wanted && mesh.isReadable && exportedMeshes.ContainsKey(mesh.GetInstanceID()))
                        sb.Append(",\"data_file\":").Append(Json(exportedMeshes[mesh.GetInstanceID()]));
                    else if (wanted && mesh.isReadable)
                    {
                        string file = "mesh_" + (meshFile++).ToString("D2") + "_" + Safe(mesh.name) + ".json";
                        WriteMesh(Path.Combine(outDir, file), mesh);
                        exportedMeshes[mesh.GetInstanceID()] = file;
                        sb.Append(",\"data_file\":").Append(Json(file));
                    }
                    sb.Append('}');
                }
                sb.Append('}');
            }
            sb.Append("]}");
            File.WriteAllText(path, sb.ToString());
        }

        static void WriteMesh(string path, Mesh mesh)
        {
            int n = mesh.vertexCount;
            var sb = new StringBuilder(1 << 22);
            sb.Append("{\"name\":").Append(Json(mesh.name)).Append(",\"id\":").Append(mesh.GetInstanceID())
              .Append(",\"vertex_count\":").Append(n)
              .Append(",\"vertices_f32\":\"").Append(B64(V3(mesh.vertices))).Append('"')
              .Append(",\"normals_f32\":\"").Append(B64(V3(mesh.normals))).Append('"')
              .Append(",\"tangents_f32\":\"").Append(B64(V4(mesh.tangents))).Append('"')
              .Append(",\"uv_f32\":\"").Append(B64(V2(mesh.uv))).Append('"')
              .Append(",\"uv2_f32\":\"").Append(B64(V2(mesh.uv2))).Append('"');
            var tri = mesh.triangles;
            var trib = new byte[tri.Length * 4];
            Buffer.BlockCopy(tri, 0, trib, 0, trib.Length);
            sb.Append(",\"triangles_i32\":\"").Append(Convert.ToBase64String(trib)).Append('"');
            var bp = mesh.bindposes;
            var bpf = new float[bp.Length * 16];
            for (int i = 0; i < bp.Length; i++) for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) bpf[i * 16 + r * 4 + c] = bp[i][r, c];
            sb.Append(",\"bindposes_f32_rowmajor\":\"").Append(B64(bpf)).Append('"');
            var bw = mesh.boneWeights;
            var wf = new float[bw.Length * 4];
            var wi = new int[bw.Length * 4];
            for (int i = 0; i < bw.Length; i++)
            {
                wf[i * 4] = bw[i].weight0; wf[i * 4 + 1] = bw[i].weight1; wf[i * 4 + 2] = bw[i].weight2; wf[i * 4 + 3] = bw[i].weight3;
                wi[i * 4] = bw[i].boneIndex0; wi[i * 4 + 1] = bw[i].boneIndex1; wi[i * 4 + 2] = bw[i].boneIndex2; wi[i * 4 + 3] = bw[i].boneIndex3;
            }
            sb.Append(",\"bone_weights_f32\":\"").Append(B64(wf)).Append("\",\"bone_indices_i32\":\"");
            var wib = new byte[wi.Length * 4];
            Buffer.BlockCopy(wi, 0, wib, 0, wib.Length);
            sb.Append(Convert.ToBase64String(wib)).Append("\",\"blendshapes\":[");
            var dv = new Vector3[n]; var dn = new Vector3[n]; var dt = new Vector3[n];
            for (int s = 0; s < mesh.blendShapeCount; s++)
            {
                if (s > 0) sb.Append(',');
                sb.Append("{\"name\":").Append(Json(mesh.GetBlendShapeName(s))).Append(",\"frames\":[");
                int fcount = mesh.GetBlendShapeFrameCount(s);
                for (int f = 0; f < fcount; f++)
                {
                    mesh.GetBlendShapeFrameVertices(s, f, dv, dn, dt);
                    if (f > 0) sb.Append(',');
                    sb.Append("{\"weight\":").Append(mesh.GetBlendShapeFrameWeight(s, f).ToString("R", CultureInfo.InvariantCulture))
                      .Append(",\"dv_f32\":\"").Append(B64(V3(dv))).Append("\",\"dn_f32\":\"").Append(B64(V3(dn)))
                      .Append("\",\"dt_f32\":\"").Append(B64(V3(dt))).Append("\"}");
                }
                sb.Append("]}");
            }
            sb.Append("]}");
            File.WriteAllText(path, sb.ToString());
        }

        static float[] V3(Vector3[] v) { var f = new float[v.Length * 3]; for (int i = 0; i < v.Length; i++) { f[3 * i] = v[i].x; f[3 * i + 1] = v[i].y; f[3 * i + 2] = v[i].z; } return f; }
        static float[] V2(Vector2[] v) { var f = new float[v.Length * 2]; for (int i = 0; i < v.Length; i++) { f[2 * i] = v[i].x; f[2 * i + 1] = v[i].y; } return f; }
        static float[] V4(Vector4[] v) { var f = new float[v.Length * 4]; for (int i = 0; i < v.Length; i++) { f[4 * i] = v[i].x; f[4 * i + 1] = v[i].y; f[4 * i + 2] = v[i].z; f[4 * i + 3] = v[i].w; } return f; }
        // Per-transform rows in static.json order (frame exports and, since 6.1.0, every bridge
        // render's pre_cull record). Getters only; nothing is written to the transforms.
        internal const string TransformLayout = "localPosition3 localRotation4 localScale3 position3 rotation4 localToWorld16(row-major)";

        internal static float[] TransformRows(Transform[] transforms)
        {
            var f = new float[transforms.Length * 33];
            for (int i = 0; i < transforms.Length; i++)
            {
                var t = transforms[i];
                int o = i * 33;
                Put(f, ref o, t.localPosition); Put(f, ref o, t.localRotation); Put(f, ref o, t.localScale);
                Put(f, ref o, t.position); Put(f, ref o, t.rotation);
                var m = t.localToWorldMatrix;
                for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) f[o++] = m[r, c];
            }
            return f;
        }

        internal static byte[] ActiveFlags(Transform[] transforms)
        {
            var act = new byte[transforms.Length];
            for (int i = 0; i < transforms.Length; i++) act[i] = (byte)(transforms[i].gameObject.activeInHierarchy ? 1 : 0);
            return act;
        }

        static void Put(float[] f, ref int o, Vector3 v) { f[o++] = v.x; f[o++] = v.y; f[o++] = v.z; }
        static void Put(float[] f, ref int o, Quaternion q) { f[o++] = q.x; f[o++] = q.y; f[o++] = q.z; f[o++] = q.w; }
        internal static string B64(float[] f) { var b = new byte[f.Length * 4]; Buffer.BlockCopy(f, 0, b, 0, b.Length); return Convert.ToBase64String(b); }
        static string Safe(string s) { var sb = new StringBuilder(); foreach (var c in s) sb.Append(char.IsLetterOrDigit(c) ? c : '_'); return sb.ToString(); }
        static byte[] Hex(string s) { var b = new byte[s.Length / 2]; for (int i = 0; i < b.Length; i++) b[i] = Convert.ToByte(s.Substring(2 * i, 2), 16); return b; }

        internal static string Json(string s)
        {
            if (s == null) return "null";
            var sb = new StringBuilder("\"");
            foreach (var c in s)
            {
                if (c == '"' || c == '\\') sb.Append('\\').Append(c);
                else if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4"));
                else sb.Append(c);
            }
            return sb.Append('"').ToString();
        }
    }

    // Official in-app API of the renderdoc.dll already loaded by the audited early
    // bootstrap. Never loads a library; table offsets per renderdoc_app.h v1.6.0
    // (0 GetAPIVersion .. 13 GetNumCaptures, 14 GetCapture, 15 TriggerCapture, 20 IsFrameCapturing).
    internal sealed class RenderDocApi
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern IntPtr GetModuleHandleW(string name);
        [DllImport("kernel32.dll", CharSet = CharSet.Ansi)] static extern IntPtr GetProcAddress(IntPtr module, string name);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern uint GetModuleFileNameW(IntPtr module, StringBuilder name, uint size);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int GetApiFn(int version, out IntPtr api);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void VersionFn(out int major, out int minor, out int patch);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void SetTemplateFn([MarshalAs(UnmanagedType.LPStr)] string path);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate IntPtr GetTemplateFn();
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate uint NumFn();
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate uint GetCaptureFn(uint idx, byte[] name, ref uint len, ref ulong ts);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void TriggerFn();
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate uint IsCapturingFn();

        readonly string _path, _sha;
        readonly int _major, _minor, _patch;
        readonly SetTemplateFn _set; readonly GetTemplateFn _get; readonly NumFn _num; readonly GetCaptureFn _cap; readonly TriggerFn _trigger; readonly IsCapturingFn _capturing;

        public RenderDocApi(string expectedSha, List<string> log)
        {
            IntPtr module = GetModuleHandleW("renderdoc.dll");
            if (module == IntPtr.Zero) throw new InvalidOperationException("Official renderdoc.dll is not loaded in this process");
            var name = new StringBuilder(1024);
            GetModuleFileNameW(module, name, 1024);
            _path = name.ToString();
            using (var sha = SHA256.Create())
                _sha = BitConverter.ToString(sha.ComputeHash(File.ReadAllBytes(_path))).Replace("-", "").ToLowerInvariant();
            if (_sha != expectedSha.ToLowerInvariant()) throw new InvalidOperationException("Loaded renderdoc.dll hash " + _sha + " differs");
            var proc = GetProcAddress(module, "RENDERDOC_GetAPI");
            if (proc == IntPtr.Zero) throw new InvalidOperationException("RENDERDOC_GetAPI missing");
            IntPtr api;
            if (((GetApiFn)Marshal.GetDelegateForFunctionPointer(proc, typeof(GetApiFn)))(10600, out api) != 1 || api == IntPtr.Zero)
                throw new InvalidOperationException("RENDERDOC_GetAPI 1.6.0 rejected");
            Func<int, Type, Delegate> fn = (i, t) =>
            {
                var p = Marshal.ReadIntPtr(api, i * IntPtr.Size);
                if (p == IntPtr.Zero) throw new InvalidOperationException("Null API slot " + i);
                return Marshal.GetDelegateForFunctionPointer(p, t);
            };
            ((VersionFn)fn(0, typeof(VersionFn)))(out _major, out _minor, out _patch);
            _set = (SetTemplateFn)fn(11, typeof(SetTemplateFn));
            _get = (GetTemplateFn)fn(12, typeof(GetTemplateFn));
            _num = (NumFn)fn(13, typeof(NumFn));
            _cap = (GetCaptureFn)fn(14, typeof(GetCaptureFn));
            _trigger = (TriggerFn)fn(15, typeof(TriggerFn));
            _capturing = (IsCapturingFn)fn(20, typeof(IsCapturingFn));   // IsFrameCapturing (diagnostic only: core counter, see present_retry_lifecycle_failure.md)
            log.Add("renderdoc module " + _path);
        }

        public string Describe() { return _path + "|sha256=" + _sha + "|api=" + _major + "." + _minor + "." + _patch; }
        public string GetTemplate() { return Marshal.PtrToStringAnsi(_get()); }
        public void SetTemplate(string t) { _set(t); }
        public uint NumCaptures() { return _num(); }
        public void Trigger() { _trigger(); }
        public uint IsFrameCapturing() { return _capturing(); }

        public void GetCapture(uint idx, out string path, out ulong ts)
        {
            uint len = 0; ulong t = 0;
            if (_cap(idx, null, ref len, ref t) != 1) throw new InvalidOperationException("GetCapture index invalid");
            var buf = new byte[len + 1];
            _cap(idx, buf, ref len, ref t);
            path = Encoding.UTF8.GetString(buf, 0, (int)Math.Max(0, (int)len - 1));
            ts = t;
        }
    }
}
