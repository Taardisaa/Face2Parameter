// Same-frame source-state / RDC pairing probe (read-only with respect to the game).
//
// On an explicit request file it, for a short window of consecutive frames and
// at Unity's end-of-frame point (after all Update/animation/LateUpdate writers and
// the frame's CPU skinning/rendering, before PresentAfterDraw):
//   * exports every transform of the single active ChaControl (local TRS, world
//     position/rotation, float32 localToWorldMatrix bits) and every skinned
//     renderer's blendshape weights, flags and bounds;
//   * uploads a 64-byte marker texture holding magic + nonce + Time.frameCount,
//     so the GPU command stream of that same frame names its export;
//   * calls the already-loaded official RenderDoc 1.46 in-app API TriggerCapture
//     once (a Present-boundary whole-frame capture; no Start/End scope).
// Static mesh data (vertices, normals, tangents, bind poses, bone weights and all
// blendshape frames) of the requested meshes is exported once. Nothing is set on
// the character, no render is issued, no card is touched.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using AIChara;
using BepInEx;
using UnityEngine;

namespace HS2Codex.FramePairProbe
{
    [BepInPlugin("codex.hs2.framepairprobe", "HS2 Frame Pair Probe", "1.0.0")]
    public class FramePairProbe : BaseUnityPlugin
    {
        const string Magic = "HS2PAIRMARKERv01";
        const string Tail = "HS2PAIRMARKEREND";
        string _requestPath;
        bool _busy;
        Texture2D _marker;

        void Awake()
        {
            _requestPath = Path.Combine(Path.GetDirectoryName(Info.Location), "HS2_FramePairProbe.request.ini");
            Logger.LogInfo("FramePairProbe ready; request path " + _requestPath);
        }

        void Update()
        {
            if (_busy || Time.frameCount % 15 != 0 || !File.Exists(_requestPath)) return;
            Dictionary<string, string> req;
            try
            {
                req = ParseIni(File.ReadAllText(_requestPath));
                File.Delete(_requestPath);
            }
            catch (Exception e) { Logger.LogError("Request unreadable: " + e); return; }
            _busy = true;
            StartCoroutine(Run(req));
        }

        static Dictionary<string, string> ParseIni(string text)
        {
            var d = new Dictionary<string, string>();
            foreach (var raw in text.Split('\n'))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line.StartsWith("#")) continue;
                int i = line.IndexOf('=');
                if (i <= 0) throw new FormatException("Bad request line: " + line);
                d[line.Substring(0, i).Trim()] = line.Substring(i + 1).Trim();
            }
            return d;
        }

        IEnumerator Run(Dictionary<string, string> req)
        {
            string outDir = req["OutDir"];
            var log = new List<string>();
            Exception failure = null;
            IEnumerator inner = null;
            try
            {
                if (Directory.Exists(outDir)) throw new IOException("Fresh OutDir required: " + outDir);
                Directory.CreateDirectory(Path.Combine(outDir, "frames"));
                File.WriteAllText(Path.Combine(outDir, "request.ini"), string.Join("\n", Dict(req)));
                inner = Window(req, outDir, log);
            }
            catch (Exception e) { failure = e; }
            while (failure == null)
            {
                bool more;
                try { more = inner.MoveNext(); }
                catch (Exception e) { failure = e; break; }
                if (!more) break;
                yield return inner.Current;
            }
            if (failure != null)
            {
                log.Add("FAILURE " + failure);
                Logger.LogError(failure.ToString());
            }
            try { File.WriteAllText(Path.Combine(outDir, "probe_log.txt"), string.Join("\n", log.ToArray())); }
            catch (Exception e) { Logger.LogError(e.ToString()); }
            _busy = false;
        }

        static IEnumerable<string> Dict(Dictionary<string, string> d)
        {
            foreach (var kv in d) yield return kv.Key + "=" + kv.Value;
        }

        IEnumerator Window(Dictionary<string, string> req, string outDir, List<string> log)
        {
            byte[] nonce = Hex(req["Nonce"]);
            if (nonce.Length != 16) throw new ArgumentException("Nonce must be 32 hex digits");
            int frames = int.Parse(req["Frames"], CultureInfo.InvariantCulture);
            int trigger = int.Parse(req["TriggerIndex"], CultureInfo.InvariantCulture);
            float poll = float.Parse(req["PollSeconds"], CultureInfo.InvariantCulture);
            if (frames < 2 || frames > 120 || trigger < 0 || trigger >= frames) throw new ArgumentException("Bad window");
            var meshNames = new HashSet<string>(req["Meshes"].Split(','));

            var charas = UnityEngine.Object.FindObjectsOfType<ChaControl>();
            if (charas.Length != 1) throw new InvalidOperationException("Exactly one active ChaControl required, found " + charas.Length);
            var cha = charas[0];
            var transforms = cha.transform.GetComponentsInChildren<Transform>(true);
            var smrs = cha.transform.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            var index = new Dictionary<Transform, int>();
            for (int i = 0; i < transforms.Length; i++) index[transforms[i]] = i;

            var rd = new RenderDocApi(req["ExpectedRenderDocSha256"], log);
            string oldTemplate = rd.GetTemplate();
            uint capturesBefore = rd.NumCaptures();
            rd.SetTemplate(req["CapturePrefix"]);
            log.Add("renderdoc " + rd.Describe() + " captures_before=" + capturesBefore + " old_template=" + oldTemplate);

            WriteStatic(Path.Combine(outDir, "static.json"), cha, transforms, smrs, index, meshNames, outDir, req);
            string manifestPath;
            req.TryGetValue("ShaderManifest", out manifestPath);
            var manifest = ShaderManifest.Load(manifestPath);
            string sceneEvery;
            bool scene = req.TryGetValue("SceneState", out sceneEvery) && sceneEvery == "1";

            _marker = new Texture2D(16, 1, TextureFormat.RGBA32, false, true);
            _marker.hideFlags = HideFlags.HideAndDontSave;
            _marker.name = "HS2_FramePairProbe_marker";
            var eof = new WaitForEndOfFrame();
            int triggeredFrame = -1;
            for (int k = 0; k < frames; k++)
            {
                yield return eof;
                int fc = Time.frameCount;
                WriteFrame(Path.Combine(outDir, "frames", "frame_" + k.ToString("D3") + ".json"), k, transforms, smrs);
                if (scene)
                {
                    // Same end-of-frame point as the transform export; recorded per frame
                    // because jitter/noise counters and effect history advance every frame.
                    try { SceneState.Write(Path.Combine(outDir, "frames", "scene_" + k.ToString("D3") + ".json"), manifest, k); }
                    catch (Exception e) { log.Add("scene state failed at index " + k + ": " + e); }
                }
                WriteMarker(nonce, fc, k);
                string auditPath;
                if (k == trigger && req.TryGetValue("RenderAuditDll", out auditPath) && auditPath.Length > 0)
                {
                    // Existing read-only helper: materials/SH/probes/effects plus raw native texture jobs.
                    try
                    {
                        var asm = System.Reflection.Assembly.LoadFrom(auditPath);
                        var type = asm.GetType("HS2_RenderAudit.RenderAudit", true);
                        var result = type.GetMethod("Capture").Invoke(null, null);
                        SceneState.WriteObject(Path.Combine(outDir, "render_audit_" + k.ToString("D3") + ".json"), result);
                        log.Add("RenderAudit.Capture at frame " + fc);
                    }
                    catch (Exception e) { log.Add("RenderAudit failed: " + e); }
                }
                if (k == trigger)
                {
                    rd.Trigger();
                    triggeredFrame = fc;
                    log.Add("TriggerCapture called at end of frame " + fc + " (index " + k + ")");
                }
            }
            float deadline = Time.realtimeSinceStartup + poll;
            uint after = rd.NumCaptures();
            while (after == capturesBefore && Time.realtimeSinceStartup < deadline)
            {
                yield return null;
                after = rd.NumCaptures();
            }
            var sb = new StringBuilder();
            sb.Append("{\"schema_version\":1,\"nonce\":\"").Append(req["Nonce"]).Append("\",\"triggered_frame\":").Append(triggeredFrame)
              .Append(",\"captures_before\":").Append(capturesBefore).Append(",\"captures_after\":").Append(after)
              .Append(",\"renderdoc\":").Append(Json(rd.Describe())).Append(",\"captures\":[");
            for (uint i = capturesBefore; i < after; i++)
            {
                string path; ulong ts;
                rd.GetCapture(i, out path, out ts);
                if (i > capturesBefore) sb.Append(',');
                sb.Append("{\"index\":").Append(i).Append(",\"path\":").Append(Json(path)).Append(",\"timestamp\":").Append(ts).Append('}');
            }
            sb.Append("],\"end_frame\":").Append(Time.frameCount).Append('}');
            rd.SetTemplate(oldTemplate);
            File.WriteAllText(Path.Combine(outDir, "capture_receipt.json"), sb.ToString());
            UnityEngine.Object.Destroy(_marker);
            _marker = null;
            log.Add("done; captures " + capturesBefore + " -> " + after);
        }

        void WriteMarker(byte[] nonce, int frameCount, int k)
        {
            var bytes = new byte[64];
            Encoding.ASCII.GetBytes(Magic, 0, 16, bytes, 0);
            Buffer.BlockCopy(nonce, 0, bytes, 16, 16);
            Buffer.BlockCopy(BitConverter.GetBytes(frameCount), 0, bytes, 32, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(k), 0, bytes, 36, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(Time.renderedFrameCount), 0, bytes, 40, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(~frameCount), 0, bytes, 44, 4);
            Encoding.ASCII.GetBytes(Tail, 0, 16, bytes, 48);
            var px = new Color32[16];
            for (int i = 0; i < 16; i++) px[i] = new Color32(bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]);
            _marker.SetPixels32(px);
            _marker.Apply(false, false);
        }

        static void WriteFrame(string path, int k, Transform[] transforms, SkinnedMeshRenderer[] smrs)
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
            var sb = new StringBuilder(1 << 20);
            sb.Append("{\"index\":").Append(k).Append(",\"frame_count\":").Append(Time.frameCount)
              .Append(",\"rendered_frame_count\":").Append(Time.renderedFrameCount)
              .Append(",\"realtime\":").Append(Time.realtimeSinceStartup.ToString("R", CultureInfo.InvariantCulture))
              .Append(",\"time\":").Append(Time.time.ToString("R", CultureInfo.InvariantCulture))
              .Append(",\"delta_time\":").Append(Time.deltaTime.ToString("R", CultureInfo.InvariantCulture))
              .Append(",\"transform_layout\":\"localPosition3 localRotation4 localScale3 position3 rotation4 localToWorld16(row-major)\"")
              .Append(",\"transforms_f32\":\"").Append(B64(f)).Append("\",\"active\":\"");
            var act = new byte[transforms.Length];
            for (int i = 0; i < transforms.Length; i++) act[i] = (byte)(transforms[i].gameObject.activeInHierarchy ? 1 : 0);
            sb.Append(Convert.ToBase64String(act)).Append("\",\"renderers\":[");
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
            Dictionary<Transform, int> index, HashSet<string> meshNames, string outDir, Dictionary<string, string> req)
        {
            var sb = new StringBuilder(1 << 20);
            sb.Append("{\"schema_version\":1,\"unity_version\":").Append(Json(Application.unityVersion))
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
                    if (meshNames.Contains(mesh.name) && mesh.isReadable)
                    {
                        string file = "mesh_" + (meshFile++).ToString("D2") + "_" + Safe(mesh.name) + ".json";
                        WriteMesh(Path.Combine(outDir, file), mesh);
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
        static void Put(float[] f, ref int o, Vector3 v) { f[o++] = v.x; f[o++] = v.y; f[o++] = v.z; }
        static void Put(float[] f, ref int o, Quaternion q) { f[o++] = q.x; f[o++] = q.y; f[o++] = q.z; f[o++] = q.w; }
        static string B64(float[] f) { var b = new byte[f.Length * 4]; Buffer.BlockCopy(f, 0, b, 0, b.Length); return Convert.ToBase64String(b); }
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
    // bootstrap. Never loads a library; table offsets per renderdoc_app.h v1.6.0.
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

        readonly string _path, _sha;
        readonly int _major, _minor, _patch;
        readonly SetTemplateFn _set; readonly GetTemplateFn _get; readonly NumFn _num; readonly GetCaptureFn _cap; readonly TriggerFn _trigger;

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
            log.Add("renderdoc module " + _path);
        }

        public string Describe() { return _path + "|sha256=" + _sha + "|api=" + _major + "." + _minor + "." + _patch; }
        public string GetTemplate() { return Marshal.PtrToStringAnsi(_get()); }
        public void SetTemplate(string t) { _set(t); }
        public uint NumCaptures() { return _num(); }
        public void Trigger() { _trigger(); }

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
