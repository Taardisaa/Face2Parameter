// Mode "renderstyle": issues the installed bridge's own /maker/renderstyle/capture
// request so that its sixteen Camera.Render() calls land in a known frame.
//
// Source chain (installed HS2_McpBridge.dll efa943cb…, decompiled):
//   HttpServer.HandleRequest "/maker/renderstyle/capture" -> RunOnMainThread(func, 5)
//   (the RenderDoc dispatch branch is taken only if HS2_RenderDocScope.request.json
//   exists; the probe refuses to post in that case) -> MainThreadDispatcher.Invoke ->
//   KoikatuAPI.SynchronizedInvoke -> BepInEx ThreadingHelper.StartSyncInvoke, which
//   appends to ThreadingHelper._invokeList under _invokeLock. ThreadingHelper.Update
//   swaps the list out and runs every action, i.e. the capture (and its sixteen
//   renders) runs inside ScriptRunBehaviourUpdate of the first frame whose
//   ThreadingHelper.Update starts after the enqueue.
// The probe posts from its end-of-frame coroutine of frame P (after every Update of P)
// and blocks the main thread until the bridge's action is in _invokeList, so the
// renders run in frame P+1. RunOnMainThread's 5 s wait only decides the HTTP reply
// (504 operation_in_progress once the work has started); the probe then polls
// /operations/{id}. An action still queued after 5 s would be cancelled by the
// bridge, which cannot happen here because P+1 starts right after the block ends.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using UnityEngine;

namespace HS2Codex.FramePairProbe
{
    internal sealed class BridgeCapture
    {
        readonly ProbeRequest _req;
        public readonly BridgeBody Body;
        readonly Assembly _bridge;
        public readonly string BridgeLocation, BridgeSha256;
        readonly object _helper, _invokeLock;
        readonly FieldInfo _invokeList;
        readonly object _gate = new object();
        readonly List<string> _events = new List<string>();
        Thread _thread;
        volatile bool _done, _responded;
        public int PostFrame = -1, QueuedObservedFrame = -1;
        public double QueueWaitMs = -1;
        string _finalStatus, _finalBody, _initialStatus, _initialBody;

        BridgeCapture(ProbeRequest req, BridgeBody body, Assembly bridge, string location, string sha, object helper, object invokeLock, FieldInfo invokeList)
        {
            _req = req; Body = body; _bridge = bridge; BridgeLocation = location; BridgeSha256 = sha;
            _helper = helper; _invokeLock = invokeLock; _invokeList = invokeList;
        }

        public static BridgeCapture Prepare(ProbeRequest req)
        {
            if (File.Exists(req.BridgeDispatchRequestFile))
                throw new InvalidOperationException("Installed bridge would route the capture through HS2_RenderDocDispatch because " + req.BridgeDispatchRequestFile + " exists");
            var bridge = AppDomain.CurrentDomain.GetAssemblies().FirstOrDefault(a => a.GetName().Name == "HS2_McpBridge");
            if (bridge == null) throw new InvalidOperationException("HS2_McpBridge assembly is not loaded");
            string sha;
            using (var h = SHA256.Create()) sha = Mem.Hex(h.ComputeHash(File.ReadAllBytes(bridge.Location)));
            if (sha != req.ExpectedBridgeSha256) throw new InvalidOperationException("Loaded bridge " + bridge.Location + " hash " + sha + " differs from " + req.ExpectedBridgeSha256);
            var body = BridgeBody.Parse(File.ReadAllText(req.BridgeBodyFile), p => File.Exists(p) || Directory.Exists(p));
            var helperType = typeof(BepInEx.ThreadingHelper);
            var helper = helperType.GetProperty("Instance", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(null, null);
            var list = helperType.GetField("_invokeList", BindingFlags.Instance | BindingFlags.NonPublic);
            var lockField = helperType.GetField("_invokeLock", BindingFlags.Instance | BindingFlags.NonPublic);
            if (helper == null || list == null || lockField == null || list.FieldType != typeof(Action))
                throw new InvalidOperationException("BepInEx ThreadingHelper contract differs (Instance/_invokeList/_invokeLock)");
            return new BridgeCapture(req, body, bridge, bridge.Location, sha, helper, lockField.GetValue(helper), list);
        }

        // Same rule as MakerRenderStyleService.Capture: 16 renders when an enabled CTAA_PC is on Camera.main.
        public static int ExpectedRenders(Camera main)
        {
            return main.GetComponents<Behaviour>().Any(c => c != null && c.GetType().FullName == "CTAA_PC" && c.enabled) ? 16 : 1;
        }

        public string QueueMatch;

        // A queued ThreadingHelper action of the bridge assembly. "precise" when it is the
        // MainThreadDispatcher.Invoke closure whose 'action' delegate captured this request's
        // body (HttpServer.HandleRequest closure field 'body', the same contract the installed
        // HS2_RenderDocDispatch checks); "bridge" for any other bridge action.
        string QueuedBridgeAction()
        {
            lock (_invokeLock)
            {
                var list = (Action)_invokeList.GetValue(_helper);
                if (list == null) return null;
                string found = null;
                foreach (var d in list.GetInvocationList())
                {
                    if (d.Method.DeclaringType == null || d.Method.DeclaringType.Assembly != _bridge) continue;
                    found = found ?? "bridge";
                    var target = d.Target;
                    var actionField = target == null ? null : target.GetType().GetField("action", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                    var inner = actionField == null ? null : actionField.GetValue(target) as Delegate;
                    var bodyField = inner == null || inner.Target == null ? null : inner.Target.GetType().GetField("body", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                    if (bodyField != null && bodyField.FieldType == typeof(string) && (string)bodyField.GetValue(inner.Target) == Body.Text) return "precise";
                }
                return found;
            }
        }

        bool BridgeActionQueued() { return QueuedBridgeAction() != null; }

        // Main thread, end of frame. Returns true once the bridge's dispatcher action is queued.
        public bool PostAndWaitQueued(float seconds)
        {
            if (File.Exists(_req.BridgeDispatchRequestFile)) throw new InvalidOperationException("RenderDoc dispatch request file appeared: " + _req.BridgeDispatchRequestFile);
            if (BridgeActionQueued()) throw new InvalidOperationException("A bridge action is already queued before the probe's post");
            PostFrame = Time.frameCount;
            _thread = new Thread(Worker) { IsBackground = true, Name = "HS2_FramePairProbe.bridge" };
            var clock = System.Diagnostics.Stopwatch.StartNew();
            _thread.Start();
            while (clock.Elapsed.TotalSeconds < seconds)
            {
                var match = QueuedBridgeAction();
                if (match != null)
                {
                    QueueMatch = match;
                    QueueWaitMs = clock.Elapsed.TotalMilliseconds; QueuedObservedFrame = Time.frameCount;
                    Event("bridge action observed in ThreadingHelper._invokeList after " + QueueWaitMs.ToString("F3") + " ms (match: " + match + ")");
                    return true;
                }
                if (_responded) { Event("bridge replied before its action was queued (request rejected)"); return false; }
                Thread.Sleep(1);
            }
            Event("timeout waiting for the bridge action to be queued");
            return false;
        }

        public bool Done { get { return _done; } }

        void Event(string s) { lock (_gate) _events.Add(DateTime.UtcNow.ToString("o") + " " + s); }

        void Worker()
        {
            try
            {
                int status; string body;
                Http("POST", _req.BridgeUrl, Body.Text, out status, out body);
                lock (_gate) { _initialStatus = status.ToString(); _initialBody = body; }
                _responded = true;
                Event("POST returned " + status);
                if (status == 504)
                {
                    var err = MiniJson.Parse(body) as Dictionary<string, object>;
                    object op;
                    if (err == null || !err.TryGetValue("operation_id", out op) || !(op is string) || ((string)op).Length == 0)
                        throw new InvalidOperationException("504 without operation_id: " + body);
                    var url = "http://127.0.0.1:43127/operations/" + (string)op;
                    var deadline = DateTime.UtcNow.AddSeconds(_req.BridgeTimeoutSeconds);
                    while (true)
                    {
                        Http("GET", url, null, out status, out body);
                        var row = MiniJson.Parse(body) as Dictionary<string, object>;
                        object st;
                        string s = row != null && row.TryGetValue("status", out st) ? st as string : null;
                        if (s != "queued" && s != "running") break;
                        if (DateTime.UtcNow > deadline) throw new TimeoutException("operation " + (string)op + " still " + s);
                        Thread.Sleep(250);
                    }
                    Event("operation poll ended with HTTP " + status);
                }
                lock (_gate) { _finalStatus = status.ToString(); _finalBody = body; }
            }
            catch (Exception e) { Event("worker failure: " + e); _responded = true; }
            finally { _done = true; }
        }

        void Http(string method, string url, string body, out int status, out string text)
        {
            var request = (HttpWebRequest)WebRequest.Create(url);
            request.Method = method;
            request.Proxy = null;
            request.KeepAlive = false;
            request.Timeout = (int)(_req.BridgeTimeoutSeconds * 1000);
            request.ReadWriteTimeout = request.Timeout;
            if (body != null)
            {
                var bytes = new UTF8Encoding(false).GetBytes(body);
                request.ContentType = "application/json";
                request.ContentLength = bytes.Length;
                using (var s = request.GetRequestStream()) s.Write(bytes, 0, bytes.Length);
            }
            HttpWebResponse response;
            try { response = (HttpWebResponse)request.GetResponse(); }
            catch (WebException e)
            {
                response = e.Response as HttpWebResponse;
                if (response == null) throw;
            }
            using (response)
            using (var reader = new StreamReader(response.GetResponseStream(), Encoding.UTF8))
            {
                status = (int)response.StatusCode;
                text = reader.ReadToEnd();
            }
        }

        public void Write(JsonOut w)
        {
            lock (_gate)
            {
                w.Begin('{');
                w.Key("bridge_assembly"); w.Str(BridgeLocation); w.Key("bridge_sha256"); w.Str(BridgeSha256);
                w.Key("url"); w.Str(_req.BridgeUrl);
                w.Key("request_body"); w.Str(Body.Text);
                using (var h = SHA256.Create()) { w.Key("request_body_sha256"); w.Str(Mem.Hex(h.ComputeHash(new UTF8Encoding(false).GetBytes(Body.Text)))); }
                w.Key("body_width"); w.Int(Body.Width); w.Key("body_height"); w.Int(Body.Height); w.Key("png_path"); w.Str(Body.PngPath);
                w.Key("body_yaw_degrees"); w.Raw(Body.Yaw.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
                w.Key("post_frame"); w.Int(PostFrame);
                w.Key("queued_observed_frame"); w.Int(QueuedObservedFrame);
                w.Key("queue_match"); w.Str(QueueMatch);
                w.Key("queue_wait_ms"); w.Raw(QueueWaitMs.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
                w.Key("done"); w.Bool(_done);
                w.Key("initial_http_status"); w.Str(_initialStatus); w.Key("initial_http_body"); w.Str(_initialBody);
                w.Key("final_http_status"); w.Str(_finalStatus); w.Key("final_http_body"); w.Str(_finalBody);
                w.Key("events"); SceneState.Names(w, _events);
                w.End('}');
            }
        }
    }
}
