// Raw texel export through HS2_RenderAudit's native readback DLL
// (tools/offline_renderer/native_audit/d3d11_readback.cpp, installed as
// BepInEx/Plugins/HS2_McpBridge/HS2_RenderReadback.dll). The DLL is loaded from the
// request path after its SHA-256 is checked; its exports are used directly so the
// output directory comes from the request (HS2_RenderAudit.Capture() reads a sidecar
// config beside the installed DLL and covers only head renderers, so it is not used).
// QueueTexture AddRefs the ID3D11Resource; the copy to a staging resource, the Map
// and the file writes run on Unity's render thread inside GL.IssuePluginEvent.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using UnityEngine;

namespace HS2Codex.FramePairProbe
{
    internal sealed class RawJob
    {
        public int InstanceId, TextureId, Handle = -1;
        public string Name, Type, Path, Status, Error;
        public long EstimateUpperBytes;
        public HashSet<string> Reasons = new HashSet<string>();
    }

    internal sealed class RawTextures
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] static extern IntPtr LoadLibraryW(string path);
        [DllImport("kernel32.dll", CharSet = CharSet.Ansi)] static extern IntPtr GetProcAddress(IntPtr module, string name);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate IntPtr EventFn();
        [UnmanagedFunctionPointer(CallingConvention.Cdecl, CharSet = CharSet.Unicode)] delegate int QueueFn(IntPtr resource, [MarshalAs(UnmanagedType.LPWStr)] string path);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int StatusFn(int job, byte[] error, int capacity);

        readonly QueueFn _queue; readonly StatusFn _status; readonly IntPtr _event;
        public readonly string DllPath, DllSha256;
        public readonly List<RawJob> Jobs = new List<RawJob>();
        public int QueuedFrame = -1;
        public string FreeSpaceCheck;

        public RawTextures(string path, string expectedSha)
        {
            using (var h = SHA256.Create()) DllSha256 = Mem.Hex(h.ComputeHash(File.ReadAllBytes(path)));
            if (DllSha256 != expectedSha) throw new InvalidOperationException("Readback DLL " + path + " hash " + DllSha256 + " differs from " + expectedSha);
            DllPath = path;
            IntPtr module = LoadLibraryW(path);
            if (module == IntPtr.Zero) throw new InvalidOperationException("LoadLibrary failed: " + Marshal.GetLastWin32Error());
            _queue = (QueueFn)Marshal.GetDelegateForFunctionPointer(Need(module, "QueueTexture"), typeof(QueueFn));
            _status = (StatusFn)Marshal.GetDelegateForFunctionPointer(Need(module, "GetTextureStatus"), typeof(StatusFn));
            _event = ((EventFn)Marshal.GetDelegateForFunctionPointer(Need(module, "GetRenderEventFunc"), typeof(EventFn)))();
            if (_event == IntPtr.Zero) throw new InvalidOperationException("GetRenderEventFunc returned null");
        }

        static IntPtr Need(IntPtr module, string name)
        {
            var p = GetProcAddress(module, name);
            if (p == IntPtr.Zero) throw new InvalidOperationException("Readback export missing: " + name);
            return p;
        }

        public static long UpperBound(Texture t)
        {
            long faces = t.dimension == UnityEngine.Rendering.TextureDimension.Cube ? 6 : 1;
            long depth = 1;
            var t3 = t as Texture3D; if (t3 != null) depth = t3.depth;
            var rt = t as RenderTexture; if (rt != null && rt.volumeDepth > 1) depth = rt.volumeDepth;
            long bytes = (long)t.width * t.height * depth * faces * 16;
            int mips = 1;
            var t2 = t as Texture2D; if (t2 != null) mips = t2.mipmapCount;
            var cube = t as Cubemap; if (cube != null) mips = cube.mipmapCount;
            if (rt != null && rt.useMipMap) mips = 2;
            return mips > 1 ? bytes * 4 / 3 + 16 : bytes;
        }

        // textures: candidate set with the reasons each was selected.
        public void Queue(Dictionary<Texture, HashSet<string>> textures, string outDir, long minFreeBytes, UnityNative native)
        {
            if (Jobs.Count > 0) throw new InvalidOperationException("Raw export already queued in this window");
            long estimate = textures.Keys.Sum(t => UpperBound(t));
            var root = System.IO.Path.GetPathRoot(outDir);
            var drive = new DriveInfo(root);
            long free = drive.AvailableFreeSpace;
            FreeSpaceCheck = "drive " + root + " free " + free + " bytes; upper-bound estimate " + estimate + " + reserve " + minFreeBytes;
            if (free < estimate + minFreeBytes) throw new IOException("Insufficient free space for raw export: " + FreeSpaceCheck);
            Directory.CreateDirectory(outDir);
            foreach (var kv in textures.OrderBy(k => k.Key.GetInstanceID()))
            {
                var t = kv.Key;
                var job = new RawJob { InstanceId = t.GetInstanceID(), Name = t.name, Type = t.GetType().FullName, EstimateUpperBytes = UpperBound(t) };
                foreach (var r in kv.Value) job.Reasons.Add(r);
                try { job.TextureId = native.TextureId(t); } catch (Exception e) { job.Error = "texture id: " + e.Message; }
                Jobs.Add(job);
                var rt = t as RenderTexture;
                if (rt != null && !rt.IsCreated()) { job.Status = "skipped"; job.Error = "RenderTexture not created (no native resource; not created by the probe)"; continue; }
                IntPtr ptr;
                try { ptr = t.GetNativeTexturePtr(); }
                catch (Exception e) { job.Status = "failed"; job.Error = "GetNativeTexturePtr: " + e.Message; continue; }
                if (ptr == IntPtr.Zero) { job.Status = "skipped"; job.Error = "no native resource"; continue; }
                job.Path = System.IO.Path.Combine(outDir, "texture_" + job.InstanceId.ToString(System.Globalization.CultureInfo.InvariantCulture));
                job.Handle = _queue(ptr, job.Path);
                job.Status = job.Handle < 0 ? "failed" : "pending";
                if (job.Handle < 0) job.Error = "QueueTexture rejected the resource";
            }
            QueuedFrame = Time.frameCount;
            if (Jobs.Any(j => j.Handle >= 0)) GL.IssuePluginEvent(_event, 1);
        }

        public int Pending()
        {
            int pending = 0;
            foreach (var j in Jobs)
            {
                if (j.Handle < 0 || j.Status == "complete" || j.Status == "failed") continue;
                var err = new byte[4096];
                int s = _status(j.Handle, err, err.Length);
                if (s == 0) { pending++; continue; }
                j.Status = s == 1 ? "complete" : "failed";
                if (s != 1) j.Error = Encoding.UTF8.GetString(err, 0, Math.Max(0, Array.IndexOf(err, (byte)0)));
            }
            return pending;
        }

        // Blocks the calling (main) thread. Safe: the render thread needs nothing from
        // the main thread to reach the queued plugin event.
        public bool WaitAll(float seconds)
        {
            var deadline = DateTime.UtcNow.AddSeconds(seconds);
            while (Pending() > 0)
            {
                if (DateTime.UtcNow > deadline) return false;
                Thread.Sleep(5);
            }
            return true;
        }

        public void Write(JsonOut w)
        {
            w.Begin('{');
            w.Key("dll"); w.Str(DllPath); w.Key("dll_sha256"); w.Str(DllSha256);
            w.Key("queued_frame"); w.Int(QueuedFrame);
            w.Key("free_space_check"); w.Str(FreeSpaceCheck);
            w.Key("layout"); w.Str("<path>/descriptor.json + subresource_<D3D11CalcSubresource>.bin, D3D11 subresource order, rows tightly packed (d3d11_readback.cpp)");
            w.Key("counts"); w.Begin('{');
            foreach (var g in Jobs.GroupBy(j => j.Status ?? "unknown")) { w.Key(g.Key); w.Int(g.Count()); }
            w.End('}');
            w.Key("jobs"); w.Begin('[');
            foreach (var j in Jobs)
            {
                w.Begin('{');
                w.Key("instance_id"); w.Int(j.InstanceId); w.Key("name"); w.Str(j.Name); w.Key("type"); w.Str(j.Type);
                w.Key("texture_id"); w.Int(j.TextureId);
                w.Key("reasons"); SceneState.Names(w, j.Reasons.OrderBy(x => x, StringComparer.Ordinal));
                w.Key("path"); w.Str(j.Path); w.Key("status"); w.Str(j.Status); w.Key("error"); w.Str(j.Error);
                w.Key("estimate_upper_bytes"); w.Int(j.EstimateUpperBytes);
                w.End('}');
            }
            w.End(']');
            w.End('}');
        }
    }
}
