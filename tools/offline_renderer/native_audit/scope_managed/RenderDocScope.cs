using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json.Linq;

// Independent, opt-in hook shim; never loads renderdoc.dll or changes rendering.
// The existing game Capture owns all Camera.Render/ReadPixels/finally restoration.
public static partial class HS2_RenderDocScope
{
    public sealed class Request
    {
        public bool Enabled;
        public uint ExpectedPid;
        public string ExactPngPath, CapturePrefix, ReceiptPath;
        public string ExpectedNativeSha256, ExpectedRenderDocSha256;
        public string GeometryBeforePath;
        public string GeometryAfterPath;
        public string ExpectedBridgeSha256;
        public string CaptureMode, ExpectedUnityCoreSha256;
    }
    static readonly object Gate = new object();
    static readonly string DirectoryPath = Path.GetDirectoryName(typeof(HS2_RenderDocScope).Assembly.Location);

    static Request active;
    static ulong token;
    static IntPtr library;
    static BeginTexture begin;
    static Finish end, abort;
    static Receipt receipt;
    static readonly List<object> events = new List<object>();
    static string loadedNativeSha;
    static GeometryCapture geometry;
    public static string LastReceipt { get; private set; } = "{\"status\":\"not_started\"}";

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int BeginTexture(uint pid, byte[] prefix, IntPtr texture, out ulong outputToken);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int Finish(uint pid, ulong scopeToken);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate uint Receipt(IntPtr buffer, uint capacity);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr LoadLibraryExW(string path, IntPtr reserved, uint flags);
    [DllImport("kernel32.dll", CharSet = CharSet.Ansi, ExactSpelling = true)]
    static extern IntPtr GetProcAddress(IntPtr module, string name);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    static extern IntPtr GetModuleHandleW(string name);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern uint GetModuleFileNameW(IntPtr module, StringBuilder path, uint capacity);

    static T Export<T>(string name) where T : class
    {
        var address = GetProcAddress(library, name);
        if (address == IntPtr.Zero) throw new InvalidOperationException("Missing own helper export " + name);
        return (T)(object)Marshal.GetDelegateForFunctionPointer(address, typeof(T));
    }
    static void VerifyHash(string path, string expected)
    {
        if (expected == null || expected.Length != 64) throw new InvalidOperationException("Explicit SHA256 required for " + path);
        string actual;
        using (var hash = SHA256.Create()) using (var input = File.OpenRead(path))
            actual = BitConverter.ToString(hash.ComputeHash(input)).Replace("-", "").ToLowerInvariant();
        if (!String.Equals(actual, expected, StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("Pinned binary SHA256 mismatch: " + path);
    }
    static void Validate(Request request, string png)
    {
        if (request.ExpectedPid != (uint)Process.GetCurrentProcess().Id) throw new InvalidOperationException("Expected HS2 process mismatch");
        foreach (var path in new[] { request.ExactPngPath, request.CapturePrefix, request.ReceiptPath })
            if (String.IsNullOrEmpty(path) || !Path.IsPathRooted(path) || !Directory.Exists(Path.GetDirectoryName(path)))
                throw new InvalidOperationException("Absolute existing output parent required");
        if (!String.Equals(Path.GetFullPath(png), Path.GetFullPath(request.ExactPngPath), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Capture PNG does not match the explicit request");
        if (String.Equals(Path.GetFullPath(request.ReceiptPath), Path.GetFullPath(png), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Receipt must not overwrite the native PNG");
        var module = GetModuleHandleW("renderdoc.dll");
        if (module == IntPtr.Zero) throw new InvalidOperationException("RenderDoc must already be loaded by the official launcher");
        var modulePath = new StringBuilder(32768);
        var count = GetModuleFileNameW(module, modulePath, (uint)modulePath.Capacity);
        if (count == 0 || count == modulePath.Capacity) throw new InvalidOperationException("Loaded RenderDoc module path unavailable");
        VerifyHash(modulePath.ToString(), request.ExpectedRenderDocSha256);
        VerifyHash(Path.Combine(DirectoryPath, "hs2_renderdoc_scope_native.dll"), request.ExpectedNativeSha256);
    }
    static void LoadOwnHelper()
    {
        if (library != IntPtr.Zero && begin != null && end != null && abort != null && receipt != null) {
            if (!String.Equals(loadedNativeSha, active.ExpectedNativeSha256, StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("Loaded helper differs from requested binary; restart required");
            return;
        }
        // Absolute own-module path, restricted to its directory/system dependencies.
        // This does not load RenderDoc. Keep module loaded for delegate lifetime.
        library = LoadLibraryExW(Path.Combine(DirectoryPath, "hs2_renderdoc_scope_native.dll"), IntPtr.Zero, 0x00000100 | 0x00000800);
        if (library == IntPtr.Zero) throw new InvalidOperationException("Own helper load failed: " + Marshal.GetLastWin32Error());
        begin = Export<BeginTexture>("HS2_RDOC_BeginFromNativeTexture");
        end = Export<Finish>("HS2_RDOC_End"); abort = Export<Finish>("HS2_RDOC_AbortOwned");
        receipt = Export<Receipt>("HS2_RDOC_GetReceipt");
        loadedNativeSha = active.ExpectedNativeSha256;
    }
    static string ReadReceipt()
    {
        var bytes = receipt(IntPtr.Zero, 0);
        if (bytes == 0 || bytes > 1024 * 1024) throw new InvalidOperationException("Native receipt size invalid");
        var buffer = Marshal.AllocHGlobal((int)bytes);
        try {
            if (receipt(buffer, bytes) != bytes) throw new InvalidOperationException("Native receipt changed during serialized hook");
            var data = new byte[bytes - 1]; Marshal.Copy(buffer, data, 0, data.Length);
            return Encoding.UTF8.GetString(data);
        } finally { Marshal.FreeHGlobal(buffer); }
    }
    static bool NoLongerOwned(string native)
    {
        var values = ScopeJson.ParseObject(native); var value = values["owned"];
        if (value == null || value.Type != JTokenType.Boolean) throw new InvalidOperationException("Native receipt must declare boolean ownership");
        return !value.Value<bool>();
    }
    static void Save(string stage, int code, string native, Exception error = null, JToken originalResult = null)
    {
        var entry = new Dictionary<string, object> {
            { "schema_version", 1 }, { "stage", stage }, { "code", code },
            { "pid", Process.GetCurrentProcess().Id }, { "owned_token", token.ToString() },
            { "utc", DateTime.UtcNow.ToString("o") },
            { "native", native == null ? null : ScopeJson.ParseObject(native) },
            { "error", error == null ? null : error.ToString() },
            { "request", ScopeJson.RequestFields(active) }
        };
        if(originalResult!=null)entry.Add("original_result",originalResult);
        if (geometry != null) entry.Add("geometry", geometry.Describe());
        events.Add(entry);
        LastReceipt = ScopeJson.Object(new Dictionary<string, object> { { "schema_version", 1 }, { "events", events } });
        if (active != null && !String.IsNullOrEmpty(active.ReceiptPath)) File.WriteAllText(active.ReceiptPath, LastReceipt, new UTF8Encoding(false));
    }
    static void SafeSave(string stage, Exception error)
    {
        try { Save(stage, -100, null, error); }
        catch (Exception saveError) {
            try { LastReceipt = ScopeJson.Object(new Dictionary<string, object> {
                { "schema_version", 1 }, { "stage", stage }, { "code", -100 }, { "error", error.ToString() }, { "receipt_write_error", saveError.ToString() },
                { "previous_receipt", ScopeJson.ParseObject(LastReceipt) }, { "request", ScopeJson.RequestFields(active) }, { "failed_event_count", events.Count }
            }); }
            catch(Exception fallbackError) { LastReceipt=ScopeJson.Emergency(stage,error.ToString(),saveError.ToString(),fallbackError.ToString(),LastReceipt); }
            // A locked/unwritable primary receipt must not erase diagnostics or
            // throw through native restoration. A separate fresh failure file is
            // recoverable when its original parent remains writable.
            if(active!=null && !String.IsNullOrEmpty(active.ReceiptPath)) {
                string path=active.ReceiptPath+".failure."+Guid.NewGuid().ToString("N")+".json";
                try {using(var file=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.Read))
                    using(var writer=new StreamWriter(file,new UTF8Encoding(false)))writer.Write(LastReceipt);}
                catch(Exception fallbackWrite) { LastReceipt=ScopeJson.Emergency(stage,error.ToString(),saveError.ToString(),fallbackWrite.ToString(),LastReceipt); }
            }
        }
    }

    // IL: after assigning actual RT to Camera.targetTexture, before ResetTemporal.
    public static void BeginFromNativeTexture(IntPtr resource, string capturePngPath)
    {
        lock (Gate) {
            if (active != null && active.CaptureMode == "official_present") { if(ExactPresentPath(capturePngPath)) BindPresentTarget(resource, capturePngPath); return; }
            if (token != 0) { SafeSave("begin_rejected_owned", new InvalidOperationException("Previous owned scope still active")); return; }
            active = null;
            geometry = null;
            events.Clear();
            try {
                string path = Path.Combine(DirectoryPath, "HS2_RenderDocScope.request.json");
                if (!File.Exists(path)) return;
                var request = ScopeJson.ReadRequest(File.ReadAllText(path));
                if (request == null || !request.Enabled) return;
                // Ignore other ordinary Captures; exact opt-in pathname selects one.
                if (String.IsNullOrEmpty(request.ExactPngPath) || !String.Equals(Path.GetFullPath(request.ExactPngPath), Path.GetFullPath(capturePngPath), StringComparison.OrdinalIgnoreCase)) return;
                if (request.CaptureMode == "official_present") throw new InvalidOperationException("Present mode requires outer dispatch Arm/ready/Enter before original full Capture");
                Validate(request, capturePngPath); active = request;
                geometry = GeometryCapture.Prepare(request); LoadOwnHelper();
                // Same frozen actor/current temporary camera, immediately before SDK Start.
                if (geometry != null) geometry.Before();
                int code = begin(request.ExpectedPid, Encoding.UTF8.GetBytes(request.CapturePrefix + "\0"), resource, out token);
                Save("begin", code, ReadReceipt());
            } catch (Exception error) { SafeSave("begin_failure", error); }
        }
    }
    // IL: after saved PNG/readback/inner finally, before result dictionary.
    public static void EndAfterSavedPng(string capturePngPath)
    {
        lock (Gate) {
            if (active != null && active.CaptureMode == "official_present") { if(ExactPresentPath(capturePngPath)) CompletePresentOriginal(capturePngPath); return; }
            if (token == 0) return;
            try {
                if (!String.Equals(Path.GetFullPath(capturePngPath), Path.GetFullPath(active.ExactPngPath), StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("End pathname mismatch");
                if (!File.Exists(capturePngPath) || new FileInfo(capturePngPath).Length == 0) throw new InvalidOperationException("Actual synchronized PNG missing");
                int code = end(active.ExpectedPid, token);
                string native = ReadReceipt();
                if (NoLongerOwned(native)) token = 0;
                if (geometry != null && code == 2) {
                    Save("end_native_pending_geometry", code, native);
                    // SDK End and saved PNG succeeded; existing outer restoration has not run.
                    geometry.After();
                    if (!geometry.Complete) throw new InvalidOperationException("Paired geometry export incomplete");
                }
                Save("end", code, native);
            } catch (Exception error) { SafeSave("end_failure", error); }
        }
    }
    // IL: first instruction of existing outer restoration finally. Never throws.
    public static void AbortOwnedSafely()
    {
        lock (Gate) {
            if (active != null && active.CaptureMode == "official_present") {
                if(presentInsideOriginal){presentInsideOriginal=false;if (!presentOriginalComplete) AbortPresentSafely();}
                return;
            }
            if (token == 0) return;
            try {
                int code = abort(active.ExpectedPid, token);
                string native = ReadReceipt();
                if (NoLongerOwned(native)) token = 0;
                Save("abort", code, native);
            } catch (Exception error) { SafeSave("abort_failure", error); }
        }
    }
}
