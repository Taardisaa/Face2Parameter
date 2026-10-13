using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json;

// Calls the pinned installed read-only exporter. Never rebuilds game code or
// rewrites its geometry JSON. Optional metadata is tied to the same freeze scope.
internal sealed class GeometryCapture
{
    static readonly Type[] ParameterTypes = { typeof(string), typeof(string[]), typeof(bool), typeof(string), typeof(bool) };
    static readonly string[] ParameterNames = { "absoluteJsonPath", "meshes", "includeBlendShapeFrames", "materialTextureDirectory", "includeActorTransforms" };
    const string TypeName = "HS2Mod.McpBridge.MakerGeometryService";
    readonly HS2_RenderDocScope.Request request;
    readonly Assembly assembly;
    readonly MethodInfo export;
    readonly ConstructorInfo constructor;
    object service;
    Dictionary<string, object> before, after;
    public bool Complete { get { return before != null && after != null; } }

    internal static string HashFile(string path)
    {
        using (var hash = SHA256.Create()) using (var input = File.OpenRead(path))
            return BitConverter.ToString(hash.ComputeHash(input)).Replace("-", "").ToLowerInvariant();
    }
    static bool AnyOptions(HS2_RenderDocScope.Request r)
    {
        return r.GeometryBeforePath != null || r.GeometryAfterPath != null || r.ExpectedBridgeSha256 != null;
    }
    static void Fresh(string path)
    {
        if (String.IsNullOrEmpty(path) || !Path.IsPathRooted(path) || !String.Equals(Path.GetExtension(path), ".json", StringComparison.OrdinalIgnoreCase)
            || !Directory.Exists(Path.GetDirectoryName(path)) || File.Exists(path) || Directory.Exists(path))
            throw new InvalidOperationException("Geometry export requires fresh absolute JSON path in existing parent: " + path);
    }
    internal static GeometryCapture Prepare(HS2_RenderDocScope.Request request)
    {
        if (!AnyOptions(request)) return null;
        Fresh(request.GeometryBeforePath); Fresh(request.GeometryAfterPath);
        string first = Path.GetFullPath(request.GeometryBeforePath), second = Path.GetFullPath(request.GeometryAfterPath);
        if (String.Equals(first, second, StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("Geometry before/after paths must be distinct");
        foreach (string path in new[] { request.ExactPngPath, request.ReceiptPath, request.CapturePrefix })
            if (!String.IsNullOrEmpty(path) && (String.Equals(first, Path.GetFullPath(path), StringComparison.OrdinalIgnoreCase)
                || String.Equals(second, Path.GetFullPath(path), StringComparison.OrdinalIgnoreCase)))
                throw new InvalidOperationException("Geometry paths must not alias PNG, receipt or capture prefix");
        if (request.ExpectedBridgeSha256 == null || request.ExpectedBridgeSha256.Length != 64)
            throw new InvalidOperationException("Explicit loaded bridge SHA256 required for geometry export");
        var loaded = AppDomain.CurrentDomain.GetAssemblies().Where(a => a.GetName().Name == "HS2_McpBridge").ToArray();
        if (loaded.Length != 1) throw new InvalidOperationException("Exactly one already-loaded HS2_McpBridge assembly required; no load attempted");
        var actual = loaded[0];
        if (String.IsNullOrEmpty(actual.Location) || !String.Equals(HashFile(actual.Location), request.ExpectedBridgeSha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Loaded bridge SHA256 mismatch");
        var type = actual.GetType(TypeName, true, false);
        if (type.Attributes != (TypeAttributes.Sealed | TypeAttributes.BeforeFieldInit))
            throw new InvalidOperationException("Installed internal sealed geometry type metadata differs");
        var ctors = type.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (ctors.Length != 1 || ctors[0].GetParameters().Length != 0 || ctors[0].Attributes !=
            (MethodAttributes.Public | MethodAttributes.HideBySig | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName))
            throw new InvalidOperationException("Installed internal service constructor metadata differs");
        var body = ctors[0].GetMethodBody(); var il = body == null ? null : body.GetILAsByteArray();
        if (il == null || il.Length != 7 || il[0] != 0x02 || il[1] != 0x28 || il[6] != 0x2a || body.ExceptionHandlingClauses.Count != 0
            || ctors[0].Module.ResolveMethod(BitConverter.ToInt32(il, 2)).DeclaringType != typeof(object)
            || ctors[0].Module.ResolveMethod(BitConverter.ToInt32(il, 2)).Name != ".ctor")
            throw new InvalidOperationException("Installed constructor is not the audited object-only initializer");
        var method = type.GetMethod("ExportGeometry", BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly, null, ParameterTypes, null);
        if (method == null || method.ReturnType != typeof(Dictionary<string, object>) || method.IsGenericMethod
            || method.Attributes != (MethodAttributes.Public | MethodAttributes.HideBySig))
            throw new InvalidOperationException("Installed ExportGeometry method signature differs");
        var parameters = method.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
            if (parameters[i].Name != ParameterNames[i] || parameters[i].ParameterType != ParameterTypes[i])
                throw new InvalidOperationException("Installed ordered ExportGeometry parameters differ");
        return new GeometryCapture(request, actual, method, ctors[0]);
    }
    GeometryCapture(HS2_RenderDocScope.Request r, Assembly a, MethodInfo m, ConstructorInfo c)
    {
        request = r; assembly = a; export = m; constructor = c;
    }
    public Dictionary<string, object> Describe()
    {
        return new Dictionary<string, object> {
            { "required", true }, { "complete", Complete }, { "before", before }, { "after", after },
            { "bridge_path", assembly.Location }, { "bridge_sha256", request.ExpectedBridgeSha256 },
            { "bridge_assembly", assembly.FullName }, { "bridge_mvid", assembly.ManifestModule.ModuleVersionId.ToString() },
            { "export_method", TypeName + ".ExportGeometry(System.String,System.String[],System.Boolean,System.String,System.Boolean)->System.Collections.Generic.Dictionary<System.String,System.Object>" },
            { "export_attributes", export.Attributes.ToString() }, { "constructor_attributes", constructor.Attributes.ToString() },
            { "arguments_policy", "(absoluteJsonPath,null,true,null,true); installed native serializer unchanged" }
        };
    }
    Dictionary<string, object> Export(string path)
    {
        Fresh(path);
        // Recheck file provenance at both boundaries; hot replacement is rejected.
        if (!String.Equals(HashFile(assembly.Location), request.ExpectedBridgeSha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Pinned loaded bridge file changed during scope");
        if (service == null) service = constructor.Invoke(new object[0]);
        var invokedUtc = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture);
        Dictionary<string, object> native;
        try { native = (Dictionary<string, object>)export.Invoke(service, new object[] { path, null, true, null, true }); }
        catch (TargetInvocationException error) { throw new InvalidOperationException("Installed native geometry export failed", error.InnerException ?? error); }
        if (native == null || !File.Exists(path)) throw new InvalidOperationException("Native exporter did not save geometry");
        var bytes = new FileInfo(path).Length; var sha = HashFile(path);
        object value;
        if (bytes == 0 || !native.TryGetValue("sha256", out value) || !String.Equals(sha, Convert.ToString(value), StringComparison.OrdinalIgnoreCase)
            || !native.ContainsKey("bytes") || Convert.ToInt64(native["bytes"]) != bytes
            || !native.ContainsKey("path") || !String.Equals(Path.GetFullPath(Convert.ToString(native["path"])), Path.GetFullPath(path), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Native geometry file/receipt provenance mismatch");
        var metadata = new Dictionary<string, object>();
        // Read root scalar metadata without materializing full mesh/blendframe arrays.
        using (var text = File.OpenText(path)) using (var reader = new JsonTextReader(text)) {
            reader.DateParseHandling = DateParseHandling.None;
            while (reader.Read()) {
                if (reader.TokenType != JsonToken.PropertyName || reader.Depth != 1) continue;
                string key = Convert.ToString(reader.Value);
                if (key != "frame_count" && key != "frame_count_end" && key != "captured_utc" && key != "snapshot_kind"
                    && key != "include_actor_transforms" && key != "include_blendshape_frames") continue;
                if (!reader.Read() || (reader.TokenType != JsonToken.Integer && reader.TokenType != JsonToken.String && reader.TokenType != JsonToken.Boolean))
                    throw new InvalidOperationException("Native root geometry metadata type invalid: " + key);
                metadata.Add(key, reader.Value);
            }
        }
        if (!metadata.ContainsKey("frame_count") || !metadata.ContainsKey("frame_count_end") || !metadata.ContainsKey("captured_utc")
            || !metadata.ContainsKey("snapshot_kind") || Convert.ToString(metadata["snapshot_kind"]) != "maker_live_skinned_geometry"
            || !metadata.ContainsKey("include_actor_transforms") || !Equals(metadata["include_actor_transforms"], true)
            || !metadata.ContainsKey("include_blendshape_frames") || !Equals(metadata["include_blendshape_frames"], true))
            throw new InvalidOperationException("Native geometry snapshot missing required full actor/blendframe metadata");
        if (!native.ContainsKey("frame_count") || Convert.ToInt64(native["frame_count"]) != Convert.ToInt64(metadata["frame_count"])
            || !native.ContainsKey("captured_utc") || Convert.ToString(native["captured_utc"]) != Convert.ToString(metadata["captured_utc"]))
            throw new InvalidOperationException("Native geometry snapshot/receipt frame timestamp mismatch");
        metadata.Add("path", path); metadata.Add("bytes", bytes); metadata.Add("sha256", sha);
        metadata.Add("invoked_utc", invokedUtc); metadata.Add("returned_utc", DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture));
        metadata.Add("native_export_receipt", native); return metadata;
    }
    public void Before() { if (before != null) throw new InvalidOperationException("Before geometry already exported"); before = Export(request.GeometryBeforePath); }
    public void After() { if (before == null || after != null) throw new InvalidOperationException("Geometry boundary order invalid"); after = Export(request.GeometryAfterPath); }
}
