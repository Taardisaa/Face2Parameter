using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

// Doorstop entry wrapper; references mscorlib only. No BepInEx/Unity/JSON dependency
// is loaded before its original preloader installs the normal assembly resolver.
public static class HS2_RenderDocBootstrap
{
    const string OriginalSha = "9678c74f47353a11afde4ab05887388bcbb69dffc4e28db93af1d1ce0b6ee635";
    const string RenderDocSha = "809da38e3867d9fd09cc5c30dd5310500dee75e999166d7a14ad1ef6e9eca65d";
    const string OriginalType = "BepInEx.Preloader.Entrypoint";
    static bool entered;
    static IntPtr retainedRenderDoc;
    internal sealed class Request
    {
        internal bool Enabled;
        internal string RenderDocPath, ExpectedRenderDocSha256, ExpectedPreloaderSha256, ExpectedProcessPath, ExpectedBackend, ReceiptPath;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern IntPtr LoadLibraryW(string filename);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode)]
    static extern IntPtr GetModuleHandleW(string name);
    [DllImport("kernel32.dll", CharSet=CharSet.Ansi, ExactSpelling=true)]
    static extern IntPtr GetProcAddress(IntPtr module, string name);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern uint GetModuleFileNameW(IntPtr module, StringBuilder path, uint capacity);
    [DllImport("kernel32.dll")]
    static extern uint GetCurrentProcessId();
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode)]
    static extern IntPtr GetCommandLineW();
    [DllImport("user32.dll", CharSet=CharSet.Unicode)]
    static extern int GetClassNameW(IntPtr window, StringBuilder name, int capacity);
    [DllImport("user32.dll")]
    static extern bool IsWindowVisible(IntPtr window);
    [DllImport("user32.dll")]
    static extern uint GetWindowThreadProcessId(IntPtr window, out uint pid);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    delegate bool EnumCallback(IntPtr window, IntPtr parameter);
    [DllImport("user32.dll")]
    static extern bool EnumWindows(EnumCallback callback, IntPtr parameter);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int GetApi(int version, out IntPtr api);

    internal static string Hash(string path)
    {
        using (var sha=new SHA256Managed()) using(var file=File.OpenRead(path))
            return BitConverter.ToString(sha.ComputeHash(file)).Replace("-","").ToLowerInvariant();
    }
    internal static Request Parse(string path)
    {
        if(!File.Exists(path))return null;
        var fields=new Dictionary<string,string>(StringComparer.Ordinal);
        foreach(string raw in File.ReadAllLines(path)){
            string line=raw.Trim();if(line.Length==0 || line[0]=='#')continue;
            int split=line.IndexOf('=');if(split<1)throw new InvalidOperationException("Bootstrap sentinel must use explicit Key=Value lines");
            string key=line.Substring(0,split).Trim();if(fields.ContainsKey(key))throw new InvalidOperationException("Duplicate bootstrap sentinel key: "+key);
            fields.Add(key,line.Substring(split+1).Trim());
        }
        string enabled;if(!fields.TryGetValue("Enabled",out enabled) || (enabled!="true" && enabled!="false"))throw new InvalidOperationException("Explicit Enabled=true/false required");
        var request=new Request();request.Enabled=enabled=="true";
        if(!request.Enabled)return request;
        string[] keys={"RenderDocPath","ExpectedRenderDocSha256","ExpectedPreloaderSha256","ExpectedProcessPath","ExpectedBackend","ReceiptPath"};
        foreach(string key in keys)if(!fields.ContainsKey(key) || fields[key].Length==0)throw new InvalidOperationException("Missing enabled bootstrap field: "+key);
        foreach(string key in fields.Keys)if(key!="Enabled" && Array.IndexOf(keys,key)<0)throw new InvalidOperationException("Unknown enabled bootstrap field: "+key);
        request.RenderDocPath=fields["RenderDocPath"];request.ExpectedRenderDocSha256=fields["ExpectedRenderDocSha256"];
        request.ExpectedPreloaderSha256=fields["ExpectedPreloaderSha256"];request.ExpectedProcessPath=fields["ExpectedProcessPath"];
        request.ExpectedBackend=fields["ExpectedBackend"];request.ReceiptPath=fields["ReceiptPath"];return request;
    }
    internal static void ValidateFiles(Request request,string original,string actualProcess,string commandLine)
    {
        if(request.ExpectedBackend!="D3D11" || !Flag(commandLine,"-force-d3d11"))throw new InvalidOperationException("Opt-in capture requires explicit D3D11 backend/-force-d3d11; actual driver acceptance remains pending");
        foreach(string flag in new[]{"-force-d3d12","-force-vulkan","-force-glcore","-force-opengl"})
            if(Flag(commandLine,flag))throw new InvalidOperationException("Conflicting graphics backend flag: "+flag);
        if(!Path.IsPathRooted(request.ExpectedProcessPath) || !String.Equals(Path.GetFullPath(actualProcess),Path.GetFullPath(request.ExpectedProcessPath),StringComparison.OrdinalIgnoreCase))throw new InvalidOperationException("Expected native executable path mismatch");
        if(request.ExpectedPreloaderSha256!=OriginalSha || !String.Equals(Hash(original),request.ExpectedPreloaderSha256,StringComparison.OrdinalIgnoreCase))throw new InvalidOperationException("Original installed preloader SHA256 mismatch");
        if(!Path.IsPathRooted(request.RenderDocPath) || !String.Equals(Path.GetFileName(request.RenderDocPath),"renderdoc.dll",StringComparison.OrdinalIgnoreCase)
            || request.ExpectedRenderDocSha256!=RenderDocSha || !String.Equals(Hash(request.RenderDocPath),request.ExpectedRenderDocSha256,StringComparison.OrdinalIgnoreCase))throw new InvalidOperationException("Pinned absolute official RenderDoc SHA256/path mismatch");
        using(var file=File.OpenRead(request.RenderDocPath))using(var reader=new BinaryReader(file)){
            if(reader.ReadUInt16()!=0x5a4d)throw new InvalidOperationException("Official library has no DOS header");file.Position=0x3c;uint pe=reader.ReadUInt32();file.Position=pe;
            if(reader.ReadUInt32()!=0x4550 || reader.ReadUInt16()!=0x8664)throw new InvalidOperationException("Official RenderDoc library must be AMD64 PE");file.Position=pe+24;
            if(reader.ReadUInt16()!=0x20b)throw new InvalidOperationException("Official RenderDoc library must be PE32+");
        }
        if(String.IsNullOrEmpty(request.ReceiptPath) || !Path.IsPathRooted(request.ReceiptPath) || !Directory.Exists(Path.GetDirectoryName(request.ReceiptPath)) || File.Exists(request.ReceiptPath) || Directory.Exists(request.ReceiptPath))throw new InvalidOperationException("Fresh absolute bootstrap receipt in existing parent required");
        foreach(string protectedPath in new[]{original,request.RenderDocPath,actualProcess})if(String.Equals(Path.GetFullPath(request.ReceiptPath),Path.GetFullPath(protectedPath),StringComparison.OrdinalIgnoreCase))throw new InvalidOperationException("Receipt cannot alias a binary");
    }
    internal static bool Flag(string command,string flag)
    {
        int index=0;while((index=command.IndexOf(flag,index,StringComparison.OrdinalIgnoreCase))>=0){int end=index+flag.Length;
            if((index==0 || Char.IsWhiteSpace(command[index-1]) || command[index-1]=='"') && (end==command.Length || Char.IsWhiteSpace(command[end]) || command[end]=='"'))return true;index=end;
        }return false;
    }
    static string NativePath(IntPtr module){var buffer=new StringBuilder(32768);uint n=GetModuleFileNameW(module,buffer,(uint)buffer.Capacity);if(n==0 || n==buffer.Capacity)throw new InvalidOperationException("Native module path unavailable");return buffer.ToString();}
    static void EarlyGuard()
    {
        if(IntPtr.Size!=8)throw new InvalidOperationException("HS2 bootstrap requires native x64 process");
        if(Environment.GetEnvironmentVariable("DOORSTOP_INITIALIZED")!="TRUE")throw new InvalidOperationException("Bootstrap must already be inside Doorstop's initial Mono invocation");
        foreach(var assembly in AppDomain.CurrentDomain.GetAssemblies())
            if(assembly.GetName().Name=="BepInEx" || assembly.GetName().Name=="BepInEx.Preloader")throw new InvalidOperationException("Original BepInEx already loaded; early RenderDoc startup opportunity missed");
        bool windowFound=false;uint own=GetCurrentProcessId();EnumCallback callback=(window,unused)=>{
            uint pid;GetWindowThreadProcessId(window,out pid);if(pid!=own)return true;
            var name=new StringBuilder(256);GetClassNameW(window,name,name.Capacity);
            if(IsWindowVisible(window) || name.ToString().IndexOf("Unity",StringComparison.OrdinalIgnoreCase)>=0)windowFound=true;return true;
        };
        if(!EnumWindows(callback,IntPtr.Zero))throw new InvalidOperationException("Early native window enumeration failed");GC.KeepAlive(callback);
        if(windowFound)throw new InvalidOperationException("Native game window already exists; do not treat a late capture as complete");
        foreach(string module in new[]{"nvwgf2umx.dll","atidxx64.dll","igd10iumd64.dll","igd12umd64.dll","vulkan-1.dll","d3d12.dll"})
            if(GetModuleHandleW(module)!=IntPtr.Zero)throw new InvalidOperationException("Graphics driver/device initialization already observed: "+module);
        if(GetModuleHandleW("renderdoc.dll")!=IntPtr.Zero)throw new InvalidOperationException("RenderDoc already loaded; bootstrap will not adopt an unknown late injection");
    }
    static void LoadOfficial(Request request,List<object> events)
    {
        EarlyGuard();
        events.Add(new Dictionary<string,object>{{"stage","official_library_load_invoking"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"original_preloader_loaded",false},{"path",request.RenderDocPath}});
        retainedRenderDoc=LoadLibraryW(request.RenderDocPath);
        if(retainedRenderDoc==IntPtr.Zero)throw new InvalidOperationException("Pinned official LoadLibraryW failed Win32="+Marshal.GetLastWin32Error());
        string actual=NativePath(retainedRenderDoc);
        if(!String.Equals(Path.GetFullPath(actual),Path.GetFullPath(request.RenderDocPath),StringComparison.OrdinalIgnoreCase) || Hash(actual)!=request.ExpectedRenderDocSha256)throw new InvalidOperationException("Loaded official module does not match explicit path/hash");
        var entry=GetProcAddress(retainedRenderDoc,"RENDERDOC_GetAPI");if(entry==IntPtr.Zero)throw new InvalidOperationException("Official application SDK export absent");
        var get=(GetApi)Marshal.GetDelegateForFunctionPointer(entry,typeof(GetApi));IntPtr api;
        if(get(10600,out api)!=1 || api==IntPtr.Zero)throw new InvalidOperationException("Official v1.6.0 C SDK unavailable");
        events.Add(new Dictionary<string,object>{{"stage","official_sdk_loaded_before_original_preloader"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"path",actual},{"sha256",Hash(actual)},{"sdk_table",api.ToInt64().ToString("x",CultureInfo.InvariantCulture)}});
    }
    internal static void InvokeOriginal(string path,List<object> events)
    {
        if(!String.Equals(Hash(path),OriginalSha,StringComparison.OrdinalIgnoreCase))throw new InvalidOperationException("Pinned original preloader SHA256 mismatch before Assembly.LoadFrom");
        InvokeEntry(path,events);
    }
    // CPU fixture exercises the same invocation mechanics; production can only
    // reach it through InvokeOriginal's unconditional installed-source hash gate.
    internal static void InvokeEntry(string path,List<object> events)
    {
        // Preserve the exact original directory/path input to EnvVars.LoadVars.
        string previous=Environment.GetEnvironmentVariable("DOORSTOP_INVOKE_DLL_PATH");
        Environment.SetEnvironmentVariable("DOORSTOP_INVOKE_DLL_PATH",path);
        events.Add(new Dictionary<string,object>{{"stage","doorstop_original_invocation_path_restored"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"before",previous},{"restored",Environment.GetEnvironmentVariable("DOORSTOP_INVOKE_DLL_PATH")}});
        var assembly=Assembly.LoadFrom(path);var type=assembly.GetType(OriginalType,true,false);
        var main=type.GetMethod("Main",BindingFlags.Public|BindingFlags.Static|BindingFlags.DeclaredOnly,null,Type.EmptyTypes,null);
        if(main==null || main.ReturnType!=typeof(void) || main.IsGenericMethod || main.Attributes!=(MethodAttributes.Public|MethodAttributes.Static|MethodAttributes.HideBySig))throw new InvalidOperationException("Audited original preloader Main signature differs");
        events.Add(new Dictionary<string,object>{{"stage","original_preloader_main_invoking"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"path",path},{"sha256",Hash(path)},{"assembly",assembly.FullName},{"mvid",assembly.ManifestModule.ModuleVersionId.ToString()},{"method",OriginalType+".Main()->System.Void"}});
        try{main.Invoke(null,null);}catch(TargetInvocationException error){throw new InvalidOperationException("Original preloader invocation failed",error.InnerException ?? error);}
        events.Add(new Dictionary<string,object>{{"stage","original_preloader_main_returned"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"doorstop_invocation_path",Environment.GetEnvironmentVariable("DOORSTOP_INVOKE_DLL_PATH")},{"plugins_ready_verified",false}});
    }
    // Pure CPU-testable order/error policy. SDK failure never skips the original
    // full preloader chain and never changes renderer/plugin feature settings.
    internal static bool Sequence(bool enabled,Action optIn,Action original,List<object> events)
    {
        bool loaded=false;
        if(enabled){try{optIn();loaded=true;}catch(Exception error){events.Add(new Dictionary<string,object>{{"stage","opt_in_rejected_original_preloader_continues"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"error",error.ToString()}});}}
        else events.Add(new Dictionary<string,object>{{"stage","ordinary_no_capture_requested"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)}});
        original();return loaded;
    }
    static string Escape(string value){var text=new StringBuilder("\"");foreach(char c in value){if(c=='"'||c=='\\'){text.Append('\\');text.Append(c);}else if(c<32)text.Append("\\u"+((int)c).ToString("x4",CultureInfo.InvariantCulture));else text.Append(c);}return text.Append('"').ToString();}
    static string Json(object value){if(value==null)return "null";if(value is string)return Escape((string)value);if(value is bool)return (bool)value?"true":"false";
        var map=value as Dictionary<string,object>;if(map!=null){var entries=new List<string>();foreach(var pair in map)entries.Add(Escape(pair.Key)+":"+Json(pair.Value));return "{"+String.Join(",",entries.ToArray())+"}";}
        var list=value as List<object>;if(list!=null){var entries=new List<string>();foreach(var item in list)entries.Add(Json(item));return "["+String.Join(",",entries.ToArray())+"]";}
        return Convert.ToString(value,CultureInfo.InvariantCulture);
    }
    static void WriteFresh(string path,string text){using(var file=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.Read))using(var writer=new StreamWriter(file,new UTF8Encoding(false)))writer.Write(text);}
    public static void Main()
    {
        if(entered)throw new InvalidOperationException("Bootstrap/preloader may run only once");entered=true;
        string directory=Path.GetDirectoryName(typeof(HS2_RenderDocBootstrap).Assembly.Location);
        string original=Path.Combine(directory,"BepInEx.Preloader.dll");
        string fallback=Path.Combine(directory,"HS2_RenderDocBootstrap.receipt."+GetCurrentProcessId()+"."+Guid.NewGuid().ToString("N")+".json");string destination=fallback;
        var events=new List<object>();bool loaded=false;Request request=null;Exception originalFailure=null;
        try{
            try{request=Parse(Path.Combine(directory,"HS2_RenderDocBootstrap.request.ini"));}
            catch(Exception error){events.Add(new Dictionary<string,object>{{"stage","request_rejected_original_preloader_continues"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"diagnostic_fallback_receipt",fallback},{"error",error.ToString()}});}
            loaded=Sequence(request!=null && request.Enabled,()=>{
                try{ValidateFiles(request,original,NativePath(IntPtr.Zero),Marshal.PtrToStringUni(GetCommandLineW()));destination=request.ReceiptPath;}
                catch(Exception error){events.Add(new Dictionary<string,object>{{"stage","validation_failed_diagnostic_fallback"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"requested_receipt",request.ReceiptPath},{"diagnostic_fallback_receipt",fallback}});throw new InvalidOperationException("Opt-in validation failed; diagnostic fallback="+fallback,error);}
                LoadOfficial(request,events);
            },()=>InvokeOriginal(original,events),events);
        }catch(Exception error){originalFailure=error;events.Add(new Dictionary<string,object>{{"stage","original_preloader_failure"},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"error",error.ToString()}});}
        finally{
            var result=new Dictionary<string,object>{{"schema_version",1},{"utc",DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture)},{"pid",GetCurrentProcessId()},
                {"sdk_early_load_succeeded",loaded},{"capture_accepted",false},{"native_plugins_verified",false},{"requested_backend",request==null?null:request.ExpectedBackend},
                {"receipt_path",destination},{"diagnostic_fallback_receipt",fallback},{"events",events}};
            try{WriteFresh(destination,Json(result));}catch(Exception error){result["receipt_path"]=fallback;result["receipt_write_error"]=error.ToString();
                try{WriteFresh(fallback,Json(result));}catch(Exception fallbackError){try{WriteFresh(fallback+".error.txt",fallbackError+"\n"+Json(result));}catch{}}}
        }
        if(originalFailure!=null)throw new InvalidOperationException("Original preloader failed; no capture acceptance",originalFailure);
    }
}
