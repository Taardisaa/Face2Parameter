using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Newtonsoft.Json.Linq;
internal static class ScopeJsonCpuTest {
    static int checks;
    static bool forbiddenLoaded;
    static void Check(bool value){checks++;if(!value)throw new Exception("Streaming JSON check failed "+checks);}
    static void Reject(Action action,string expected){try{action();}catch(Exception error){Check(error.ToString().Contains(expected));return;}throw new Exception("Expected JSON rejection absent");}
    public static int Main(){try{
        AppDomain.CurrentDomain.AssemblyLoad+=(sender,args)=>{if(args.LoadedAssembly.GetName().Name=="System.Data")forbiddenLoaded=true;};
        AppDomain.CurrentDomain.AssemblyResolve+=(sender,args)=>{if(args.Name.StartsWith("System.Data,"))throw new Exception("System.Data forbidden in streaming fixture");return null;};
        foreach(var assembly in AppDomain.CurrentDomain.GetAssemblies())Check(assembly.GetName().Name!="System.Data");
        const string input="{\"Enabled\":true,\"ExpectedPid\":4294967295,\"ExactPngPath\":\"C:/程儿/a.png\",\"CapturePrefix\":\"C:/a\",\"ReceiptPath\":\"C:/r.json\",\"ExpectedNativeSha256\":\"native\",\"ExpectedRenderDocSha256\":\"renderdoc\",\"GeometryBeforePath\":\"C:/before.json\",\"GeometryAfterPath\":\"C:/after.json\",\"ExpectedBridgeSha256\":\"bridge\"}";
        var request=ScopeJson.ReadRequest(input);Check(request.ExpectedPid==UInt32.MaxValue && request.Enabled && request.ExactPngPath=="C:/程儿/a.png");
        Check(JToken.DeepEquals(ScopeJson.ParseObject(input),ScopeJson.ParseObject(ScopeJson.Object(ScopeJson.RequestFields(request)))));
        var present=ScopeJson.ReadRequest("{\"CaptureMode\":\"official_present\",\"ExpectedUnityCoreSha256\":\"core\"}");
        Check(present.CaptureMode=="official_present" && present.ExpectedUnityCoreSha256=="core");
        Check(ScopeJson.RequestFields(present)["CaptureMode"].ToString()=="official_present");
        Reject(()=>ScopeJson.ReadRequest("{\"CaptureMode\":\"guessed_frame\"}"),"Unsupported CaptureMode");
        Reject(()=>ScopeJson.ReadRequest("{\"ExpectedUnityCoreSha256\":4}"),"string or null");
        string stamp="2026-10-12T03:04:05.1234567Z";var native=ScopeJson.ParseObject("{\"owned\":false,\"nested\":[1,2.5,null,true,\""+stamp+"\"]}");
        Check(native["nested"][4].Type==JTokenType.String);
        var values=new Dictionary<string,object>{{"native",native},{"unicode","程儿\\\"\n"},{"ints",new object[]{(byte)2,(sbyte)-3,(short)-4,(ushort)5,6,(uint)7,(long)-8,(ulong)9}},{"float",1.25f},{"double",2.5},{"decimal",3.125m},{"char",'字'},{"bool",true},{"null",null}};
        var round=ScopeJson.ParseObject(ScopeJson.Object(values));Check(JToken.DeepEquals(round["native"],native));Check(((JArray)round["ints"]).Count==8);
        Reject(()=>ScopeJson.ReadRequest("{\"Enabled\":\"true\"}"),"boolean");
        Reject(()=>ScopeJson.ReadRequest("{\"ExpectedPid\":-1}"),"uint32");
        Reject(()=>ScopeJson.ReadRequest("{\"ExpectedPid\":1.5}"),"uint32");
        Reject(()=>ScopeJson.ReadRequest("{\"ExpectedPid\":4294967296}"),"uint32");
        Reject(()=>ScopeJson.ReadRequest("{\"ReceiptPath\":[]}"),"string or null");
        Reject(()=>ScopeJson.ReadRequest("{\"unknown\":true}"),"Unsupported scope request field");
        Reject(()=>ScopeJson.ParseObject("{\"owned\":true,\"owned\":false}"),"already exists");
        Reject(()=>ScopeJson.ParseObject("{} {}"),"Additional text");
        Reject(()=>ScopeJson.Object(new Dictionary<string,object>{{"opaque",new object()}}),"Unsupported JSON response value");
        Reject(()=>ScopeJson.Object(new Dictionary<string,object>{{"bad",Double.NaN}}),"finite");
        Reject(()=>ScopeJson.Object(new Dictionary<string,object>{{"bad",new Hashtable{{1,"value"}}}}),"keys must be strings");
        // Force receipt file writing to fail; SafeSave must retain parseable previous
        // full receipt/error context without reflection serialization or System.Data.
        var active=typeof(HS2_RenderDocScope).GetField("active",BindingFlags.Static|BindingFlags.NonPublic);
        string directory=Path.Combine(Path.GetTempPath(),"HS2_streaming_json_cpu_"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(directory);
        string impossible=Path.Combine(directory,"primary_is_directory");Directory.CreateDirectory(impossible);
        active.SetValue(null,new HS2_RenderDocScope.Request{ReceiptPath=impossible});
        var safe=typeof(HS2_RenderDocScope).GetMethod("SafeSave",BindingFlags.Static|BindingFlags.NonPublic);
        safe.Invoke(null,new object[]{"synthetic_write_failure",new InvalidOperationException("original failure retained")});
        var failed=ScopeJson.ParseObject(HS2_RenderDocScope.LastReceipt);
        Check(failed["error"].Value<string>().Contains("original failure retained") && failed["receipt_write_error"]!=null && failed["previous_receipt"]!=null);
        Check(Directory.GetFiles(directory,"*.failure.*.json").Length==1);
        // Force both streaming attempts to reject unsupported/broken data; the
        // emergency envelope must remain parseable and preserve all raw errors.
        var events=(List<object>)typeof(HS2_RenderDocScope).GetField("events",BindingFlags.Static|BindingFlags.NonPublic).GetValue(null);
        events.Add(new object());typeof(HS2_RenderDocScope).GetField("<LastReceipt>k__BackingField",BindingFlags.Static|BindingFlags.NonPublic).SetValue(null,"{} {}");
        safe.Invoke(null,new object[]{"synthetic_double_failure",new InvalidOperationException("both errors retained\n程儿")});
        var emergency=ScopeJson.ParseObject(HS2_RenderDocScope.LastReceipt);
        Check(emergency["previous_receipt_raw"].Value<string>()=="{} {}" && emergency["streaming_fallback_error"]!=null && !emergency["capture_accepted"].Value<bool>());
        foreach(string path in Directory.GetFiles(directory,"*.failure.*.json")){ScopeJson.ParseObject(File.ReadAllText(path));File.Delete(path);}
        TestPresentManaged.Run(Check,directory);
        Directory.Delete(impossible);Directory.Delete(directory);
        foreach(var assembly in AppDomain.CurrentDomain.GetAssemblies())Check(assembly.GetName().Name!="System.Data");Check(!forbiddenLoaded);
        Console.WriteLine("Streaming request/receipt/native-token/error JSON passed "+checks+" checks with no System.Data load; no game/GPU/native capture executed");return 0;
    }catch(Exception error){Console.Error.WriteLine(error);return 1;}}
}
