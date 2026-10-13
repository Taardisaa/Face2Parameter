using System;
using System.Collections.Generic;
using System.IO;
internal static class BootstrapCpuTest {
    static int checks;
    static void Check(bool value) {checks++;if(!value)throw new Exception("CPU bootstrap check failed "+checks);}
    static void Reject(Action action,string message){try{action();}catch(Exception error){Check(error.Message.Contains(message));return;}throw new Exception("Expected rejection absent");}
    public static int Main(string[] args){try{
        if(args.Length!=2)throw new Exception("Fresh fixture directory and original fixture DLL required");
        var root=Path.GetFullPath(args[0]);if(Directory.Exists(root))throw new Exception("Fresh CPU directory required");Directory.CreateDirectory(root);
        var original=Path.GetFullPath(args[1]);var list=new List<object>();var order=new List<string>();
        Check(!HS2_RenderDocBootstrap.Sequence(false,()=>{throw new Exception("disabled SDK invoked");},()=>order.Add("original"),list));Check(String.Join(",",order.ToArray())=="original");
        order.Clear();Check(HS2_RenderDocBootstrap.Sequence(true,()=>order.Add("sdk"),()=>order.Add("original"),list));Check(String.Join(",",order.ToArray())=="sdk,original");
        order.Clear();Check(!HS2_RenderDocBootstrap.Sequence(true,()=>{order.Add("sdk_failed");throw new Exception("synthetic SDK load failure");},()=>order.Add("original"),list));Check(String.Join(",",order.ToArray())=="sdk_failed,original");
        Check(HS2_RenderDocBootstrap.Parse(Path.Combine(root,"missing.ini"))==null);
        string disabled=Path.Combine(root,"disabled.ini");File.WriteAllText(disabled,"Enabled=false\nRenderDocPath=invalid\n");Check(!HS2_RenderDocBootstrap.Parse(disabled).Enabled);
        string bad=Path.Combine(root,"bad.ini");File.WriteAllText(bad,"Enabled=true\n");Reject(()=>HS2_RenderDocBootstrap.Parse(bad),"Missing enabled");
        File.WriteAllText(bad,"Enabled=true\nEnabled=false\n");Reject(()=>HS2_RenderDocBootstrap.Parse(bad),"Duplicate");
        Check(HS2_RenderDocBootstrap.Flag("\"C:/game path/game.exe\" -force-d3d11","-force-d3d11"));
        Check(!HS2_RenderDocBootstrap.Flag("game.exe -force-d3d11-suffix","-force-d3d11"));
        var request=new HS2_RenderDocBootstrap.Request{Enabled=true,ExpectedBackend="Vulkan"};
        Reject(()=>HS2_RenderDocBootstrap.ValidateFiles(request,original,"game.exe","game.exe -force-vulkan"),"D3D11 backend");
        request.ExpectedBackend="D3D11";Reject(()=>HS2_RenderDocBootstrap.ValidateFiles(request,original,"game.exe","game.exe -force-d3d11 -force-vulkan"),"Conflicting");
        request.ExpectedProcessPath="C:/expected.exe";Reject(()=>HS2_RenderDocBootstrap.ValidateFiles(request,original,"C:/wrong.exe","game.exe -force-d3d11"),"executable path mismatch");
        request.ExpectedProcessPath=original;request.ExpectedPreloaderSha256="wrong";Reject(()=>HS2_RenderDocBootstrap.ValidateFiles(request,original,original,"game.exe -force-d3d11"),"preloader SHA256");
        string log=Path.Combine(root,"original_calls.txt");Environment.SetEnvironmentVariable("HS2_BOOTSTRAP_CPU_LOG",log);
        Reject(()=>HS2_RenderDocBootstrap.InvokeOriginal(original,list),"Pinned original preloader SHA256");
        Environment.SetEnvironmentVariable("DOORSTOP_INVOKE_DLL_PATH","synthetic-wrapper-path");HS2_RenderDocBootstrap.InvokeEntry(original,list);
        Check(File.ReadAllText(log)=="original:"+original+"\n");Check(Environment.GetEnvironmentVariable("DOORSTOP_INVOKE_DLL_PATH")==original);
        Console.WriteLine("CPU bootstrap sequence/entrypoint fixture passed "+checks+" checks; no LoadLibrary, graphics or installed preloader execution");return 0;
    }catch(Exception error){Console.Error.WriteLine(error);return 1;}}
}
