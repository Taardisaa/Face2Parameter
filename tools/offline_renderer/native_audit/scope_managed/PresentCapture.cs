using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

public static partial class HS2_RenderDocScope
{
    static BeginTexture queuePresent;
    static Finish pollPresent, enterPresent, completePresent, abortPresent;
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate IntPtr RenderFunction();
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int EventId(uint pid, ulong ticket, int kind);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int BindTarget(uint pid, ulong ticket, IntPtr resource);
    static BindTarget bindTarget;
    static RenderFunction renderFunction;
    static EventId eventId;
    static MethodInfo issuePluginEvent;
    static bool presentOriginalComplete;
    static bool presentInsideOriginal;
    static int lastPresentPollCode = Int32.MinValue;

    static void LoadPresentExports()
    {
        LoadOwnHelper();
        queuePresent = Export<BeginTexture>("HS2_RDOC_QueuePresentFromNativeTexture");
        pollPresent = Export<Finish>("HS2_RDOC_PollPresent");
        enterPresent = Export<Finish>("HS2_RDOC_EnterOriginal");
        bindTarget = Export<BindTarget>("HS2_RDOC_BindPresentTarget");
        completePresent = Export<Finish>("HS2_RDOC_CompleteOriginal");
        abortPresent = Export<Finish>("HS2_RDOC_AbortPresent");
        renderFunction = Export<RenderFunction>("HS2_RDOC_GetPresentRenderEvent");
        eventId = Export<EventId>("HS2_RDOC_GetPresentEventId");
        Assembly core = null;
        foreach(var assembly in AppDomain.CurrentDomain.GetAssemblies())
            if(assembly.GetName().Name == "UnityEngine.CoreModule") {
                if(core!=null)throw new InvalidOperationException("Ambiguous Unity Core module");core=assembly;
            }
        if(core==null)throw new InvalidOperationException("Actual Unity Core module must already be loaded");
        VerifyHash(core.Location,active.ExpectedUnityCoreSha256);
        var gl=core.GetType("UnityEngine.GL",true,false);
        issuePluginEvent=gl.GetMethod("IssuePluginEvent",BindingFlags.Public|BindingFlags.Static|BindingFlags.DeclaredOnly,null,new[]{typeof(IntPtr),typeof(int)},null);
        if(issuePluginEvent==null || issuePluginEvent.ReturnType!=typeof(void) || issuePluginEvent.IsGenericMethod)
            throw new InvalidOperationException("Installed exact GL.IssuePluginEvent(IntPtr,int) signature absent");
        Save("present_unity_api",0,null);
    }
    static void EnqueuePresentEvent(int kind)
    {
        int id=eventId(active.ExpectedPid,token,kind);IntPtr callback=renderFunction();
        if(id<0 || callback==IntPtr.Zero)throw new InvalidOperationException("Owned Unity render event ID/function unavailable");
        // Native callback is invoked by the actual engine on its rendering thread.
        // Never call a managed/native fake callback directly in production.
        issuePluginEvent.Invoke(null,new object[]{callback,id});
    }
    static bool ExactPresentPath(string png)
    {
        return active!=null && active.CaptureMode=="official_present" &&
            String.Equals(Path.GetFullPath(png),Path.GetFullPath(active.ExactPngPath),StringComparison.OrdinalIgnoreCase);
    }
    public static int ArmPresentFromNativeTexture(IntPtr existingRealUnityTexture,string exactPngPath)
    {
        lock(Gate){
            if(token!=0){SafeSave("present_arm_busy",new InvalidOperationException("Previous reservation retained"));return -1;}
            active=null;geometry=null;events.Clear();presentOriginalComplete=false;presentInsideOriginal=false;lastPresentPollCode=Int32.MinValue;
            try{
                string path=Path.Combine(DirectoryPath,"HS2_RenderDocScope.request.json");if(!File.Exists(path))return 0;
                var request=ScopeJson.ReadRequest(File.ReadAllText(path));
                if(request==null || !request.Enabled || request.CaptureMode!="official_present")return 0;
                if(String.IsNullOrEmpty(request.ExactPngPath) || !String.Equals(Path.GetFullPath(request.ExactPngPath),Path.GetFullPath(exactPngPath),StringComparison.OrdinalIgnoreCase))return 0;
                Validate(request,exactPngPath);active=request;
                if(File.Exists(request.ReceiptPath) || File.Exists(request.ExactPngPath) || Directory.Exists(request.ReceiptPath) || Directory.Exists(request.ExactPngPath))
                    throw new InvalidOperationException("Present mode requires fresh receipt/native PNG");
                geometry=GeometryCapture.Prepare(request);LoadPresentExports();
                int code=queuePresent(request.ExpectedPid,Encoding.UTF8.GetBytes(request.CapturePrefix+"\0"),existingRealUnityTexture,out token);
                Save("present_arm",code,ReadReceipt());return code;
            }catch(Exception error){SafeSave("present_arm_failure",error);return -100;}
        }
    }
    public static int PollPresentCapture()
    {
        lock(Gate){
            if(active==null || active.CaptureMode!="official_present" || token==0)return 0;
            try{
                int code=pollPresent(active.ExpectedPid,token);string native=ReadReceipt();
                if(NoLongerOwned(native))token=0;
                if(code!=lastPresentPollCode || code<0 || code==2){Save("present_poll",code,native);lastPresentPollCode=code;}
                return code;
            }catch(Exception error){SafeSave("present_poll_failure",error);return -100;}
        }
    }
    public static int EnterOriginalPresentCapture(string exactPngPath)
    {
        lock(Gate){
            if(active==null || active.CaptureMode!="official_present")return 0;
            try{
                if(token==0 || !ExactPresentPath(exactPngPath))throw new InvalidOperationException("Original pathname/reservation mismatch");
                int code=enterPresent(active.ExpectedPid,token);Save("present_original_enter",code,ReadReceipt());
                if(code!=6)return code;
                presentInsideOriginal=true;
                EnqueuePresentEvent(0);Save("present_begin_event_queued",6,ReadReceipt());return 6;
            }catch(Exception error){SafeSave("present_original_enter_failure",error);return -100;}
        }
    }
    static void BindPresentTarget(IntPtr resource,string png)
    {
        try{
            if(token==0 || !ExactPresentPath(png))throw new InvalidOperationException("Actual original target pathname/reservation mismatch");
            int code=bindTarget(active.ExpectedPid,token,resource);Save("present_actual_target",code,ReadReceipt());
            if(code!=7)throw new InvalidOperationException("Native target bind rejected code="+code);
            if(geometry!=null)geometry.Before();
            EnqueuePresentEvent(1);Save("present_target_event_queued",7,ReadReceipt());
        }catch(Exception error){SafeSave("present_target_failure",error);}
    }
    static void CompletePresentOriginal(string png)
    {
        try{
            if(token==0 || !ExactPresentPath(png) || !File.Exists(png) || new FileInfo(png).Length==0)
                throw new InvalidOperationException("Actual saved original PNG/path/reservation missing");
            if(geometry!=null){geometry.After();if(!geometry.Complete)throw new InvalidOperationException("Same-freeze pair incomplete");}
            int code=completePresent(active.ExpectedPid,token);Save("present_original_complete",code,ReadReceipt());
            if(code!=8)throw new InvalidOperationException("Native complete rejected code="+code);
            EnqueuePresentEvent(2);presentOriginalComplete=true;
            Save("present_end_event_queued",8,ReadReceipt());
        }catch(Exception error){presentOriginalComplete=false;SafeSave("present_original_complete_failure",error);}
    }
    public static void AbortPresentSafely()
    {
        lock(Gate){if(active==null || active.CaptureMode!="official_present" || token==0)return;
            try{int code=abortPresent(active.ExpectedPid,token);string native=ReadReceipt();if(NoLongerOwned(native))token=0;
                presentOriginalComplete=false;presentInsideOriginal=false;Save("present_abort",code,native);
            }catch(Exception error){presentOriginalComplete=false;presentInsideOriginal=false;SafeSave("present_abort_failure",error);}}
    }
}
