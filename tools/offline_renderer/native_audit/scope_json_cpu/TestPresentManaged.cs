// Source hook sequencing fixture; fake GL only records IDs, never executes pointers.
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
internal static class TestPresentManaged
{
    static string native;
    static int completeCount,abortCount;
    static readonly List<int> queuedEvents=new List<int>();
    static FieldInfo Field(string name){return typeof(HS2_RenderDocScope).GetField(name,BindingFlags.Static|BindingFlags.NonPublic);}
    static void DelegateField(string name,string method){var field=Field(name);field.SetValue(null,Delegate.CreateDelegate(field.FieldType,typeof(TestPresentManaged).GetMethod(method,BindingFlags.Static|BindingFlags.NonPublic)));}
    static uint ReadReceipt(IntPtr buffer,uint capacity){byte[] data=Encoding.UTF8.GetBytes(native+"\0");if(buffer!=IntPtr.Zero && capacity>=data.Length)Marshal.Copy(data,0,buffer,data.Length);return (uint)data.Length;}
    static int Enter(uint pid,ulong token){native="{\"owned\":true,\"code\":6}";return 6;}
    static int Bind(uint pid,ulong token,IntPtr resource){native="{\"owned\":true,\"code\":7}";return resource==new IntPtr(33)?7:-1;}
    static int Complete(uint pid,ulong token){completeCount++;native="{\"owned\":true,\"code\":8}";return 8;}
    static int Abort(uint pid,ulong token){abortCount++;native="{\"owned\":false,\"code\":3}";return 3;}
    static IntPtr Render(){return new IntPtr(999);}
    static int Event(uint pid,ulong token,int kind){return 4+kind;}
    public static void CpuGlOnly(IntPtr pointer,int eventID){if(pointer!=new IntPtr(999))throw new Exception("Synthetic event pointer mismatch");queuedEvents.Add(eventID);}
    internal static void Run(Action<bool> check,string directory)
    {
        var png=Path.Combine(directory,"synthetic.png");File.WriteAllText(png,"CPU path fixture, not a PNG/native acceptance");
        var request=new HS2_RenderDocScope.Request{CaptureMode="official_present",ExactPngPath=png,ReceiptPath=Path.Combine(directory,"present_receipt.json"),ExpectedPid=123};
        Field("active").SetValue(null,request);Field("token").SetValue(null,(ulong)1);Field("geometry").SetValue(null,null);
        Field("presentOriginalComplete").SetValue(null,false);((List<object>)Field("events").GetValue(null)).Clear();
        DelegateField("receipt","ReadReceipt");DelegateField("enterPresent","Enter");DelegateField("bindTarget","Bind");DelegateField("completePresent","Complete");DelegateField("abortPresent","Abort");DelegateField("renderFunction","Render");DelegateField("eventId","Event");
        Field("issuePluginEvent").SetValue(null,typeof(TestPresentManaged).GetMethod("CpuGlOnly"));
        check(HS2_RenderDocScope.EnterOriginalPresentCapture(png)==6 && queuedEvents.Count==1 && queuedEvents[0]==4);
        HS2_RenderDocScope.BeginFromNativeTexture(new IntPtr(33),png);check(queuedEvents.Count==2 && queuedEvents[1]==5);
        HS2_RenderDocScope.EndAfterSavedPng(png);check(queuedEvents.Count==3 && queuedEvents[2]==6 && completeCount==1);
        HS2_RenderDocScope.AbortOwnedSafely();check(abortCount==0 && (ulong)Field("token").GetValue(null)==1);
        HS2_RenderDocScope.BeginFromNativeTexture(new IntPtr(77),png+".ordinary");HS2_RenderDocScope.EndAfterSavedPng(png+".ordinary");HS2_RenderDocScope.AbortOwnedSafely();
        check(queuedEvents.Count==3 && abortCount==0);
        HS2_RenderDocScope.AbortPresentSafely();check(abortCount==1 && (ulong)Field("token").GetValue(null)==0);
        check(ScopeJson.ParseObject(HS2_RenderDocScope.LastReceipt)["events"]!=null);
        var result=new Dictionary<string,object>{{"path",png},{"capture",new Dictionary<string,object>{{"renders",16},{"SSS",true},{"nested",new object[]{1,"unchanged",null}}}}};
        HS2_RenderDocScope.RecordOriginalCallResult(result);
        var saved=(Newtonsoft.Json.Linq.JArray)ScopeJson.ParseObject(HS2_RenderDocScope.LastReceipt)["events"];
        check(saved[saved.Count-1]["stage"].ToString()=="original_call_returned" && saved[saved.Count-1]["original_result"]["capture"]["renders"].ToString()=="16");
        File.Delete(png);File.Delete(request.ReceiptPath);
        Field("active").SetValue(null,null);
        check(HS2_RenderDocScope.EnterOriginalPresentCapture(png)==0 && HS2_RenderDocScope.PollPresentCapture()==0);
    }
}
