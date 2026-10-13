// Included once by renderdoc_scope.cpp; no graphics creation or API interception.
// TriggerCapture is recorded at official WrappedDevice::Present, not mid-frame.
static_assert(offsetof(RENDERDOC_API_1_7_0,TriggerCapture)==15*sizeof(void*),"Official SDK order");
static_assert(offsetof(RENDERDOC_API_1_7_0,SetCommandAnnotation)==28*sizeof(void*),"Official SDK order");
enum class PresentPhase { None, Queued, Ready, Entered, Bound, AwaitPresent };
static PresentPhase presentPhase=PresentPhase::None;
static bool markerBegin=false,markerTarget=false,markerEnd=false;
static bool presentInvalid=false;
static uint32_t markerThreads[3]={};
static ID3D11Resource* presentTarget=nullptr;
static std::string presentNonce;
static int presentEvents[3]={};

static const char* presentPhaseName(){switch(presentPhase){
  case PresentPhase::Queued:return "queued";case PresentPhase::Ready:return "ready";
  case PresentPhase::Entered:return "original_entered";case PresentPhase::Bound:return "target_bound";
  case PresentPhase::AwaitPresent:return "await_present";default:return "none";}}
static void presentMessage(const char* op,int code,const std::string& detail){
  message(op,code,detail);receipt.pop_back();
  receipt+=",\"mode\":\"official_present\",\"phase\":"+jsonQuote(presentPhaseName())+
    ",\"nonce\":"+jsonQuote(presentNonce)+",\"capture_accepted\":false,"+
    "\"actual_device_pointer\":"+jsonQuote(std::to_string(reinterpret_cast<uintptr_t>(heldDevice)))+
    ",\"actual_target_pointer\":"+jsonQuote(std::to_string(reinterpret_cast<uintptr_t>(presentTarget)))+
    ",\"marker_callbacks\":["+(markerBegin?std::string("true"):"false")+","+(markerTarget?"true":"false")+","+(markerEnd?"true":"false")+
    "],\"marker_thread_ids\":["+std::to_string(markerThreads[0])+","+std::to_string(markerThreads[1])+","+std::to_string(markerThreads[2])+"],"+
    "\"rdc_proof_required\":\"matching begin/target/end annotation chunks and real target readback; ordinary retry file is insufficient\"}";
}
static void releasePresent(){
  if(presentTarget){presentTarget->Release();presentTarget=nullptr;}
  presentPhase=PresentPhase::None;releaseScope();
}
static bool extendedApi(){
  if(!resolveApi())return false;
#ifndef HS2_RDOC_SCOPE_TEST
  auto get=(pRENDERDOC_GetAPI)GetProcAddress(GetModuleHandleW(L"renderdoc.dll"),"RENDERDOC_GetAPI");
  if(!get || !get(eRENDERDOC_API_Version_1_7_0,(void**)&api))return false;
#endif
  return api && api->TriggerCapture && api->SetCommandAnnotation;
}
static int presentOwned(uint32_t pid,uint64_t supplied){
  if(pid!=GetCurrentProcessId() || !owned || supplied!=token || presentPhase==PresentPhase::None){
    presentMessage("present_guard",-20,"No matching Present reservation; other captures untouched");return -20;}
  return 0;
}
static int reservePresent(uint32_t pid,const char* requested,ID3D11Device* actual,uint64_t* output){
  if(!output)throw std::runtime_error("Output token required");*output=0;
  if(pid!=GetCurrentProcessId() || owned)throw std::runtime_error("Expected PID/exclusive owner mismatch");
  if(!extendedApi()){presentMessage("present_queue",-2,"Pinned official v1.7 annotation/trigger API unavailable");return -2;}
  if(api->IsFrameCapturing()){presentMessage("present_queue",-3,"Existing capture not adopted");return -3;}
  if(!requested || !*requested || !std::filesystem::u8path(requested).is_absolute())throw std::runtime_error("Fresh absolute prefix required");
  auto guard=std::make_unique<CaptureGuard>(requested);
  if(serial>=0x1ffffffe)throw std::runtime_error("Render event token range exhausted");
  const char* old=api->GetCaptureFilePathTemplate();if(!old)throw std::runtime_error("Capture template unavailable");
  previousPath=old;prefix=requested;beginCount=api->GetNumCaptures();beginTime=epoch();
  api->SetCaptureFilePathTemplate(prefix.c_str());templateChanged=true;
  const char* changed=api->GetCaptureFilePathTemplate();if(!changed){releaseScope();throw std::runtime_error("Capture template unavailable after set");}
  installedPath=changed;fileGuard=std::move(guard);heldDevice=actual;if(actual)actual->AddRef();device=actual;window=nullptr;
  token=++serial;owned=true;presentReserved=true;presentPhase=PresentPhase::Queued;presentNonce=prefix+"#"+std::to_string(token);
  markerBegin=markerTarget=markerEnd=false;presentInvalid=false;std::memset(markerThreads,0,sizeof(markerThreads));
  for(int i=0;i<3;i++)presentEvents[i]=int(token*4+i);
  *output=token;
  // The public API has no cancel-pending-trigger operation. Retain reservation
  // ownership on cancellation until an actual active capture can be discarded.
  api->TriggerCapture();presentMessage("present_queue",4,"Reserved one trigger; waiting for official Present start; no SDK Start called");return 4;
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_QueuePresentFromNativeTexture(uint32_t pid,const char* prefix,void* resource,uint64_t* output){
  std::lock_guard<std::mutex> lock(gate);ID3D11Device* actual=nullptr;
  try{if(!output)throw std::runtime_error("Output token required");*output=0;
    if(pid!=GetCurrentProcessId() || owned)throw std::runtime_error("Expected PID/exclusive owner mismatch");
    if(!extendedApi()){presentMessage("present_queue",-2,"Official v1.7 API unavailable");return -2;}
    if(api->IsFrameCapturing()){presentMessage("present_queue",-3,"Existing capture not adopted");return -3;}
    if(!resource)throw std::runtime_error("Existing real Unity texture resource required");
    static_cast<ID3D11Resource*>(resource)->GetDevice(&actual);
    if(!actual)throw std::runtime_error("Actual resource device absent");
    int result=reservePresent(pid,prefix,actual,output);actual->Release();return result;
  }catch(const std::exception& e){if(actual)actual->Release();presentMessage("present_queue",-1,e.what());return -1;}
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_PollPresent(uint32_t pid,uint64_t supplied){
  std::lock_guard<std::mutex> lock(gate);try{
    if(auto bad=presentOwned(pid,supplied))return bad;
    auto count=api->GetNumCaptures();bool active=api->IsFrameCapturing()!=0;
    if(presentInvalid){presentMessage("present_poll",-28,"Render annotation/protocol failure retained; must abort own reservation");return -28;}
    if(count==beginCount){
      if(presentPhase==PresentPhase::Queued && active)presentPhase=PresentPhase::Ready;
      if(presentPhase==PresentPhase::Ready && !active){presentMessage("present_poll",-22,"Ready capture ended before original invocation");return -22;}
      int code=presentPhase==PresentPhase::Ready?5:4;
      presentMessage("present_poll",code,active?"Official capture active; original may execute once only when ready":"No saved entry yet; GPU Present and retry timing not inferred from Unity frame count");return code;
    }
    if(count!=beginCount+1 || active || presentPhase!=PresentPhase::AwaitPresent || !markerBegin || !markerTarget || !markerEnd){
      if(!active)owned=false;
      presentMessage("present_poll",-23,"Saved capture did not match one complete marked original call; retained for rejection");
      if(!active)releasePresent();return -23;
    }
    uint32_t n=0;uint64_t stamp=0;if(!api->GetCapture(beginCount,nullptr,&n,&stamp) || !n || n>1<<20)throw std::runtime_error("Saved metadata absent");
    std::vector<char> path(n+1,0);if(!api->GetCapture(beginCount,path.data(),&n,&stamp))throw std::runtime_error("Saved metadata changed");
    auto size=std::filesystem::file_size(std::filesystem::u8path(path.data()));std::string reason;
    if(!fileGuard->accepts_file(path.data(),stamp,size,beginTime,reason))throw std::runtime_error(reason);
    auto hash=hashFile(std::filesystem::u8path(path.data()));
    owned=false;presentMessage("present_file_saved",2,"Fresh file saved; annotation/readback/frame/source matching remains required before acceptance");
    receipt.pop_back();receipt+=",\"path\":"+jsonQuote(path.data())+",\"bytes\":"+std::to_string(size)+",\"sha256\":"+jsonQuote(hash)+
      ",\"timestamp\":"+std::to_string(stamp)+",\"capture_count_before\":"+std::to_string(beginCount)+",\"capture_count_after\":"+std::to_string(count)+"}";
    releasePresent();return 2;
  }catch(const std::exception& e){presentMessage("present_poll",-1,e.what());return -1;}
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_EnterOriginal(uint32_t pid,uint64_t supplied){
  std::lock_guard<std::mutex> lock(gate);
  if(auto bad=presentOwned(pid,supplied))return bad;
  if(presentPhase!=PresentPhase::Ready || !api->IsFrameCapturing() || api->GetNumCaptures()!=beginCount){presentMessage("original_enter",-24,"Original entry requires active ready reservation exactly once");return -24;}
  presentPhase=PresentPhase::Entered;presentMessage("original_enter",6,"Original full Capture entered; enqueue begin event before any original GPU work");return 6;
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_BindPresentTarget(uint32_t pid,uint64_t supplied,void* resource){
  std::lock_guard<std::mutex> lock(gate);ID3D11Device* actual=nullptr;IUnknown *a=nullptr,*b=nullptr;
  try{if(auto bad=presentOwned(pid,supplied))return bad;
    if(presentPhase!=PresentPhase::Entered || !api->IsFrameCapturing() || api->GetNumCaptures()!=beginCount || !resource)throw std::runtime_error("Target bind requires still-active entered original");
    static_cast<ID3D11Resource*>(resource)->GetDevice(&actual);
    if(!actual || !heldDevice || FAILED(actual->QueryInterface(__uuidof(IUnknown),(void**)&a)) || FAILED(heldDevice->QueryInterface(__uuidof(IUnknown),(void**)&b)) || a!=b)throw std::runtime_error("Actual original RT device identity differs from reserved Unity resource");
    a->Release();a=nullptr;b->Release();b=nullptr;actual->Release();actual=nullptr;
    presentTarget=static_cast<ID3D11Resource*>(resource);presentTarget->AddRef();presentPhase=PresentPhase::Bound;
    presentMessage("original_target",7,"Actual original camera RT retained; enqueue target APIObject annotation before sixteen renders");return 7;
  }catch(const std::exception& e){if(a)a->Release();if(b)b->Release();if(actual)actual->Release();presentMessage("original_target",-1,e.what());return -1;}
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_CompleteOriginal(uint32_t pid,uint64_t supplied){
  std::lock_guard<std::mutex> lock(gate);if(auto bad=presentOwned(pid,supplied))return bad;
  if(presentPhase!=PresentPhase::Bound || !api->IsFrameCapturing() || api->GetNumCaptures()!=beginCount){presentMessage("original_complete",-25,"Original exited without still-active target-bound reservation");return -25;}
  presentPhase=PresentPhase::AwaitPresent;presentMessage("original_complete",8,"PNG and paired geometry caller complete; enqueue end annotation, then let normal Present end capture");return 8;
}
// Actual UnityRenderingEvent ABI (2018.4 GL.IssuePluginEvent), x64 StdCall.
// Called only by the engine on the render thread, never by a managed delegate.
static void __stdcall presentRenderEvent(int eventID){
  std::lock_guard<std::mutex> lock(gate);
  try{
    if(!owned || presentPhase==PresentPhase::None)return;
    int kind=-1;for(int i=0;i<3;i++)if(eventID==presentEvents[i])kind=i;
    if(kind<0)return; // delayed events from a different reservation cannot mutate this one
    bool* flag=kind==0?&markerBegin:(kind==1?&markerTarget:&markerEnd);
    if(*flag || !api->IsFrameCapturing() || api->GetNumCaptures()!=beginCount ||
       (kind==1 && (!markerBegin || !presentTarget)) || (kind==2 && (!markerBegin || !markerTarget || presentPhase!=PresentPhase::AwaitPresent))){
      presentInvalid=true;presentMessage("render_annotation",-26,"Stale, repeated, reordered or inactive render event rejected");return;}
    RENDERDOC_AnnotationValue value={};value.string=presentNonce.c_str();
    const char* key=kind==0?"hs2_scope.begin":(kind==1?"hs2_scope.target":"hs2_scope.end");
    auto type=eRENDERDOC_String;if(kind==1){type=eRENDERDOC_APIObject;value.apiObject=presentTarget;}
    if(api->SetCommandAnnotation(heldDevice,nullptr,key,type,1,&value)!=0)throw std::runtime_error("Official streamed command annotation rejected");
    *flag=true;markerThreads[kind]=GetCurrentThreadId();
    presentMessage("render_annotation",9,"Actual render-thread command annotation streamed; RDC occurrence must still be verified");
  }catch(const std::exception& e){presentInvalid=true;presentMessage("render_annotation",-1,e.what());}
}
extern "C" __declspec(dllexport) void* __cdecl HS2_RDOC_GetPresentRenderEvent(){return reinterpret_cast<void*>(&presentRenderEvent);}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_GetPresentEventId(uint32_t pid,uint64_t supplied,int kind){
  std::lock_guard<std::mutex> lock(gate);if(presentOwned(pid,supplied) || kind<0 || kind>2)return -1;return presentEvents[kind];
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_AbortPresent(uint32_t pid,uint64_t supplied){
  std::lock_guard<std::mutex> lock(gate);if(auto bad=presentOwned(pid,supplied))return bad;
  if(api->GetNumCaptures()!=beginCount){bool running=api->IsFrameCapturing()!=0;if(!running)owned=false;presentMessage("present_abort",-27,"Capture already saved/changed; no other active scope discarded");if(!running)releasePresent();return -27;}
  if(!api->IsFrameCapturing()){
    // In pinned core End decrements the global active counter even when the
    // D3D11 ordinary driver remains active for retry. False therefore does NOT
    // prove completion in any phase (including Entered/Bound/AwaitPresent).
    presentMessage("present_abort",-21,"SDK active counter false with unchanged file count cannot exclude retry/write; reservation and prefix retained until authoritative completion or owned process shutdown");return -21;
  }
  if(!api->DiscardFrameCapture(device,nullptr) || api->IsFrameCapturing()){presentMessage("present_abort",-8,"Owned active discard failed; reservation retained");return -8;}
  owned=false;presentMessage("present_abort",3,"Discarded only reserved active capture; no forced SDK End");releasePresent();return 3;
}
