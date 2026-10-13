// CPU COM-identity/table fixture only. No RenderDoc DLL or graphics device.
#define HS2_RDOC_SCOPE_TEST
#include "renderdoc_scope.cpp"
#include <iostream>
struct CpuIdentity : IUnknown {
  ULONG refs=1;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id,void** out) override {if(!out)return E_POINTER;*out=nullptr;if(id!=__uuidof(IUnknown))return E_NOINTERFACE;*out=this;AddRef();return S_OK;}
  ULONG STDMETHODCALLTYPE AddRef() override{return ++refs;}
  ULONG STDMETHODCALLTYPE Release() override{return --refs;}
};
struct CpuResource : ID3D11Resource {
  CpuIdentity* identity;ULONG refs=1;
  explicit CpuResource(CpuIdentity* value):identity(value){}
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id,void** out) override {if(!out)return E_POINTER;*out=nullptr;if(id!=__uuidof(IUnknown))return E_NOINTERFACE;*out=this;AddRef();return S_OK;}
  ULONG STDMETHODCALLTYPE AddRef() override{return ++refs;}
  ULONG STDMETHODCALLTYPE Release() override{return --refs;}
  void STDMETHODCALLTYPE GetDevice(ID3D11Device** out) override {*out=reinterpret_cast<ID3D11Device*>(identity);identity->AddRef();}
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID,UINT*,void*) override{return E_NOTIMPL;}
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID,UINT,const void*) override{return E_NOTIMPL;}
  HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID,const IUnknown*) override{return E_NOTIMPL;}
  void STDMETHODCALLTYPE GetType(D3D11_RESOURCE_DIMENSION* out) override{*out=D3D11_RESOURCE_DIMENSION_TEXTURE2D;}
  void STDMETHODCALLTYPE SetEvictionPriority(UINT) override{}
  UINT STDMETHODCALLTYPE GetEvictionPriority() override{return 0;}
};
static bool running=false;
static int queued=0,legacyStarts=0,legacyEnds=0,discarded=0,annotations=0;
static std::string captureTemplate;
static std::vector<std::string> files;
static std::vector<uint64_t> stamps;
static void RENDERDOC_CC version(int* a,int* b,int* c){*a=1;*b=7;*c=0;}
static void RENDERDOC_CC setPath(const char* p){captureTemplate=p;}
static const char* RENDERDOC_CC getPath(){return captureTemplate.c_str();}
static uint32_t RENDERDOC_CC count(){return (uint32_t)files.size();}
static uint32_t RENDERDOC_CC getCapture(uint32_t i,char* out,uint32_t* len,uint64_t* stamp){if(i>=files.size())return 0;if(len)*len=(uint32_t)files[i].size()+1;if(stamp)*stamp=stamps[i];if(out)memcpy(out,files[i].c_str(),files[i].size()+1);return 1;}
static void RENDERDOC_CC start(RENDERDOC_DevicePointer,RENDERDOC_WindowHandle){legacyStarts++;running=true;}
static uint32_t RENDERDOC_CC active(){return running?1:0;}
static uint32_t RENDERDOC_CC end(RENDERDOC_DevicePointer,RENDERDOC_WindowHandle){legacyEnds++;return 0;}
static uint32_t RENDERDOC_CC discard(RENDERDOC_DevicePointer,RENDERDOC_WindowHandle){discarded++;running=false;return 1;}
static void RENDERDOC_CC trigger(){queued++;}
static uint32_t RENDERDOC_CC annotation(RENDERDOC_DevicePointer dev,void* queue,const char* key,RENDERDOC_AnnotationType type,uint32_t width,const RENDERDOC_AnnotationValue* val){
  if(dev!=heldDevice || queue || width!=1 || !val)return 1;
  if(std::string(key)=="hs2_scope.target"){if(type!=eRENDERDOC_APIObject || val->apiObject!=presentTarget)return 1;}
  else if(type!=eRENDERDOC_String || presentNonce!=val->string)return 1;
  annotations++;return 0;
}
static void fakePresentSave(){running=false;auto path=captureTemplate+"_frame1.rdc";std::ofstream(path)<<"CPU synthetic file is not a valid RDC";files.push_back(path);stamps.push_back(epoch());}
int main(int argc,char** argv){try{
  if(argc!=2)throw std::runtime_error("Fresh CPU output required");auto root=std::filesystem::absolute(argv[1]);if(std::filesystem::exists(root))throw std::runtime_error("Fresh CPU output required");std::filesystem::create_directories(root);
  int checks=0;auto check=[&](bool value){checks++;if(!value)throw std::runtime_error("CPU Present check "+std::to_string(checks)+" failed");};auto path=[&](const char* n){return (root/n).u8string();};auto pid=GetCurrentProcessId();uint64_t ticket=0;
  RENDERDOC_API_1_7_0 table={};table.GetAPIVersion=version;table.SetCaptureFilePathTemplate=setPath;table.GetCaptureFilePathTemplate=getPath;table.GetNumCaptures=count;table.GetCapture=getCapture;table.StartFrameCapture=start;table.IsFrameCapturing=active;table.EndFrameCapture=end;table.DiscardFrameCapture=discard;table.TriggerCapture=trigger;table.SetCommandAnnotation=annotation;HS2_RDOC_TestApi(&table);
  CpuIdentity deviceA,deviceB;CpuResource source(&deviceA),target(&deviceA),wrongTarget(&deviceB);
  running=true;check(HS2_RDOC_QueuePresentFromNativeTexture(pid,path("external").c_str(),(void*)1,&ticket)==-3 && ticket==0 && queued==0);running=false;
  check(HS2_RDOC_QueuePresentFromNativeTexture(pid,path("good").c_str(),&source,&ticket)==4 && ticket && owned && queued==1 && deviceA.refs==2);
  check(HS2_RDOC_PollPresent(pid,ticket)==4 && !running);
  check(HS2_RDOC_AbortPresent(pid,ticket)==-21 && owned && discarded==0);
  check(HS2_RDOC_End(pid,ticket)==-29 && legacyEnds==0);
  running=true;check(HS2_RDOC_PollPresent(pid,ticket)==5);
  check(HS2_RDOC_EnterOriginal(pid,ticket)==6);
  check(HS2_RDOC_EnterOriginal(pid,ticket)==-24);
  presentRenderEvent(HS2_RDOC_GetPresentEventId(pid,ticket,0));check(markerBegin && annotations==1);
  check(HS2_RDOC_BindPresentTarget(pid,ticket,&wrongTarget)==-1 && wrongTarget.refs==1 && deviceB.refs==1);
  check(HS2_RDOC_BindPresentTarget(pid,ticket,&target)==7 && target.refs==2 && deviceA.refs==2);
  presentRenderEvent(HS2_RDOC_GetPresentEventId(pid,ticket,1));check(markerTarget && annotations==2);
  check(HS2_RDOC_CompleteOriginal(pid,ticket)==8);
  presentRenderEvent(HS2_RDOC_GetPresentEventId(pid,ticket,2));check(markerEnd && annotations==3);
  check(HS2_RDOC_PollPresent(pid,ticket)==4 && running); // no immediate SDK End
  fakePresentSave();check(HS2_RDOC_PollPresent(pid,ticket)==2 && !owned && deviceA.refs==1 && target.refs==1);
  check(receipt.find("\"capture_accepted\":false")!=std::string::npos && receipt.find("\"owned\":false")!=std::string::npos && receipt.find("\"sha256\":")!=std::string::npos);
  check(legacyStarts==0 && legacyEnds==0);
  check(HS2_RDOC_QueuePresentFromNativeTexture(pid,path("early").c_str(),&source,&ticket)==4);fakePresentSave();check(HS2_RDOC_PollPresent(pid,ticket)==-23 && !owned);
  check(HS2_RDOC_QueuePresentFromNativeTexture(pid,path("badorder").c_str(),&source,&ticket)==4);running=true;check(HS2_RDOC_PollPresent(pid,ticket)==5);check(HS2_RDOC_EnterOriginal(pid,ticket)==6);check(HS2_RDOC_BindPresentTarget(pid,ticket,&target)==7);
  presentRenderEvent(HS2_RDOC_GetPresentEventId(pid,ticket,1));check(presentInvalid && HS2_RDOC_PollPresent(pid,ticket)==-28);
  check(HS2_RDOC_AbortPresent(pid,ticket)==3 && !owned && discarded==1 && deviceA.refs==1 && target.refs==1);
  check(HS2_RDOC_AbortPresent(pid,ticket)==-20 && discarded==1);
  check(HS2_RDOC_QueuePresentFromNativeTexture(pid,path("cancel").c_str(),&source,&ticket)==4);running=true;check(HS2_RDOC_AbortPresent(pid,ticket)==3 && !owned && deviceA.refs==1);
  check(HS2_RDOC_QueuePresentFromNativeTexture(pid,path("inactive_entered_retry").c_str(),&source,&ticket)==4);
  running=true;check(HS2_RDOC_PollPresent(pid,ticket)==5);check(HS2_RDOC_EnterOriginal(pid,ticket)==6);
  running=false;auto oldTemplate=captureTemplate;int oldDiscardCount=discarded;
  check(HS2_RDOC_AbortPresent(pid,ticket)==-21 && owned && captureTemplate==oldTemplate && discarded==oldDiscardCount && deviceA.refs==2);
  // Simulate authoritative eventual file completion, not a GPU/readiness claim.
  fakePresentSave();check(HS2_RDOC_PollPresent(pid,ticket)==-23 && !owned && deviceA.refs==1);
  std::cout<<"CPU Present/table/COM-identity protocol passed "<<checks<<" checks; no Unity callback, GPU, RenderDoc DLL or real RDC execution\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
