// Strict C ABI capture scope helper. Never loads/injects RenderDoc or creates a
// graphics device. The caller owns the real native rendering/ReadPixels scope.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <bcrypt.h>
#include "renderdoc_app.h"
#include "../capture_guard.h"
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static_assert(sizeof(void*)==8,"HS2 is x64");
static_assert(offsetof(RENDERDOC_API_1_6_0,GetCapture)==14*sizeof(void*),"Official SDK table order");
static_assert(offsetof(RENDERDOC_API_1_6_0,StartFrameCapture)==19*sizeof(void*),"Official SDK table order");
static_assert(offsetof(RENDERDOC_API_1_6_0,IsFrameCapturing)==20*sizeof(void*),"Official SDK table order");
static_assert(offsetof(RENDERDOC_API_1_6_0,EndFrameCapture)==21*sizeof(void*),"Official SDK table order");
static_assert(offsetof(RENDERDOC_API_1_6_0,DiscardFrameCapture)==24*sizeof(void*),"Official SDK table order");

static std::mutex gate;
static RENDERDOC_API_1_6_0* api=nullptr;
static std::string receipt="{\"status\":\"not_started\"}";
static uint64_t serial=0,token=0,beginTime=0;
static uint32_t beginCount=0;
static bool owned=false;
static bool presentReserved=false;
static void* device=nullptr;static void* window=nullptr;
static ID3D11Device* heldDevice=nullptr;
static std::string previousPath,prefix,modulePath,installedPath;
static bool templateChanged=false;
static std::unique_ptr<CaptureGuard> fileGuard;

static std::string jsonQuote(const std::string& s){
  std::string r="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){r+='\\';r+=c;}else if(c<32){char b[7];sprintf_s(b,"\\u%04x",c);r+=b;}else r+=c;}return r+'"';
}
static uint64_t epoch(){return (uint64_t)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
static std::string hashFile(const std::filesystem::path& path){
  BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
  if(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)throw std::runtime_error("SHA256 provider failed");
  try{
    if(BCryptCreateHash(algorithm,&hash,nullptr,0,nullptr,0,0)<0)throw std::runtime_error("SHA256 hash failed");
    std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("SHA256 input inaccessible");
    std::vector<char> chunk(1<<20);while(input){input.read(chunk.data(),chunk.size());auto n=input.gcount();if(n && BCryptHashData(hash,(PUCHAR)chunk.data(),(ULONG)n,0)<0)throw std::runtime_error("SHA256 update failed");}
    if(input.bad())throw std::runtime_error("SHA256 file read failed");
    unsigned char bytes[32];if(BCryptFinishHash(hash,bytes,32,0)<0)throw std::runtime_error("SHA256 finish failed");
    std::ostringstream out;for(auto b:bytes)out<<std::hex<<std::setfill('0')<<std::setw(2)<<(unsigned)b;
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);return out.str();
  }catch(...){if(hash)BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);throw;}
}
static void message(const char* operation,int code,const std::string& detail){
  receipt="{\"schema_version\":1,\"operation\":"+jsonQuote(operation)+",\"code\":"+std::to_string(code)+",\"pid\":"+std::to_string(GetCurrentProcessId())+",\"owned\":"+(owned?"true":"false")+",\"token\":"+std::to_string(token)+",\"detail\":"+jsonQuote(detail)+"}";
}
static void releaseScope(){
  // Do not overwrite a template another writer changed while this scope ran.
  if(api && templateChanged){
    const char* current=api->GetCaptureFilePathTemplate();
    if(current && installedPath==current)api->SetCaptureFilePathTemplate(previousPath.c_str());
  }
  templateChanged=false;
  if(heldDevice){heldDevice->Release();heldDevice=nullptr;}
  owned=false;presentReserved=false;device=nullptr;window=nullptr;fileGuard.reset();
}

#ifdef HS2_RDOC_SCOPE_TEST
static RENDERDOC_API_1_6_0* testApi=nullptr;
extern "C" __declspec(dllexport) void __cdecl HS2_RDOC_TestApi(RENDERDOC_API_1_6_0* value){testApi=value;}
#endif
static bool resolveApi(){
#ifdef HS2_RDOC_SCOPE_TEST
  api=testApi;modulePath="CPU SDK table mock";
  if(!api)return false;
#else
  HMODULE module=GetModuleHandleW(L"renderdoc.dll");if(!module)return false;
  auto get=(pRENDERDOC_GetAPI)GetProcAddress(module,"RENDERDOC_GetAPI");
  if(!get || !get(eRENDERDOC_API_Version_1_6_0,(void**)&api))return false;
  wchar_t path[32768];auto n=GetModuleFileNameW(module,path,32768);if(!n || n==32768)throw std::runtime_error("Loaded RenderDoc module path unavailable");
  modulePath=std::filesystem::path(path).u8string();
#endif
  if(!api || !api->GetAPIVersion || !api->GetCaptureFilePathTemplate || !api->SetCaptureFilePathTemplate || !api->GetNumCaptures || !api->GetCapture || !api->StartFrameCapture || !api->IsFrameCapturing || !api->EndFrameCapture || !api->DiscardFrameCapture)
    throw std::runtime_error("Official SDK table lacks required capture functions");
  return true;
}
static int begin(uint32_t expectedPid,const char* requestedPrefix,void* suppliedDevice,void* suppliedWindow,ID3D11Device* retained,uint64_t* outputToken){
  if(!outputToken)throw std::runtime_error("Output token pointer required");*outputToken=0;
  if(expectedPid!=GetCurrentProcessId())throw std::runtime_error("Expected process mismatch");
  if(owned)throw std::runtime_error("This helper already owns a capture scope");
  if(!resolveApi()){message("begin",-2,"RenderDoc is not already loaded by the official launcher");return -2;}
  if(api->IsFrameCapturing()){message("begin",-3,"Existing capture is not taken over");return -3;}
  if(!requestedPrefix || !*requestedPrefix)throw std::runtime_error("Explicit UTF8 fresh prefix required");
  if(!std::filesystem::u8path(requestedPrefix).is_absolute())throw std::runtime_error("Absolute fresh prefix required");
  auto guard=std::make_unique<CaptureGuard>(requestedPrefix);
  const char* old=api->GetCaptureFilePathTemplate();if(!old)throw std::runtime_error("Original capture template unavailable");
  previousPath=old;prefix=requestedPrefix;beginCount=api->GetNumCaptures();beginTime=epoch();
  device=suppliedDevice;window=suppliedWindow;fileGuard=std::move(guard);
  api->SetCaptureFilePathTemplate(prefix.c_str());
  templateChanged=true;
  const char* installed=api->GetCaptureFilePathTemplate();
  if(!installed){releaseScope();throw std::runtime_error("Installed capture template unavailable");}
  installedPath=installed;
  // The public SDK has no atomic begin-if-idle. Caller must remain the sole
  // capture writer, including no concurrent Trigger/hotkey/other SDK scope.
  if(api->IsFrameCapturing()){releaseScope();message("begin",-3,"Another capture began before Start; not taken over");return -3;}
  api->StartFrameCapture(device,window);
  if(!api->IsFrameCapturing()){releaseScope();message("begin",-4,"SDK Start did not start a supported device/window capture");return -4;}
  owned=true;token=++serial;heldDevice=retained;if(heldDevice)heldDevice->AddRef();*outputToken=token;
  message("begin",1,"Owned native capture started; real rendering remains caller-controlled");return 1;
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_Begin(uint32_t pid,const char* utf8Prefix,void* dev,void* wnd,uint64_t* outputToken){
  std::lock_guard<std::mutex> lock(gate);try{return begin(pid,utf8Prefix,dev,wnd,nullptr,outputToken);}catch(const std::exception& e){message("begin",-1,e.what());return -1;}
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_BeginFromNativeTexture(uint32_t pid,const char* utf8Prefix,void* nativeResource,uint64_t* outputToken){
  std::lock_guard<std::mutex> lock(gate);ID3D11Device* actualDevice=nullptr;
  try{
    if(!outputToken)throw std::runtime_error("Output token pointer required");*outputToken=0;
    if(pid!=GetCurrentProcessId())throw std::runtime_error("Expected process mismatch");
    if(owned)throw std::runtime_error("This helper already owns a capture scope");
    if(!resolveApi()){message("begin",-2,"RenderDoc is not already loaded by the official launcher");return -2;}
    if(api->IsFrameCapturing()){message("begin",-3,"Existing capture is not taken over");return -3;}
    if(!nativeResource)throw std::runtime_error("Caller must supply actual live Unity D3D11 resource");
    static_cast<ID3D11Resource*>(nativeResource)->GetDevice(&actualDevice);
    int result=begin(pid,utf8Prefix,actualDevice,nullptr,actualDevice,outputToken);if(actualDevice)actualDevice->Release();return result;
  }catch(const std::exception& e){if(actualDevice)actualDevice->Release();message("begin",-1,e.what());return -1;}
}
static int verifyOwned(uint32_t pid,uint64_t supplied){
  if(pid!=GetCurrentProcessId() || !owned || supplied!=token){message("end",-5,"No matching owned scope; external capture untouched");return -5;}
  if(!api->IsFrameCapturing() || api->GetNumCaptures()!=beginCount){releaseScope();message("end",-6,"Owned scope was externally ended/replaced; active capture untouched");return -6;}
  return 0;
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_End(uint32_t pid,uint64_t supplied){
  std::lock_guard<std::mutex> lock(gate);
  try{
    if(presentReserved){message("end",-29,"Present reservation cannot be ended through legacy mid-frame SDK End");return -29;}
    if(auto failure=verifyOwned(pid,supplied))return failure;
    auto success=api->EndFrameCapture(device,window);
    if(!success){if(api->IsFrameCapturing()){message("end",-7,"SDK End failed and owned capture remains active; caller must AbortOwned");return -7;}releaseScope();message("end",-7,"SDK End failed");return -7;}
    if(api->IsFrameCapturing()){message("end",-7,"SDK End returned success but capture remains active; caller must AbortOwned");return -7;}
    auto count=api->GetNumCaptures();if(count!=beginCount+1)throw std::runtime_error("Expected exactly one newly saved owned capture");
    uint32_t n=0;uint64_t stamp=0;if(!api->GetCapture(beginCount,nullptr,&n,&stamp) || !n || n>1<<20)throw std::runtime_error("New capture path metadata unavailable");
    std::vector<char> path(n+1,0);if(!api->GetCapture(beginCount,path.data(),&n,&stamp))throw std::runtime_error("New capture metadata changed");
    auto actual=std::filesystem::u8path(path.data());auto size=std::filesystem::file_size(actual);std::string reason;
    if(!fileGuard->accepts_file(path.data(),stamp,size,beginTime,reason))throw std::runtime_error(reason);
    auto sha=hashFile(actual);int major=0,minor=0,patch=0;api->GetAPIVersion(&major,&minor,&patch);
    auto savedToken=token;releaseScope();
    receipt="{\"schema_version\":1,\"operation\":\"end\",\"code\":2,\"owned\":false,\"pid\":"+std::to_string(pid)+",\"token\":"+std::to_string(savedToken)+",\"path\":"+jsonQuote(path.data())+",\"bytes\":"+std::to_string(size)+",\"sha256\":"+jsonQuote(sha)+",\"timestamp\":"+std::to_string(stamp)+",\"capture_count_before\":"+std::to_string(beginCount)+",\"capture_count_after\":"+std::to_string(count)+",\"renderdoc_module\":"+jsonQuote(modulePath)+",\"sdk_version\":["+std::to_string(major)+","+std::to_string(minor)+","+std::to_string(patch)+"],\"frame_driver_verification\":\"RDC parsing required; SDK GetCapture does not expose driver\"}";return 2;
  }catch(const std::exception& e){if(owned && api && !api->IsFrameCapturing())releaseScope();message("end",-1,e.what());return -1;}
}
extern "C" __declspec(dllexport) int __cdecl HS2_RDOC_AbortOwned(uint32_t pid,uint64_t supplied){
  std::lock_guard<std::mutex> lock(gate);try{
    if(presentReserved){message("abort",-29,"Use owned Present abort protocol; no legacy discard of pending trigger");return -29;}
    if(auto failure=verifyOwned(pid,supplied))return failure;
    if(!api->DiscardFrameCapture(device,window) || api->IsFrameCapturing()){message("abort",-8,"Owned discard failed; active scope retained, not silently cleared");return -8;}
    releaseScope();message("abort",3,"Only the matching owned capture was discarded");return 3;
  }catch(const std::exception& e){message("abort",-1,e.what());return -1;}
}
extern "C" __declspec(dllexport) uint32_t __cdecl HS2_RDOC_GetReceipt(char* output,uint32_t capacity){
  std::lock_guard<std::mutex> lock(gate);auto required=(uint32_t)receipt.size()+1;if(output && capacity>=required)memcpy(output,receipt.c_str(),required);return required;
}

// Separate opt-in Present-boundary protocol. Legacy SDK scope signatures remain.
#include "renderdoc_present_scope.hpp"
