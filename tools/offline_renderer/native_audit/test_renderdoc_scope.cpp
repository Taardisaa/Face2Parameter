// CPU-only official C table mock; never loads RenderDoc or creates graphics.
#define HS2_RDOC_SCOPE_TEST
#include "renderdoc_scope.cpp"
#include <iostream>
static bool capturing=false,noopStart=false,discardFails=false;
static int starts=0,ends=0,discards=0;
static std::string pathTemplate;
static std::vector<std::string> paths;
static std::vector<uint64_t> times;
static void RENDERDOC_CC version(int* a,int* b,int* c){*a=1;*b=6;*c=0;}
static void RENDERDOC_CC setPath(const char* p){pathTemplate=p;}
static const char* RENDERDOC_CC getPath(){return pathTemplate.c_str();}
static uint32_t RENDERDOC_CC count(){return (uint32_t)paths.size();}
static uint32_t RENDERDOC_CC getCapture(uint32_t index,char* out,uint32_t* len,uint64_t* timestamp){
  if(index>=paths.size())return 0;if(len)*len=(uint32_t)paths[index].size()+1;
  if(out)memcpy(out,paths[index].c_str(),paths[index].size()+1);if(timestamp)*timestamp=times[index];return 1;
}
static void RENDERDOC_CC start(RENDERDOC_DevicePointer,RENDERDOC_WindowHandle){starts++;if(!noopStart)capturing=true;}
static uint32_t RENDERDOC_CC isCapturing(){return capturing?1:0;}
static uint32_t RENDERDOC_CC endCapture(RENDERDOC_DevicePointer,RENDERDOC_WindowHandle){
  ends++;if(!capturing)return 0;capturing=false;
  auto path=pathTemplate+"_frame1.rdc";std::ofstream(path,std::ios::binary)<<"CPU synthetic file, not a real RDC";
  paths.push_back(path);times.push_back(epoch());return 1;
}
static uint32_t RENDERDOC_CC discard(RENDERDOC_DevicePointer,RENDERDOC_WindowHandle){discards++;if(discardFails)return 0;capturing=false;return 1;}
int main(int argc,char** argv){try{
  if(argc!=2)throw std::runtime_error("Fresh CPU mock directory required");
  auto root=std::filesystem::absolute(argv[1]);if(std::filesystem::exists(root))throw std::runtime_error("Directory must be fresh");std::filesystem::create_directories(root);
  int checks=0;auto check=[&](bool value){checks++;if(!value)throw std::runtime_error("CPU mock check "+std::to_string(checks)+" failed");};
  uint64_t out=99;auto pid=GetCurrentProcessId();auto name=[&](const char* n){return (root/n).u8string();};
  check(HS2_RDOC_Begin(pid,name("absent").c_str(),nullptr,nullptr,&out)==-2 && out==0);
  RENDERDOC_API_1_6_0 table={};table.GetAPIVersion=version;table.SetCaptureFilePathTemplate=setPath;table.GetCaptureFilePathTemplate=getPath;table.GetNumCaptures=count;table.GetCapture=getCapture;table.StartFrameCapture=start;table.IsFrameCapturing=isCapturing;table.EndFrameCapture=endCapture;table.DiscardFrameCapture=discard;HS2_RDOC_TestApi(&table);
  check(HS2_RDOC_Begin(pid+1,name("pid").c_str(),nullptr,nullptr,&out)==-1 && starts==0 && out==0);
  capturing=true;
  check(HS2_RDOC_BeginFromNativeTexture(pid,name("busy").c_str(),(void*)1,&out)==-3 && out==0 && starts==0 && ends==0 && discards==0);
  capturing=false;noopStart=true;pathTemplate="";
  check(HS2_RDOC_Begin(pid,name("noop").c_str(),nullptr,nullptr,&out)==-4 && out==0 && pathTemplate.empty());
  noopStart=false;
  check(HS2_RDOC_Begin(pid,name("success").c_str(),nullptr,nullptr,&out)==1 && out!=0 && owned);
  auto good=out;
  check(HS2_RDOC_End(pid,good+1)==-5 && capturing && ends==0);
  check(HS2_RDOC_AbortOwned(pid+1,good)==-5 && capturing && discards==0);
  check(HS2_RDOC_End(pid,good)==2 && !capturing && !owned && pathTemplate.empty());
  check(receipt.find("\"sha256\":")!=std::string::npos && receipt.find("RDC parsing required")!=std::string::npos);
  check(HS2_RDOC_Begin(pid,name("success").c_str(),nullptr,nullptr,&out)==-1 && !capturing && out==0);
  pathTemplate="old";
  check(HS2_RDOC_Begin(pid,name("abort").c_str(),nullptr,nullptr,&out)==1);
  discardFails=true;check(HS2_RDOC_AbortOwned(pid,out)==-8 && capturing && owned);
  discardFails=false;check(HS2_RDOC_AbortOwned(pid,out)==3 && !capturing && !owned && pathTemplate=="old");
  check(HS2_RDOC_Begin(pid,name("external_template").c_str(),nullptr,nullptr,&out)==1);
  pathTemplate="external";check(HS2_RDOC_AbortOwned(pid,out)==3 && pathTemplate=="external");
  check(HS2_RDOC_Begin(pid,name("external_end").c_str(),nullptr,nullptr,&out)==1);
  capturing=false;auto calls=ends;
  check(HS2_RDOC_End(pid,out)==-6 && !owned && ends==calls);
  check(HS2_RDOC_AbortOwned(pid,out)==-5 && ends==calls);
  check(HS2_RDOC_Begin(pid,name("replaced").c_str(),nullptr,nullptr,&out)==1);
  paths.push_back("external-other.rdc");times.push_back(epoch());auto discardCount=discards;
  check(HS2_RDOC_AbortOwned(pid,out)==-6 && capturing && !owned && discards==discardCount);
  capturing=false;uint32_t bytes=HS2_RDOC_GetReceipt(nullptr,0);std::vector<char> shortBuffer(2,'X');
  check(HS2_RDOC_GetReceipt(shortBuffer.data(),2)==bytes && shortBuffer[0]=='X');
  std::vector<char> full(bytes);check(HS2_RDOC_GetReceipt(full.data(),bytes)==bytes && std::strlen(full.data())+1==bytes);
  std::cout<<"CPU official SDK scope mock passed "<<checks<<" checks; synthetic files are not native/RDC acceptance\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
