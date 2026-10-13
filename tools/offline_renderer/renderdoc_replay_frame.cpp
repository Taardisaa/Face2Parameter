// Official driver execution for the entire recorded frame, not a selected pass.
// Build only with installed x64 MSVC and exactly matching pinned official headers.
#define RENDERDOC_PLATFORM_WIN32
#if !defined(_MSC_VER) || defined(__MINGW32__)
#error "The official Windows replay C++ interface requires matching MSVC ABI"
#endif
#include "renderdoc_replay.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstring>
#include <type_traits>
REPLAY_PROGRAM_MARKER();
static_assert(sizeof(RENDERDOC_ProgressCallback)==64,"Pinned MSVC callback layout");
static_assert(sizeof(ResourceId)==8 && std::is_trivially_copyable<ResourceId>::value,"Pinned opaque resource ID representation");
static uint64_t idValue(const ResourceId& id){uint64_t value;std::memcpy(&value,&id,8);return value;}
static std::string quoteJson(const std::string& text){std::string out="\"";for(unsigned char c:text){if(c=='"'||c=='\\'){out+='\\';out+=c;}else if(c<32){char b[7];sprintf_s(b,"\\u%04x",c);out+=b;}else out+=c;}return out+'"';}
static void require(ResultDetails result,const char* operation){
  if(!result.OK())throw std::runtime_error(std::string(operation)+" ResultCode="+std::to_string((uint32_t)result.code)+(result.internal_msg?(" "+std::string(result.internal_msg->c_str())):""));
}
static void fresh(const char* path){auto p=std::filesystem::u8path(path);if(!p.is_absolute() || !std::filesystem::is_directory(p.parent_path()) || std::filesystem::exists(p))throw std::runtime_error("Absolute fresh output in existing directory required");}
static void actions(const rdcarray<ActionDescription>& values,const SDFile& structured,uint32_t selected,bool& found){
  for(const auto& action:values){
    if(action.eventId==selected && !action.IsFakeMarker())found=true;
    std::cout<<"{\"event\":\"action\",\"event_id\":"<<action.eventId<<",\"action_id\":"<<action.actionId<<",\"flags\":"<<(uint32_t)action.flags<<",\"name\":"<<quoteJson(action.GetName(structured).c_str())<<",\"fake_marker\":"<<(action.IsFakeMarker()?"true":"false")<<"}"<<std::endl;
    actions(action.children,structured,selected,found);
  }
}
struct FileOwner{ICaptureFile* p=nullptr;~FileOwner(){if(p)p->Shutdown();}};
struct ReplayOwner{IReplayController* p=nullptr;~ReplayOwner(){if(p)p->Shutdown();}};
int main(int argc,char** argv){
  try{
    if(argc<3)throw std::runtime_error("convert INPUT TYPE OUTPUT TYPE | inspect RDC | replay RDC EVENT RESOURCE_ID PNG RAW");
    std::string mode=argv[1];
    if((mode=="convert" && argc!=6)||(mode=="inspect" && argc!=3)||(mode=="replay" && argc!=7)||(mode!="convert" && mode!="inspect" && mode!="replay"))throw std::runtime_error("Invalid explicit mode/arguments");
    if(!std::filesystem::is_regular_file(std::filesystem::u8path(argv[2])))throw std::runtime_error("Actual input capture required");
    if(mode=="convert"){
      fresh(argv[4]);
      if(std::string(argv[3])!="rdc" && std::string(argv[3])!="zip.xml")throw std::runtime_error("Input must explicitly be rdc or zip.xml; XML-only cannot be imported");
      if(std::string(argv[5])!="rdc" && std::string(argv[5])!="zip.xml")throw std::runtime_error("Output must explicitly be rdc or complete zip.xml");
      if(std::string(argv[5])=="zip.xml"){auto external=std::filesystem::u8path(argv[4]);external.replace_extension();if(std::filesystem::exists(external))throw std::runtime_error("XML external ZIP output already exists");}
    }
    if(mode=="replay"){fresh(argv[5]);fresh(argv[6]);if(std::filesystem::absolute(std::filesystem::u8path(argv[5]))==std::filesystem::absolute(std::filesystem::u8path(argv[6])))throw std::runtime_error("PNG and official readback output paths must differ");}
    // Exact release matched before invoking non-trivial C++ interface methods.
    const char* commit=RENDERDOC_GetCommitHash();
    if(!commit || std::string(commit)!="e4bd23b671d3d5a747ff5221dbe08a63eb6ca200")throw std::runtime_error("Pinned official v1.46 DLL commit mismatch");
    struct Session{Session(){GlobalEnvironment env;env.enumerateGPUs=false;RENDERDOC_InitialiseReplay(env,{});}~Session(){RENDERDOC_ShutdownReplay();}} session;
    FileOwner file;file.p=RENDERDOC_OpenCaptureFile();if(!file.p)throw std::runtime_error("Official capture factory returned null");
    require(file.p->OpenFile(argv[2],mode=="convert"?argv[3]:"rdc",nullptr),"OpenFile");
    auto driver=file.p->DriverName();if(driver!="D3D11")throw std::runtime_error("Actual recorded driver is not D3D11");
    std::cout<<"{\"event\":\"opened\",\"driver\":"<<quoteJson(driver.c_str())<<",\"version\":"<<quoteJson(RENDERDOC_GetVersionString())<<",\"commit\":"<<quoteJson(commit)<<"}"<<std::endl;
    if(mode=="convert"){
      require(file.p->Convert(argv[4],argv[5],nullptr,nullptr),"Convert");
      if(!std::filesystem::is_regular_file(std::filesystem::u8path(argv[4])) || !std::filesystem::file_size(std::filesystem::u8path(argv[4])))throw std::runtime_error("Converted output absent/empty");
      std::cout<<"{\"event\":\"converted\",\"path\":"<<quoteJson(argv[4])<<",\"format\":"<<quoteJson(argv[5])<<"}"<<std::endl;return 0;
    }
    ReplayOptions options;options.optimisation=ReplayOptimisationLevel::NoOptimisation;
    ReplayOwner replay;auto opened=file.p->OpenCapture(options,nullptr);require(opened.first,"OpenCapture");replay.p=opened.second;if(!replay.p)throw std::runtime_error("Official replay returned null");
    auto api=replay.p->GetAPIProperties();if(api.pipelineType!=GraphicsAPI::D3D11 || api.remoteReplay)throw std::runtime_error("Replay must be actual local D3D11");
    std::cout<<"{\"event\":\"replay_opened\",\"optimisation\":\"NoOptimisation\",\"local_renderer\":"<<(uint32_t)api.localRenderer<<",\"vendor\":"<<(uint32_t)api.vendor<<"}"<<std::endl;
    uint32_t event=mode=="replay"?(uint32_t)std::stoul(argv[3]):0;bool found=false;
    actions(replay.p->GetRootActions(),replay.p->GetStructuredFile(),event,found);
    uint64_t wanted=mode=="replay"?std::stoull(argv[4]):0;ResourceId selected;bool textureFound=false;
    for(const auto& texture:replay.p->GetTextures()){
      auto value=idValue(texture.resourceId);if(value==wanted){selected=texture.resourceId;textureFound=true;if(texture.dimension!=2 || texture.arraysize!=1 || texture.msSamp!=1)throw std::runtime_error("Selected synchronized PNG target must be single-sample 2D slice0");}
      std::cout<<"{\"event\":\"texture\",\"resource_id\":"<<value<<",\"width\":"<<texture.width<<",\"height\":"<<texture.height<<",\"depth\":"<<texture.depth<<",\"array_size\":"<<texture.arraysize<<",\"mips\":"<<texture.mips<<",\"samples\":"<<texture.msSamp<<",\"format_type\":"<<(uint32_t)texture.format.type<<",\"component_type\":"<<(uint32_t)texture.format.compType<<",\"component_count\":"<<(uint32_t)texture.format.compCount<<",\"component_bytes\":"<<(uint32_t)texture.format.compByteWidth<<",\"srgb_corrected\":"<<(texture.format.SRGBCorrected()?"true":"false")<<"}"<<std::endl;
    }
    if(mode=="inspect")return 0;
    if(!event || !found || !textureFound)throw std::runtime_error("Explicit actual event/texture not present; no default output selection");
    replay.p->SetFrameEvent(event,true);
    Subresource sub;sub.mip=0;sub.slice=0;sub.sample=0;auto data=replay.p->GetTextureData(selected,sub);
    if(data.empty())throw std::runtime_error("Official selected texture readback empty");
    std::ofstream raw(std::filesystem::u8path(argv[6]),std::ios::binary);raw.write((const char*)data.data(),data.size());raw.close();if(!raw)throw std::runtime_error("Official readback write failed");
    TextureSave save;save.resourceId=selected;save.destType=FileType::PNG;save.mip=0;save.slice.sliceIndex=0;save.sample.sampleIndex=0;
    require(replay.p->SaveTexture(save,argv[5]),"SaveTexture");
    std::cout<<"{\"event\":\"replayed\",\"event_id\":"<<event<<",\"resource_id\":"<<wanted<<",\"png\":"<<quoteJson(argv[5])<<",\"official_texture_data\":"<<quoteJson(argv[6])<<",\"readback_bytes\":"<<data.size()<<",\"candidate_geometry_complete\":false}"<<std::endl;return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
