// Official driver execution for the entire recorded frame, not a selected pass.
// Build only with installed x64 MSVC and exactly matching pinned official headers.
#define RENDERDOC_PLATFORM_WIN32
#if !defined(_MSC_VER) || defined(__MINGW32__)
#error "The official Windows replay C++ interface requires matching MSVC ABI"
#endif
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#include "renderdoc_replay.h"
// Pinned v1.46 renderdoc/3rdparty/lz4 (the same codec source compiled into the official DLL).
#include "lz4.h"
#include <chrono>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstring>
#include <type_traits>
#include <set>
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
    std::cout<<"{\"event\":\"action\",\"event_id\":"<<action.eventId<<",\"action_id\":"<<action.actionId<<",\"flags\":"<<(uint32_t)action.flags<<",\"name\":"<<quoteJson(action.GetName(structured).c_str())<<",\"fake_marker\":"<<(action.IsFakeMarker()?"true":"false")
      <<",\"copy_source\":"<<idValue(action.copySource)<<",\"copy_destination\":"<<idValue(action.copyDestination)<<",\"api_events\":[";
    for(size_t k=0;k<action.events.size();k++){if(k)std::cout<<",";const auto& e=action.events[k];std::cout<<"{\"event_id\":"<<e.eventId<<",\"chunk_index\":"<<e.chunkIndex<<"}";}
    std::cout<<"]}"<<std::endl;
    actions(action.children,structured,selected,found);
  }
}
static const ActionDescription* findAction(const rdcarray<ActionDescription>& values,uint32_t event){
  for(const auto& action:values){if(action.eventId==event && !action.IsFakeMarker())return &action;if(auto child=findAction(action.children,event))return child;}return nullptr;
}
static void drawInputs(IReplayController* replay,const ActionDescription& action,const char* directory){
  if(((uint32_t)action.flags&(uint32_t)ActionFlags::Drawcall)==0)throw std::runtime_error("Explicit actual draw action required for source correspondence");
  auto root=std::filesystem::u8path(directory);std::filesystem::create_directory(root);
  replay->SetFrameEvent(action.eventId,true);auto state=replay->GetD3D11PipelineState();if(!state)throw std::runtime_error("Actual D3D11 input state unavailable");
  std::ofstream file(root/"draw_inputs.jsonl",std::ios::binary);if(!file)throw std::runtime_error("Input receipt write failed");
  file<<"{\"event\":\"draw_arguments\",\"event_id\":"<<action.eventId<<",\"flags\":"<<(uint32_t)action.flags<<",\"num_indices\":"<<action.numIndices<<",\"num_instances\":"<<action.numInstances<<",\"base_vertex\":"<<action.baseVertex<<",\"index_offset\":"<<action.indexOffset<<",\"vertex_offset\":"<<action.vertexOffset<<",\"instance_offset\":"<<action.instanceOffset<<",\"layout_resource\":"<<idValue(state->inputAssembly.resourceId)<<",\"topology\":"<<(uint32_t)state->inputAssembly.topology<<",\"vertex_shader\":"<<idValue(state->vertexShader.resourceId)<<",\"pixel_shader\":"<<idValue(state->pixelShader.resourceId)<<"}\n";
  std::set<uint64_t> dumped;
  auto buffer=[&](const ResourceId& id){auto value=idValue(id);if(!value || !dumped.insert(value).second)return;
    auto data=replay->GetBufferData(id,0,0);if(data.empty())throw std::runtime_error("Bound actual buffer data unavailable");
    auto name="buffer_"+std::to_string(value)+".bin";std::ofstream out(root/name,std::ios::binary);out.write((const char*)data.data(),data.size());out.close();if(!out)throw std::runtime_error("Bound buffer write failed");
    file<<"{\"event\":\"buffer_data\",\"resource_id\":"<<value<<",\"path\":"<<quoteJson(name)<<",\"bytes\":"<<data.size()<<"}\n";
  };
  for(size_t i=0;i<state->inputAssembly.vertexBuffers.size();i++){
    const auto& vb=state->inputAssembly.vertexBuffers[i];if(vb.resourceId==ResourceId::Null())continue;
    file<<"{\"event\":\"vertex_binding\",\"slot\":"<<i<<",\"resource_id\":"<<idValue(vb.resourceId)<<",\"offset\":"<<vb.byteOffset<<",\"stride\":"<<vb.byteStride<<"}\n";buffer(vb.resourceId);
  }
  const auto& ib=state->inputAssembly.indexBuffer;
  file<<"{\"event\":\"index_binding\",\"resource_id\":"<<idValue(ib.resourceId)<<",\"offset\":"<<ib.byteOffset<<",\"stride\":"<<ib.byteStride<<"}\n";buffer(ib.resourceId);
  for(const auto& layout:state->inputAssembly.layouts){
    file<<"{\"event\":\"layout_element\",\"semantic\":"<<quoteJson(layout.semanticName.c_str())<<",\"semantic_index\":"<<layout.semanticIndex<<",\"input_slot\":"<<layout.inputSlot<<",\"byte_offset\":"<<layout.byteOffset<<",\"per_instance\":"<<(layout.perInstance?"true":"false")<<",\"instance_step\":"<<layout.instanceDataStepRate<<",\"format_type\":"<<(uint32_t)layout.format.type<<",\"component_type\":"<<(uint32_t)layout.format.compType<<",\"component_count\":"<<(uint32_t)layout.format.compCount<<",\"component_bytes\":"<<(uint32_t)layout.format.compByteWidth<<",\"bgra_order\":"<<(layout.format.BGRAOrder()?"true":"false")<<"}\n";
  }
  file<<"{\"event\":\"scope\",\"all_bound_vertex_index_bytes_dumped\":true,\"surface_correspondence_accepted\":false,\"previous_shadow_motion_cache_coverage_accepted\":false}\n";
  file.close();if(!file)throw std::runtime_error("Draw input receipt write failed");
}
struct FileOwner{ICaptureFile* p=nullptr;~FileOwner(){if(p)p->Shutdown();}};
struct ReplayOwner{IReplayController* p=nullptr;~ReplayOwner(){if(p)p->Shutdown();}};

// ---------------------------------------------------------------------------------------------
// Patched-capture pipeline (2026-10-20). Exact layout sources, pinned v1.46 e4bd23b6:
//   serialise/rdcfile.cpp    FileHeader, BinaryThumbnail, CaptureMetaData, BinarySectionHeader, Init(), ReadSection()
//   serialise/lz4io.cpp      1 MiB pages; [int32 compSize][LZ4_compress_fast_continue block]; one page of history
//   serialise/serialiser.cpp Reading BeginChunk/EndChunk (flag-dependent header, 64-byte chunk alignment);
//                            DoSerialise(SDObject*) and WriteStructuredFile (layout of every structured value)
//   serialise/serialiser.h   byte buffer = uint64 size, AlignTo<64>, raw bytes; strings = 32-bit length + bytes
// The offset index is derived from the official structured data and verified against the official
// decompressed section. Patched candidates are complete standard RDC files replayed by the official DLL.
// ---------------------------------------------------------------------------------------------
using Clock=std::chrono::steady_clock;
static double msSince(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Sha256{
  BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
  Sha256(){if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)!=0 || BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0)!=0)throw std::runtime_error("BCrypt SHA256 unavailable");}
  ~Sha256(){if(hash)BCryptDestroyHash(hash);if(alg)BCryptCloseAlgorithmProvider(alg,0);}
  Sha256(const Sha256&)=delete;
  void update(const void* data,uint64_t size){auto p=(PUCHAR)data;while(size){ULONG n=(ULONG)std::min<uint64_t>(size,1u<<30);if(BCryptHashData(hash,p,n,0)!=0)throw std::runtime_error("BCrypt hash failed");p+=n;size-=n;}}
  std::string hex(){UCHAR d[32];if(BCryptFinishHash(hash,d,32,0)!=0)throw std::runtime_error("BCrypt finish failed");static const char* x="0123456789abcdef";std::string s;for(auto c:d){s+=x[c>>4];s+=x[c&15];}return s;}
};
static std::string sha256(const void* data,uint64_t size){Sha256 s;s.update(data,size);return s.hex();}
static std::vector<uint8_t> readAll(const std::filesystem::path& p){
  std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Cannot read "+p.u8string());
  auto n=(size_t)f.tellg();std::vector<uint8_t> out(n);f.seekg(0);if(n && !f.read((char*)out.data(),n))throw std::runtime_error("Short read "+p.u8string());return out;
}
template<class T> static T rd(const uint8_t* base,uint64_t size,uint64_t off){if(off+sizeof(T)>size)throw std::runtime_error("Truncated structure");T v;std::memcpy(&v,base+off,sizeof(T));return v;}
static uint64_t alignUp64(uint64_t v){return (v+63)&~uint64_t(63);}

// rdcfile.cpp container. ASCII sections are rejected explicitly (not produced by D3D11 capture).
struct RdcSection{uint64_t headerOffset=0,dataOffset=0,compressed=0,uncompressed=0,version=0;uint32_t type=0,flags=0;std::string name;};
struct RdcContainer{uint32_t version=0,headerLength=0;uint64_t machineIdent=0;uint32_t driverId=0;std::string driverName;std::vector<RdcSection> sections;int frame=-1;};
static RdcContainer parseContainer(const std::vector<uint8_t>& f){
  RdcContainer c;const uint8_t* b=f.data();uint64_t n=f.size();
  if(rd<uint64_t>(b,n,0)!=0x434F4452ull)throw std::runtime_error("Not an RDOC container");
  c.version=rd<uint32_t>(b,n,8);c.headerLength=rd<uint32_t>(b,n,12);
  if(c.version!=0x100 && c.version!=0x101 && c.version!=0x102)throw std::runtime_error("Unsupported RDC container version");
  uint32_t thumbLen=rd<uint32_t>(b,n,32+4);uint64_t meta=32+8+(uint64_t)thumbLen;
  c.machineIdent=rd<uint64_t>(b,n,meta);c.driverId=rd<uint32_t>(b,n,meta+8);uint8_t nameLen=rd<uint8_t>(b,n,meta+12);
  if(meta+13+nameLen>n || !nameLen)throw std::runtime_error("Bad driver name");
  c.driverName.assign((const char*)b+meta+13,nameLen);while(!c.driverName.empty() && c.driverName.back()==0)c.driverName.pop_back();
  uint64_t off=c.headerLength;
  while(off<n){
    RdcSection s;s.headerOffset=off;if(rd<uint8_t>(b,n,off)!=0)throw std::runtime_error("ASCII section is not supported by the patch pipeline");
    s.type=rd<uint32_t>(b,n,off+4);s.compressed=rd<uint64_t>(b,n,off+8);s.uncompressed=rd<uint64_t>(b,n,off+16);s.version=rd<uint64_t>(b,n,off+24);s.flags=rd<uint32_t>(b,n,off+32);
    uint32_t nl=rd<uint32_t>(b,n,off+36);if(!nl || nl>2048)throw std::runtime_error("Invalid section name length");
    s.name.assign((const char*)b+off+40,nl-1);s.dataOffset=off+40+nl;if(s.dataOffset+s.compressed>n)throw std::runtime_error("Truncated section data");
    if(s.type==(uint32_t)SectionType::FrameCapture){if(c.frame!=-1)throw std::runtime_error("Duplicate frame capture section");c.frame=(int)c.sections.size();}
    off=s.dataOffset+s.compressed;c.sections.push_back(s);
  }
  if(c.frame<0)throw std::runtime_error("No frame capture section");
  return c;
}

// lz4io.cpp page model: page k covers section bytes [k MiB, (k+1) MiB).
static const uint64_t kPage=1024*1024;
struct Lz4Block{uint64_t fileOffset;int32_t compSize;};
struct FrameStore{
  const std::vector<uint8_t>* file=nullptr;RdcSection sec;bool lz4=false,replayOnly=false;
  std::vector<Lz4Block> blocks;std::vector<std::vector<char>> tails; // last <=64 KiB (LZ4_DISTANCE_MAX) of each original decoded page
  uint64_t pages()const{return (sec.uncompressed+kPage-1)/kPage;}
  uint64_t pageSize(uint64_t k)const{return std::min<uint64_t>(kPage,sec.uncompressed-k*kPage);}
  void decode(uint64_t k,std::vector<char>& out)const{
    out.resize((size_t)pageSize(k));
    if(!lz4){std::memcpy(out.data(),file->data()+sec.dataOffset+k*kPage,out.size());return;}
    const auto& bl=blocks[(size_t)k];const char* src=(const char*)file->data()+bl.fileOffset+4;
    int got=k?LZ4_decompress_safe_usingDict(src,out.data(),bl.compSize,(int)kPage,tails[(size_t)k-1].data(),(int)tails[(size_t)k-1].size())
             :LZ4_decompress_safe(src,out.data(),bl.compSize,(int)kPage);
    if(got<0 || (uint64_t)got!=out.size())throw std::runtime_error("LZ4 page decode mismatch");
  }
};
static std::string loadFrame(FrameStore& fs,const std::vector<uint8_t>& file,const RdcSection& sec){
  fs.file=&file;fs.sec=sec;fs.blocks.clear();fs.tails.clear();
  if(sec.flags==(uint32_t)SectionFlags::LZ4Compressed)fs.lz4=true;
  else if(sec.flags==(uint32_t)SectionFlags::NoFlags){fs.lz4=false;if(sec.compressed!=sec.uncompressed)throw std::runtime_error("Uncompressed section length mismatch");}
  else throw std::runtime_error("Frame section is not LZ4/uncompressed; zstd block splicing is not implemented");
  if(fs.lz4){
    uint64_t off=sec.dataOffset,end=sec.dataOffset+sec.compressed;
    while(off<end){int32_t cs=rd<int32_t>(file.data(),file.size(),off);if(cs<0 || cs>LZ4_COMPRESSBOUND((int)kPage) || off+4+(uint64_t)cs>end)throw std::runtime_error("Invalid LZ4 block");fs.blocks.push_back({off,cs});off+=4+(uint64_t)cs;}
    if(fs.blocks.size()!=fs.pages())throw std::runtime_error("LZ4 block count disagrees with uncompressed length");
  }
  Sha256 h;std::vector<char> page;
  for(uint64_t k=0;k<fs.pages();k++){
    fs.decode(k,page);h.update(page.data(),page.size());
    size_t t=std::min<size_t>(page.size(),65536);fs.tails.emplace_back(page.end()-t,page.end());
  }
  return h.hex();
}

// Section offsets of structured byte buffers, mirroring DoSerialise(SDObject*) exactly.
struct BufferLoc{uint64_t sdIndex,sizeOffset,dataOffset,bytes;bool opaque;};
struct Walker{
  const SDFile& file;uint64_t off;std::vector<BufferLoc> out;
  void walk(SDObject* o){
    if(o->type.flags & SDTypeFlags::Nullable)off+=1;
    switch(o->type.basetype){
      case SDBasic::Chunk:throw std::runtime_error("Unexpected chunk inside object");
      case SDBasic::Struct:for(size_t i=0;i<o->NumChildren();i++)walk(o->GetChild(i));break;
      case SDBasic::Array:off+=8;for(size_t i=0;i<o->NumChildren();i++)walk(o->GetChild(i));break;
      case SDBasic::Null:if(!(o->type.flags & SDTypeFlags::Nullable))throw std::runtime_error("Null without Nullable");break;
      case SDBasic::Buffer:{uint64_t id=o->data.basic.u;if(id>=file.buffers.size())throw std::runtime_error("Buffer index out of range");
        uint64_t size=file.buffers[(size_t)id]->size();uint64_t sizeOff=off;off=alignUp64(off+8);out.push_back({id,sizeOff,off,size,false});off+=size;break;}
      case SDBasic::String:off+=4;if(!(o->type.flags & SDTypeFlags::NullString))off+=o->data.str.size();break;
      case SDBasic::Boolean:case SDBasic::Character:off+=1;break;
      case SDBasic::Resource:off+=8;break;
      case SDBasic::Enum:case SDBasic::GPUAddress:case SDBasic::UnsignedInteger:case SDBasic::SignedInteger:
        if(o->type.byteSize!=1 && o->type.byteSize!=2 && o->type.byteSize!=4 && o->type.byteSize!=8)throw std::runtime_error("Unexpected integer width");off+=o->type.byteSize;break;
      case SDBasic::Float:if(o->type.byteSize!=4 && o->type.byteSize!=8)throw std::runtime_error("Unexpected float width");off+=o->type.byteSize;break;
      default:throw std::runtime_error("Unknown structured basetype");
    }
  }
};
struct ChunkLoc{uint64_t header,payload,length;uint32_t id,rawFlags;};
static std::vector<ChunkLoc> parseChunks(const bytebuf& s){
  std::vector<ChunkLoc> out;uint64_t off=0,n=s.size();const uint8_t* b=s.data();
  while(off<n){
    ChunkLoc c;c.header=off;uint32_t v=rd<uint32_t>(b,n,off);off+=4;if(!v)throw std::runtime_error("Chunk ID 0 in stream");
    c.rawFlags=v&~0xffffu;c.id=v&0xffffu;
    if(v&0x00010000u){uint32_t frames=rd<uint32_t>(b,n,off);off+=4;if(frames>=4096)throw std::runtime_error("Invalid callstack");off+=8ull*frames;} // ChunkCallstack
    if(v&0x00020000u)off+=8; // ChunkThreadID
    if(v&0x00040000u)off+=8; // ChunkDuration
    if(v&0x00080000u)off+=8; // ChunkTimestamp
    if(v&0x00100000u){c.length=rd<uint64_t>(b,n,off);off+=8;}else{c.length=rd<uint32_t>(b,n,off);off+=4;} // Chunk64BitSize
    c.payload=off;if(c.payload+c.length>n)throw std::runtime_error("Chunk overruns section");
    out.push_back(c);off=alignUp64(c.payload+c.length);
  }
  return out;
}
static int sectionsMode(const char* path){
  FileOwner file;file.p=RENDERDOC_OpenCaptureFile();if(!file.p)throw std::runtime_error("Official capture factory returned null");require(file.p->OpenFile(path,"rdc",nullptr),"OpenFile");
  for(int32_t i=0;i<file.p->GetSectionCount();i++){
    auto props=file.p->GetSectionProperties(i);auto t=Clock::now();auto data=file.p->GetSectionContents(i);double rm=msSince(t);
    if(data.size()!=props.uncompressedSize)throw std::runtime_error("Official section contents size mismatch");
    std::cout<<"{\"event\":\"section\",\"index\":"<<i<<",\"type\":"<<(uint32_t)props.type<<",\"name\":"<<quoteJson(props.name.c_str())<<",\"flags\":"<<(uint32_t)props.flags<<",\"version\":"<<props.version
      <<",\"compressed_bytes\":"<<props.compressedSize<<",\"uncompressed_bytes\":"<<props.uncompressedSize<<",\"contents_sha256\":\""<<sha256(data.data(),data.size())<<"\",\"official_read_ms\":"<<rm<<"}"<<std::endl;
  }
  return 0;
}
static int indexMode(const char* path,const char* outPath){
  fresh(outPath);auto t0=Clock::now();
  auto raw=readAll(std::filesystem::u8path(path));auto fileSha=sha256(raw.data(),raw.size());auto cont=parseContainer(raw);
  FrameStore fs;auto ownDecodeSha=loadFrame(fs,raw,cont.sections[(size_t)cont.frame]);double ownMs=msSince(t0);
  FileOwner file;file.p=RENDERDOC_OpenCaptureFile();if(!file.p)throw std::runtime_error("Official capture factory returned null");require(file.p->OpenFile(path,"rdc",nullptr),"OpenFile");
  if(file.p->DriverName()!="D3D11")throw std::runtime_error("Actual recorded driver is not D3D11");
  int idx=file.p->FindSectionByType(SectionType::FrameCapture);if(idx!=cont.frame)throw std::runtime_error("Official and parsed frame section index differ");
  auto t1=Clock::now();bytebuf section=file.p->GetSectionContents(idx);double offMs=msSince(t1);
  auto officialSha=sha256(section.data(),section.size());
  if(officialSha!=ownDecodeSha)throw std::runtime_error("Pinned LZ4 page decode disagrees with official section contents");
  auto chunks=parseChunks(section);
  auto t2=Clock::now();const SDFile& sd=file.p->GetStructuredData();double sdMs=msSince(t2);
  if(sd.chunks.size()!=chunks.size())throw std::runtime_error("Structured chunk count != stream chunk count");
  std::ofstream out(std::filesystem::u8path(outPath),std::ios::binary);
  out<<"{\"schema_version\":1,\"scope\":\"official structured-data byte-buffer section offsets, verified against the official decompressed frame section\",\n";
  out<<"\"rdc\":"<<quoteJson(path)<<",\"rdc_bytes\":"<<raw.size()<<",\"rdc_sha256\":\""<<fileSha<<"\",\"frame_section_index\":"<<idx<<",\"frame_section_flags\":"<<cont.sections[(size_t)idx].flags
     <<",\"frame_section_uncompressed_bytes\":"<<section.size()<<",\"frame_section_sha256\":\""<<officialSha<<"\",\"pages\":"<<fs.pages()<<",\"chunk_count\":"<<chunks.size()<<",\n\"chunks\":[\n";
  uint64_t buffersTotal=0,exactLength=0,slackChunks=0,maxSlack=0;
  for(size_t i=0;i<chunks.size();i++){
    SDChunk* ch=sd.chunks[i];const auto& loc=chunks[i];
    if(ch->metadata.chunkID!=loc.id || ch->metadata.length!=loc.length)throw std::runtime_error("Chunk metadata mismatch at "+std::to_string(i));
    Walker w{sd,loc.payload,{}};
    bool opaque=(ch->metadata.flags & SDChunkFlags::OpaqueChunk)!=SDChunkFlags::NoFlags;
    if(opaque){if(ch->NumChildren()!=1 || ch->GetChild(0)->type.basetype!=SDBasic::Buffer)throw std::runtime_error("Opaque chunk layout");
      uint64_t id=ch->GetChild(0)->data.basic.u;uint64_t size=sd.buffers[(size_t)id]->size();w.out.push_back({id,UINT64_MAX,loc.payload,size,true});w.off=loc.payload+size;}
    else for(size_t c=0;c<ch->NumChildren();c++)w.walk(ch->GetChild(c));
    uint64_t walked=w.off-loc.payload;if(walked>loc.length)throw std::runtime_error("Walked layout overruns chunk "+std::to_string(i));
    if(walked==loc.length)exactLength++;else{slackChunks++;maxSlack=std::max(maxSlack,loc.length-walked);}
    out<<(i?",\n":"")<<"{\"chunk_index\":"<<i<<",\"name\":"<<quoteJson(ch->name.c_str())<<",\"chunk_id\":"<<loc.id<<",\"payload_offset\":"<<loc.payload<<",\"length\":"<<loc.length<<",\"walked_bytes\":"<<walked<<",\"resources\":[";
    bool first=true;for(size_t c=0;c<ch->NumChildren();c++){SDObject* o=ch->GetChild(c);if(o->type.basetype==SDBasic::Resource){out<<(first?"":",")<<"{\"name\":"<<quoteJson(o->name.c_str())<<",\"id\":"<<idValue(o->data.basic.id)<<"}";first=false;}}
    out<<"],\"buffers\":[";
    for(size_t k=0;k<w.out.size();k++){
      const auto& bl=w.out[k];const bytebuf& blob=*sd.buffers[(size_t)bl.sdIndex];
      if(bl.dataOffset+bl.bytes>section.size() || (bl.bytes && std::memcmp(section.data()+bl.dataOffset,blob.data(),(size_t)bl.bytes)))throw std::runtime_error("Buffer bytes not at walked offset in chunk "+std::to_string(i));
      if(!bl.opaque){if(rd<uint64_t>(section.data(),section.size(),bl.sizeOffset)!=bl.bytes)throw std::runtime_error("Buffer size prefix mismatch in chunk "+std::to_string(i));
        for(uint64_t p=bl.sizeOffset+8;p<bl.dataOffset;p++)if(section[(size_t)p])throw std::runtime_error("Non-zero alignment padding in chunk "+std::to_string(i));}
      out<<(k?",":"")<<"{\"ordinal\":"<<k<<",\"sd_buffer_index\":"<<bl.sdIndex<<",\"size_offset\":"<<(bl.opaque?-1:(int64_t)bl.sizeOffset)<<",\"data_offset\":"<<bl.dataOffset<<",\"bytes\":"<<bl.bytes
         <<",\"sha256\":\""<<sha256(blob.data(),blob.size())<<"\",\"opaque\":"<<(bl.opaque?"true":"false")<<"}";
      buffersTotal++;
    }
    out<<"]}";
  }
  out<<"\n],\"buffers_verified\":"<<buffersTotal<<",\"chunks_walk_exact_length\":"<<exactLength<<",\"chunks_with_length_slack\":"<<slackChunks<<",\"max_length_slack\":"<<maxSlack
     <<",\"timing_ms\":{\"read_and_pinned_lz4_decode\":"<<ownMs<<",\"official_section_contents\":"<<offMs<<",\"official_structured_data\":"<<sdMs<<",\"total\":"<<msSince(t0)<<"}}\n";
  out.close();if(!out)throw std::runtime_error("Index write failed");
  std::cout<<"{\"event\":\"indexed\",\"chunks\":"<<chunks.size()<<",\"buffers\":"<<buffersTotal<<",\"exact_length_chunks\":"<<exactLength<<",\"slack_chunks\":"<<slackChunks<<",\"frame_section_sha256\":\""<<officialSha<<"\"}"<<std::endl;
  return 0;
}

// Persistent session: the original compressed RDC stays in memory. Each job verifies the exact
// original bytes, re-encodes only touched LZ4 pages plus each following page (whose history
// changed), writes a complete standard RDC and executes it through the official full D3D11 replay.
static std::vector<std::string> splitTabs(const std::string& line){std::vector<std::string> v;std::string cur;for(char c:line){if(c=='\t'){v.push_back(cur);cur.clear();}else if(c!='\r')cur+=c;}v.push_back(cur);return v;}
struct PatchReq{uint64_t sizeOffset=0;bool hasSize=false;uint64_t dataOffset=0,blobBytes=0,byteOffset=0;std::string blobSha,rangeSha,path,replacementSha;std::vector<char> bytes;};
struct TargetReq{std::string kind,name,out;uint32_t event=0;uint64_t resource=0,offset=0,length=0;uint32_t mip=0,slice=0;};
static void collectEvents(const rdcarray<ActionDescription>& v,std::set<uint32_t>& ev){for(const auto& a:v){if(!a.IsFakeMarker())ev.insert(a.eventId);for(const auto& e:a.events)ev.insert(e.eventId);collectEvents(a.children,ev);}}
static std::string runJob(const std::string& id,const std::filesystem::path& candidate,bool keep,std::vector<PatchReq>& patches,const std::vector<TargetReq>& targets,
                          const std::vector<uint8_t>& raw,FrameStore& fs,const std::string& originalPath){
  std::ostringstream js;auto t0=Clock::now();std::map<std::string,double> tm;
  auto tv=Clock::now();
  for(const auto& t:targets){auto p=std::filesystem::u8path(t.out);if(!p.is_absolute() || !std::filesystem::is_directory(p.parent_path()) || std::filesystem::exists(p))throw std::runtime_error("Fresh absolute target output required: "+t.out);}
  std::map<uint64_t,std::vector<char>> pages; // decoded from the pristine original for this job only
  auto page=[&](uint64_t k)->std::vector<char>&{auto it=pages.find(k);if(it!=pages.end())return it->second;auto& v=pages[k];fs.decode(k,v);return v;};
  auto hashRange=[&](uint64_t off,uint64_t n){Sha256 h;while(n){uint64_t k=off/kPage,in=off%kPage;auto& pg=page(k);uint64_t c=std::min<uint64_t>(n,pg.size()-in);h.update(pg.data()+in,c);off+=c;n-=c;}return h.hex();};
  std::vector<std::pair<uint64_t,uint64_t>> ranges;
  if(fs.replayOnly && !patches.empty())throw std::runtime_error("This session's capture is replay-only (zstd section); patches require the LZ4 original");
  for(auto& p:patches){
    if(p.bytes.empty() || p.dataOffset+p.blobBytes>fs.sec.uncompressed || p.byteOffset+p.bytes.size()>p.blobBytes)throw std::runtime_error("Patch range outside the selected recorded buffer");
    if(p.hasSize){uint64_t v=0;for(int b=0;b<8;b++){uint64_t o=p.sizeOffset+b;v|=(uint64_t)(uint8_t)page(o/kPage)[(size_t)(o%kPage)]<<(8*b);}if(v!=p.blobBytes)throw std::runtime_error("Recorded buffer size prefix mismatch");}
    if(hashRange(p.dataOffset,p.blobBytes)!=p.blobSha)throw std::runtime_error("Original selected blob SHA256 mismatch");
    if(hashRange(p.dataOffset+p.byteOffset,p.bytes.size())!=p.rangeSha)throw std::runtime_error("Original selected byte range SHA256 mismatch");
    if(sha256(p.bytes.data(),p.bytes.size())!=p.replacementSha)throw std::runtime_error("Replacement bytes SHA256 mismatch");
    uint64_t s=p.dataOffset+p.byteOffset,e=s+p.bytes.size();for(auto& r:ranges)if(s<r.second && e>r.first)throw std::runtime_error("Overlapping patches");ranges.push_back({s,e});
  }
  tm["verify"]=msSince(tv);
  std::string replayPath=originalPath,candidateSha;bool wrote=false;uint64_t changedPages=0,reencodedPages=0,candidateBytes=raw.size();
  if(!patches.empty()){
    if(!candidate.is_absolute() || !std::filesystem::is_directory(candidate.parent_path()) || std::filesystem::exists(candidate))throw std::runtime_error("Fresh absolute candidate RDC path required");
    if(!fs.lz4)throw std::runtime_error("Uncompressed frame sections: in-place splice not implemented in this build");
    auto ts=Clock::now();std::set<uint64_t> touched;
    for(auto& p:patches){uint64_t s=p.dataOffset+p.byteOffset,e=s+p.bytes.size();
      uint64_t off=s;size_t src=0;while(off<e){uint64_t k=off/kPage,in=off%kPage;touched.insert(k);auto& pg=page(k);uint64_t c=std::min<uint64_t>(e-off,pg.size()-in);std::memcpy(pg.data()+in,p.bytes.data()+src,(size_t)c);off+=c;src+=(size_t)c;}}
    changedPages=touched.size();
    std::set<uint64_t> enc;for(auto k:touched){enc.insert(k);if(k+1<fs.pages())enc.insert(k+1);}
    for(auto k:enc)page(k);
    reencodedPages=enc.size();std::map<uint64_t,std::vector<char>> blocks;
    // history used by the official decoder for page k = (possibly patched) decoded page k-1
    auto history=[&](uint64_t k,const char*& d,int& n){auto it=pages.find(k-1);if(it!=pages.end()){size_t t=std::min<size_t>(it->second.size(),65536);d=it->second.data()+it->second.size()-t;n=(int)t;}else{d=fs.tails[(size_t)k-1].data();n=(int)fs.tails[(size_t)k-1].size();}};
    LZ4_stream_t* stream=LZ4_createStream();if(!stream)throw std::runtime_error("LZ4 stream alloc");
    struct StreamFree{LZ4_stream_t* s;~StreamFree(){LZ4_freeStream(s);}} streamFree{stream};
    for(auto k:enc){
      auto& pg=page(k);std::vector<char> dst((size_t)LZ4_COMPRESSBOUND((int)kPage));const char* dict=nullptr;int dictSize=0;
      LZ4_resetStream_fast(stream);if(k){history(k,dict,dictSize);LZ4_loadDict(stream,dict,dictSize);}
      int cs=LZ4_compress_fast_continue(stream,pg.data(),dst.data(),(int)pg.size(),(int)dst.size(),20);
      if(cs<=0)throw std::runtime_error("LZ4 re-encode failed");dst.resize((size_t)cs);
      std::vector<char> check(pg.size());
      int got=k?LZ4_decompress_safe_usingDict(dst.data(),check.data(),cs,(int)kPage,dict,dictSize):LZ4_decompress_safe(dst.data(),check.data(),cs,(int)kPage);
      if(got!=(int)pg.size() || std::memcmp(check.data(),pg.data(),pg.size()))throw std::runtime_error("LZ4 re-encode self-check failed");
      blocks[k]=std::move(dst);
    }
    tm["patch_and_reencode"]=msSince(ts);
    auto tw=Clock::now();uint64_t newCompressed=fs.sec.compressed;
    for(auto& kv:blocks)newCompressed=newCompressed-(4+(uint64_t)fs.blocks[(size_t)kv.first].compSize)+(4+(uint64_t)kv.second.size());
    {
      std::ofstream o(candidate,std::ios::binary);if(!o)throw std::runtime_error("Candidate RDC create failed");Sha256 h;
      auto put=[&](const void* d,uint64_t n){o.write((const char*)d,(std::streamsize)n);h.update(d,n);};
      std::vector<uint8_t> head(raw.begin(),raw.begin()+(size_t)fs.sec.dataOffset);std::memcpy(head.data()+fs.sec.headerOffset+8,&newCompressed,8);put(head.data(),head.size());
      uint64_t k=0,np=fs.pages();
      while(k<np){
        auto it=blocks.find(k);
        if(it!=blocks.end()){int32_t cs=(int32_t)it->second.size();put(&cs,4);put(it->second.data(),it->second.size());k++;continue;}
        uint64_t k2=k;while(k2<np && !blocks.count(k2))k2++;
        uint64_t from=fs.blocks[(size_t)k].fileOffset,to=(k2<np)?fs.blocks[(size_t)k2].fileOffset:fs.sec.dataOffset+fs.sec.compressed;put(raw.data()+from,to-from);k=k2;
      }
      uint64_t rest=fs.sec.dataOffset+fs.sec.compressed;put(raw.data()+rest,raw.size()-rest);
      o.close();if(!o)throw std::runtime_error("Candidate RDC write failed");candidateSha=h.hex();
    }
    candidateBytes=std::filesystem::file_size(candidate);tm["write_candidate_rdc"]=msSince(tw);replayPath=candidate.u8string();wrote=true;
  }
  js<<"{\"event\":\"job_result\",\"job\":"<<quoteJson(id)<<",\"replayed_rdc\":"<<quoteJson(replayPath)<<",\"candidate_written\":"<<(wrote?"true":"false")
    <<",\"candidate_rdc_sha256\":"<<(wrote?quoteJson(candidateSha):std::string("null"))<<",\"candidate_rdc_bytes\":"<<candidateBytes<<",\"changed_pages\":"<<changedPages<<",\"reencoded_pages\":"<<reencodedPages<<",\"targets\":[";
  {
    auto to=Clock::now();FileOwner file;file.p=RENDERDOC_OpenCaptureFile();if(!file.p)throw std::runtime_error("Official capture factory returned null");
    require(file.p->OpenFile(replayPath.c_str(),"rdc",nullptr),"OpenFile");
    if(file.p->DriverName()!="D3D11")throw std::runtime_error("Replayed driver is not D3D11");tm["official_open_file"]=msSince(to);
    auto tc=Clock::now();ReplayOptions options;options.optimisation=ReplayOptimisationLevel::NoOptimisation;
    ReplayOwner replay;auto opened=file.p->OpenCapture(options,nullptr);require(opened.first,"OpenCapture");replay.p=opened.second;if(!replay.p)throw std::runtime_error("Official replay returned null");
    auto api=replay.p->GetAPIProperties();if(api.pipelineType!=GraphicsAPI::D3D11 || api.remoteReplay)throw std::runtime_error("Replay must be actual local D3D11");
    tm["official_open_capture"]=msSince(tc);
    std::set<uint32_t> events;collectEvents(replay.p->GetRootActions(),events);
    std::map<uint64_t,TextureDescription> textures;for(const auto& t:replay.p->GetTextures())textures[idValue(t.resourceId)]=t;
    std::map<uint64_t,BufferDescription> buffers;for(const auto& b:replay.p->GetBuffers())buffers[idValue(b.resourceId)]=b;
    for(const auto& t:targets){if(!events.count(t.event))throw std::runtime_error("Target event is not an actual API event: "+std::to_string(t.event));
      if(t.kind=="texture" && !textures.count(t.resource))throw std::runtime_error("Target texture absent: "+std::to_string(t.resource));
      if(t.kind=="buffer" && !buffers.count(t.resource))throw std::runtime_error("Target buffer absent: "+std::to_string(t.resource));}
    uint32_t current=0;bool haveCurrent=false;double setMs=0,readMs=0,writeMs=0;
    for(size_t i=0;i<targets.size();i++){
      const auto& t=targets[i];
      auto ts=Clock::now();if(!haveCurrent || current!=t.event){replay.p->SetFrameEvent(t.event,true);current=t.event;haveCurrent=true;}setMs+=msSince(ts);
      auto tr=Clock::now();bytebuf data;std::ostringstream extra;
      if(t.kind=="texture"){
        const auto& d=textures[t.resource];
        if(d.msSamp!=1)throw std::runtime_error("MSAA read-back is not supported by this pipeline");
        uint32_t layers=d.dimension==3?std::max(1u,d.depth>>t.mip):d.arraysize;
        if(t.mip>=d.mips || t.slice>=layers)throw std::runtime_error("Target mip/slice out of range");
        Subresource sub;sub.mip=t.mip;sub.slice=t.slice;sub.sample=0;data=replay.p->GetTextureData(d.resourceId,sub);
        extra<<",\"width\":"<<std::max(1u,d.width>>t.mip)<<",\"height\":"<<std::max(1u,d.height>>t.mip)<<",\"depth\":"<<d.depth<<",\"dimension\":"<<d.dimension<<",\"array_size\":"<<d.arraysize<<",\"mips\":"<<d.mips
             <<",\"format_type\":"<<(uint32_t)d.format.type<<",\"component_type\":"<<(uint32_t)d.format.compType<<",\"component_count\":"<<(uint32_t)d.format.compCount<<",\"component_bytes\":"<<(uint32_t)d.format.compByteWidth
             <<",\"bgra_order\":"<<(d.format.BGRAOrder()?"true":"false")<<",\"srgb_corrected\":"<<(d.format.SRGBCorrected()?"true":"false")<<",\"mip\":"<<t.mip<<",\"slice\":"<<t.slice;
      }else{
        const auto& b=buffers[t.resource];if(t.offset+t.length>b.length)throw std::runtime_error("Buffer read-back range outside resource");
        data=replay.p->GetBufferData(b.resourceId,t.offset,t.length);extra<<",\"buffer_length\":"<<b.length<<",\"offset\":"<<t.offset<<",\"requested_length\":"<<t.length;
      }
      readMs+=msSince(tr);if(data.empty())throw std::runtime_error("Official read-back empty for target "+t.name);
      auto tw2=Clock::now();{std::ofstream o(std::filesystem::u8path(t.out),std::ios::binary);o.write((const char*)data.data(),data.size());o.close();if(!o)throw std::runtime_error("Target write failed");}writeMs+=msSince(tw2);
      js<<(i?",":"")<<"{\"name\":"<<quoteJson(t.name)<<",\"kind\":"<<quoteJson(t.kind)<<",\"event_id\":"<<t.event<<",\"resource_id\":"<<t.resource<<",\"raw\":"<<quoteJson(t.out)<<",\"bytes\":"<<data.size()<<",\"sha256\":\""<<sha256(data.data(),data.size())<<"\""<<extra.str()<<"}";
    }
    tm["set_frame_event"]=setMs;tm["readback"]=readMs;tm["write_targets"]=writeMs;
    auto tsd=Clock::now();replay.p->Shutdown();replay.p=nullptr;file.p->Shutdown();file.p=nullptr;tm["shutdown"]=msSince(tsd);
  }
  if(wrote && !keep){auto td=Clock::now();std::filesystem::remove(candidate);tm["delete_candidate"]=msSince(td);}
  js<<"],\"candidate_retained\":"<<((wrote&&keep)?"true":"false")<<",\"timing_ms\":{";bool first=true;for(auto& kv:tm){js<<(first?"":",")<<quoteJson(kv.first)<<":"<<kv.second;first=false;}
  js<<",\"total\":"<<msSince(t0)<<"},\"candidate_geometry_complete\":false}";
  return js.str();
}
static int sessionMode(const char* path,const char* expectedRdcSha,const char* expectedSectionSha){
  auto t0=Clock::now();auto raw=readAll(std::filesystem::u8path(path));double readMs=msSince(t0);
  auto th=Clock::now();auto fileSha=sha256(raw.data(),raw.size());double hashMs=msSince(th);
  if(fileSha!=expectedRdcSha)throw std::runtime_error("Original RDC SHA256 differs from the index");
  auto cont=parseContainer(raw);if(cont.driverName!="D3D11")throw std::runtime_error("Container driver is not D3D11");
  FrameStore fs;auto td=Clock::now();std::string sectionSha;bool replayOnly=false;
  if(cont.sections[(size_t)cont.frame].flags==(uint32_t)SectionFlags::ZstdCompressed){
    // e.g. the official zip.xml->rdc conversion output: replay/read-back only, verified through the official decoder
    replayOnly=true;FileOwner file;file.p=RENDERDOC_OpenCaptureFile();if(!file.p)throw std::runtime_error("Official capture factory returned null");require(file.p->OpenFile(path,"rdc",nullptr),"OpenFile");
    auto data=file.p->GetSectionContents(cont.frame);sectionSha=sha256(data.data(),data.size());fs.sec=cont.sections[(size_t)cont.frame];fs.file=&raw;
  }else sectionSha=loadFrame(fs,raw,cont.sections[(size_t)cont.frame]);
  double decodeMs=msSince(td);
  if(sectionSha!=expectedSectionSha)throw std::runtime_error("Decoded frame section SHA256 differs from the official index");
  fs.replayOnly=replayOnly;
  std::cout<<"{\"event\":\"ready\",\"rdc_sha256\":\""<<fileSha<<"\",\"frame_section_sha256\":\""<<sectionSha<<"\",\"lz4\":"<<(fs.lz4?"true":"false")<<",\"replay_only\":"<<(replayOnly?"true":"false")<<",\"pages\":"<<fs.pages()
           <<",\"timing_ms\":{\"read\":"<<readMs<<",\"sha256\":"<<hashMs<<",\"decode_and_history_cache\":"<<decodeMs<<"}}"<<std::endl;
  std::string line,id;std::filesystem::path candidate;bool keep=false,inJob=false;std::vector<PatchReq> patches;std::vector<TargetReq> targets;std::string jobError;
  while(std::getline(std::cin,line)){
    auto f=splitTabs(line);if(f.empty() || f[0].empty())continue;
    try{
      if(f[0]=="quit")break;
      if(f[0]=="job"){if(f.size()!=4)throw std::runtime_error("job ID CANDIDATE_RDC KEEP");id=f[1];candidate=std::filesystem::u8path(f[2]);keep=f[3]=="1";patches.clear();targets.clear();inJob=true;jobError.clear();continue;}
      if(!inJob)throw std::runtime_error("Directive outside job");
      if(!jobError.empty() && f[0]!="run")continue;
      if(f[0]=="patch"){if(f.size()!=10 || f[9]!="end")throw std::runtime_error("patch SIZE_OFFSET|- DATA_OFFSET BLOB_BYTES BLOB_SHA BYTE_OFFSET RANGE_SHA PATH REPLACEMENT_SHA end");
        PatchReq p;p.hasSize=f[1]!="-";p.sizeOffset=p.hasSize?std::stoull(f[1]):0;p.dataOffset=std::stoull(f[2]);p.blobBytes=std::stoull(f[3]);p.blobSha=f[4];p.byteOffset=std::stoull(f[5]);p.rangeSha=f[6];p.path=f[7];p.replacementSha=f[8];
        auto bytes=readAll(std::filesystem::u8path(p.path));p.bytes.assign(bytes.begin(),bytes.end());patches.push_back(std::move(p));continue;}
      if(f[0]=="texture"){if(f.size()!=7)throw std::runtime_error("texture NAME EVENT RESOURCE MIP SLICE RAW");TargetReq t;t.kind="texture";t.name=f[1];t.event=(uint32_t)std::stoul(f[2]);t.resource=std::stoull(f[3]);t.mip=(uint32_t)std::stoul(f[4]);t.slice=(uint32_t)std::stoul(f[5]);t.out=f[6];targets.push_back(t);continue;}
      if(f[0]=="buffer"){if(f.size()!=7)throw std::runtime_error("buffer NAME EVENT RESOURCE OFFSET LENGTH RAW");TargetReq t;t.kind="buffer";t.name=f[1];t.event=(uint32_t)std::stoul(f[2]);t.resource=std::stoull(f[3]);t.offset=std::stoull(f[4]);t.length=std::stoull(f[5]);t.out=f[6];targets.push_back(t);continue;}
      if(f[0]=="run"){inJob=false;if(!jobError.empty())throw std::runtime_error(jobError);if(targets.empty())throw std::runtime_error("At least one explicit read-back target required");
        std::cout<<runJob(id,candidate,keep,patches,targets,raw,fs,path)<<std::endl;continue;}
      throw std::runtime_error("Unknown directive "+f[0]);
    }catch(const std::exception& e){
      if(inJob && f[0]!="run"){jobError=e.what();continue;}
      inJob=false;std::cout<<"{\"event\":\"job_rejected\",\"job\":"<<quoteJson(id)<<",\"error\":"<<quoteJson(e.what())<<",\"candidate_rdc_retained_if_written\":"<<quoteJson(candidate.u8string())<<"}"<<std::endl;
    }
  }
  std::cout<<"{\"event\":\"session_end\"}"<<std::endl;return 0;
}
int main(int argc,char** argv){
  try{
    if(argc<3)throw std::runtime_error("convert INPUT TYPE OUTPUT TYPE | inspect RDC | replay RDC EVENT RESOURCE_ID PNG RAW | drawinputs RDC EVENT NEW_DIR | sections RDC | index RDC NEW_INDEX_JSON | session RDC RDC_SHA256 FRAME_SECTION_SHA256");
    std::string mode=argv[1];
    if(mode=="sections" || mode=="index" || mode=="session"){
      if((mode=="sections" && argc!=3)||(mode=="index" && argc!=4)||(mode=="session" && argc!=5))throw std::runtime_error("Invalid explicit mode/arguments");
      if(!std::filesystem::is_regular_file(std::filesystem::u8path(argv[2])))throw std::runtime_error("Actual input capture required");
      const char* commit=RENDERDOC_GetCommitHash();
      if(!commit || std::string(commit)!="e4bd23b671d3d5a747ff5221dbe08a63eb6ca200")throw std::runtime_error("Pinned official v1.46 DLL commit mismatch");
      struct Session{Session(){GlobalEnvironment env;env.enumerateGPUs=false;RENDERDOC_InitialiseReplay(env,{});}~Session(){RENDERDOC_ShutdownReplay();}} session;
      std::cout<<"{\"event\":\"tool\",\"mode\":"<<quoteJson(mode)<<",\"version\":"<<quoteJson(RENDERDOC_GetVersionString())<<",\"commit\":"<<quoteJson(commit)<<"}"<<std::endl;
      if(mode=="sections")return sectionsMode(argv[2]);
      if(mode=="index")return indexMode(argv[2],argv[3]);
      return sessionMode(argv[2],argv[3],argv[4]);
    }
    if((mode=="convert" && argc!=6)||(mode=="inspect" && argc!=3)||(mode=="replay" && argc!=7)||(mode=="drawinputs" && argc!=5)||(mode!="convert" && mode!="inspect" && mode!="replay" && mode!="drawinputs"))throw std::runtime_error("Invalid explicit mode/arguments");
    if(!std::filesystem::is_regular_file(std::filesystem::u8path(argv[2])))throw std::runtime_error("Actual input capture required");
    if(mode=="convert"){
      fresh(argv[4]);
      if(std::string(argv[3])!="rdc" && std::string(argv[3])!="zip.xml")throw std::runtime_error("Input must explicitly be rdc or zip.xml; XML-only cannot be imported");
      if(std::string(argv[5])!="rdc" && std::string(argv[5])!="zip.xml")throw std::runtime_error("Output must explicitly be rdc or complete zip.xml");
      if(std::string(argv[5])=="zip.xml"){auto external=std::filesystem::u8path(argv[4]);external.replace_extension();if(std::filesystem::exists(external))throw std::runtime_error("XML external ZIP output already exists");}
    }
    if(mode=="replay"){fresh(argv[5]);fresh(argv[6]);if(std::filesystem::absolute(std::filesystem::u8path(argv[5]))==std::filesystem::absolute(std::filesystem::u8path(argv[6])))throw std::runtime_error("PNG and official readback output paths must differ");}
    if(mode=="drawinputs")fresh(argv[4]);
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
    uint32_t event=(mode=="replay" || mode=="drawinputs")?(uint32_t)std::stoul(argv[3]):0;bool found=false;
    actions(replay.p->GetRootActions(),replay.p->GetStructuredFile(),event,found);
    if(mode=="drawinputs"){
      auto action=findAction(replay.p->GetRootActions(),event);if(!event || !action)throw std::runtime_error("Explicit existing actual draw event required");
      drawInputs(replay.p,*action,argv[4]);std::cout<<"{\"event\":\"draw_inputs_saved\",\"event_id\":"<<event<<",\"directory\":"<<quoteJson(argv[4])<<"}"<<std::endl;return 0;
    }
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
