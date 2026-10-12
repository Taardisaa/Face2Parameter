// Raw native texture capture scheduled on Unity's render thread.
// GetNativeTexturePtr is ID3D11Resource on Unity 2018 D3D11. Resources are
// AddRef'ed at queue time, copied only to a staging resource, never modified.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <memory>
#include <mutex>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>

template<class T> struct Com{T* p=nullptr;~Com(){if(p)p->Release();}T** addr(){return &p;}T* operator->(){return p;}};
static void check(HRESULT hr,const char* name){if(FAILED(hr))throw std::runtime_error(std::string(name)+" HRESULT "+std::to_string((unsigned)hr));}
struct Job{ID3D11Resource* source=nullptr;std::filesystem::path path;int state=0;std::string error;~Job(){if(source)source->Release();}};
static std::mutex mutex;
static std::vector<std::unique_ptr<Job>> jobs;
static unsigned pixelBytes(DXGI_FORMAT f){
    unsigned n=(unsigned)f;
    if(n>=1&&n<=4)return 16;if(n>=5&&n<=8)return 12;
    if(n>=9&&n<=22)return 8;if(n>=23&&n<=47)return 4;
    if(n>=48&&n<=59)return 2;if(n>=60&&n<=66)return 1;
    if(n==85||n==86)return 2;if(n>=87&&n<=93)return 4;
    throw std::runtime_error("Unsupported raw DXGI texel format "+std::to_string(n));
}
static bool blockFormat(unsigned f){return (f>=70&&f<=84)||(f>=94&&f<=99);}
static unsigned blockBytes(unsigned f){return (f>=70&&f<=72)||(f>=79&&f<=81)?8:16;}
static std::string stamp(){SYSTEMTIME s;GetSystemTime(&s);char b[50];sprintf(b,"%04u-%02u-%02uT%02u:%02u:%02u.%03uZ",s.wYear,s.wMonth,s.wDay,s.wHour,s.wMinute,s.wSecond,s.wMilliseconds);return b;}
static void capture(Job& job){
    if(std::filesystem::exists(job.path))throw std::runtime_error("Fresh raw resource output directory required");
    D3D11_RESOURCE_DIMENSION dimension;job.source->GetType(&dimension);
    Com<ID3D11Device> device;job.source->GetDevice(device.addr());
    Com<ID3D11DeviceContext> context;device->GetImmediateContext(context.addr());
    Com<ID3D11Resource> staging;Com<ID3D11Texture2D> resolved;
    unsigned width=0,height=1,depth=1,mips=0,arrays=1,samples=1,misc=0,bind=0;
    DXGI_FORMAT format=DXGI_FORMAT_UNKNOWN;
    if(dimension==D3D11_RESOURCE_DIMENSION_TEXTURE2D){
        Com<ID3D11Texture2D> source;check(job.source->QueryInterface(IID_ID3D11Texture2D,(void**)source.addr()),"QI Texture2D");
        D3D11_TEXTURE2D_DESC d;source->GetDesc(&d);width=d.Width;height=d.Height;mips=d.MipLevels;arrays=d.ArraySize;samples=d.SampleDesc.Count;format=d.Format;misc=d.MiscFlags;bind=d.BindFlags;
        ID3D11Resource* copy=job.source;
        if(samples>1){
            if(d.ArraySize!=1||d.MipLevels!=1||(bind&D3D11_BIND_DEPTH_STENCIL))throw std::runtime_error("MSAA depth/array raw capture not implemented; not silently resolved");
            auto r=d;r.SampleDesc.Count=1;r.SampleDesc.Quality=0;r.Usage=D3D11_USAGE_DEFAULT;r.CPUAccessFlags=0;r.BindFlags=0;r.MiscFlags=0;
            check(device->CreateTexture2D(&r,nullptr,resolved.addr()),"Create resolve Texture2D");
            UINT support=0;check(device->CheckFormatSupport(format,&support),"Format resolve support");
            if(!(support&D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE))throw std::runtime_error("Source format cannot resolve MSAA");
            context->ResolveSubresource(resolved.p,0,source.p,0,format);copy=resolved.p;
        }
        d.SampleDesc.Count=1;d.SampleDesc.Quality=0;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;d.BindFlags=0;d.MiscFlags=0;
        ID3D11Texture2D* texture=nullptr;check(device->CreateTexture2D(&d,nullptr,&texture),"Create staging Texture2D");staging.p=texture;context->CopyResource(staging.p,copy);
    }else if(dimension==D3D11_RESOURCE_DIMENSION_TEXTURE3D){
        Com<ID3D11Texture3D> source;check(job.source->QueryInterface(IID_ID3D11Texture3D,(void**)source.addr()),"QI Texture3D");
        D3D11_TEXTURE3D_DESC d;source->GetDesc(&d);width=d.Width;height=d.Height;depth=d.Depth;mips=d.MipLevels;format=d.Format;misc=d.MiscFlags;bind=d.BindFlags;
        d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;d.BindFlags=0;d.MiscFlags=0;
        ID3D11Texture3D* texture=nullptr;check(device->CreateTexture3D(&d,nullptr,&texture),"Create staging Texture3D");staging.p=texture;context->CopyResource(staging.p,source.p);
    }else throw std::runtime_error("Unsupported native resource dimension "+std::to_string((unsigned)dimension));
    if(!width||!height||!mips)throw std::runtime_error("Empty resource descriptor");
    std::filesystem::create_directories(job.path);
    std::ofstream manifest(job.path/"descriptor.json");
    manifest<<"{\"schema_version\":1,\"captured_utc\":\""<<stamp()<<"\",\"dimension\":"<<(unsigned)dimension
        <<",\"width\":"<<width<<",\"height\":"<<height<<",\"depth\":"<<depth<<",\"mip_levels\":"<<mips
        <<",\"array_size\":"<<arrays<<",\"dxgi_format\":"<<(unsigned)format<<",\"source_sample_count\":"<<samples
        <<",\"source_misc_flags\":"<<misc<<",\"source_bind_flags\":"<<bind
        <<",\"msaa_resolved\":"<<(samples>1?"true":"false")<<",\"subresources\":[";
    bool first=true;
    for(unsigned array=0;array<arrays;array++)for(unsigned mip=0;mip<mips;mip++){
        unsigned w=std::max(1u,width>>mip),h=std::max(1u,height>>mip),z=std::max(1u,depth>>mip);
        unsigned sub=D3D11CalcSubresource(mip,array,mips),rows=h,pitch;
        if(blockFormat((unsigned)format)){pitch=std::max(1u,(w+3)/4)*blockBytes((unsigned)format);rows=std::max(1u,(h+3)/4);}
        else if((unsigned)format==67)pitch=(w+7)/8;
        else pitch=w*pixelBytes(format);
        D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(staging.p,sub,D3D11_MAP_READ,0,&mapped),"Map staging mip");
        try{
            if(mapped.RowPitch<pitch||(z>1&&mapped.DepthPitch<mapped.RowPitch*rows))throw std::runtime_error("Mapped pitches insufficient");
            std::string name="subresource_"+std::to_string(sub)+".bin";
            std::ofstream data(job.path/name,std::ios::binary);
            for(unsigned layer=0;layer<z;layer++)for(unsigned row=0;row<rows;row++)
                data.write((char*)mapped.pData+layer*mapped.DepthPitch+row*mapped.RowPitch,pitch);
            if(!data)throw std::runtime_error("Failed raw mip write");
            if(!first)manifest<<",";first=false;
            manifest<<"{\"index\":"<<sub<<",\"array_slice\":"<<array<<",\"mip\":"<<mip
                <<",\"width\":"<<w<<",\"height\":"<<h<<",\"depth\":"<<z<<",\"row_pitch\":"<<pitch
                <<",\"slice_pitch\":"<<pitch*rows<<",\"bytes\":"<<(uint64_t)pitch*rows*z<<",\"path\":\""<<name<<"\"}";
        }catch(...){context->Unmap(staging.p,sub);throw;}
        context->Unmap(staging.p,sub);
    }
    manifest<<"],\"scope\":\"Raw resource bytes on render thread; no Blit, format/color conversion, mip regeneration or source mutation. Shader resource view gamma/typeless interpretation must be recorded separately.\"}";
    if(!manifest)throw std::runtime_error("Failed descriptor write");
}
static void __stdcall renderEvent(int){
    std::lock_guard<std::mutex> lock(mutex);
    for(auto& job:jobs)if(job->state==0){
        try{capture(*job);job->state=1;}catch(const std::exception& e){job->error=e.what();job->state=2;}
        if(job->source){job->source->Release();job->source=nullptr;}
    }
}
extern "C" __declspec(dllexport) void* __cdecl GetRenderEventFunc(){return (void*)renderEvent;}
extern "C" __declspec(dllexport) int __cdecl QueueTexture(ID3D11Resource* resource,const wchar_t* path){
    if(!resource||!path)return -1;
    std::lock_guard<std::mutex> lock(mutex);
    auto job=std::make_unique<Job>();job->source=resource;resource->AddRef();job->path=path;
    jobs.push_back(std::move(job));return (int)jobs.size()-1;
}
extern "C" __declspec(dllexport) int __cdecl GetTextureStatus(int id,char* error,int capacity){
    std::lock_guard<std::mutex> lock(mutex);
    if(id<0||id>=(int)jobs.size())return -1;
    if(error&&capacity>0){auto& text=jobs[id]->error;size_t n=std::min(text.size(),(size_t)capacity-1);memcpy(error,text.data(),n);error[n]=0;}
    return jobs[id]->state;
}
