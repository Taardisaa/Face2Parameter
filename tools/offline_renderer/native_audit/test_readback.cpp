// Analytic WARP raw-resource fixtures. No installed game or Unity invocation.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
static void check(HRESULT hr,const char* where){if(FAILED(hr))throw std::runtime_error(std::string(where)+" HRESULT "+std::to_string((unsigned)hr));}
template<class T>struct Com{T* p=nullptr;~Com(){if(p)p->Release();}T** addr(){return &p;}T* operator->(){return p;}};
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("Usage test_readback native_dll fresh_output");
    std::filesystem::path out=std::filesystem::absolute(argv[2]);
    if(std::filesystem::exists(out))throw std::runtime_error("Fresh fixture directory required");
    std::filesystem::create_directories(out);
    HMODULE library=LoadLibraryW(std::filesystem::absolute(argv[1]).c_str());if(!library)throw std::runtime_error("LoadLibrary failed");
    auto queue=(int(__cdecl*)(ID3D11Resource*,const wchar_t*))GetProcAddress(library,"QueueTexture");
    auto status=(int(__cdecl*)(int,char*,int))GetProcAddress(library,"GetTextureStatus");
    auto getEvent=(void*(__cdecl*)())GetProcAddress(library,"GetRenderEventFunc");
    if(!queue||!status||!getEvent)throw std::runtime_error("Native export signature missing");
    auto event=(void(__stdcall*)(int))getEvent();
    Com<ID3D11Device> device;Com<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL feature;
    check(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,device.addr(),&feature,context.addr()),"WARP create");
    for(bool volume:{false,true}){
        unsigned arrays=volume?1:6,mips=4;
        std::vector<std::vector<unsigned short>> bytes;std::vector<D3D11_SUBRESOURCE_DATA> init;
        bytes.reserve(arrays*mips);init.reserve(arrays*mips);
        std::string prefix=volume?"volume":"cube";
        std::filesystem::create_directories(out/(prefix+"_expected"));
        for(unsigned face=0;face<arrays;face++)for(unsigned mip=0;mip<mips;mip++){
            unsigned w=std::max(1u,8u>>mip),h=w,d=volume?std::max(1u,4u>>mip):1,sub=D3D11CalcSubresource(mip,face,mips);
            bytes.emplace_back(w*h*d*4);
            auto& row=bytes.back();for(size_t i=0;i<row.size();i++)row[i]=(unsigned short)(sub*500+i);
            D3D11_SUBRESOURCE_DATA data{};data.pSysMem=row.data();data.SysMemPitch=w*8;data.SysMemSlicePitch=w*h*8;init.push_back(data);
            std::ofstream expected(out/(prefix+"_expected")/("subresource_"+std::to_string(sub)+".bin"),std::ios::binary);
            expected.write((char*)row.data(),row.size()*2);
        }
        Com<ID3D11Resource> texture;
        if(volume){D3D11_TEXTURE3D_DESC d{};d.Width=8;d.Height=8;d.Depth=4;d.MipLevels=mips;d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            ID3D11Texture3D* t=nullptr;check(device->CreateTexture3D(&d,init.data(),&t),"Create fixture volume");texture.p=t;
        }else{D3D11_TEXTURE2D_DESC d{};d.Width=8;d.Height=8;d.MipLevels=mips;d.ArraySize=6;d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;d.SampleDesc.Count=1;d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;d.MiscFlags=D3D11_RESOURCE_MISC_TEXTURECUBE;
            ID3D11Texture2D* t=nullptr;check(device->CreateTexture2D(&d,init.data(),&t),"Create fixture cube");texture.p=t;
        }
        auto path=out/prefix;int job=queue(texture.p,path.c_str());if(job<0)throw std::runtime_error("Queue fixture failed");event(1);
        char error[2048];if(status(job,error,sizeof(error))!=1)throw std::runtime_error(error);
        // Read the original a second time: no resource mutation is permitted.
        auto again=out/(prefix+"_again");job=queue(texture.p,again.c_str());event(1);
        if(status(job,error,sizeof(error))!=1)throw std::runtime_error(error);
    }
    std::cout<<"Raw HDR cube and 3D mip fixtures exported twice from WARP\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}
