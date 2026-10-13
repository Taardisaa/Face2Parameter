// Independent D3D11 WARP pass executor. Runs original installed DXBC shaders.
// No Unity/game process, CUDA allocation, substituted AIT BRDF or light fitting.
// Packet paths are relative to a caller-created fresh packet directory.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#ifdef HS2_RENDERDOC_SUPPORT
#include "renderdoc_app.h"
#endif

static void check(HRESULT hr,const char* where){
    if(FAILED(hr)) throw std::runtime_error(std::string(where)+" HRESULT "+std::to_string((unsigned)hr));
}
template<class T> struct Com {
    T* p=nullptr;
    ~Com(){if(p)p->Release();}
    T** address(){return &p;}
    T* operator->(){return p;}
};
static std::vector<char> read(const std::filesystem::path& path){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f)throw std::runtime_error("Missing input "+path.string());
    auto length=f.tellg(); if(length<0)throw std::runtime_error("Invalid input size");
    std::vector<char> data((size_t)length);f.seekg(0);f.read(data.data(),length);
    if(!f)throw std::runtime_error("Short input "+path.string());
    return data;
}
static ID3D11Buffer* buffer(ID3D11Device* device,const std::vector<char>& bytes,UINT flags){
    if(bytes.empty())throw std::runtime_error("Empty buffer");
    D3D11_BUFFER_DESC desc{};desc.ByteWidth=(UINT)bytes.size();desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=flags;
    D3D11_SUBRESOURCE_DATA data{};data.pSysMem=bytes.data();
    ID3D11Buffer* out=nullptr;check(device->CreateBuffer(&desc,&data,&out),"CreateBuffer");return out;
}
static D3D11_TEXTURE_ADDRESS_MODE address(int v){
    if(v<1||v>5)throw std::runtime_error("Explicit D3D texture address required");
    return (D3D11_TEXTURE_ADDRESS_MODE)v;
}
int main(int argc,char** argv){
    try{
        if(argc!=2)throw std::runtime_error("Usage: d3d11_pass packet_directory");
        std::filesystem::path root=std::filesystem::absolute(argv[1]);
        std::ifstream packet(root/"packet.txt");if(!packet)throw std::runtime_error("Missing packet.txt");
        UINT width,height,samples,vertexCount,stride,indexCount;
        int cull,depthEnable,depthWrite,depthFunction,blendEnable,srcBlend,dstBlend,blendOperation;
        packet>>width>>height>>samples>>vertexCount>>stride>>indexCount;
        packet>>cull>>depthEnable>>depthWrite>>depthFunction>>blendEnable>>srcBlend>>dstBlend>>blendOperation;
        float clear[4];for(float& c:clear)packet>>c;
        if(!packet||!width||!height||!samples||!stride||!vertexCount||!indexCount)throw std::runtime_error("Incomplete pass packet");
        if(std::filesystem::exists(root/"rgba32f.bin"))throw std::runtime_error("Fresh output required");
        Com<ID3D11Device> device;Com<ID3D11DeviceContext> context;
        D3D_FEATURE_LEVEL feature;
        check(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,
            device.address(),&feature,context.address()),"D3D11CreateDevice WARP");
#ifdef HS2_RENDERDOC_SUPPORT
        RENDERDOC_API_1_6_0* captureApi=nullptr;
        if(auto module=GetModuleHandleA("renderdoc.dll")){
            auto getApi=(pRENDERDOC_GetAPI)GetProcAddress(module,"RENDERDOC_GetAPI");
            if(!getApi||!getApi(eRENDERDOC_API_Version_1_6_0,(void**)&captureApi))throw std::runtime_error("RenderDoc SDK API unavailable");
            captureApi->StartFrameCapture(device.p,nullptr);
        }
#endif
        auto vsBytes=read(root/"vertex.dxbc"),psBytes=read(root/"pixel.dxbc");
        Com<ID3D11VertexShader> vs;Com<ID3D11PixelShader> ps;
        check(device->CreateVertexShader(vsBytes.data(),vsBytes.size(),nullptr,vs.address()),"CreateVertexShader");
        check(device->CreatePixelShader(psBytes.data(),psBytes.size(),nullptr,ps.address()),"CreatePixelShader");
        Com<ID3D11ShaderReflection> reflection;
        check(D3DReflect(vsBytes.data(),vsBytes.size(),IID_ID3D11ShaderReflection,(void**)reflection.address()),"D3DReflect");
        D3D11_SHADER_DESC shader{};check(reflection->GetDesc(&shader),"Shader GetDesc");
        std::vector<D3D11_INPUT_ELEMENT_DESC> elements;
        for(UINT i=0;i<shader.InputParameters;i++){
            D3D11_SIGNATURE_PARAMETER_DESC parameter{};check(reflection->GetInputParameterDesc(i,&parameter),"Input signature");
            if(parameter.SystemValueType!=D3D_NAME_UNDEFINED)continue;
            DXGI_FORMAT format=parameter.ComponentType==D3D_REGISTER_COMPONENT_FLOAT32?DXGI_FORMAT_R32G32B32A32_FLOAT:
                parameter.ComponentType==D3D_REGISTER_COMPONENT_UINT32?DXGI_FORMAT_R32G32B32A32_UINT:DXGI_FORMAT_R32G32B32A32_SINT;
            if(parameter.Register*16+16>stride)throw std::runtime_error("Vertex register exceeds explicit stride");
            elements.push_back({parameter.SemanticName,parameter.SemanticIndex,format,0,parameter.Register*16,
                               D3D11_INPUT_PER_VERTEX_DATA,0});
        }
        Com<ID3D11InputLayout> layout;
        check(device->CreateInputLayout(elements.data(),(UINT)elements.size(),vsBytes.data(),vsBytes.size(),layout.address()),"CreateInputLayout");
        auto vertices=read(root/"vertices.bin"),indices=read(root/"indices.bin");
        if(vertices.size()!=size_t(vertexCount)*stride||indices.size()!=size_t(indexCount)*4)throw std::runtime_error("Geometry packet length mismatch");
        Com<ID3D11Buffer> vertex,index;vertex.p=buffer(device.p,vertices,D3D11_BIND_VERTEX_BUFFER);index.p=buffer(device.p,indices,D3D11_BIND_INDEX_BUFFER);
        context->IASetInputLayout(layout.p);UINT offset=0;context->IASetVertexBuffers(0,1,&vertex.p,&stride,&offset);
        context->IASetIndexBuffer(index.p,DXGI_FORMAT_R32_UINT,0);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.p,nullptr,0);context->PSSetShader(ps.p,nullptr,0);
        std::vector<ID3D11Buffer*> ownedBuffers;
        for(int stage=0;stage<2;stage++){
            UINT count;packet>>count;
            for(UINT i=0;i<count;i++){
                UINT slot;std::string path;packet>>slot>>path;
                auto data=read(root/path);if(data.size()%16)throw std::runtime_error("CBuffer not float4 aligned");
                auto* b=buffer(device.p,data,D3D11_BIND_CONSTANT_BUFFER);ownedBuffers.push_back(b);
                if(stage==0)context->VSSetConstantBuffers(slot,1,&b);else context->PSSetConstantBuffers(slot,1,&b);
            }
        }
        UINT textureCount;packet>>textureCount;
        std::vector<ID3D11Texture2D*> ownedTextures;std::vector<ID3D11ShaderResourceView*> ownedViews;std::vector<ID3D11SamplerState*> ownedSamplers;
        for(UINT i=0;i<textureCount;i++){
            UINT slot,samplerSlot,tw,th,mips,arraySize,format,filter,aniso;int wrapU,wrapV,wrapW;
            packet>>slot>>samplerSlot>>tw>>th>>mips>>arraySize>>format>>filter>>aniso>>wrapU>>wrapV>>wrapW;
            if(!tw||!th||!mips||(arraySize!=1&&arraySize!=6))throw std::runtime_error("Explicit 2D or cube texture descriptor required");
            std::vector<std::vector<char>> storage;std::vector<D3D11_SUBRESOURCE_DATA> init;
            storage.reserve(arraySize*mips);init.reserve(arraySize*mips);
            for(UINT face=0;face<arraySize;face++)for(UINT mip=0;mip<mips;mip++){
                std::string path;UINT pitch;packet>>path>>pitch;storage.push_back(read(root/path));
                D3D11_SUBRESOURCE_DATA data{};data.pSysMem=storage.back().data();data.SysMemPitch=pitch;data.SysMemSlicePitch=(UINT)storage.back().size();init.push_back(data);
            }
            D3D11_TEXTURE2D_DESC desc{};desc.Width=tw;desc.Height=th;desc.MipLevels=mips;desc.ArraySize=arraySize;
            desc.Format=(DXGI_FORMAT)format;desc.SampleDesc.Count=1;desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            desc.MiscFlags=arraySize==6?D3D11_RESOURCE_MISC_TEXTURECUBE:0;
            ID3D11Texture2D* texture=nullptr;check(device->CreateTexture2D(&desc,init.data(),&texture),"CreateTexture2D input");ownedTextures.push_back(texture);
            ID3D11ShaderResourceView* view=nullptr;check(device->CreateShaderResourceView(texture,nullptr,&view),"CreateShaderResourceView");ownedViews.push_back(view);context->PSSetShaderResources(slot,1,&view);
            D3D11_SAMPLER_DESC sampler{};sampler.Filter=(D3D11_FILTER)filter;sampler.AddressU=address(wrapU);sampler.AddressV=address(wrapV);sampler.AddressW=address(wrapW);
            sampler.MaxAnisotropy=aniso;sampler.ComparisonFunc=D3D11_COMPARISON_NEVER;sampler.MinLOD=0;sampler.MaxLOD=D3D11_FLOAT32_MAX;
            ID3D11SamplerState* state=nullptr;check(device->CreateSamplerState(&sampler,&state),"CreateSamplerState");ownedSamplers.push_back(state);context->PSSetSamplers(samplerSlot,1,&state);
        }
        if(!packet)throw std::runtime_error("Truncated pass packet");
        D3D11_TEXTURE2D_DESC colorDesc{};colorDesc.Width=width;colorDesc.Height=height;colorDesc.MipLevels=1;colorDesc.ArraySize=1;
        colorDesc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;colorDesc.SampleDesc.Count=samples;colorDesc.BindFlags=D3D11_BIND_RENDER_TARGET;colorDesc.Usage=D3D11_USAGE_DEFAULT;
        Com<ID3D11Texture2D> color;check(device->CreateTexture2D(&colorDesc,nullptr,color.address()),"CreateTexture2D target");
        Com<ID3D11RenderTargetView> target;check(device->CreateRenderTargetView(color.p,nullptr,target.address()),"CreateRenderTargetView");
        D3D11_TEXTURE2D_DESC depthDesc=colorDesc;depthDesc.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;depthDesc.BindFlags=D3D11_BIND_DEPTH_STENCIL;
        Com<ID3D11Texture2D> depth;check(device->CreateTexture2D(&depthDesc,nullptr,depth.address()),"CreateTexture2D depth");
        Com<ID3D11DepthStencilView> depthView;check(device->CreateDepthStencilView(depth.p,nullptr,depthView.address()),"CreateDepthStencilView");
        context->OMSetRenderTargets(1,&target.p,depthView.p);context->ClearRenderTargetView(target.p,clear);context->ClearDepthStencilView(depthView.p,D3D11_CLEAR_DEPTH,1,0);
        D3D11_RASTERIZER_DESC rasterDesc{};rasterDesc.FillMode=D3D11_FILL_SOLID;rasterDesc.CullMode=(D3D11_CULL_MODE)cull;
        rasterDesc.FrontCounterClockwise=FALSE;rasterDesc.DepthClipEnable=TRUE;rasterDesc.MultisampleEnable=samples>1;
        Com<ID3D11RasterizerState> raster;check(device->CreateRasterizerState(&rasterDesc,raster.address()),"CreateRasterizerState");context->RSSetState(raster.p);
        D3D11_DEPTH_STENCIL_DESC depthState{};depthState.DepthEnable=depthEnable;depthState.DepthWriteMask=depthWrite?D3D11_DEPTH_WRITE_MASK_ALL:D3D11_DEPTH_WRITE_MASK_ZERO;
        depthState.DepthFunc=(D3D11_COMPARISON_FUNC)depthFunction;
        Com<ID3D11DepthStencilState> ds;check(device->CreateDepthStencilState(&depthState,ds.address()),"CreateDepthStencilState");context->OMSetDepthStencilState(ds.p,0);
        D3D11_BLEND_DESC blend{};auto& rt=blend.RenderTarget[0];rt.BlendEnable=blendEnable;rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
        rt.SrcBlend=(D3D11_BLEND)srcBlend;rt.DestBlend=(D3D11_BLEND)dstBlend;rt.BlendOp=(D3D11_BLEND_OP)blendOperation;
        rt.SrcBlendAlpha=D3D11_BLEND_ONE;rt.DestBlendAlpha=D3D11_BLEND_ZERO;rt.BlendOpAlpha=D3D11_BLEND_OP_ADD;
        Com<ID3D11BlendState> bs;check(device->CreateBlendState(&blend,bs.address()),"CreateBlendState");context->OMSetBlendState(bs.p,nullptr,0xffffffff);
        D3D11_VIEWPORT viewport{0,0,(float)width,(float)height,0,1};context->RSSetViewports(1,&viewport);context->DrawIndexed(indexCount,0,0);
        Com<ID3D11Texture2D> resolved;
        if(samples>1){auto desc=colorDesc;desc.SampleDesc.Count=1;check(device->CreateTexture2D(&desc,nullptr,resolved.address()),"CreateTexture2D resolve");context->ResolveSubresource(resolved.p,0,color.p,0,colorDesc.Format);}
        auto stageDesc=colorDesc;stageDesc.SampleDesc.Count=1;stageDesc.Usage=D3D11_USAGE_STAGING;stageDesc.BindFlags=0;stageDesc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        Com<ID3D11Texture2D> staging;check(device->CreateTexture2D(&stageDesc,nullptr,staging.address()),"CreateTexture2D staging");context->CopyResource(staging.p,samples>1?resolved.p:color.p);
        D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(staging.p,0,D3D11_MAP_READ,0,&mapped),"Map output");
        std::ofstream output(root/"rgba32f.bin",std::ios::binary);
        for(UINT y=0;y<height;y++)output.write((char*)mapped.pData+y*mapped.RowPitch,width*16);
        context->Unmap(staging.p,0);if(!output)throw std::runtime_error("Failed output write");
#ifdef HS2_RENDERDOC_SUPPORT
        if(captureApi&&!captureApi->EndFrameCapture(device.p,nullptr))throw std::runtime_error("RenderDoc fixture capture failed");
#endif
        context->ClearState();for(auto* x:ownedBuffers)x->Release();for(auto* x:ownedViews)x->Release();for(auto* x:ownedSamplers)x->Release();for(auto* x:ownedTextures)x->Release();
        std::cout<<"WARP original DXBC pass executed "<<width<<"x"<<height<<" samples="<<samples<<"\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}
