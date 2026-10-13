// Independent D3D11 WARP window/present fixture for external target-control validation.
// This does not access any game files/process/state.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
static void check(HRESULT r){if(FAILED(r))throw std::runtime_error("D3D11 fixture failed");}
int main(int argc,char** argv){
  try {
    if(argc!=2)throw std::runtime_error("Usage: fixture ready_file");
    WNDCLASSA wc{};wc.lpfnWndProc=DefWindowProcA;wc.hInstance=GetModuleHandleA(nullptr);wc.lpszClassName="OfflineRenderDocFixture";
    if(!RegisterClassA(&wc))throw std::runtime_error("Fixture class failed");
    HWND window=CreateWindowA(wc.lpszClassName,"Independent offline fixture",WS_OVERLAPPEDWINDOW,0,0,128,128,nullptr,nullptr,wc.hInstance,nullptr);
    if(!window)throw std::runtime_error("Fixture window failed");
    DXGI_SWAP_CHAIN_DESC desc{};desc.BufferDesc.Width=32;desc.BufferDesc.Height=32;desc.BufferDesc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count=1;desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.BufferCount=1;desc.OutputWindow=window;desc.Windowed=TRUE;
    desc.SwapEffect=DXGI_SWAP_EFFECT_DISCARD;
    ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;IDXGISwapChain* swap=nullptr;
    check(D3D11CreateDeviceAndSwapChain(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&desc,&swap,&device,nullptr,&context));
    ID3D11Texture2D* back=nullptr;check(swap->GetBuffer(0,__uuidof(ID3D11Texture2D),(void**)&back));
    ID3D11RenderTargetView* rtv=nullptr;check(device->CreateRenderTargetView(back,nullptr,&rtv));
    std::ofstream ready(argv[1]);ready<<GetCurrentProcessId();ready.close();
    for(unsigned frame=0;frame<1200;++frame){
      MSG msg;while(PeekMessageA(&msg,nullptr,0,0,PM_REMOVE)){TranslateMessage(&msg);DispatchMessageA(&msg);}
      const float color[4]={float(frame%255)/255,.25f,.5f,1};context->ClearRenderTargetView(rtv,color);check(swap->Present(0,0));Sleep(25);
    }
    rtv->Release();back->Release();swap->Release();context->Release();device->Release();DestroyWindow(window);return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
