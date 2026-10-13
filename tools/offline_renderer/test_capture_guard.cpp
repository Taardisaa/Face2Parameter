// CPU policy test. Files are synthetic and are never loaded by RenderDoc.
#include "capture_guard.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){
  try {
    if(argc!=2)throw std::runtime_error("Fresh policy fixture directory required");
    auto root=std::filesystem::absolute(argv[1]);
    if(std::filesystem::exists(root))throw std::runtime_error("Fresh directory required");
    std::filesystem::create_directories(root);auto prefix=root/"expected";
    CaptureGuard guard(prefix.u8string());auto capture=root/"expected_frame1.rdc";
    std::ofstream(capture,std::ios::binary).write("synthetic-policy-fixture",24);
    std::string reason;
    auto check=[&](bool result,bool expected){if(result!=expected)throw std::runtime_error("Capture policy disagrees");};
    check(guard.accepts(capture.u8string(),"D3D11",true,100,24,100,reason),true);
    check(guard.accepts(capture.u8string(),"D3D11",true,99,24,100,reason),false);
    check(guard.accepts(capture.u8string(),"D3D12",true,100,24,100,reason),false);
    check(guard.accepts(capture.u8string(),"D3D11",false,100,24,100,reason),false);
    check(guard.accepts(capture.u8string(),"D3D11",true,100,25,100,reason),false);
    check(guard.accepts((root/"wrong_frame1.rdc").u8string(),"D3D11",true,100,24,100,reason),false);
    check(guard.accepts((root/"other/expected_frame1.rdc").u8string(),"D3D11",true,100,24,100,reason),false);
    bool rejected=false;try{CaptureGuard old(prefix.u8string());}catch(const std::exception&){rejected=true;}
    check(rejected,true);
    std::cout<<"Capture policy passed eight CPU checks: prefix, old timestamp, API, local flag, actual byte size, wrong paths and prefix reuse\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
