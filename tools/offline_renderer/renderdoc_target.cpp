// External target-control client. Never launches/injects a process or forces a connection.
// Headers and DLL must both come from the pinned official RenderDoc release.
#define RENDERDOC_PLATFORM_WIN32
#if defined(__MINGW32__)
#error "Use the installed MSVC x64 toolchain for the official MSVC C++ replay interface; GNU prototype is not accepted"
#endif
#include "renderdoc_replay.h"
#include <iostream>
#include <string>
#include <chrono>
#include <stdexcept>
#include <set>
#include <memory>
#include "capture_guard.h"
REPLAY_PROGRAM_MARKER();

// The official Windows DLL uses the MS C++ ABI. MinGW's non-trivial aggregate
// member returns put sret before this; MSVC puts it after this. Plain calls such
// as Connected/GetPID and the extern-C factory do not have this discrepancy.
// The previously built GNU client passed an output object as this to GetTarget,
// which explains why a successful native handshake was followed by a crash.
// Use MSVC directly rather than shipping an unaccepted cross-toolchain shim.
// Native acceptance remains pending; do not run against HS2 before it passes.
static rdcstr target_name(ITargetControl* object) {
  return object->GetTarget();
}
static rdcstr target_api(ITargetControl* object) {
  return object->GetAPI();
}
static TargetControlMessage receive(ITargetControl* object) {
  return object->ReceiveMessage(nullptr);
}

static std::string json(const char* p) {
  std::string out="\"";
  for(const unsigned char* c=(const unsigned char*)p;*c;++c) {
    if(*c=='"'||*c=='\\') {out+='\\';out+=char(*c);}
    else if(*c<32) {char b[7];snprintf(b,sizeof(b),"\\u%04x",*c);out+=b;}
    else out+=char(*c);
  }
  return out+'"';
}
int main(int argc,char** argv) {
  struct Session {
    Session(){GlobalEnvironment env;env.enumerateGPUs=false;RENDERDOC_InitialiseReplay(env,{});}
    ~Session(){RENDERDOC_ShutdownReplay();}
  };
  std::unique_ptr<Session> session;
  ITargetControl* target=nullptr;
  try {
    if(argc==2 && std::string(argv[1])=="list") {
      session=std::make_unique<Session>();
      uint32_t ident=0;
      while((ident=RENDERDOC_EnumerateRemoteTargets("",ident)))
        std::cout<<"{\"event\":\"target_ident\",\"ident\":"<<ident<<"}\n";
      return 0;
    }
    if(argc!=5 && argc!=6) throw std::runtime_error("Usage: renderdoc_target list | ident expected_pid status timeout_seconds | ident expected_pid trigger timeout_seconds fresh_capture_prefix");
    uint32_t ident=(uint32_t)std::stoul(argv[1]),pid=(uint32_t)std::stoul(argv[2]);
    std::string action=argv[3];int seconds=std::stoi(argv[4]);
    if(!pid || seconds<1 || (action!="status"&&action!="trigger")) throw std::runtime_error("Explicit expected PID, action and positive deadline required");
    if((action=="trigger" && argc!=6)||(action=="status" && argc!=5))throw std::runtime_error("Trigger requires explicit fresh output prefix");
    std::unique_ptr<CaptureGuard> guard;
    if(action=="trigger")guard=std::make_unique<CaptureGuard>(argv[5]);
    session=std::make_unique<Session>();
    target=RENDERDOC_CreateTargetControl("",ident,"HS2 offline renderer audit",false);
    if(!target || !target->Connected()) throw std::runtime_error("Target unavailable or another client owns it; no forced connection attempted");
    if(target->GetPID()!=pid) throw std::runtime_error("Expected PID mismatch; capture not triggered");
    std::cout<<"{\"event\":\"connected\",\"pid\":"<<target->GetPID()<<",\"ident\":"<<ident
      <<",\"target\":"<<json(target_name(target).c_str())<<",\"api\":"<<json(target_api(target).c_str())<<"}"<<std::endl;
    uint64_t requested=(uint64_t)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // Trigger is deliberately explicit. Caller must first establish Maker/card/SSS readiness.
    if(action=="trigger") {target->TriggerCapture(1);std::cout<<"{\"event\":\"trigger_requested\",\"frames\":1}"<<std::endl;}
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
    bool completed=action=="status";
    while(target->Connected() && std::chrono::steady_clock::now()<deadline) {
      TargetControlMessage message=receive(target);
      if(message.type==TargetControlMessageType::NewCapture) {
        const auto &c=message.newCapture;
        std::cout<<"{\"event\":\"new_capture\",\"capture_id\":"<<c.captureId<<",\"frame\":"<<c.frameNumber
          <<",\"timestamp\":"<<c.timestamp<<",\"bytes\":"<<c.byteSize<<",\"path\":"<<json(c.path.c_str())
          <<",\"api\":"<<json(c.api.c_str())<<",\"local\":"<<(c.local?"true":"false")<<"}"<<std::endl;
        if(action=="trigger") {
          std::string reason;
          if(guard->accepts(c.path.c_str(),c.api.c_str(),c.local,c.timestamp,c.byteSize,requested,reason)){completed=true;break;}
          std::cout<<"{\"event\":\"capture_rejected\",\"capture_id\":"<<c.captureId<<",\"reason\":"<<json(reason.c_str())<<"}"<<std::endl;
        }
      } else if(message.type!=TargetControlMessageType::Noop && message.type!=TargetControlMessageType::Unknown) {
        std::cout<<"{\"event\":\"target_message\",\"type\":"<<(uint32_t)message.type<<"}"<<std::endl;
      }
      if(action=="status") break;
    }
    target->Shutdown();target=nullptr;
    if(!completed) throw std::runtime_error("No capture received within deadline; do not assume capture success");
    return 0;
  } catch(const std::exception& e) {
    if(target)target->Shutdown();
    std::cerr<<e.what()<<std::endl;return 1;
  }
}
