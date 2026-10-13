// External target-control client. Never launches/injects a process or forces a connection.
// Headers and DLL must both come from the pinned official RenderDoc release.
#define RENDERDOC_PLATFORM_WIN32
#include "renderdoc_replay.h"
#include <iostream>
#include <string>
#include <chrono>
#include <stdexcept>
#include <set>
REPLAY_PROGRAM_MARKER();

// The official Windows DLL uses the MS C++ ABI. MinGW's non-trivial aggregate
// member returns put sret before this; MSVC puts it after this. Plain calls such
// as Connected/GetPID and the extern-C factory do not have this discrepancy.
// The previously built GNU client passed an output object as this to GetTarget,
// which explains why a successful native handshake was followed by a crash.
// This source repair has NOT received graphical/native acceptance: do not run
// it against HS2 until the later authorized independent fixture check passes.
template<class T,class... Args> static T member_result(ITargetControl* object,size_t slot,Args... args) {
  T result;
  auto table=*reinterpret_cast<void***>(object);
  using ExplicitMSMember=T* (__cdecl *)(ITargetControl*,T*,Args...);
  reinterpret_cast<ExplicitMSMember>(table[slot])(object,&result,args...);
  return result;
}
static rdcstr target_name(ITargetControl* object) {
#if defined(__GNUC__) && defined(_WIN64)
  return member_result<rdcstr>(object,2);
#else
  return object->GetTarget();
#endif
}
static rdcstr target_api(ITargetControl* object) {
#if defined(__GNUC__) && defined(_WIN64)
  return member_result<rdcstr>(object,3);
#else
  return object->GetAPI();
#endif
}
static TargetControlMessage receive(ITargetControl* object) {
#if defined(__GNUC__) && defined(_WIN64)
  return member_result<TargetControlMessage>(object,10,(RENDERDOC_ProgressCallback)nullptr);
#else
  return object->ReceiveMessage(nullptr);
#endif
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
  } session;
  ITargetControl* target=nullptr;
  try {
    if(argc==2 && std::string(argv[1])=="list") {
      uint32_t ident=0;
      while((ident=RENDERDOC_EnumerateRemoteTargets("",ident)))
        std::cout<<"{\"event\":\"target_ident\",\"ident\":"<<ident<<"}\n";
      return 0;
    }
    if(argc!=5) throw std::runtime_error("Usage: renderdoc_target list | ident expected_pid status|trigger timeout_seconds");
    uint32_t ident=(uint32_t)std::stoul(argv[1]),pid=(uint32_t)std::stoul(argv[2]);
    std::string action=argv[3];int seconds=std::stoi(argv[4]);
    if(!pid || seconds<1 || (action!="status"&&action!="trigger")) throw std::runtime_error("Explicit expected PID, action and positive deadline required");
    target=RENDERDOC_CreateTargetControl("",ident,"HS2 offline renderer audit",false);
    if(!target || !target->Connected()) throw std::runtime_error("Target unavailable or another client owns it; no forced connection attempted");
    if(target->GetPID()!=pid) throw std::runtime_error("Expected PID mismatch; capture not triggered");
    std::cout<<"{\"event\":\"connected\",\"pid\":"<<target->GetPID()<<",\"ident\":"<<ident
      <<",\"target\":"<<json(target_name(target).c_str())<<",\"api\":"<<json(target_api(target).c_str())<<"}"<<std::endl;
    std::set<uint32_t> existing;
    // Drain initial notifications so a pre-existing capture cannot satisfy this request.
    auto drainDeadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    bool drained=false;
    while(target->Connected() && std::chrono::steady_clock::now()<drainDeadline) {
      auto m=receive(target);
      if(m.type==TargetControlMessageType::NewCapture) {
        existing.insert(m.newCapture.captureId);
        std::cout<<"{\"event\":\"existing_capture\",\"capture_id\":"<<m.newCapture.captureId
          <<",\"path\":"<<json(m.newCapture.path.c_str())<<"}"<<std::endl;
      }
      if(m.type==TargetControlMessageType::Noop) {drained=true;break;}
    }
    if(!drained)throw std::runtime_error("Initial target notification queue did not drain; capture not triggered");
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
        if(action=="trigger" && !existing.count(c.captureId)) {completed=true;break;}
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
