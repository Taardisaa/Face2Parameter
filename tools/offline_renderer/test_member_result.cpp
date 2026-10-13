// Non-graphics ABI fixture. No RenderDoc headers, DLL, device, window or process.
#include "msvc_member_result.h"
#include <cstdint>
#include <stdexcept>
#include <iostream>
struct Small {
  uint64_t words[3]{};
  Small()=default;Small(const Small& other){for(int i=0;i<3;i++)words[i]=other.words[i];}
  ~Small(){}
};
struct Large {
  uint64_t words[64]{};
  Large()=default;Large(const Large& other){for(int i=0;i<64;i++)words[i]=other.words[i];}
  ~Large(){}
};
struct Object {void** table;uint64_t magic;};
static Object* expected=nullptr;
static Small* name(Object* object,Small* result) {
  if(object!=expected || object->magic!=0x0123456789abcdefULL || !result)
    throw std::runtime_error("this/output ABI mismatch");
  result->words[0]=object->magic;result->words[1]=2;result->words[2]=3;return result;
}
static Large* receive(Object* object,Large* result,void* callback) {
  if(object!=expected || callback!=(void*)0x1234 || !result)
    throw std::runtime_error("this/output/argument ABI mismatch");
  for(uint64_t i=0;i<64;i++)result->words[i]=i^object->magic;return result;
}
int main(){
  try {
    void* table[11]{};table[2]=(void*)name;table[10]=(void*)receive;
    Object object{table,0x0123456789abcdefULL};expected=&object;
    auto n=msvc_member_result<Small>(&object,2);
    auto m=msvc_member_result<Large>(&object,10,(void*)0x1234);
    if(n.words[0]!=object.magic || n.words[1]!=2 || n.words[2]!=3)throw std::runtime_error("Small result changed");
    for(uint64_t i=0;i<64;i++)if(m.words[i]!=(i^object.magic))throw std::runtime_error("Large result changed");
    std::cout<<"MS member ABI result fixture passed: this/output/argument and 24/512-byte nontrivial results\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
