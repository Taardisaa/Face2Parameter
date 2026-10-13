#pragma once
#include <cstddef>

// Microsoft x64 non-trivial member-result ABI: this, return buffer, arguments.
// This explicit C function signature avoids GNU C++ sret-before-this lowering.
template<class T,class Object,class... Args>
static T msvc_member_result(Object* object,std::size_t slot,Args... args) {
  T result;
  auto table=*reinterpret_cast<void***>(object);
  using ExplicitMSMember=T* (__cdecl *)(Object*,T*,Args...);
  reinterpret_cast<ExplicitMSMember>(table[slot])(object,&result,args...);
  return result;
}
