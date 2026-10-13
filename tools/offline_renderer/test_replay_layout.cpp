// CPU layout report only: no RenderDoc function/object/device is invoked.
#define RENDERDOC_PLATFORM_WIN32
#include "renderdoc_replay.h"
#include <cstdio>
int main(){
  std::printf("rdcstr=%zu TargetControlMessage=%zu std_function_progress=%zu\n",
    sizeof(rdcstr),sizeof(TargetControlMessage),sizeof(RENDERDOC_ProgressCallback));
}
