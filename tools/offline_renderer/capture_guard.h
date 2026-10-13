#pragma once
#include <filesystem>
#include <string>
#include <stdexcept>
#include <cstdint>
// Noop is not a barrier for the server's asynchronous list of old captures.
class CaptureGuard {
  std::filesystem::path prefix;
public:
  explicit CaptureGuard(const std::string& value):prefix(std::filesystem::absolute(std::filesystem::u8path(value)).lexically_normal()) {
    if(!std::filesystem::is_directory(prefix.parent_path()))throw std::runtime_error("Capture prefix parent must exist");
    for(auto& entry:std::filesystem::directory_iterator(prefix.parent_path()))
      if(matches(entry.path()))throw std::runtime_error("Fresh capture prefix already has files; request not triggered");
  }
  bool matches(const std::filesystem::path& path) const {
    std::error_code error;
    if(!std::filesystem::equivalent(path.parent_path(),prefix.parent_path(),error)||error)return false;
    auto name=path.filename().u8string(),base=prefix.filename().u8string();
    return name.size()>base.size() && name.compare(0,base.size(),base)==0 && (name[base.size()]=='_'||name[base.size()]=='.');
  }
  bool accepts(const std::string& value,const std::string& api,bool local,uint64_t timestamp,uint64_t bytes,uint64_t requested,std::string& reason) const {
    auto path=std::filesystem::absolute(std::filesystem::u8path(value)).lexically_normal();
    if(!local || api!="D3D11"){reason="Capture must be local D3D11";return false;}
    if(timestamp<requested){reason="Capture timestamp predates explicit trigger";return false;}
    if(!matches(path) || path.extension()!=".rdc"){reason="Capture path does not belong to fresh output prefix";return false;}
    if(!bytes || !std::filesystem::is_regular_file(path) || std::filesystem::file_size(path)!=bytes){reason="Actual RDC file missing, empty or size disagrees with target";return false;}
    return true;
  }
};
