// HIP host loading/cache/stream glue; all RMSNorm arithmetic is NineToothed.
#include <hip/hip_runtime_api.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
namespace {
struct Module { hipModule_t module; hipFunction_t function; };
std::mutex mutex; std::unordered_map<std::string,Module> modules;
std::array<std::atomic<uint64_t>,7> stats{};
std::array<std::array<std::atomic<uint64_t>,17>,64> per_device{};
thread_local std::string error;
void check(hipError_t status) { if(status!=hipSuccess) throw std::string(hipGetErrorString(status)); }
int pair_index(const std::string &path) {
    const std::array<std::string,7> pairs{"f32_f32","f16_f16","f16_bf16","f16_f32","bf16_bf16","bf16_f16","bf16_f32"};
    for(int i=0;i<7;++i) if(path.find("/"+pairs[i]+"/")!=std::string::npos) return i;
    throw std::string("Unknown RMSNorm pair path");
}
}
extern "C" const char *minimax_nine_rms_norm_error() { return error.c_str(); }
extern "C" void minimax_nine_rms_norm_statistics(uint64_t *out) { for(size_t i=0;i<7;++i) out[i]=stats[i].load(); }
extern "C" void minimax_nine_rms_norm_device_statistics(int device,uint64_t *out) {
    for(size_t i=0;i<17;++i) out[i]=(device>=0 && device<64)?per_device[device][i].load():0;
}
extern "C" int minimax_nine_rms_norm_launch(const char *binary,int tile,unsigned grid,unsigned width,void *stream,void **args) {
    error.clear();
    try {
        const int route=tile==128?0:tile==1024?1:tile==8192?2:-1;
        if(!binary || !stream || !args || !grid || !width || width>unsigned(tile) || route<0) throw std::string("Invalid RMSNorm host arguments");
        const unsigned shared=0;
        int device=-1; check(hipGetDevice(&device)); if(device<0 || device>=64) throw std::string("RMSNorm diagnostic capacity exceeded");
        const int pair=pair_index(binary); auto &local=per_device[device]; hipFunction_t function=nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex); const auto key=std::to_string(device)+":"+binary; auto found=modules.find(key);
            if(found==modules.end()) {
                hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop,device)); const std::string arch=prop.gcnArchName;
                if(arch.substr(0,arch.find(':'))!="gfx936" || prop.warpSize!=64) throw std::string("Expected gfx936/warp64");
                hipModule_t module=nullptr; check(hipModuleLoad(&module,binary));
                auto status=hipModuleGetFunction(&function,module,"minimax_nine_rms_norm");
                if(status!=hipSuccess) { (void)hipModuleUnload(module); check(status); }
                modules.emplace(key,Module{module,function}); ++stats[1]; ++local[1];
            } else function=found->second.function;
        }
        check(hipModuleLaunchKernel(function,grid,1,1,64,1,1,shared,static_cast<hipStream_t>(stream),args,nullptr));
        ++stats[0]; stats[2]=reinterpret_cast<uintptr_t>(stream); stats[3]=device; stats[4]=grid; stats[5]=width; stats[6]=tile;
        ++local[0]; local[2]=reinterpret_cast<uintptr_t>(stream); local[3]=device; local[4]=grid; local[5]=width; local[6]=tile; ++local[7+pair]; ++local[14+route];
        return 0;
    } catch(const std::string &value) { error=value; } catch(const std::exception &value) { error=value.what(); } catch(...) { error="Unknown RMSNorm host exception"; }
    return 1;
}
