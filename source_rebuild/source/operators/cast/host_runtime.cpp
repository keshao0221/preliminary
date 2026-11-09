// HIP host loading/cache/launch only. Device conversion is NineToothed-generated.
#include <hip/hip_runtime_api.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
struct Module { hipModule_t module; hipFunction_t function; };
std::mutex mutex;
std::unordered_map<std::string,Module> modules;
std::array<std::atomic<uint64_t>,7> statistics{};
// Additional host-only evidence. The seven-field original statistics ABI and
// all module loading/launch behavior are retained. No device code changes.
std::array<std::array<std::atomic<uint64_t>,16>,64> device_statistics{};
int pair_index(const std::string &path) {
    const std::array<std::string,9> names{"f32_to_f32","f32_to_f16","f32_to_bf16","f16_to_f32","f16_to_f16","f16_to_bf16","bf16_to_f32","bf16_to_f16","bf16_to_bf16"};
    for (size_t i=0;i<names.size();++i) if (path.find("/"+names[i]+"/")!=std::string::npos) return static_cast<int>(i);
    throw std::string("Unknown observed Cast pair");
}
thread_local std::string error;
void check(hipError_t status) {
    if (status!=hipSuccess) throw std::string(hipGetErrorString(status));
}
}

extern "C" const char *minimax_nine_cast_error() { return error.c_str(); }
extern "C" void minimax_nine_cast_statistics(uint64_t *result) {
    for (size_t i=0;i<statistics.size();++i) result[i]=statistics[i].load();
}
extern "C" void minimax_nine_cast_device_statistics(int device,uint64_t *result) {
    for (size_t i=0;i<16;++i) result[i]=(device>=0 && device<64) ? device_statistics[device][i].load() : 0;
}
extern "C" int minimax_nine_cast_launch(const char *binary,int rank,unsigned grid,void *stream,void **arguments) {
    error.clear();
    try {
        if (!binary || !stream || !arguments || !grid || (rank!=1 && rank!=8)) throw std::string("Invalid host launch arguments");
        int device=-1; check(hipGetDevice(&device));
        if (device<0 || device>=64) throw std::string("Observed Cast device index exceeds 64-device diagnostic capacity");
        const int pair=pair_index(binary);
        auto &local=device_statistics[device];
        const std::string key=std::to_string(device)+":"+binary;
        hipFunction_t function=nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto found=modules.find(key);
            if (found==modules.end()) {
                hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties,device));
                const std::string arch=properties.gcnArchName;
                if (arch.substr(0,arch.find(':'))!="gfx936" || properties.warpSize!=64)
                    throw std::string("Precompiled Cast requires gfx936 with warp size 64");
                hipModule_t module=nullptr; check(hipModuleLoad(&module,binary));
                const auto status=hipModuleGetFunction(&function,module,"minimax_nine_cast");
                if (status!=hipSuccess) { (void)hipModuleUnload(module); check(status); }
                modules.emplace(key,Module{module,function}); ++statistics[1]; ++local[1];
            } else function=found->second.function;
        }
        check(hipModuleLaunchKernel(function,grid,1,1,256,1,1,0,static_cast<hipStream_t>(stream),arguments,nullptr));
        // The framework owns stream ordering/synchronization. Retained modules
        // remain valid until process exit; no per-call synchronization/unload.
        ++statistics[0];
        statistics[2]=reinterpret_cast<uintptr_t>(stream);
        statistics[3]=static_cast<uint64_t>(device);
        ++statistics[rank==1 ? 4 : 5]; statistics[6]=grid;
        ++local[0]; local[2]=reinterpret_cast<uintptr_t>(stream); local[3]=static_cast<uint64_t>(device);
        ++local[rank==1 ? 4 : 5]; local[6]=grid; ++local[7+pair];
        return 0;
    } catch (const std::string &message) { error=message; }
      catch (const std::exception &exception) { error=exception.what(); }
      catch (...) { error="Unknown HIP host exception"; }
    return 1;
}
