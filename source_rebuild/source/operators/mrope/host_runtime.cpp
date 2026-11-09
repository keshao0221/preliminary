// Pure host module/cache/stream glue. No device numerical implementation.
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
std::array<std::atomic<uint64_t>,5> stats{};
std::array<std::array<std::atomic<uint64_t>,11>,64> per_device{};
std::array<std::array<std::atomic<uint64_t>,11>,64> geometry{};
thread_local std::string error;
void check(hipError_t status) { if(status!=hipSuccess) throw std::string(hipGetErrorString(status)); }
int pair_index(const std::string &path) {
    const std::array<std::string,3> data{"f16","bf16","f32"};
    const std::array<std::string,2> pos{"i32","i64"};
    for(int i=0;i<3;++i) for(int j=0;j<2;++j) if(path.find("/"+data[i]+"_"+pos[j]+"/")!=std::string::npos) return 2*i+j;
    throw std::string("Unknown MRoPE dtype pair");
}
}
extern "C" const char *minimax_nine_mrope_error() { return error.c_str(); }
extern "C" void minimax_nine_mrope_statistics(uint64_t *values) { for(size_t i=0;i<5;++i) values[i]=stats[i].load(); }
extern "C" void minimax_nine_mrope_device_statistics(int device,uint64_t *values) {
    for(size_t i=0;i<11;++i) values[i]=(device>=0 && device<64)?per_device[device][i].load():0;
}
extern "C" void minimax_nine_mrope_argument_geometry(void **arguments,uint64_t *values) {
    // Read scalar ABI metadata only; never dereference a device pointer.
    const std::array<int,11> slots{7,8,9,0,10,17,23,33,34,5,4};
    for(size_t i=0;i<slots.size();++i) values[i]=static_cast<uint64_t>(*static_cast<int64_t *>(arguments[slots[i]]));
}
extern "C" void minimax_nine_mrope_device_geometry(int device,uint64_t *values) {
    for(size_t i=0;i<11;++i) values[i]=(device>=0 && device<64)?geometry[device][i].load():0;
}
extern "C" int minimax_nine_mrope_launch(const char *binary,unsigned grid,void *stream,void **arguments) {
    error.clear();
    try {
        if(!binary || !grid || !stream || !arguments) throw std::string("Invalid MRoPE host arguments");
        int device=-1; check(hipGetDevice(&device));
        if(device<0 || device>=64) throw std::string("MRoPE diagnostic capacity exceeded");
        const int pair=pair_index(binary); auto &local=per_device[device]; hipFunction_t function=nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto key=std::to_string(device)+":"+binary;
            const auto found=modules.find(key);
            if(found==modules.end()) {
                hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop,device));
                const std::string arch=prop.gcnArchName;
                if(arch.substr(0,arch.find(':'))!="gfx936" || prop.warpSize!=64) throw std::string("Expected gfx936/warp64");
                hipModule_t module=nullptr; check(hipModuleLoad(&module,binary));
                const auto result=hipModuleGetFunction(&function,module,"minimax_nine_mrope");
                if(result!=hipSuccess) { (void)hipModuleUnload(module); check(result); }
                modules.emplace(key,Module{module,function}); ++stats[1]; ++local[1];
            } else function=found->second.function;
        }
        check(hipModuleLaunchKernel(function,grid,1,1,256,1,1,0,static_cast<hipStream_t>(stream),arguments,nullptr));
        std::array<uint64_t,11> description{}; minimax_nine_mrope_argument_geometry(arguments,description.data());
        for(size_t i=0;i<11;++i) geometry[device][i]=description[i];
        ++stats[0]; stats[2]=reinterpret_cast<uintptr_t>(stream); stats[3]=device; stats[4]=grid;
        ++local[0]; local[2]=reinterpret_cast<uintptr_t>(stream); local[3]=device; local[4]=grid; ++local[5+pair];
        return 0;
    } catch(const std::string &value) { error=value; }
      catch(const std::exception &value) { error=value.what(); }
      catch(...) { error="Unknown MRoPE host exception"; }
    return 1;
}
