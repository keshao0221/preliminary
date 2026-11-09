// Module/cache/statistics/launch glue only; no attention device mathematics.
#include <hip/hip_runtime_api.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
namespace {
struct Module{hipModule_t module;hipFunction_t function;};
std::mutex mutex;std::unordered_map<std::string,Module> modules;
std::array<std::array<std::atomic<uint64_t>,21>,64> per_device{};
thread_local std::string error;
void check(hipError_t code){if(code!=hipSuccess)throw std::string(hipGetErrorString(code));}
}
extern "C" const char *minimax_nine_mha_error(){return error.c_str();}
extern "C" void minimax_nine_mha_device_statistics(int device,uint64_t *out){for(int i=0;i<21;++i)out[i]=device>=0&&device<64?per_device[device][i].load():0;}
extern "C" int minimax_nine_mha_launch(const char *binary,unsigned grid,unsigned shared,unsigned tile,int dtype,const int64_t *meta,void *stream,void **arguments){
    error.clear();try{
        if(!binary||!grid||!stream||!meta||!arguments||shared>65536||(tile!=64&&tile!=128)||dtype<0||dtype>2)throw std::string("Invalid MHA host arguments");
        int device=-1;check(hipGetDevice(&device));if(device<0||device>=64)throw std::string("MHA device capacity exceeded");
        auto &stat=per_device[device];hipFunction_t function=nullptr;
        {std::lock_guard<std::mutex> lock(mutex);auto key=std::to_string(device)+":"+binary;auto it=modules.find(key);
            if(it==modules.end()){
                hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,device));std::string arch=prop.gcnArchName;
                if(arch.substr(0,arch.find(':'))!="gfx936"||prop.warpSize!=64)throw std::string("Expected gfx936/warp64");
                hipModule_t module=nullptr;check(hipModuleLoad(&module,binary));auto code=hipModuleGetFunction(&function,module,"minimax_nine_mha");
                if(code!=hipSuccess){(void)hipModuleUnload(module);check(code);}modules.emplace(key,Module{module,function});++stat[1];
            }else function=it->second.function;
        }
        check(hipModuleLaunchKernel(function,grid,1,1,256,1,1,shared,static_cast<hipStream_t>(stream),arguments,nullptr));
        ++stat[0];stat[2]=reinterpret_cast<uintptr_t>(stream);stat[3]=device;stat[4]=grid;stat[5]=shared;stat[6]=tile;
        for(int i=0;i<11;++i){stat[7+i]=meta[i];}
        ++stat[18+dtype];return 0;
    }catch(const std::string &s){error=s;}catch(const std::exception &s){error=s.what();}catch(...){error="Unknown MHA host exception";}return 1;
}
