"""Add SwiGLU diagnostics while retaining the validated four-backend checks."""
import ctypes
from pathlib import Path
from quad_counters import Counter as QuadCounter,assess as assess_quad,GEOMETRY

FIELDS=('launches','module_loads','last_stream','last_device','rank1_launches','rank8_launches','last_grid')
TYPES=('f32','f16','bf16')

class Counter:
    def __init__(self,cast_root,rotary_root,mrope_root,silu_root,swiglu_root):
        self.quad=QuadCounter(cast_root,rotary_root,mrope_root,silu_root)
        self.swiglu=ctypes.CDLL(str(Path(swiglu_root)/'libminimax_nine_swiglu_runtime.so'))
        self.swiglu.minimax_nine_swiglu_device_statistics.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
        self.swiglu.minimax_nine_swiglu_device_statistics.restype=None
    def snapshot(self):
        result=self.quad.snapshot(); result['swiglu']={}
        for device in range(8):
            values=(ctypes.c_uint64*10)(); self.swiglu.minimax_nine_swiglu_device_statistics(device,values)
            result['swiglu'][str(device)]=dict(zip(FIELDS,map(int,values[:7])),types=dict(zip(TYPES,map(int,values[7:]))))
        return result

def assess(before,after,position_device,position_type,rows,frequency_dim,expected_silu_calls,expected_swiglu_calls):
    result=assess_quad(before,after,position_device,position_type,rows,frequency_dim,expected_silu_calls); swiglu={}
    if expected_swiglu_calls<=0: raise RuntimeError('Expected positive frozen H3 SwiGLU count')
    for device in map(str,range(8)):
        a,b=before['swiglu'][device],after['swiglu'][device]
        delta={name:b[name]-a[name] for name in ('launches','module_loads','rank1_launches','rank8_launches')}
        delta['types']={name:b['types'][name]-a['types'][name] for name in TYPES}
        if any(v<0 for k,v in delta.items() if k!='types') or any(v<0 for v in delta['types'].values()):
            raise RuntimeError('SwiGLU counters moved backwards on '+device)
        if delta['launches']!=expected_swiglu_calls or delta['launches']!=sum(delta['types'].values()) or delta['launches']!=delta['rank1_launches']+delta['rank8_launches']:
            raise RuntimeError('Actual H3 SwiGLU count/type/layout matrix differs on '+device)
        if delta['types']['bf16']!=delta['launches'] or b['last_device']!=int(device) or not b['last_stream'] or not b['last_grid']:
            raise RuntimeError('Actual H3 BF16 SwiGLU device/stream evidence missing on '+device)
        swiglu[device]=delta
    result['swiglu']=swiglu; return result
