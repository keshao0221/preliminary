"""Four-backend counters; all reads concern metadata and host diagnostics."""
import ctypes
from pathlib import Path
from trio_observer_base import Counter as TrioCounter,assess as assess_trio,GEOMETRY

SILU_FIELDS=('launches','module_loads','last_stream','last_device','rank1_launches','rank8_launches','last_grid')
SILU_TYPES=('f32','f16','bf16')

class Counter:
    def __init__(self,cast_root,rotary_root,mrope_root,silu_root):
        self.trio=TrioCounter(cast_root,rotary_root,mrope_root)
        self.silu=ctypes.CDLL(str(Path(silu_root)/'libminimax_nine_silu_runtime.so'))
        self.silu.minimax_nine_silu_device_statistics.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
        self.silu.minimax_nine_silu_device_statistics.restype=None
    def snapshot(self):
        result=self.trio.snapshot(); result['silu']={}
        for device in range(8):
            values=(ctypes.c_uint64*10)(); self.silu.minimax_nine_silu_device_statistics(device,values)
            result['silu'][str(device)]=dict(zip(SILU_FIELDS,map(int,values[:7])),types=dict(zip(SILU_TYPES,map(int,values[7:]))))
        return result

def assess(before,after,position_device,position_type,rows,frequency_dim,expected_silu_calls):
    result=assess_trio(before,after,position_device,position_type,rows,frequency_dim); silu={}
    if expected_silu_calls<=0: raise RuntimeError('Expected positive frozen H3 SiLU count')
    for device in map(str,range(8)):
        a,b=before['silu'][device],after['silu'][device]
        delta={name:b[name]-a[name] for name in ('launches','module_loads','rank1_launches','rank8_launches')}
        delta['types']={name:b['types'][name]-a['types'][name] for name in SILU_TYPES}
        if any(value<0 for name,value in delta.items() if name!='types') or any(value<0 for value in delta['types'].values()):
            raise RuntimeError('SiLU counters moved backwards on device '+device)
        if delta['launches']!=expected_silu_calls or delta['launches']!=sum(delta['types'].values()) or delta['launches']!=delta['rank1_launches']+delta['rank8_launches']:
            raise RuntimeError('Actual H3 SiLU count/type/layout matrix differs on device '+device)
        if delta['types']['f32']!=delta['launches'] or b['last_device']!=int(device) or not b['last_stream'] or not b['last_grid']:
            raise RuntimeError('Actual H3 FP32 SiLU device/stream evidence missing on '+device)
        silu[device]=delta
    result['silu']=silu; return result
