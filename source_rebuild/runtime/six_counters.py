"""Six-backend diagnostics; preserve the validated five-backend assessment."""
import ctypes
from pathlib import Path
from five_counters import Counter as FiveCounter,assess as assess_five,GEOMETRY

FIELDS=('launches','module_loads','last_stream','last_device','last_grid','last_width','last_tile')
PAIRS=('f32_f32','f16_f16','f16_bf16','f16_f32','bf16_bf16','bf16_f16','bf16_f32')
TILES=('128','1024','8192')

class Counter:
    def __init__(self,cast,rotary,mrope,silu,swiglu,rms):
        self.five=FiveCounter(cast,rotary,mrope,silu,swiglu)
        self.rms=ctypes.CDLL(str(Path(rms)/'libminimax_nine_rms_norm_runtime.so'))
        self.rms.minimax_nine_rms_norm_device_statistics.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
        self.rms.minimax_nine_rms_norm_device_statistics.restype=None
    def snapshot(self):
        result=self.five.snapshot(); result['rms']={}
        for device in range(8):
            values=(ctypes.c_uint64*17)(); self.rms.minimax_nine_rms_norm_device_statistics(device,values)
            result['rms'][str(device)]=dict(zip(FIELDS,map(int,values[:7])),pairs=dict(zip(PAIRS,map(int,values[7:14]))),tiles=dict(zip(TILES,map(int,values[14:]))))
        return result

def assess(before,after,position_device,position_type,rows,frequency_dim,expected_silu_calls,expected_swiglu_calls,expected_rms_counts):
    result=assess_five(before,after,position_device,position_type,rows,frequency_dim,expected_silu_calls,expected_swiglu_calls); rms={}
    for device in map(str,range(8)):
        a,b=before['rms'][device],after['rms'][device]
        delta={k:b[k]-a[k] for k in ('launches','module_loads')}
        for group in ('pairs','tiles'): delta[group]={k:b[group][k]-a[group][k] for k in b[group]}
        if any(v<0 for k,v in delta.items() if k not in ('pairs','tiles')) or any(v<0 for group in ('pairs','tiles') for v in delta[group].values()): raise RuntimeError('RMS counters moved backwards')
        if delta['launches']!=expected_rms_counts['rms_norm'] or delta['launches']!=sum(delta['pairs'].values()) or delta['launches']!=sum(delta['tiles'].values()): raise RuntimeError('Actual RMS count/pair/tile matrix differs on '+device)
        if delta['pairs']['bf16_bf16']!=delta['launches'] or delta['tiles']!=expected_rms_counts['rms_tiles']: raise RuntimeError('Actual RMS BF16 or width routes differ on '+device)
        if b['last_device']!=int(device) or not b['last_stream'] or b['last_grid']!=rows or b['last_width']!=expected_rms_counts['hidden_size'] or b['last_tile']!=expected_rms_counts['hidden_tile']: raise RuntimeError('Actual RMS last device/stream/geometry differs on '+device)
        rms[device]=delta
    result['rms']=rms; return result
