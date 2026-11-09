"""Observe two real NineToothed backends at the actual H3 Runner boundary."""
import ctypes
import functools
import inspect
from pathlib import Path
from cast_observer_base import Counter as CastCounter,assess as assess_cast,mapped_core_paths,sha256,write_json

INPUTS=('i32','i64','f16','bf16','f32','f64')
OUTPUTS=('f16','bf16','f32')
ROTARY_PAIRS=tuple(a+'_to_'+b for a in INPUTS for b in OUTPUTS)
ROTARY_FIELDS=('launches','module_loads','last_stream','last_device','last_grid')

class Counter:
    def __init__(self,cast_root,rotary_root):
        self.cast=CastCounter(cast_root)
        self.rotary=ctypes.CDLL(str(Path(rotary_root)/'libminimax_nine_rotary_runtime.so'))
        self.rotary.minimax_nine_rotary_device_statistics.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
        self.rotary.minimax_nine_rotary_device_statistics.restype=None
    def snapshot(self):
        rotary={}
        for device in range(8):
            values=(ctypes.c_uint64*23)(); self.rotary.minimax_nine_rotary_device_statistics(device,values)
            rotary[str(device)]=dict(zip(ROTARY_FIELDS,map(int,values[:5])),pairs=dict(zip(ROTARY_PAIRS,map(int,values[5:]))))
        return dict(cast=self.cast.snapshot(),rotary=rotary)

def assess(before,after,position_device,position_type):
    cast=assess_cast(before['cast'],after['cast'])
    rotary={}
    for device in map(str,range(8)):
        a,b=before['rotary'][device],after['rotary'][device]
        delta={k:b[k]-a[k] for k in ('launches','module_loads')}
        delta['pairs']={k:b['pairs'][k]-a['pairs'][k] for k in ROTARY_PAIRS}
        if delta['launches']<0 or delta['module_loads']<0 or any(v<0 for v in delta['pairs'].values()) or delta['launches']!=sum(delta['pairs'].values()):
            raise RuntimeError('Rotary counter matrix inconsistent on device '+device)
        if delta['launches'] and (not b['last_stream'] or b['last_device']!=int(device)):
            raise RuntimeError('Rotary stream/device evidence missing on '+device)
        rotary[device]=delta
    expected=rotary[str(position_device)]
    if expected['launches']<=0 or sum(v for k,v in expected['pairs'].items() if k.startswith(position_type+'_to_'))<=0:
        raise RuntimeError('No real H3 rotary launch on the input-position device/type')
    return dict(cast=cast,rotary=rotary)

def install(cast_root,rotary_root,report_path,config):
    import torch
    from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
    cast_root,rotary_root=Path(cast_root).resolve(),Path(rotary_root).resolve()
    expected={rotary_root/'libinfinicore_cpp_api.so':config['core_sha256'],
        cast_root/'libminimax_nine_cast_runtime.so':config['cast_host_sha256'],rotary_root/'libminimax_nine_rotary_runtime.so':config['rotary_host_sha256']}
    for path,digest in expected.items():
        if sha256(path)!=digest: raise RuntimeError('Combined native hash mismatch: '+path.name)
    source=Path(inspect.getfile(MiniMaxH3TransformerRunner))
    if sha256(source)!=config['model_source_sha256']: raise RuntimeError('Frozen H3 runner differs')
    if mapped_core_paths()!=[str(rotary_root/'libinfinicore_cpp_api.so')]: raise RuntimeError('Framework loaded an unintended core library')
    counter=Counter(cast_root,rotary_root)
    report=dict(schema_version=1,status='COMBINED_OBSERVER_INSTALLED_MODEL_PENDING',configuration=config,mapped_core_paths=mapped_core_paths(),calls=[],
        h3_cast_usage_verified=False,h3_rotary_usage_verified=False,formal_first_frame_verified=False,performance_speedup_verified=False)
    write_json(report_path,report)
    original=MiniMaxH3TransformerRunner.__call__
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):
        inputs=args[0] if args else kwargs['inputs']; positions=inputs.position_ids
        position_type={'torch.int32':'i32','torch.int64':'i64','torch.float16':'f16','torch.bfloat16':'bf16','torch.float32':'f32','torch.float64':'f64'}[str(positions.dtype)]
        position_device=positions.device.index or 0
        before=counter.snapshot()
        call=dict(status='RUNNING',before=before,positions=dict(shape=list(positions.shape),dtype=str(positions.dtype),device=str(positions.device),stride=list(positions.stride())),
            requested_frequency_dim=int(self.rope_freq_dim),requested_theta=float(self.rope_theta))
        report['calls'].append(call); write_json(report_path,report)
        try:
            result=original(self,*args,**kwargs)
            for device in range(8): torch.cuda.synchronize(device)
            after=counter.snapshot(); delta=assess(before,after,position_device,position_type)
            outputs={}
            for name in ('sample','audio_sample'):
                value=getattr(result,name)
                if not isinstance(value,torch.Tensor) or not bool(torch.isfinite(value).all()): raise RuntimeError('Invalid/nonfinite H3 output: '+name)
                outputs[name]=dict(shape=list(value.shape),dtype=str(value.dtype),device=str(value.device),all_finite=True)
            call.update(status='PASSED',after=after,delta=delta,outputs=outputs)
            report.update(status='REAL_H3_NINETOOTHED_CAST_AND_ROTARY_OBSERVED',h3_cast_usage_verified=True,h3_rotary_usage_verified=True)
        except Exception as error:
            call.update(status='FAILED',error=str(error),after=counter.snapshot()); report['status']='FAILED_COMBINED_H3_OBSERVATION'; raise
        finally: write_json(report_path,report)
        return result
    MiniMaxH3TransformerRunner.__call__=wrapped
    return report
