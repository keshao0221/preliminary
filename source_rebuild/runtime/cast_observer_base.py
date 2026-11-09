"""Observe real H3 runner calls and host Cast counters without changing math."""
import ctypes
import functools
import hashlib
import json
from pathlib import Path

FIELDS=('launches','module_loads','last_stream','last_device','rank1_launches','rank8_launches','last_grid')
PAIRS=tuple(a+'_to_'+b for a in ('f32','f16','bf16') for b in ('f32','f16','bf16'))

def sha256(path):
    with Path(path).open('rb') as source: return hashlib.file_digest(source,'sha256').hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

class Counter:
    def __init__(self,root):
        self.host=ctypes.CDLL(str(Path(root)/'libminimax_nine_cast_runtime.so'))
        self.host.minimax_nine_cast_device_statistics.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
        self.host.minimax_nine_cast_device_statistics.restype=None
    def snapshot(self):
        result={}
        for device in range(8):
            data=(ctypes.c_uint64*16)(); self.host.minimax_nine_cast_device_statistics(device,data)
            result[str(device)]=dict(zip(FIELDS,map(int,data[:7])),pairs=dict(zip(PAIRS,map(int,data[7:]))))
        return result

def assess(before,after):
    deltas={}
    for device in map(str,range(8)):
        first,last=before[device],after[device]
        delta={name:last[name]-first[name] for name in ('launches','module_loads','rank1_launches','rank8_launches')}
        delta['pairs']={name:last['pairs'][name]-first['pairs'][name] for name in PAIRS}
        if any(value<0 for key,value in delta.items() if key!='pairs') or any(v<0 for v in delta['pairs'].values()):
            raise RuntimeError('Cast counter moved backwards on device '+device)
        if delta['launches']!=sum(delta['pairs'].values()) or delta['launches']!=delta['rank1_launches']+delta['rank8_launches']:
            raise RuntimeError('Cast counter matrix inconsistent on device '+device)
        if delta['launches']<=0 or not last['last_stream'] or last['last_device']!=int(device):
            raise RuntimeError('Real H3 call has no NineToothed Cast evidence on device '+device)
        deltas[device]=delta
    return deltas

def mapped_core_paths():
    paths=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        if 'libinfinicore_cpp_api.so' in line:
            paths.add(str(Path(line.split()[-1]).resolve()))
    return sorted(paths)

def install(root,report_path,expected_core_sha256,expected_host_sha256,expected_model_source_sha256):
    import torch
    from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
    import inspect
    root=Path(root).resolve()
    core=root/'libinfinicore_cpp_api.so'; host=root/'libminimax_nine_cast_runtime.so'
    if sha256(core)!=expected_core_sha256 or sha256(host)!=expected_host_sha256: raise RuntimeError('Observed native library hash mismatch')
    source=Path(inspect.getfile(MiniMaxH3TransformerRunner))
    if sha256(source)!=expected_model_source_sha256: raise RuntimeError('H3 Python runner differs from frozen baseline')
    paths=mapped_core_paths()
    if paths!=[str(core)]: raise RuntimeError('Framework loaded another core library: '+str(paths))
    counter=Counter(root)
    report=dict(schema_version=1,status='OBSERVER_INSTALLED_MODEL_PENDING',core_sha256=sha256(core),host_sha256=sha256(host),
        mapped_core_paths=paths,runner_source_sha256=sha256(source),calls=[],h3_cast_usage_verified=False,
        first_frame_verified=False,competition_submission_ready=False,performance_speedup_verified=False)
    write_json(report_path,report)
    original=MiniMaxH3TransformerRunner.__call__
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):
        before=counter.snapshot()
        call=dict(status='RUNNING',before=before)
        report['calls'].append(call); write_json(report_path,report)
        try:
            result=original(self,*args,**kwargs)
            for device in range(8): torch.cuda.synchronize(device)
            after=counter.snapshot(); deltas=assess(before,after)
            outputs={}
            for name in ('sample','audio_sample'):
                tensor=getattr(result,name)
                if not isinstance(tensor,torch.Tensor): raise RuntimeError('Expected Torch output for integration observer')
                finite=bool(torch.isfinite(tensor).all())
                if not finite: raise RuntimeError('Nonfinite H3 output: '+name)
                outputs[name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),device=str(tensor.device),all_finite=True)
            call.update(status='PASSED',after=after,delta=deltas,outputs=outputs)
            report['status']='REAL_H3_NINETOOTHED_CAST_OBSERVED'
            report['h3_cast_usage_verified']=True
        except Exception as error:
            call.update(status='FAILED',error=str(error),after=counter.snapshot())
            report['status']='FAILED_H3_CAST_OBSERVATION'
            raise
        finally: write_json(report_path,report)
        return result
    MiniMaxH3TransformerRunner.__call__=wrapped
    return report
