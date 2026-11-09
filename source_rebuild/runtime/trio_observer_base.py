"""Observe three NineToothed backends around the actual frozen H3 Runner."""
import ctypes
import functools
import inspect
from pathlib import Path
from pair_observer_base import Counter as PairCounter,assess as assess_pair
from cast_observer_base import mapped_core_paths,sha256,write_json

MROPE_FIELDS=('launches','module_loads','last_stream','last_device','last_grid')
MROPE_PAIRS=tuple(a+'_'+b for a in ('f16','bf16','f32') for b in ('i32','i64'))
GEOMETRY=('rows','heads','head_dim','half_rotary_dim','output_token_stride','input_token_stride',
          'cache_row_stride','position_axis_stride','position_token_stride','table_rows','interleaved')

class Counter:
    def __init__(self,cast_root,rotary_root,mrope_root):
        self.pair=PairCounter(cast_root,rotary_root)
        self.mrope=ctypes.CDLL(str(Path(mrope_root)/'libminimax_nine_mrope_runtime.so'))
        for name in ('minimax_nine_mrope_device_statistics','minimax_nine_mrope_device_geometry'):
            func=getattr(self.mrope,name); func.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]; func.restype=None
    def snapshot(self):
        result=self.pair.snapshot(); result['mrope']={}
        for device in range(8):
            values=(ctypes.c_uint64*11)(); geometry=(ctypes.c_uint64*11)()
            self.mrope.minimax_nine_mrope_device_statistics(device,values)
            self.mrope.minimax_nine_mrope_device_geometry(device,geometry)
            result['mrope'][str(device)]=dict(zip(MROPE_FIELDS,map(int,values[:5])),pairs=dict(zip(MROPE_PAIRS,map(int,values[5:]))),
                last_launch_geometry=dict(zip(GEOMETRY,map(int,geometry))))
        return result

def assess(before,after,position_device,position_type,rows,frequency_dim):
    result=assess_pair(before,after,position_device,position_type)
    used=[name.split('_to_')[1] for name,count in result['rotary'][str(position_device)]['pairs'].items() if count]
    if len(used)!=1: raise RuntimeError('Actual H3 cache output dtype is ambiguous')
    expected_pair=used[0]+'_i64'; mrope={}
    for device in map(str,range(8)):
        a,b=before['mrope'][device],after['mrope'][device]
        delta={k:b[k]-a[k] for k in ('launches','module_loads')}
        delta['pairs']={k:b['pairs'][k]-a['pairs'][k] for k in MROPE_PAIRS}
        if any(v<0 for k,v in delta.items() if k!='pairs') or any(v<0 for v in delta['pairs'].values()): raise RuntimeError('MRoPE counter moved backwards')
        if delta['launches']<=0 or delta['launches']%2 or delta['launches']!=sum(delta['pairs'].values()) or delta['pairs'][expected_pair]!=delta['launches']:
            raise RuntimeError('Missing or inconsistent actual H3 MRoPE dtype/dispatch evidence on device '+device)
        if b['last_device']!=int(device) or not b['last_stream']: raise RuntimeError('MRoPE device/stream evidence missing')
        g=b['last_launch_geometry']
        if g['rows']!=rows or g['table_rows']!=rows or g['heads']<=0 or g['head_dim']<6*frequency_dim or g['half_rotary_dim']!=3*frequency_dim:
            raise RuntimeError('MRoPE actual token/head/cache geometry differs on device '+device)
        if g['interleaved']!=0 or g['position_axis_stride']!=0 or g['position_token_stride']!=1 or g['cache_row_stride']!=6*frequency_dim:
            raise RuntimeError('MRoPE actual cache/position layout differs on device '+device)
        if g['output_token_stride']<g['heads']*g['head_dim'] or g['input_token_stride']<g['heads']*g['head_dim']:
            raise RuntimeError('MRoPE actual token strides differ')
        if b['last_grid']!=rows*g['heads']*((g['head_dim']+127)//128): raise RuntimeError('MRoPE geometry/grid evidence differs')
        mrope[device]=delta
    result['mrope']=mrope
    return result

def install(cast_root,rotary_root,mrope_root,report_path,config):
    import torch
    from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
    cast_root,rotary_root,mrope_root=map(lambda p:Path(p).resolve(),(cast_root,rotary_root,mrope_root))
    expected={mrope_root/'libinfinicore_cpp_api.so':config['core_sha256'],cast_root/'libminimax_nine_cast_runtime.so':config['cast_host_sha256'],
        rotary_root/'libminimax_nine_rotary_runtime.so':config['rotary_host_sha256'],mrope_root/'libminimax_nine_mrope_runtime.so':config['mrope_host_sha256']}
    for path,digest in expected.items():
        if sha256(path)!=digest: raise RuntimeError('Triple native hash differs: '+path.name)
    if mapped_core_paths()!=[str(mrope_root/'libinfinicore_cpp_api.so')]: raise RuntimeError('Unexpected core mapping')
    if sha256(Path(inspect.getfile(MiniMaxH3TransformerRunner)))!=config['model_source_sha256']: raise RuntimeError('Frozen H3 Runner differs')
    counter=Counter(cast_root,rotary_root,mrope_root)
    report=dict(schema_version=1,status='TRIPLE_OBSERVER_INSTALLED_MODEL_PENDING',configuration=config,mapped_core_paths=mapped_core_paths(),calls=[],
        actual_model_executed=False,h3_cast_usage_verified=False,h3_rotary_usage_verified=False,h3_mrope_usage_verified=False,
        formal_first_frame_verified=False,performance_speedup_verified=False)
    write_json(report_path,report); original=MiniMaxH3TransformerRunner.__call__
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):
        inputs=args[0] if args else kwargs['inputs']; positions=inputs.position_ids
        position_type={'torch.int32':'i32','torch.int64':'i64','torch.float16':'f16','torch.bfloat16':'bf16','torch.float32':'f32','torch.float64':'f64'}[str(positions.dtype)]
        device=positions.device.index or 0; before=counter.snapshot()
        call=dict(status='RUNNING',before=before,positions=dict(shape=list(positions.shape),dtype=str(positions.dtype),device=str(positions.device),stride=list(positions.stride())),
            requested_frequency_dim=int(self.rope_freq_dim),requested_theta=float(self.rope_theta))
        hf_config=self.engine.hf_config
        call['h3_config_geometry']={key:hf_config[key] for key in ('num_layers','num_hidden_layers','num_attention_heads','attention_head_dim','head_dim','hidden_size','rope_freq_dim','rope_inv_freq_len') if key in hf_config}
        report['calls'].append(call); write_json(report_path,report)
        print('H3_TRIPLE_REAL_H3_CALL_BEGIN',len(report['calls']),flush=True)
        try:
            result=original(self,*args,**kwargs)
            for gpu in range(8): torch.cuda.synchronize(gpu)
            after=counter.snapshot(); delta=assess(before,after,device,position_type,int(positions.shape[0]),int(self.rope_freq_dim))
            outputs={}
            for name in ('sample','audio_sample'):
                value=getattr(result,name)
                if not isinstance(value,torch.Tensor) or not bool(torch.isfinite(value).all()): raise RuntimeError('Nonfinite H3 output: '+name)
                outputs[name]=dict(shape=list(value.shape),dtype=str(value.dtype),device=str(value.device),all_finite=True)
            call.update(status='PASSED',after=after,delta=delta,outputs=outputs)
            report.update(status='REAL_H3_THREE_NINETOOTHED_BACKENDS_OBSERVED',actual_model_executed=True,
                h3_cast_usage_verified=True,h3_rotary_usage_verified=True,h3_mrope_usage_verified=True)
            print('H3_TRIPLE_REAL_H3_CALL_COMPLETED',len(report['calls']),flush=True)
        except Exception as error:
            call.update(status='FAILED',error=str(error),after=counter.snapshot()); report['status']='FAILED_TRIPLE_H3_OBSERVATION'; raise
        finally: write_json(report_path,report)
        return result
    MiniMaxH3TransformerRunner.__call__=wrapped
    return report
