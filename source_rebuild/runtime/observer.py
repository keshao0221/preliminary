"""Observe each real denoising call under a fixed six-op route."""
import functools
import inspect
from pathlib import Path
from cast_observer_base import mapped_core_paths,sha256,write_json
from six_counters import Counter,assess,GEOMETRY
from model_counts import expected_counts
from modes import activate,zero_candidates

def install(cast,rotary,mrope,silu,swiglu,rms,report_path,config):
    import torch
    from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
    roots=tuple(Path(p).resolve() for p in (cast,rotary,mrope,silu,swiglu,rms));cast,rotary,mrope,silu,swiglu,rms=roots
    expected={rms/'libinfinicore_cpp_api.so':config['core_sha256'],rms/'libminimax_nine_rms_norm_runtime.so':config['rms_host_sha256'],
        cast/'libminimax_nine_cast_runtime.so':config['cast_host_sha256'],rotary/'libminimax_nine_rotary_runtime.so':config['rotary_host_sha256'],
        mrope/'libminimax_nine_mrope_runtime.so':config['mrope_host_sha256'],silu/'libminimax_nine_silu_runtime.so':config['silu_host_sha256'],swiglu/'libminimax_nine_swiglu_runtime.so':config['swiglu_host_sha256']}
    for path,digest in expected.items():
        if sha256(path)!=digest:raise RuntimeError('Repeated-model native fingerprint differs: '+path.name)
    if mapped_core_paths()!=[str(rms/'libinfinicore_cpp_api.so')] or sha256(Path(inspect.getfile(MiniMaxH3TransformerRunner)))!=config['model_source_sha256']:raise RuntimeError('Active core/model differs')
    mode=config.get('model_mode','candidate');activate(mode,roots);counter=Counter(*roots)
    report=dict(status='MULTISTEP_OBSERVER_INSTALLED',mode=mode,configuration=config,mapped_core_paths=mapped_core_paths(),calls=[],actual_model_executed=False,training_executed=False,optimization_accepted=False)
    write_json(report_path,report);original=MiniMaxH3TransformerRunner.__call__
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):
        inputs=args[0] if args else kwargs['inputs'];positions=inputs.position_ids;device=positions.device.index or 0
        kind={'torch.int32':'i32','torch.int64':'i64','torch.float16':'f16','torch.bfloat16':'bf16','torch.float32':'f32','torch.float64':'f64'}[str(positions.dtype)]
        for gpu in range(8):torch.cuda.synchronize(gpu)
        before=counter.snapshot();index=len(report['calls'])+1
        if index>config['expected_h3_calls']:raise RuntimeError('Unexpected extra denoising call')
        if index==1 and mode=='candidate':
            for gpu,v in before['rms'].items():
                if any(v[k]!=config['encoder_rms_expected_per_device'][gpu][k] for k in ('launches','pairs','tiles')):raise RuntimeError('Condition encoder RMS evidence differs')
        hf=self.engine.hf_config;counts=expected_counts(hf)
        call=dict(status='RUNNING',index=index,before=before,positions=dict(shape=list(positions.shape),dtype=str(positions.dtype),device=str(positions.device),stride=list(positions.stride())),
            frequency_dim=int(self.rope_freq_dim),theta=float(self.rope_theta),resolved_operation_counts=counts,
            h3_config_geometry={k:hf[k] for k in ('num_layers','num_refiner_layers','token_refiner_num_layers','num_attention_heads','attention_head_dim','hidden_size','ffn_dim','rope_freq_dim') if k in hf})
        report['calls'].append(call);write_json(report_path,report);print('H3_MULTI_REAL_H3_CALL_BEGIN',mode,index,flush=True)
        try:
            result=original(self,*args,**kwargs)
            for gpu in range(8):torch.cuda.synchronize(gpu)
            after=counter.snapshot()
            if mode=='candidate':delta=assess(before,after,device,kind,int(positions.shape[0]),int(self.rope_freq_dim),counts['silu'],counts['swiglu'],counts)
            else:zero_candidates(before,after);delta={}
            outputs={}
            for name in ('sample','audio_sample'):
                value=getattr(result,name)
                if not isinstance(value,torch.Tensor) or not bool(torch.isfinite(value).all()):raise RuntimeError('Nonfinite H3 output')
                outputs[name]=dict(shape=list(value.shape),dtype=str(value.dtype),all_finite=True)
            call.update(status='PASSED',after=after,delta=delta,outputs=outputs);report.update(status='ACTUAL_MULTISTEP_H3_OBSERVED',actual_model_executed=True)
            print('H3_MULTI_REAL_H3_CALL_COMPLETED',mode,index,flush=True);return result
        except Exception as error:call.update(status='FAILED',error=str(error),after=counter.snapshot());report['status']='FAILED_MULTISTEP_H3';raise
        finally:write_json(report_path,report)
    MiniMaxH3TransformerRunner.__call__=wrapped
    return report
