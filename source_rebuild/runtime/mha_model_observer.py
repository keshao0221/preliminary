"""Observe real native MHA launches around the existing H3 observer."""
import ctypes
import functools
import os
from pathlib import Path
from cast_observer_base import mapped_core_paths,sha256,write_json
from mha_contract import FIELDS,assess,complete,require

class Counter:
    def __init__(self,root):
        self.lib=ctypes.CDLL(str(Path(root)/'libminimax_nine_mha_runtime.so'))
        self.lib.minimax_nine_mha_device_statistics.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
        self.lib.minimax_nine_mha_device_statistics.restype=None
    def snapshot(self):
        result={}
        for d in range(8):
            values=(ctypes.c_uint64*21)();self.lib.minimax_nine_mha_device_statistics(d,values)
            result[str(d)]=dict(zip(FIELDS,map(int,values)))
        return result

def install(directory,config,runner_class=None):
    if runner_class is None:
        from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
        runner_class=MiniMaxH3TransformerRunner
    root=Path(os.environ['INFINICORE_NINETOOTHED_MHA_ROOT']).resolve();counter=Counter(root)
    paths=mapped_core_paths()
    require(len(paths)==1 and sha256(Path(paths[0]))==config['core_sha256'],'loaded core differs')
    require(sha256(root/'libminimax_nine_mha_runtime.so')==config['mha_runtime_sha256'],'MHA runtime differs')
    report=dict(status='NINE_MHA_MODEL_OBSERVER_INSTALLED',calls=[],core_sha256=config['core_sha256'],
        runtime_sha256=config['mha_runtime_sha256'],mapped_core_paths=paths,root=str(root),
        root_variable='INFINICORE_NINETOOTHED_MHA_ROOT',fallback_supported=False,
        actual_model_executed=False,training_executed=False)
    target=Path(directory)/'mha_model_observer.json';write_json(target,report);original=runner_class.__call__
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):
        import torch
        inputs=args[0] if args else kwargs['inputs'];hf=self.engine.hf_config
        for d in range(8):torch.cuda.synchronize(d)
        call=dict(status='RUNNING',index=len(report['calls'])+1,rows=int(inputs.position_ids.shape[0]),
            h3_config_geometry={k:hf[k] for k in ('num_layers','num_refiner_layers','token_refiner_num_layers','num_attention_heads','attention_head_dim','hidden_size','ffn_dim','rope_freq_dim') if k in hf},before=counter.snapshot())
        require(call['index']<=9,'extra H3 call');report['calls'].append(call);write_json(target,report)
        try:
            result=original(self,*args,**kwargs)
            for d in range(8):torch.cuda.synchronize(d)
            after=counter.snapshot();delta=assess(call['before'],after,call['h3_config_geometry'],call['rows'])
            call.update(status='PASSED',after=after,delta=delta);report['actual_model_executed']=True
            print('H3_PUBLIC_MHA_CALL_COMPLETED candidate',call['index'],flush=True)
            return result
        except BaseException as error:
            call.update(status='FAILED',error=str(error),after=counter.snapshot());report['status']='FAILED_NINE_MHA_MODEL';raise
        finally:write_json(target,report)
    runner_class.__call__=wrapped
    def finish(state):
        report['final_statistics']=counter.snapshot();proof=complete(report,state,config)
        report['status']='NINE_MHA_MODEL_DISPATCH_COMPLETE';write_json(target,report)
        write_json(Path(directory)/'mha_model_acceptance.json',proof)
        print('H3_PUBLIC_MHA_COMPLETED',proof['total_launches'],flush=True)
        return proof
    return finish,report,(runner_class,original)
