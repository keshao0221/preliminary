"""Execute one immutable public case with checked image-conditioning hooks."""
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
from case_contract import CASE
from inventory import PhaseInventory
from vae_silu_inventory import VideoSiluInventory,assess_vae
from sample_observer import install,assess_complete
from observer import write_json,sha256
from modes import ROOT_VARIABLES
from rocblas_runtime import model_binding

def main():
    directory=Path(os.environ['MINIMAX_SMOKE_LOG_DIRECTORY']);config=json.loads(Path(os.environ['MINIMAX_PUBLIC_CONFIG']).read_text())
    mode=config['public_mode'];audit=VideoSiluInventory(os.environ['INFINICORE_NINETOOTHED_SILU_ROOT']);state=None
    if mode!='candidate':raise RuntimeError('VAE SiLU entry is candidate only')
    try:
        image=Path(config['first_frame_path'])
        if sha256(image)!=CASE['image_sha256']:raise RuntimeError('Actual model first frame differs')
        binding=model_binding(os.environ['MINIMAX_EXPECTED_ROCBLAS_CATALOG'])
        roots=[os.environ[n] for n in ROOT_VARIABLES]
        state,_=install(roots,directory,config,mode,audit)
        from mha_model_observer import install as install_mha
        finish_mha,_,_=install_mha(directory,config)
        sys.argv=sys.argv[1:]
        from diffusers.models.attention_dispatch import attention_backend
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with audit,attention_backend('_native_math'),sdpa_kernel(SDPBackend.MATH):
            try:runpy.run_path(sys.argv[0],run_name='__main__');code=0
            except SystemExit as error:code=error.code if isinstance(error.code,int) else 0 if error.code is None else 1
        if code:raise RuntimeError('Model exited '+str(code))
        assess_complete(state)
        finish_mha(state)
        proof=assess_vae(audit.report())
        write_json(directory/'vae_silu_acceptance.json',proof)
        print('H3_PUBLIC_VAE_SILU_COMPLETED',json.dumps(proof['counts']),flush=True)
        binding=model_binding(binding['catalog_directory'])
        write_json(directory/'model_wrapper_report.json',dict(status='PUBLIC_CASE_MODEL_RETURNED',mode=mode,h3_calls=len(state['calls']),exit_code=0,rocblas_binding=binding,attention_backend='_native_math'))
        return 0
    except BaseException:
        write_json(directory/'model_wrapper_report.json',dict(status='FAILED_PUBLIC_CASE_MODEL',traceback=traceback.format_exc(),mode=mode,
            completed_h3_calls=sum(c['status']=='PASSED' for c in state['calls']) if state else 0))
        raise
    finally:
        write_json(directory/'compute_inventory.json',audit.report())

if __name__=='__main__':raise SystemExit(main())
