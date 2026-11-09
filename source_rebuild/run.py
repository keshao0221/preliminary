"""Run the unchanged public workload using this checkout's source-built outputs."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from tools.build_outputs import load_build,runtime_environment,sha
from tools.platform_environment import check

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
from case_contract import CASE,arguments

CPU_BINDING_PROBE=r'''
import json
from pathlib import Path
import torch,infinicore,infinilm
import sys
sys.path.insert(0,sys.argv[2])
from native_bridge import install_bridge_fix
install_bridge_fix()
from infinicore.utils import to_infinicore_dtype
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
from infinicore.lib import _infinicore
from infinilm.lib import _infinilm
root=Path(__import__('sys').argv[1]).resolve()
if not Path(_infinicore.__file__).resolve().is_relative_to(root) or not Path(_infinilm.__file__).resolve().is_relative_to(root):
    raise RuntimeError('A binding outside this source build was imported')
records=[]
for dtype in (torch.float32,torch.float64,torch.int64):
    source=torch.arange(32,dtype=dtype).reshape(4,8)[:,1::2]
    converted=infinicore.from_torch(source)
    if converted.data_ptr()!=source.data_ptr() or list(converted.shape)!=list(source.shape) or list(converted.stride())!=list(source.stride()):
        raise RuntimeError('Actual framework input conversion pointer/shape/stride differs')
    if converted.dtype!=to_infinicore_dtype(source.dtype):
        raise RuntimeError('Actual framework dtype conversion differs')
    records.append(dict(dtype=str(dtype),shape=list(source.shape),stride=list(source.stride()),pointer_preserved=True))
print(json.dumps(dict(status='SOURCE_BINDINGS_AND_CPU_INPUT_CONVERSION_VERIFIED',records=records,gpu_abi_verified=False,model_executed=False)))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-build',type=Path,required=True)
    parser.add_argument('--first-frame',type=Path,required=True)
    parser.add_argument('--env-root',type=Path,required=True)
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if sys.platform!='linux' or sys.version_info[:2]!=(3,11):raise RuntimeError('Use competition Linux/Python 3.11')
    work=args.from_build.resolve()
    directory=args.env_root.resolve()/'toolchain_offline_runs'/(datetime.now(timezone.utc).strftime('h3source_%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True,exist_ok=False)
    print('H3_PUBLIC_EVIDENCE_DIRECTORY',directory,flush=True)
    report=dict(status='STARTED',actual_model_executed=False,source_only=True,training_executed=False,preliminary_admission_passed=False)
    started=time.monotonic()
    def interrupted(signum,frame):raise TimeoutError('Entry interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:
        report['platform_environment']=check(ROOT,'runtime')
        build=load_build(ROOT,work)
        if build.get('validation_fixture') and not args.verify_only:raise RuntimeError('CPU validation fixtures cannot run the model')
        report['build_report_sha256']=sha(work/'build_report.json')
        if args.first_frame.stat().st_size!=CASE['image_bytes'] or sha(args.first_frame)!=CASE['image_sha256']:
            raise RuntimeError('Public input differs; use the documented original first frame')
        image=directory/'first_frame.jpg';shutil.copyfile(args.first_frame,image)
        config=json.loads((ROOT/'config/run.json').read_text(encoding='utf-8'))
        assets=work/'rebuilt_assets'
        config['core_sha256']=sha(assets/'rms/libinfinicore_cpp_api.so')
        for role in ('cast','rotary','mrope','silu','swiglu','rms','mha'):
            stem='rms_norm' if role=='rms' else role
            key='mha_runtime_sha256' if role=='mha' else role+'_host_sha256'
            config[key]=sha(assets/role/('libminimax_nine_'+stem+'_runtime.so'))
        config['first_frame_path']=str(image)
        (directory/'observer_config.json').write_text(json.dumps(config,indent=2)+'\n')
        env=runtime_environment(work)
        env.update(MINIMAX_SMOKE_LOG_DIRECTORY=str(directory),MINIMAX_PUBLIC_CONFIG=str(directory/'observer_config.json'))
        def run(name,command,limit):
            with (directory/(name+'.log')).open('wb') as log:
                child=subprocess.Popen(list(map(str,command)),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=limit)
                except BaseException:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait();raise
            if code:raise RuntimeError(name+' failed; return the complete evidence ZIP')
        run('cpu_bindings',[sys.executable,'-c',CPU_BINDING_PROBE,work,ROOT/'runtime'],90)
        if args.verify_only:
            report['status']='SOURCE_BINDINGS_CPU_VERIFIED_GPU_PENDING'
        else:
            from rocblas_runtime import select_catalog
            selection=select_catalog(env);env.update(selection['environment_updates'])
            env['MINIMAX_EXPECTED_ROCBLAS_CATALOG']=selection['selected']['directory']
            (directory/'rocblas_catalog.json').write_text(json.dumps(selection,indent=2)+'\n')
            model=args.model_path or Path(config['model_path'])
            if not model.is_dir():raise RuntimeError('Competition model directory not found: '+str(model))
            model_args=arguments('candidate',image,directory/'public_dog.mp4')
            model_args[model_args.index('--model-path')+1]=str(model)
            entry=work/'src/InfiniLM/examples/minimax_h3.py'
            report['model_arguments']=model_args
            run('minimax_h3_full',[sys.executable,'-u',ROOT/'runtime/model_wrapper.py',ROOT/'runtime/native_bridge.py',entry,*model_args],max(1,1100-(time.monotonic()-started)))
            run('output_inspection',[sys.executable,ROOT/'runtime/inspect_run.py','--directory',directory],90)
            report.update(status='SOURCE_BUILT_PUBLIC_GENERATION_VERIFIED',actual_model_executed=True,actual_h3_calls=9)
    except BaseException:
        report.update(status='FAILED_SOURCE_BUILT_ENTRY',traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    report['entry_wall_seconds']=time.monotonic()-started
    (directory/'entry_report.json').write_text(json.dumps(report,indent=2)+'\n')
    evidence=directory.with_suffix('.zip')
    with zipfile.ZipFile(evidence,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(directory).as_posix())
    print('H3_PUBLIC_STATUS',report['status'],flush=True)
    print('H3_PUBLIC_EVIDENCE_ZIP',evidence,flush=True)
    return 2 if report['status']=='FAILED_SOURCE_BUILT_ENTRY' else 0


if __name__=='__main__':raise SystemExit(main())
