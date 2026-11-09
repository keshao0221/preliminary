"""Offline source build for Linux/Python3.11/DTK gfx936. No GPU/model execution."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import signal
import subprocess
import sys
import time
import zipfile
import uuid

ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def extract(path,target,prefix=None):
    with zipfile.ZipFile(path) as z:
        for item in z.infolist():
            name=PurePosixPath(item.filename)
            if name.is_absolute() or '..' in name.parts or ':' in item.filename or '\\' in item.filename:raise RuntimeError('Unsafe source member')
            if prefix:
                if not name.parts or name.parts[0]!=prefix:continue
                name=PurePosixPath(*name.parts[1:])
            if not name.parts or item.is_dir():continue
            p=target.joinpath(*name.parts);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(item))
            mode=(item.external_attr>>16)&0o777
            if mode:p.chmod(mode)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',type=Path,required=True);p.add_argument('--stage',choices=('prepare','native','operators','all'),default='all');p.add_argument('--jobs',type=int,default=2);a=p.parse_args()
    if sys.platform!='linux' or sys.version_info[:2]!=(3,11):raise RuntimeError('Build requires Linux Python3.11 in the documented DTK environment')
    if os.environ.get('MINIMAX_PRELIM_BUILD_DTK_READY')!='1':
        env=os.environ.copy();env['MINIMAX_PRELIM_BUILD_DTK_READY']='1'
        os.execve('/bin/bash',['bash','-c','set -e; . /opt/dtk/env.sh >/dev/null; exec "$@"','prelim-build',sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
    if not 1<=a.jobs<=8:raise RuntimeError('Use 1–8 compile jobs')
    work=a.workdir.resolve();statepath=work/'build_report.json';config=json.loads((ROOT/'config/build.json').read_text(encoding='utf-8'))
    if work.exists() and any(work.iterdir()) and not statepath.exists():raise RuntimeError('Use an empty workdir; existing files are preserved')
    work.mkdir(parents=True,exist_ok=True);(work/'logs').mkdir(exist_ok=True)
    state=json.loads(statepath.read_text(encoding='utf-8')) if statepath.exists() else dict(status='STARTED',commands=[],local_gpu_executed=False,model_executed=False,native_complete=False,operators_complete=False)
    policy_sha256=digest(ROOT/'config/platform_requirements.json')
    from tools.build_outputs import source_fingerprint
    inputs_sha256=source_fingerprint(ROOT)
    if state.get('prepared') and state.get('source_inputs_sha256')!=inputs_sha256:
        raise RuntimeError('Source/build inputs changed; preserve the old workdir and use a new one')
    state['source_inputs_sha256']=inputs_sha256
    if state.get('prepared') and state.get('platform_policy_sha256')!=policy_sha256:
        raise RuntimeError('Platform dependency requirements changed; use a new workdir')
    state['platform_policy_sha256']=policy_sha256
    state['source_snapshot_sha256']=config['native_sources_sha256'];state['source_only']=True;write(statepath,state)
    deadline=time.monotonic()+1450
    def run(command,name,env=None,cwd=None,timeout=1200):
        staged_install=None
        if len(command)>1 and command[1]=='install':
            staged_install=work/'staged_installs'/(name+'_'+uuid.uuid4().hex[:8])
            staged_install.mkdir(parents=True,exist_ok=False)
            command=list(command)
            command.insert(3,'--installdir='+str(staged_install))
        timeout=min(timeout,max(1,deadline-time.monotonic()))
        print('PRELIM_BUILD_BEGIN',name,flush=True);started=time.monotonic()
        logfile=work/'logs'/(name+'_'+str(len(state['commands'])).zfill(3)+'.log')
        try:
            with logfile.open('w') as log:
                child=subprocess.Popen([str(v) for v in command],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=timeout)
                except BaseException:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait();raise
                r=subprocess.CompletedProcess(command,code)
        except subprocess.TimeoutExpired:
            state['commands'].append(dict(name=name,command=list(map(str,command)),returncode=None,timed_out=True,seconds=time.monotonic()-started,log='logs/'+logfile.name));write(statepath,state);raise
        state['commands'].append(dict(name=name,command=list(map(str,command)),returncode=r.returncode,timed_out=False,seconds=time.monotonic()-started,log='logs/'+logfile.name));write(statepath,state)
        if r.returncode:
            print(logfile.read_text(encoding='utf-8',errors='replace')[-9000:]);raise RuntimeError('Build failed: '+name)
        if staged_install is not None:
            outputs=[]
            for source in sorted(staged_install.rglob('*')):
                if not source.is_file():continue
                relative=source.relative_to(staged_install)
                target=work/'infini'/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                expected=digest(source)
                if target.exists() and digest(target)==expected:
                    action='REUSED'
                else:
                    if target.exists():
                        backup=work/'install_backups'/staged_install.name/relative
                        backup.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(target,backup)
                    shutil.copyfile(source,target)
                    action='COPIED'
                if digest(target)!=expected:raise RuntimeError('Installed file differs: '+str(relative))
                outputs.append(dict(path=relative.as_posix(),sha256=expected,action=action))
            if not outputs:raise RuntimeError('Staged installation produced no files')
            state.setdefault('staged_installs',{})[name]=dict(stage=str(staged_install),outputs=outputs)
            state['commands'][-1]['seconds']=time.monotonic()-started
            write(statepath,state)
        print('PRELIM_BUILD_COMPLETED',name,flush=True)
    try:
        from tools.platform_environment import check,build_environment
        provider=check(ROOT,'compile')
        signature=dict(packages={name:dict(version=item['version'],module_sha256=item.get('module_sha256'))
                                for name,item in provider['packages'].items()},
                       apis={item['module']:item.get('module_sha256') for item in provider['api_checks']})
        if state.get('prepared') and state.get('compile_package_versions')!=signature:
            raise RuntimeError('Compilation dependency versions changed; use a new workdir')
        state['compile_package_versions']=signature
        state['platform_compile_environment']=provider
        if digest(ROOT/'source/native_sources.zip')!=config['native_sources_sha256']:raise RuntimeError('Changed native source snapshot')
        if not state.get('prepared'):
            extract(ROOT/'source/native_sources.zip',work/'src')
            state['prepared']=True;write(statepath,state)
        env=build_environment(work);python=Path(sys.executable);xmake=shutil.which('xmake')
        if xmake is None:raise RuntimeError('Select the competition build environment providing XMake')
        core=work/'src/InfiniCore';lm=work/'src/InfiniLM'
        if not state.get('private_xmake_configured'):
            run([xmake,'g','-y','--network=private','--theme=plain'],'configure_offline_xmake',env,work)
            state['private_xmake_configured']=True;write(statepath,state)
        if a.stage in ('native','all') and not state['native_complete']:
            run([xmake,'f','-y','-m','release','--hygon-dcu=y','--hygon-arch=gfx936','--cuda=/opt/dtk/cuda/cuda','--ccl=y','--aten=y','--cpu=y','--omp=y'],'configure_core',env,core)
            run([xmake,'build','-y','-j',a.jobs],'build_core',env,core)
            run([xmake,'install','-y'],'install_core',env,core)
            run([xmake,'build','-y','-j',a.jobs,'_infinicore'],'build_core_python',env,core)
            run([xmake,'install','-y','_infinicore'],'install_core_python',env,core)
            run([xmake,'f','-y','-m','release','--cxx11-abi=1'],'configure_lm',env,lm)
            run([xmake,'build','-y','-j',a.jobs,'_infinilm'],'build_lm_python',env,lm)
            run([xmake,'install','-y','_infinilm'],'install_lm_python',env,lm)
            state['native_complete']=True;write(statepath,state)
        if a.stage in ('operators','all') and not state['operators_complete']:
            if not state['native_complete']:raise RuntimeError('Run --stage native before --stage operators')
            assets=work/'rebuilt_assets';assets.mkdir(exist_ok=True)
            tasks=[('cast','--rank',v,'rank'+str(v)) for v in (1,8)]+[('rotary',None,None,''),('mrope',None,None,'')]+[(r,'--rank',v,'rank'+str(v)) for r in ('silu','swiglu') for v in (1,8)]+[('rms','--tile',v,'tile'+str(v)) for v in (128,1024,8192)]+[('mha','--head-tile',v,'tile'+str(v)) for v in (64,128)]
            for role,flag,value,subdir in tasks:
                command=[python,ROOT/'source/operators'/role/'build_kernels.py','--output',assets/role/subdir]
                if flag:command.extend((flag,value))
                run(command,'nine_'+role+'_'+(subdir or 'all'),env,timeout=600)
            for role in ('cast','rotary','mrope','silu','swiglu','rms','mha'):
                source=ROOT/'source/operators'/role;target=assets/role;target.mkdir(exist_ok=True);stem='rms_norm' if role=='rms' else role
                command=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC','-pthread','-D__HIP_PLATFORM_AMD__','-I/opt/dtk/include',source/'host_runtime.cpp','-L/opt/dtk/lib','-lamdhip64','-Wl,-rpath,/opt/dtk/lib','-o',target/('libminimax_nine_'+stem+'_runtime.so')]
                run(command,'host_'+role,env)
            for name in ('libinfinicore_cpp_api.so','libinfiniop.so','libinfinirt.so','libinfiniccl.so'):shutil.copyfile(work/'infini/lib'/name,assets/'rms'/name)
            libraries=[dict(path=p.relative_to(work).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(assets.rglob('*.so'))]
            kernels=[dict(path=p.relative_to(work).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(assets.rglob('*.hsaco'))]
            if len(kernels)!=81:raise RuntimeError('Expected 81 Nine-generated objects; got '+str(len(kernels)))
            state.update(operators_complete=True,libraries=libraries,kernels=kernels);write(statepath,state)
        if state['native_complete'] and state['operators_complete']:
            from tools.build_outputs import stage_python
            state['python_bindings']=stage_python(work)
        state['status']='OFFLINE_SOURCE_BUILD_COMPLETED' if state['native_complete'] and state['operators_complete'] else 'REQUESTED_SOURCE_BUILD_STAGE_COMPLETED'
        state.pop('error',None)
    except subprocess.TimeoutExpired as error:
        state['status']='SOURCE_BUILD_CHECKPOINT_SAVED';state['checkpoint_reason']=str(error);write(statepath,state)
        print('PRELIM_BUILD_CHECKPOINT_SAVED repeat the same command to continue',flush=True);return 75
    except BaseException as error:
        state['status']='FAILED_SOURCE_BUILD';state['error']=str(error);write(statepath,state);raise
    write(statepath,state);print('PRELIM_BUILD_STATUS',state['status'],flush=True)

if __name__=='__main__':raise SystemExit(main() or 0)
