"""Stage and verify locally built outputs. No downloaded binary fallback."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_fingerprint(root):
    root=Path(root)
    paths=[p for p in (root/'source').rglob('*') if p.is_file()]
    paths.extend(root/name for name in ('build.py','config/build.json','config/platform_requirements.json','tools/build_outputs.py','tools/platform_environment.py'))
    records=[(p.relative_to(root).as_posix(),sha(p)) for p in sorted(paths)]
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def stage_python(work):
    work=Path(work)
    bindings=[]
    for project,role in (('InfiniCore','infinicore'),('InfiniLM','infinilm')):
        source=work/'src'/project/'python'/role
        target=work/'python'/role
        shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so'))
        candidates=sorted((work/'src'/project/'build/linux/x86_64/release').glob('_'+role+'*.so'))
        if len(candidates)!=1:raise RuntimeError('Missing/ambiguous locally built '+role+' binding')
        (target/'lib').mkdir(exist_ok=True)
        destination=target/'lib'/candidates[0].name
        shutil.copyfile(candidates[0],destination)
        bindings.append(dict(path=destination.relative_to(work).as_posix(),sha256=sha(destination)))
    sources=[dict(path=p.relative_to(work).as_posix(),sha256=sha(p)) for p in sorted((work/'python').rglob('*.py'))]
    return dict(bindings=bindings,sources=sources)


def load_build(root,work):
    root,work=Path(root),Path(work).resolve()
    report=json.loads((work/'build_report.json').read_text(encoding='utf-8'))
    expected=json.loads((root/'config/build.json').read_text(encoding='utf-8'))
    if (report.get('status')!='OFFLINE_SOURCE_BUILD_COMPLETED' or not report.get('source_only')
            or not report.get('native_complete') or not report.get('operators_complete')
            or report.get('source_snapshot_sha256')!=expected['native_sources_sha256']
            or report.get('source_inputs_sha256')!=source_fingerprint(root)):
        raise RuntimeError('Complete source-only build required; run build.py with this workdir')
    records=report['libraries']+report['kernels']+report['python_bindings']['bindings']+report['python_bindings']['sources']
    if len(report['kernels'])!=81:raise RuntimeError('Expected 81 locally compiled Nine objects')
    for item in records:
        relative=PurePosixPath(item['path'])
        if relative.is_absolute() or '..' in relative.parts or ':' in item['path'] or '\\' in item['path']:raise RuntimeError('Unsafe build artifact path')
        path=work.joinpath(*relative.parts)
        if not path.is_file() or sha(path)!=item['sha256']:raise RuntimeError('Changed source build output: '+item['path'])
    return report


def runtime_environment(work):
    import torch
    work=Path(work).resolve()
    assets=work/'rebuilt_assets'
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join(filter(None,[str(work/'python'),env.get('PYTHONPATH','')]))
    env['LD_LIBRARY_PATH']=os.pathsep.join([str(assets/'rms'),str(work/'infini/lib'),str(Path(torch.__file__).parent/'lib'),env.get('LD_LIBRARY_PATH','')])
    env.update(INFINI_ROOT=str(work/'infini'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    names=dict(cast='CAST',rotary='H3_ROTARY',mrope='MROPE',silu='SILU',swiglu='SWIGLU',rms='RMS_NORM',mha='MHA')
    for role,name in names.items():env['INFINICORE_NINETOOTHED_'+name+'_ROOT']=str(assets/role)
    return env
