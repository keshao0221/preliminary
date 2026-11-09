"""Validate the APIs used by this project on the provided Python environment."""
import importlib
import importlib.metadata as metadata
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import traceback


def numpy_interop(modules):
    np,torch=modules['numpy'],modules['torch']
    source=np.arange(12,dtype=np.float32).reshape(3,4)
    converted=torch.from_numpy(source)
    np.testing.assert_array_equal((converted+1).numpy(),source+1)
    assert converted.data_ptr()==source.ctypes.data
    return dict(dtype='float32',shape=[3,4],values_preserved=True,pointer_preserved=True)


def safetensors_roundtrip(modules):
    torch=modules['torch']
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    with tempfile.TemporaryDirectory(prefix='h3_api_safetensors_') as folder:
        path=Path(folder)/'compatibility.safetensors'
        tensors={str(dtype).split('.')[-1]:torch.arange(12,dtype=dtype).reshape(3,4)
                 for dtype in (torch.float32,torch.float16,torch.bfloat16,torch.float64,torch.int64)}
        save_file(tensors,str(path),metadata={'source':'platform_api_check'})
        restored=load_file(str(path),device='cpu')
        assert set(restored)==set(tensors)
        for name,value in tensors.items():
            assert restored[name].dtype==value.dtype and torch.equal(restored[name],value)
        with safe_open(str(path),framework='pt',device='cpu') as handle:
            assert handle.metadata()=={'source':'platform_api_check'}
    return dict(dtypes=list(tensors),values_dtypes_and_metadata_preserved=True)


def pillow_roundtrip(modules):
    from PIL import Image
    image=Image.new('RGB',(4,3),(17,83,149))
    data=io.BytesIO();image.save(data,format='PNG');data.seek(0)
    with Image.open(data) as loaded:
        assert loaded.mode=='RGB' and loaded.size==image.size and loaded.tobytes()==image.tobytes()
    return dict(rgb_values_preserved=True,size=[4,3])


def ssim_api(modules):
    from skimage.metrics import structural_similarity
    np=modules['numpy'];image=np.arange(256,dtype=np.float32).reshape(16,16)
    value=float(structural_similarity(image,image,data_range=255.0))
    assert value==1.0
    return dict(identical_image_ssim=value,official_threshold_defined=False)


PROBES={'numpy_torch':numpy_interop,'safetensors':safetensors_roundtrip,'pillow':pillow_roundtrip,'ssim':ssim_api}
NEEDS={'numpy_torch':('numpy','torch'),'safetensors':('safetensors','torch'),'pillow':('pillow',),'ssim':('numpy','scikit-image')}


def check(root,role):
    import sys
    root=Path(root)
    policy=json.loads((root/'config/platform_requirements.json').read_text(encoding='utf-8'))
    selected=policy[role]
    report=dict(role=role,python=sys.executable,policy='actual_project_api_compatibility',packages={},tools={},
                api_checks=[],compatibility_probes=[],optional_packages={},issues=[],errors=[],
                automatic_installation=False,automatic_upgrade=False,external_distribution_metadata_changed=False,
                dependency_versions_recorded_without_old_lock_enforcement=True,full_fl2va_verified=False)
    modules={};broken_roots=set()
    def issue(kind,component,message):
        report['issues'].append(dict(kind=kind,component=component,message=message))
        report['errors'].append(component+': '+message)
    for name,module_name in {'torch':'torch',**selected['required_modules']}.items():
        row=dict(module=module_name,version=None,file=None)
        try:
            distribution=metadata.distribution(name)
            row.update(version=distribution.version,distribution_location=str(distribution.locate_file('')),
                       declared_requirements=distribution.requires or [])
        except metadata.PackageNotFoundError:
            row['distribution_metadata']='not installed as a distribution; check importable source instead'
        try:
            module=importlib.import_module(module_name)
            modules[name]=module
            row['file']=getattr(module,'__file__',None)
            if row['file'] and Path(row['file']).is_file():
                row['module_sha256']=hashlib.sha256(Path(row['file']).read_bytes()).hexdigest()
            if row['version'] is None:row['version']=getattr(module,'__version__',None)
            row['imported']=True
        except ModuleNotFoundError as error:
            row['imported']=False;broken_roots.add(module_name.split('.')[0])
            issue('missing_module',name,str(error))
        except Exception:
            row['imported']=False;broken_roots.add(module_name.split('.')[0])
            issue('module_import_failed',name,traceback.format_exc())
        report['packages'][name]=row
    torch=modules.get('torch')
    if torch is not None:
        report['vendor']=dict(torch=report['packages']['torch']['version'],hip=torch.version.hip,
                              cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),file=str(Path(torch.__file__).resolve()))
        for key,expected in policy['vendor'].items():
            if report['vendor'][key]!=expected:issue('native_abi_requirement',key,'expected '+repr(expected)+'; active '+repr(report['vendor'][key]))
    for name in selected.get('tools',[]):
        report['tools'][name]=shutil.which(name)
        if report['tools'][name] is None:issue('missing_tool',name,'not found on PATH')
    for module_name,symbols in selected.get('apis',{}).items():
        if module_name.split('.')[0] in broken_roots:continue
        try:
            module=importlib.import_module(module_name)
            for symbol in symbols:getattr(module,symbol)
            path=getattr(module,'__file__',None)
            report['api_checks'].append(dict(module=module_name,symbols=symbols,status='PASS',file=path,
                module_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None))
        except Exception:issue('required_api_unavailable',module_name,traceback.format_exc())
    # Missing unrelated vision/audio extras do not block this first-frame case.
    # The unchanged pipeline will still reject an unsupported audio-reference task.
    for name,module_name in selected.get('optional_modules',{}).items():
        try:
            version=metadata.version(name)
        except metadata.PackageNotFoundError:version=None
        report['optional_packages'][name]=dict(version=version,required_for_this_public_fl2va=False)
    for name in selected.get('probes',[]):
        if any(value not in modules for value in NEEDS[name]):
            report['compatibility_probes'].append(dict(name=name,status='NOT_RUN_MISSING_MODULE'))
            continue
        try:
            details=PROBES[name](modules)
            report['compatibility_probes'].append(dict(name=name,status='PASS',details=details))
        except Exception:issue('compatibility_probe_failed',name,traceback.format_exc())
    missing=any(v['kind'] in ('missing_module','missing_tool') for v in report['issues'])
    report['status']='REQUIRED_COMPONENTS_UNAVAILABLE' if missing else 'PROJECT_API_COMPATIBILITY_FAILED' if report['errors'] else 'PROJECT_API_CHECK_PASSED'
    print('H3_PLATFORM_ENVIRONMENT '+json.dumps(report),flush=True)
    if report['errors']:
        raise RuntimeError('Project API check failed; supplied package versions were not upgraded or replaced.\n'+'\n'.join(report['errors']))
    return report
