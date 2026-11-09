"""Cross-compile RMSNorm and inspect the real generated launcher and LLVM ABI."""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys

PAIRS=(('f32','f32'),('f16','f16'),('f16','bf16'),('f16','f32'),('bf16','bf16'),('bf16','f16'),('bf16','f32'))

def package_version(name):
    """Record available metadata without rejecting source-provided modules."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return getattr(sys.modules.get(name), '__version__', None)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; spec.loader.exec_module(module); return module

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--tile',type=int,choices=(128,1024,8192),required=True)
    args=parser.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    import torch,triton,ninetoothed.generation as generation
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    # Specialize the NineToothed DSL source before invoking its standard generator.
    template=Path(__file__).with_name('rms_norm_definition.py').read_text()
    import re
    specialized=re.sub(r'\bTILE\b',str(args.tile),template.replace('TILE=128',''))
    definition_path=output/'rms_norm_definition_specialized.py'; definition_path.write_text(specialized)
    definition=load('rms_norm_definition',definition_path); definition.annotate()
    generation.CACHE_DIR=output/'generated'; generation.CACHE_DIR.mkdir(exist_ok=True)
    class Generator(generation.CodeGenerator):
        def __init__(self):
            ast.NodeTransformer.__init__(self); self._min_num_elements=1; self._max_num_elements=args.tile
    source=Path(Generator()(definition.application,'torch','minimax_nine_rms_norm',1,1,1,False))
    shutil.copyfile(source,output/'ninetoothed_generated.py'); module=load('generated_rms_norm',source); kernel=module.minimax_nine_rms_norm
    launch=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='launch_minimax_nine_rms_norm')
    class View:
        def __init__(self,role,shape,strides): self.role,self.shape,self.strides=role,shape,strides
        def size(self,i): return self.shape[i]
        def stride(self,i): return self.strides[i]
    class Capture:
        def __getitem__(self,grid): self.grid=tuple(grid({})); return self.call
        def call(self,*values,**options): self.values=values
    fixtures=[]
    for shape,inputs,outputs in (((3,7,17),(357,34,1),(119,17,1)),((2,3,128),(384,128,1),(384,128,1)),((3,1,1),(2**32+1,0,1),(1,0,1))):
        width=min(shape[-1],args.tile); shape=(*shape[:2],width)
        capture=Capture(); namespace={'triton':triton,'minimax_nine_rms_norm':capture}
        exec(compile(ast.Module(body=[launch],type_ignores=[]),str(source),'exec'),namespace)
        namespace['launch_minimax_nine_rms_norm'](View('input',shape,inputs),View('weight',shape,(0,0,1)),View('output',shape,outputs),1e-5,width)
        arguments=[dict(name=n,role=v.role) if isinstance(v,View) else dict(name=n,value=v,scalar='fp32' if isinstance(v,float) else 'i64') for n,v in zip(kernel.arg_names,capture.values)]
        fixtures.append(dict(shape=list(shape),input_strides=list(inputs),output_strides=list(outputs),grid=list(capture.grid),arguments=arguments))
    kernels=[]; scalar={'f32':'fp32','f16':'fp16','bf16':'bf16'}
    for data,weight in PAIRS:
        pair=data+'_'+weight
        signature={a['name']:'*'+scalar[weight if a['role']=='weight' else data] if 'role' in a else a['scalar'] for a in fixtures[0]['arguments']}
        compiled=triton.compile(ASTSource(kernel,signature=signature),target=GPUTarget('hip','gfx936',64),options={'num_warps':1,'num_stages':1,'enable_fp_fusion':False})
        metadata=compiled.metadata._asdict()
        if metadata.get('global_scratch_size',0) or metadata.get('profile_scratch_size',0): raise RuntimeError('Unexpected scratch requirement')
        directory=output/'kernels'/pair; directory.mkdir(parents=True,exist_ok=True)
        for kind,value in compiled.asm.items(): (directory/('kernel.'+kind)).write_bytes(value if isinstance(value,bytes) else str(value).encode())
        (directory/'compiler_metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
        raw=(directory/'kernel.hsaco').read_bytes()
        if raw[:4]!=b'\x7fELF' or struct.unpack_from('<H',raw,18)[0]!=224: raise RuntimeError('Not AMDGPU ELF')
        header=next(line for line in (directory/'kernel.llir').read_text().splitlines() if line.startswith('define amdgpu_kernel'))
        actual=header.split('(',1)[1].split(') local_unnamed_addr',1)[0].split(', ')
        expected=['ptr' if 'role' in a else 'float' if a['scalar']=='fp32' else 'i64' for a in fixtures[0]['arguments']]+['ptr','ptr']
        if len(actual)!=len(expected) or any(not a.startswith(t) for a,t in zip(actual,expected)): raise RuntimeError('LLVM ABI mismatch')
        kernels.append(dict(pair=pair,binary='kernels/'+pair+'/kernel.hsaco',sha256=hashlib.sha256(raw).hexdigest(),entrypoint=metadata['name'],block=[64,1,1],shared_bytes=metadata['shared'],signature=signature))
        print('COMPILED',args.tile,pair,len(raw),metadata['shared'],flush=True)
    report=dict(schema_version=1,kernels=kernels,fixtures=fixtures,tile=args.tile,target=dict(backend='hip',arch='gfx936',warp_size=64),
        ninetoothed=package_version('ninetoothed'),triton=package_version('triton'),torch=package_version('torch'),hip=torch.version.hip,
        cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),local_visible_gpu_count=torch.cuda.device_count(),gpu_executed=False,llvm_abi_verified=True,
        generation_adaptation='Fixed host tile bound, standard NineToothed generator, no hand-edited generated device code')
    (output/'generation_report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
