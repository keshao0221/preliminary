"""Generate and cross-compile six NineToothed MRoPE objects without a GPU."""
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
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    import torch,triton,ninetoothed.generation as generation
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    definition=load('mrope_definition',Path(__file__).with_name('mrope_definition.py'))
    generation.CACHE_DIR=out/'generated'; generation.CACHE_DIR.mkdir(exist_ok=True)
    class Generator(generation.CodeGenerator):
        def __init__(self):
            ast.NodeTransformer.__init__(self); self._min_num_elements=1; self._max_num_elements=256
    source=Path(Generator()(definition.application,'torch','minimax_nine_mrope',4,1,1,False))
    shutil.copyfile(source,out/'ninetoothed_generated.py'); generated=load('generated_mrope',source); kernel=generated.minimax_nine_mrope
    launch=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='launch_minimax_nine_mrope')
    class View:
        def __init__(self,role,shape,strides): self.role,self.shape,self.strides=role,shape,strides
        def size(self,i): return self.shape[i]
        def stride(self,i): return self.strides[i]
    class Capture:
        def __getitem__(self,grid): self.grid=tuple(grid({})); return self.call
        def call(self,*values,**options): self.values=values
    capture=Capture(); namespace={'triton':triton,'minimax_nine_mrope':capture}
    exec(compile(ast.Module(body=[launch],type_ignores=[]),str(source),'exec'),namespace)
    namespace['launch_minimax_nine_mrope'](
        View('output',(17,3,128),(389,128,1)),View('source',(17,3,128),(401,128,1)),
        View('cosine',(31,48),(96,1)),View('sine',(31,48),(96,1)),View('positions',(3,17),(0,2)),48,16,16,16,0,31)
    arguments=[dict(name=n,role=v.role) if isinstance(v,View) else dict(name=n,value=int(v),scalar='i64') for n,v in zip(kernel.arg_names,capture.values)]
    if len(arguments)!=35: raise RuntimeError('Unexpected generated ABI count: '+str(len(arguments)))
    kernels=[]
    for data,scalar in [('f16','fp16'),('bf16','bf16'),('f32','fp32')]:
        for pos in ('i32','i64'):
            pair=data+'_'+pos
            signature={a['name']:('*'+(pos if a['role']=='positions' else scalar)) if 'role' in a else 'i64' for a in arguments}
            compiled=triton.compile(ASTSource(kernel,signature=signature),target=GPUTarget('hip','gfx936',64),options={'num_warps':4,'num_stages':1,'enable_fp_fusion':False})
            metadata=compiled.metadata._asdict()
            if metadata.get('global_scratch_size',0) or metadata.get('profile_scratch_size',0) or metadata['shared']: raise RuntimeError('Unexpected scratch/shared requirement')
            directory=out/'kernels'/pair; directory.mkdir(parents=True,exist_ok=True)
            for kind,value in compiled.asm.items(): (directory/('kernel.'+kind)).write_bytes(value if isinstance(value,bytes) else str(value).encode())
            (directory/'compiler_metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
            raw=(directory/'kernel.hsaco').read_bytes()
            if raw[:4]!=b'\x7fELF' or struct.unpack_from('<H',raw,18)[0]!=224: raise RuntimeError('Not AMDGPU ELF')
            header=next(line for line in (directory/'kernel.llir').read_text().splitlines() if line.startswith('define amdgpu_kernel'))
            actual=header.split('(',1)[1].split(') local_unnamed_addr',1)[0].split(', ')
            expected=['ptr' if 'role' in a else 'i64' for a in arguments]+['ptr','ptr']
            if len(actual)!=len(expected) or any(not a.startswith(t) for a,t in zip(actual,expected)): raise RuntimeError('LLVM ABI differs')
            kernels.append(dict(pair=pair,binary='kernels/'+pair+'/kernel.hsaco',sha256=hashlib.sha256(raw).hexdigest(),entrypoint=metadata['name'],block=[256,1,1],shared_bytes=0,signature=signature))
            print('COMPILED',pair,len(raw),flush=True)
    report=dict(schema_version=1,kernels=kernels,arguments=arguments,tile=list(definition.TILE),target=dict(backend='hip',arch='gfx936',warp_size=64),
        ninetoothed=package_version('ninetoothed'),triton=package_version('triton'),torch=package_version('torch'),hip=torch.version.hip,
        cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),local_visible_gpu_count=torch.cuda.device_count(),gpu_executed=False,llvm_abi_verified=True,
        generator_adaptation='Fixed host tile bounds; standard NineToothed generator retained',fp_fusion=False)
    (out/'generation_report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
