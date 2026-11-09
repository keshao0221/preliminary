"""Cross-compile the actual NineToothed SwiGLU definition; never execute a GPU."""
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
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rank',type=int,choices=(1,8),default=8)
    args=parser.parse_args()
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    import ninetoothed.generation as generation
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    import torch
    definition=load('swiglu_definition',Path(__file__).with_name('swiglu_definition.py'))
    definition.RANK=args.rank
    import ninetoothed
    definition.application.__annotations__=dict(zip(('up','gate','output'),definition.arrangement(ninetoothed.Tensor(args.rank,name='a_up'),ninetoothed.Tensor(args.rank,name='b_gate'),ninetoothed.Tensor(args.rank,name='c_out'))))
    generation.CACHE_DIR=output/'generated'; generation.CACHE_DIR.mkdir(exist_ok=True)
    class FixedTileGenerator(generation.CodeGenerator):
        def __init__(self):
            ast.NodeTransformer.__init__(self)
            self._min_num_elements=1; self._max_num_elements=definition.TILE
    source=Path(FixedTileGenerator()(definition.application,'torch','minimax_nine_swiglu',4,1,1,False))
    shutil.copyfile(source,output/'ninetoothed_generated.py')
    module=load('generated_swiglu',source)
    kernel=module.minimax_nine_swiglu
    # Capture the real generated launcher on CPU descriptors, including a rank-8
    # fixture. Scalar widths are explicitly int64, rather than inferred from a
    # small fixture; host metadata may exceed the signed 32-bit range.
    tree=ast.parse(source.read_text())
    launch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='launch_minimax_nine_swiglu')
    class View:
        def __init__(self,role,shape,strides): self.role,self.shape,self.strides=role,shape,strides
        def size(self,i): return self.shape[i]
        def stride(self,i): return self.strides[i]
    class Capture:
        def __getitem__(self,grid): self.grid=tuple(grid({})); return self.call
        def call(self,*values,**options): self.values,self.options=values,options
    fixtures=[]
    views=[([2145],[1],[1]),([2145],[2],[3]),([2145],[2**31+257],[1])] if args.rank==1 else [([1]*7+[2145],[1]*8,[1]*8),([1]*6+[33,65],[0]*6+[132,2],[0]*6+[132,2]),([2]*8,[128,64,32,16,8,4,2,1],[1,2,4,8,16,32,64,128]),([1]*7+[2**31+257],[0]*7+[1],[0]*7+[3])]
    for shape,istrides,ostrides in views:
        capture=Capture(); namespace={'triton':triton,'minimax_nine_swiglu':capture}
        exec(compile(ast.Module(body=[launch],type_ignores=[]),str(source),'exec'),namespace)
        namespace['launch_minimax_nine_swiglu'](View('up',shape,istrides),View('gate',shape,ostrides),View('output',shape,ostrides))
        arguments=[dict(name=n,role=v.role) if isinstance(v,View) else dict(name=n,value=int(v)) for n,v in zip(kernel.arg_names,capture.values)]
        fixtures.append(dict(shape=shape,input_strides=istrides,output_strides=ostrides,grid=list(capture.grid),arguments=arguments))
    kernels=[]
    types=[('f32','fp32'),('f16','fp16'),('bf16','bf16')]
    for input_name,input_scalar in types:
        for output_name,output_scalar in [(input_name,input_scalar)]:
            name=input_name
            signature={a['name']:'*'+input_scalar if 'role' in a else 'i64' for a in fixtures[0]['arguments']}
            compiled=triton.compile(ASTSource(kernel,signature=signature),target=GPUTarget('hip','gfx936',64),options={'num_warps':4,'num_stages':1,'enable_fp_fusion':False})
            metadata=compiled.metadata._asdict()
            if metadata.get('global_scratch_size',0) or metadata.get('profile_scratch_size',0): raise RuntimeError('Unexpected scratch requirement')
            directory=output/'kernels'/name; directory.mkdir(parents=True,exist_ok=True)
            for kind,value in compiled.asm.items():
                (directory/('kernel.'+kind)).write_bytes(value if isinstance(value,bytes) else str(value).encode())
            (directory/'compiler_metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
            raw=(directory/'kernel.hsaco').read_bytes()
            if raw[:4]!=b'\x7fELF' or struct.unpack_from('<H',raw,18)[0]!=224: raise RuntimeError('Not an AMDGPU ELF')
            header=next(line for line in (directory/'kernel.llir').read_text().splitlines() if line.startswith('define amdgpu_kernel'))
            arguments=header.split('(',1)[1].split(') local_unnamed_addr',1)[0].split(', ')
            expected=['ptr' if 'role' in a else 'i64' for a in fixtures[0]['arguments']]+['ptr','ptr']
            if len(arguments)!=len(expected) or any(not a.startswith(t) for a,t in zip(arguments,expected)): raise RuntimeError('LLVM ABI mismatch')
            kernels.append(dict(pair=name,input=input_name,output=output_name,binary='kernels/'+name+'/kernel.hsaco',sha256=hashlib.sha256(raw).hexdigest(),entrypoint=metadata['name'],grid_tile=256,block=[256,1,1],shared_bytes=metadata['shared'],arguments=signature,trailing_null_scratch_arguments=2))
            print('COMPILED',name,len(raw),flush=True)
    report=dict(schema_version=1,kernels=kernels,fixtures=fixtures,rank=args.rank,tile=256,
        ninetoothed=package_version('ninetoothed'),triton=package_version('triton'),torch=package_version('torch'),hip=torch.version.hip,
        cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),local_visible_gpu_count=torch.cuda.device_count(),gpu_executed=False,
        generation_adaptation='Fixed tile host candidate bounds initialized without querying absent GPU; standard NineToothed generator retained',
        target=dict(backend='hip',arch='gfx936',warp_size=64),llvm_abi_verified=True)
    (output/'generation_report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
