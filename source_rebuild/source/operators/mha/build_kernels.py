"""Generate MHA using NineToothed, then CPU cross-compile to gfx936."""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys


def package_version(name):
    """Record available metadata without rejecting source-provided modules."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return getattr(sys.modules.get(name), '__version__', None)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--head-tile',type=int,choices=(64,128),required=True);p.add_argument('--only',choices=('f32','f16','bf16'));a=p.parse_args()
    import torch,triton
    import ninetoothed.generation as generation
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    from backend_compat import install
    compatibility=install(root/'compiler_compatibility')
    template=Path(__file__).with_name('mha_definition.py').read_text()
    for name,value in (('BLOCK_M',16),('BLOCK_N',64),('BLOCK_D',a.head_tile)):
        template=re.sub(r'^'+name+r'=\d+\n','',template,flags=re.M)
        template=re.sub(r'\b'+name+r'\b',str(value),template)
    path=root/'mha_definition_specialized.py';path.write_text(template)
    definition=load('mha_definition_'+str(a.head_tile),path)
    generation.CACHE_DIR=root/'generated';generation.CACHE_DIR.mkdir(exist_ok=True)
    class Generator(generation.CodeGenerator):
        def __init__(self):
            ast.NodeTransformer.__init__(self);self._min_num_elements=1;self._max_num_elements=16*a.head_tile
    source=Path(Generator()(definition.application,'torch','minimax_nine_mha',4,1,1,False))
    shutil.copyfile(source,root/'ninetoothed_generated.py');module=load('mha_generated_'+str(a.head_tile),source);kernel=module.minimax_nine_mha
    launch=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='launch_minimax_nine_mha')
    class View:
        def __init__(self,role,shape,strides):self.role,self.shape,self.strides=role,shape,strides
        def size(self,i):return self.shape[i]
        def stride(self,i):return self.strides[i]
    class Capture:
        def __getitem__(self,grid):self.grid=tuple(grid({}));return self.call
        def call(self,*values,**options):self.values,self.options=values,options
    fixtures=[]
    for b,nq,nk,hq,hk,dim in ((1,37,53,4,2,a.head_tile),(2,7,23,8,2,33),(1,8850,8850,7,7,128)):
        if dim>a.head_tile:continue
        shapes=[(b,hq,nq,dim),(b,hq,nq,dim),(b,hk,nk,dim),(b,hk,nk,dim)]
        strides=[(nq*hq*dim,dim,hq*dim,1),(nq*(hq+2*hk)*dim,dim,(hq+2*hk)*dim,1),
                 (nk*(hq+2*hk)*dim,dim,(hq+2*hk)*dim,1),(nk*(hq+2*hk)*dim,dim,(hq+2*hk)*dim,1)]
        capture=Capture();namespace={'triton':triton,'minimax_nine_mha':capture}
        exec(compile(ast.Module(body=[launch],type_ignores=[]),str(source),'exec'),namespace)
        namespace['launch_minimax_nine_mha'](*(View(role,s,t) for role,s,t in zip(('output','query','key','value'),shapes,strides)),dim**-0.5,nk,nq,hq//hk,0)
        args=[dict(name=n,role=v.role) if isinstance(v,View) else dict(name=n,value=v) for n,v in zip(kernel.arg_names,capture.values)]
        fixtures.append(dict(shapes=shapes,strides=strides,grid=list(capture.grid),arguments=args))
    kernels=[]
    for name,scalar in (('f32','fp32'),('f16','fp16'),('bf16','bf16')):
        if a.only and a.only!=name:continue
        signature={v['name']:'*'+scalar if 'role' in v else 'fp32' if v['name']=='e_scale' else 'i64' for v in fixtures[0]['arguments']}
        print('MHA_COMPILER BEGIN',name,flush=True)
        compiled=triton.compile(ASTSource(kernel,signature=signature),target=GPUTarget('hip','gfx936',64),options={'num_warps':4,'num_stages':1,'enable_fp_fusion':False})
        metadata=compiled.metadata._asdict()
        if metadata.get('global_scratch_size',0) or metadata.get('profile_scratch_size',0):raise RuntimeError('Unexpected device scratch')
        out=root/'kernels'/name;out.mkdir(parents=True,exist_ok=True)
        for kind,value in compiled.asm.items():(out/('kernel.'+kind)).write_bytes(value if isinstance(value,bytes) else str(value).encode())
        (out/'compiler_metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
        header=next(line for line in (out/'kernel.llir').read_text().splitlines() if line.startswith('define amdgpu_kernel'))
        arguments=header.split('(',1)[1].split(') local_unnamed_addr',1)[0].split(', ')
        expected=['ptr' if 'role' in v else 'float' if v['name']=='e_scale' else 'i64' for v in fixtures[0]['arguments']]+['ptr','ptr']
        if len(arguments)!=len(expected) or any(not x.startswith(t) for x,t in zip(arguments,expected)):raise RuntimeError('Generated LLVM ABI differs')
        raw=(out/'kernel.hsaco').read_bytes()
        kernels.append(dict(dtype=name,head_tile=a.head_tile,binary='kernels/'+name+'/kernel.hsaco',sha256=hashlib.sha256(raw).hexdigest(),shared_bytes=metadata['shared'],threads=256,signature=signature,trailing_null_scratch_arguments=2))
        print('COMPILED',a.head_tile,name,len(raw),'LDS',metadata['shared'],flush=True)
    report=dict(kernels=kernels,fixtures=fixtures,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ninetoothed=package_version('ninetoothed'),triton=package_version('triton'),torch=package_version('torch'),
        local_visible_gpu_count=torch.cuda.device_count(),gpu_executed=False,llvm_abi_verified=True,
        math_source='mha_definition.py, unedited NineToothed-generated source, compiler primitives only',compiler_compatibility=compatibility)
    (root/'generation_report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
