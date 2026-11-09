"""Small gfx936 compiler adapter; no device arithmetic or generated-code edits.

The pinned vendor assembler cannot parse the newer GEP `nuw` optimization
promise emitted by its LLVM IR producer. Dropping that promise preserves
defined address arithmetic and retains all instructions and existing masks.
"""
import hashlib
import re


def normalize_gep_flags(source):
    return re.sub(r'\bgetelementptr(\s+inbounds)?\s+nuw\b',lambda m:'getelementptr'+(m.group(1) or ''),source)


def install(directory):
    from triton.backends.amd.compiler import HIPBackend
    original=HIPBackend.make_amdgcn
    records=[]
    def make_amdgcn(source,metadata,options):
        patched=normalize_gep_flags(source)
        directory.mkdir(parents=True,exist_ok=True)
        index=len(records)
        (directory/(str(index)+'_input.ll')).write_text(source)
        (directory/(str(index)+'_compatible.ll')).write_text(patched)
        records.append(dict(scope='LLVM GEP optimization flag syntax only',
            input_sha256=hashlib.sha256(source.encode()).hexdigest(),
            output_sha256=hashlib.sha256(patched.encode()).hexdigest(),
            removed_gep_nuw_promises=source.count('getelementptr inbounds nuw')+source.count('getelementptr nuw')))
        print('MHA_COMPILER LLVM_TO_ASSEMBLY',records[-1]['removed_gep_nuw_promises'],flush=True)
        return original(patched,metadata,options)
    HIPBackend.make_amdgcn=staticmethod(make_amdgcn)
    return records
