"""Explicit reference/candidate routes; baseline operations are comparison-only."""
import os

ROOT_VARIABLES=('INFINICORE_NINETOOTHED_CAST_ROOT','INFINICORE_NINETOOTHED_H3_ROTARY_ROOT','INFINICORE_NINETOOTHED_MROPE_ROOT',
    'INFINICORE_NINETOOTHED_SILU_ROOT','INFINICORE_NINETOOTHED_SWIGLU_ROOT','INFINICORE_NINETOOTHED_RMS_NORM_ROOT')
MODES=('reference','candidate')

def activate(mode,roots):
    if mode not in MODES or len(roots)!=6:raise ValueError('Invalid repeated-model route')
    for name,root in zip(ROOT_VARIABLES,roots):
        if mode=='reference':os.environ.pop(name,None)
        else:os.environ[name]=str(root)

def zero_candidates(before,after):
    groups={'cast','rotary','mrope','silu','swiglu','rms'}
    if set(before)!=groups or set(after)!=groups or any(set(v)!=set(map(str,range(8))) for v in after.values()):raise RuntimeError('Incomplete reference counter matrix')
    if before!=after:raise RuntimeError('NineToothed diagnostics changed during legacy-reference H3')
    if any(v['launches'] or v['module_loads'] for group in after.values() for v in group.values()):raise RuntimeError('Candidate operator executed in reference process')

def require_calls(count,expected):
    if count!=expected:raise RuntimeError(f'Actual H3 call count {count} differs from {expected}')
