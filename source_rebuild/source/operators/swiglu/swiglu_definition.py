"""Fused up * silu(gate), with the legacy dtype rounding, in NineToothed."""
import ninetoothed as nt
import ninetoothed.language as nl
from ninetoothed.language import libdevice

RANK=8
TILE=256

def arrangement(up,gate,output):
    tile=(1,)*(RANK-1)+(TILE,)
    return up.tile(tile),gate.tile(tile),output.tile(tile)

def application(up,gate,output):
    u=nl.cast(up,nl.float32)
    g=nl.cast(gate,nl.float32)
    exponential=libdevice.fast_expf(-g)
    if gate.dtype==nl.float16:
        rounded_exp=nl.cast(exponential,nl.float16)
        denominator=nl.cast(1.0+nl.cast(rounded_exp,nl.float32),nl.float16)
        sigmoid=nl.cast(nl.div_rn(1.0,nl.cast(denominator,nl.float32)),nl.float16)
        activated=nl.cast(g*nl.cast(sigmoid,nl.float32),nl.float16)
        output=nl.cast(activated,nl.float32)*u
    elif gate.dtype==nl.bfloat16:
        sigmoid=nl.cast(nl.div_rn(1.0,1.0+exponential),nl.bfloat16)
        output=(g*nl.cast(sigmoid,nl.float32))*u
    else:
        output=(g*nl.div_rn(1.0,1.0+exponential))*u

application.__annotations__=dict(zip(('up','gate','output'),arrangement(
    nt.Tensor(RANK,name='a_up'),nt.Tensor(RANK,name='b_gate'),nt.Tensor(RANK,name='c_out'))))
