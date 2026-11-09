"""SiLU core arithmetic expressed in the NineToothed DSL."""
import ninetoothed as nt
import ninetoothed.language as nl
from ninetoothed.language import libdevice

RANK=8
TILE=256

def arrangement(input,output):
    tile=(1,)*(RANK-1)+(TILE,)
    return input.tile(tile),output.tile(tile)

def application(input,output):
    value=nl.cast(input,nl.float32)
    denominator=1.0+libdevice.fast_expf(-value)
    if input.dtype==nl.float32:
        output=value*nl.div_rn(1.0,denominator)
    else:
        output=nl.div_rn(value,denominator)

application.__annotations__=dict(zip(('input','output'),arrangement(nt.Tensor(RANK,name='a_in'),nt.Tensor(RANK,name='b_out'))))
