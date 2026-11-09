"""Last-dimension RMSNorm; device arithmetic is entirely NineToothed DSL."""
import ninetoothed as nt
import ninetoothed.language as nl
from ninetoothed.language import libdevice

TILE=128

def arrangement(input,weight,output,epsilon,width):
    tile=(1,1,TILE)
    return input.tile(tile),weight.tile(tile),output.tile(tile),epsilon,width

def application(input,weight,output,epsilon,width):
    value=nl.cast(input,nl.float32)
    if TILE<=1024:
        partial=value*value
        grouped=nl.reshape(partial,(1,1,TILE//64,64))
    else:
        values=nl.reshape(value,(1,1,TILE//1024,1024))
        squared=values*values
        group_indices=nl.reshape(nl.arange(0,TILE//1024),(1,1,TILE//1024,1))
        partial=nl.full((1,1,1024),0,nl.float32)
        for group in nl.static_range(TILE//1024):
            selected=nl.sum(nl.where(group_indices==group,squared,0),axis=-2,keep_dims=False)
            partial=partial+selected
        grouped=nl.reshape(partial,(1,1,16,64))
    prefix=nl.cumsum(grouped,axis=-1)
    lane=nl.reshape(nl.arange(0,64),(1,1,1,64))
    warp_sums=nl.sum(nl.where(lane==63,prefix,0),axis=-1,keep_dims=False)
    warp_prefix=nl.cumsum(warp_sums,axis=-1)
    warp_index=nl.reshape(nl.arange(0,(TILE//64 if TILE<=1024 else 16)),(1,1,(TILE//64 if TILE<=1024 else 16)))
    total=nl.sum(nl.where(warp_index==((TILE//64 if TILE<=1024 else 16)-1),warp_prefix,0),axis=-1,keep_dims=True)
    mean=nl.div_rn(total,nl.cast(width,nl.float32))
    inverse=libdevice.rsqrt(mean+epsilon)
    weight_value=nl.cast(weight,nl.float32)
    result=(value*weight_value)*inverse
    if input.dtype==nl.float32:
        output=result
    else:
        # The backend may lower fmul+fptrunc to fma_mixlo_f16(...,+0),
        # which erases a negative zero. Keep finite rounded zeros' sign
        # through integer bitcasts expressed in the NineToothed DSL.
        rounded=nl.cast(result,input.dtype)
        rounded_bits=nl.cast(rounded,nl.uint16,bitcast=True)
        input_bits=nl.cast(value,nl.uint32,bitcast=True)
        weight_bits=nl.cast(weight_value,nl.uint32,bitcast=True)
        sign_bits=nl.cast(((input_bits^weight_bits)>>16)&0x8000,nl.uint16)
        output_bits=nl.where(rounded==0,sign_bits,rounded_bits)
        output=nl.cast(output_bits,input.dtype,bitcast=True)

def annotate():
    application.__annotations__=dict(zip(('input','weight','output','epsilon','width'),arrangement(
        nt.Tensor(3,name='a_input',other=0),nt.Tensor(3,name='b_weight',other=0),nt.Tensor(3,name='c_output'),
        nt.Tensor(0,name='d_epsilon',dtype=nt.float32),nt.Tensor(0,name='e_width',dtype=nt.int64))))

annotate()
