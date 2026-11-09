"""Apply multi-axis rotary embeddings entirely in the NineToothed DSL."""
import ninetoothed as nt
import ninetoothed.language as nl

TILE=(1,1,128)

def application(output,source,cosine,sine,positions,half,section_t,section_h,section_w,interleaved,table_rows):
    token=output.offsets(0)
    head=output.offsets(1)
    dim=output.offsets(2)
    frequency=dim%half
    ordinary_axis=nl.where(frequency<section_t,0,nl.where(frequency<section_t+section_h,1,2))
    mixed_axis=nl.where((frequency%3==1)&(frequency<section_h*3),1,
                        nl.where((frequency%3==2)&(frequency<section_w*3),2,0))
    axis=nl.where(interleaved!=0,mixed_axis,ordinary_axis)
    raw_position=positions[axis,token]
    position=nl.where((raw_position>=0)&(raw_position<table_rows),raw_position,0)
    c=nl.cast(cosine[position,frequency],nl.float32)
    s=nl.cast(sine[position,frequency],nl.float32)
    value=nl.cast(source[token,head,dim],nl.float32)
    partner_dim=nl.where(dim<half,dim+half,dim-half)
    partner_dim=nl.where(dim<2*half,partner_dim,dim)
    partner=nl.cast(source[token,head,partner_dim],nl.float32)
    rotated=nl.where(dim<half,value*c-partner*s,value*c+partner*s)
    output=nl.expand_dims(nl.expand_dims(nl.where(dim<2*half,rotated,value),0),0)

# The first tiled output defines the launch grid. Untiled inputs use explicit
# absolute indices; NineToothed generates their gathers and bounds masks.
application.__annotations__=dict(
    output=nt.Tensor(3,name='a_out').tile(TILE),
    source=nt.Tensor(3,name='b_in',other=0,shape_options=dict(upper_bound=2**31-1)),
    cosine=nt.Tensor(2,name='c_cos',other=0,shape_options=dict(upper_bound=2**31-1)),
    sine=nt.Tensor(2,name='d_sin',other=0,shape_options=dict(upper_bound=2**31-1)),
    positions=nt.Tensor(2,name='e_pos',other=0,shape_options=dict(upper_bound=2**31-1)),half=nt.Tensor(0,dtype=nt.int64,name='f_half'),
    section_t=nt.Tensor(0,dtype=nt.int64,name='g_t'),section_h=nt.Tensor(0,dtype=nt.int64,name='h_h'),
    section_w=nt.Tensor(0,dtype=nt.int64,name='i_w'),interleaved=nt.Tensor(0,dtype=nt.int64,name='j_interleaved'),
    table_rows=nt.Tensor(0,dtype=nt.int64,name='k_rows'))
