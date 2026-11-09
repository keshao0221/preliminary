"""Blocked online-softmax MHA. All device arithmetic is NineToothed DSL.

Source/target tensors are metadata views [batch, heads, sequence, dimension].
No dense attention matrix is materialized; original strides remain intact.
"""
import ninetoothed as nt
import ninetoothed.language as nl

BLOCK_M=16
BLOCK_N=64
BLOCK_D=128


def application(output,query,key,value,scale,key_length,query_length,groups,causal):
    batch=nl.sum(output.offsets(0),axis=0)
    head=nl.sum(output.offsets(1),axis=0)
    rows=nl.expand_dims(output.offsets(2),1)
    dimensions=nl.expand_dims(output.offsets(3),0)
    kv_head=head//groups
    q=query[batch,head,rows,dimensions]
    maximum=nl.full((BLOCK_M,1),float('-inf'),nl.float32)
    denominator=nl.full((BLOCK_M,1),0,nl.float32)
    accumulator=nl.full((BLOCK_M,BLOCK_D),0,nl.float32)
    keys=nl.expand_dims(nl.arange(0,BLOCK_N),1)
    for start in range(0,key_length,BLOCK_N):
        positions=start+keys
        k=key[batch,kv_head,positions,dimensions]
        v=value[batch,kv_head,positions,dimensions]
        scores=nl.dot(q,nl.trans(k),input_precision='ieee')*scale
        visible=(nl.trans(positions)<key_length)&((causal==0)|(nl.trans(positions)<=rows+key_length-query_length))
        scores=nl.where(visible,scores,float('-inf'))
        updated_maximum=nl.maximum(maximum,nl.max(scores,axis=1,keep_dims=True))
        safe_maximum=nl.where(updated_maximum==float('-inf'),0,updated_maximum)
        old_factor=nl.exp(maximum-safe_maximum)
        probabilities=nl.exp(scores-safe_maximum)
        denominator=denominator*old_factor+nl.sum(probabilities,axis=1,keep_dims=True)
        if value.dtype==nl.float32:
            contribution=nl.dot(probabilities,v,input_precision='ieee')
        else:
            # Two exact DSL casts retain the small probability residual rather
            # than silently dropping it in a single low-precision P*V dot.
            high=nl.cast(probabilities,value.dtype)
            low=nl.cast(probabilities-nl.cast(high,nl.float32),value.dtype)
            contribution=nl.dot(high,v,input_precision='ieee')+nl.dot(low,v,input_precision='ieee')
        accumulator=accumulator*old_factor+contribution
        maximum=updated_maximum
    result=accumulator/nl.where(denominator>0,denominator,1)
    output=nl.expand_dims(nl.expand_dims(result,0),0)


def annotate():
    application.__annotations__=dict(zip(('output','query','key','value','scale','key_length','query_length','groups','causal'),(
        nt.Tensor(4,name='a_output').tile((1,1,BLOCK_M,BLOCK_D)),
        nt.Tensor(4,name='b_query',other=0,shape_options=dict(upper_bound=2**31-1)),
        nt.Tensor(4,name='c_key',other=0,shape_options=dict(upper_bound=2**31-1)),
        nt.Tensor(4,name='d_value',other=0,shape_options=dict(upper_bound=2**31-1)),
        nt.Tensor(0,name='e_scale',dtype=nt.float32),
        nt.Tensor(0,name='f_key_length',dtype=nt.int64),
        nt.Tensor(0,name='g_query_length',dtype=nt.int64),
        nt.Tensor(0,name='h_groups',dtype=nt.int64),
        nt.Tensor(0,name='i_causal',dtype=nt.int64))))

annotate()
