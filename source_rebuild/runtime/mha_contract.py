"""Independent per-H3 MHA counter/geometry contract; no device computation."""
from model_counts import expected_counts

FIELDS=('launches','module_loads','last_stream','last_device','last_grid','last_shared','last_tile',
        'batch','query_length','key_length','query_heads','kv_heads','head_dim','causal',
        'query_sequence_stride','key_sequence_stride','value_sequence_stride','output_sequence_stride','f32','f16','bf16')

def require(value,message):
    if not value:raise RuntimeError('Real H3 Nine MHA: '+message)

def geometry(hf,rows):
    heads=hf['num_attention_heads'];dim=hf['attention_head_dim']
    require(type(heads) is int and heads>0 and heads%8==0 and type(dim) is int and 1<=dim<=128,'unsupported TP8 head geometry')
    require(type(rows) is int and rows>0,'invalid packed row count')
    heads//=8;tile=64 if dim<=64 else 128
    return dict(batch=1,query_length=rows,key_length=rows,query_heads=heads,kv_heads=heads,head_dim=dim,causal=0,
                last_grid=heads*((rows+15)//16),last_tile=tile,last_shared=8192 if tile==64 else 16384,
                query_sequence_stride=heads*dim,key_sequence_stride=heads*dim,value_sequence_stride=3*heads*dim,output_sequence_stride=heads*dim)

def assess(before,after,hf,rows):
    devices=set(map(str,range(8)));require(set(before)==set(after)==devices,'eight device matrix missing')
    count=expected_counts(hf)['swiglu'];wanted=geometry(hf,rows);deltas={}
    for device in devices:
        a,b=before[device],after[device]
        require(set(a)==set(b)==set(FIELDS),'native counter schema differs')
        require(all(type(v) is int and v>=0 for p in (a,b) for v in p.values()),'invalid counter value')
        delta={k:b[k]-a[k] for k in ('launches','module_loads','f32','f16','bf16')}
        require(delta['launches']==delta['bf16']==count and delta['f32']==delta['f16']==0,'layer-derived BF16 count differs on '+device)
        require(delta['module_loads'] in (0,1) and b['module_loads']==1,'module cache differs on '+device)
        require(b['last_device']==int(device) and b['last_stream']>0,'device or stream differs on '+device)
        require(all(b[k]==v for k,v in wanted.items()),'packed MHA geometry/stride differs on '+device)
        deltas[device]=delta
    return deltas

def complete(report,state,config):
    require(report['actual_model_executed'] and not report['training_executed'],'real H3 execution evidence missing')
    require(report['core_sha256']==config['core_sha256'] and report['runtime_sha256']==config['mha_runtime_sha256'],'core/runtime identity differs')
    require(report['mapped_core_paths']==state['mapped_core_paths'],'MHA and original observer core mapping differ')
    require(report['root_variable']=='INFINICORE_NINETOOTHED_MHA_ROOT' and report['fallback_supported'] is False,'explicit Nine route missing')
    calls=report['calls'];require(len(calls)==len(state['calls'])==9,'nine complete H3 calls required')
    totals={str(d):0 for d in range(8)}
    for index,(value,public) in enumerate(zip(calls,state['calls']),1):
        require(value['status']==public['status']=='PASSED' and value['index']==public['index']==index,'actual call sequence differs')
        require(value['h3_config_geometry']==public['h3_config_geometry'] and value['rows']==public['geometry']['positions']['shape'][0],'actual model geometry differs')
        delta=assess(value['before'],value['after'],public['h3_config_geometry'],value['rows'])
        require(delta==value['delta'],'recorded MHA delta differs')
        if index>1:require(value['before']==calls[index-2]['after'],'unobserved MHA work between H3 calls')
        for d in totals:totals[d]+=delta[d]['launches']
    require(report['final_statistics']==calls[-1]['after'],'unobserved MHA work after final H3')
    return dict(status='NINE_MHA_ALL_NINE_H3_CALLS_EIGHT_DEVICE_DISPATCH_VERIFIED',h3_calls=9,
                per_device_launches=totals,total_launches=sum(totals.values()),core_sha256=config['core_sha256'],
                actual_model_executed=True,training_executed=False,preliminary_admission_passed=False,
                per_call_geometry_and_dtype_verified=True,legacy_mha_fallback_supported=False)
