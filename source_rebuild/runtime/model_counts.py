"""Mirror the frozen C++ layer-count lookup; no model computation."""
def expected_counts(config):
    layers=config['num_layers']
    key=next((key for key in ('token_refiner_num_layers','num_refiner_layers') if key in config),None)
    refiner=config[key] if key else 2
    if type(layers) is not int or layers<=0 or type(refiner) is not int or refiner<0:
        raise RuntimeError('Invalid actual H3 layer configuration')
    head_dim=config.get('attention_head_dim',128); hidden=config.get('hidden_size',5376)
    if type(head_dim) is not int or type(hidden) is not int or not 1<=head_dim<=8192 or not 1<=hidden<=8192: raise RuntimeError('Unsupported RMS model widths')
    tile=lambda width:'128' if width<=128 else '1024' if width<=1024 else '8192'
    qk=2*(layers+refiner); norms=qk+2; tiles={t:0 for t in ('128','1024','8192')}
    tiles[tile(head_dim)]+=qk; tiles[tile(hidden)]+=norms
    return dict(num_layers=layers,token_refiner_num_layers=refiner,
                token_refiner_count_source=key or 'frozen_cpp_default_2',silu=layers+2,swiglu=layers+refiner,
                rms_norm=qk+norms,rms_qk=qk,rms_hidden=norms,rms_tiles=tiles,hidden_size=hidden,hidden_tile=int(tile(hidden)))
