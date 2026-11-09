'''Standard-library independent acceptance of real VAE SiLU launch evidence.'''
PHASES=('video_vae_encode','video_vae_decode')
from collections import Counter
from math import prod

def assess_vae(report, *, expected=None):
    expected = expected or dict(video_vae_encode=150, video_vae_decode=1512)
    items=report['ninetoothed_video_vae_silu']
    if report['video_vae_silu_counts'] != expected or dict(Counter(p['phase'] for p in items)) != expected:
        raise RuntimeError('Actual video VAE SiLU count differs from frozen public case')
    for item in items:
        before, after = item['before'], item['after']
        kind = {'torch.float32': 'f32', 'torch.float16': 'f16', 'torch.bfloat16': 'bf16'}.get(item['dtype'])
        if (kind is None or item['phase'] not in expected or item['launch_delta'] != 1
                or type(item['device']) is not int or item['device'] not in range(8)
                or not 0<=len(item['shape'])<=8 or any(type(n) is not int or n<=0 for n in item['shape'])
                or not 0<prod(item['shape'])<=2147483647
                or len(item['input_stride'])!=len(item['shape']) or len(item['output_stride'])!=len(item['shape'])
                or after['launches'] - before['launches'] != 1 or after['last_device'] != item['device']
                or not after['last_stream'] or any(after[k] - before[k] != int(k == kind) for k in ('f32', 'f16', 'bf16'))):
            raise RuntimeError('Missing VAE actual per-operation Nine dispatch')
    if any(p['phase'] in PHASES and 'aten.silu' in p['operation'] for p in report['operations']):
        raise RuntimeError('Unmigrated Torch SiLU remains in video VAE')
    if report['complete_model_migration']:
        raise RuntimeError('This change does not migrate complete model computation')
    return dict(status='VIDEO_VAE_1662_SILU_REAL_NINE_DISPATCH_VERIFIED_REMAINDER_PENDING',
                counts=report['video_vae_silu_counts'], remaining_torch_operations=len(report['operations']))
