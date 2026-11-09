"""Public POD example contract; this describes a case, not an evaluator threshold."""
from pathlib import Path

MODES = ('reference', 'candidate')
CASE = dict(case_id='POD_PUBLIC_DOG_FL2VA_TEN_POINTS', prompt='A dog falls asleep', width=576,
            height=384, frames=124, fps=24, steps=10, seed=42, expected_h3_calls=9,
            image_sha256='1e71abf29876f248088a11211a350ed3d26a0c68aa34f6e45738771eb2d45a22',
            image_bytes=212652, image_size=[800,965], latent_video_shape=[1,24,37,24,36],
            latent_audio_shape=[2,32,207], condition_shape=[1,24,1,24,36], condition_video_rows=216,
            video_rows=8208, audio_rows=414,
            source='https://gxtctab8no8.feishu.cn/docx/G6BNd1A3OoyAHdxmpJacQbuInuD',
            input_original_organizer_checksum_verified=False, official_similarity_threshold=None)

def arguments(mode, image, output):
    if mode not in MODES:
        raise ValueError('Unknown public-case mode')
    result=['fl2va','--runtime-device','hygon','--gpus','0,1,2,3,4,5,6,7',
            '--model-path','/public/home/aclsylqidf/shared/MiniMax-H3', '--text',CASE['prompt'],
            '--first-frame',str(image), '--steps',str(CASE['steps']), '--num-frames',str(CASE['frames']),
            '--height',str(CASE['height']), '--width',str(CASE['width']), '--seed',str(CASE['seed']),
            '--output',str(output)]
    if mode=='reference':
        result+=['--legacy','--legacy-text-gpus','0,1,2,3','--legacy-transformer-gpus','4,5,6,7']
    return result

def validate_geometry(value):
    text=value['text_rows']; video=value['video_rows']; audio=value['audio_rows']; positions=value['positions']
    if type(text) is not int or text<=0 or video!=CASE['video_rows'] or audio!=CASE['audio_rows']:
        raise RuntimeError('Public-case modality rows differ')
    if positions.get('device') not in {'cuda:'+str(i) for i in range(8)} or {k:v for k,v in positions.items() if k!='device'}!=dict(shape=[text+video+audio,3],dtype='torch.float64',stride=[3,1]):
        raise RuntimeError('Public-case packed F64 positions differ')

def complete_calls(calls):
    if len(calls)!=CASE['expected_h3_calls'] or any(c['status']!='PASSED' for c in calls):
        raise RuntimeError('Missing, extra, or failed actual public-case H3 calls')

def placement_devices(mapping,wanted):
    result=set()
    for value in mapping.values():
        text=str(value)
        if text.startswith('cuda:'):text=text[5:]
        if not text.isdecimal():raise RuntimeError('Reference has CPU/disk/non-CUDA placement')
        result.add(int(text))
    if result!=set(wanted):raise RuntimeError('Reference four-plus-four placement differs')
    return sorted(result)
