"""Read and validate actual public-case media and latent artifacts."""
import argparse
import json
from pathlib import Path
import traceback
from case_contract import CASE
from observer import sha256,write_json

def inspect(directory):
    import av,torch
    from safetensors.torch import load_file
    path=directory/'public_dog.mp4'
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        video=dict(width=stream.width,height=stream.height,fps=float(stream.average_rate),codec=stream.codec_context.name,
            decoded_frames=sum(1 for _ in container.decode(video=0)),bytes=path.stat().st_size,sha256=sha256(path))
    with av.open(str(path)) as container:
        stream=container.streams.audio[0];frames=list(container.decode(audio=0))
        audio=dict(channels=len(stream.codec_context.layout.channels),sample_rate=stream.codec_context.sample_rate,codec=stream.codec_context.name,
            decoded_samples=sum(f.samples for f in frames),decoded_finite=all(bool(__import__('numpy').isfinite(f.to_ndarray()).all()) for f in frames))
    if [video[k] for k in ('width','height','fps','decoded_frames')]!=[576,384,24.,124] or audio['channels']!=2 or audio['sample_rate']!=32000 or audio['decoded_samples']<=0 or not audio['decoded_finite']:
        raise RuntimeError('Public media specification differs')
    result={}
    for filename,expected in (('denoised_latents.safetensors',{'latents':CASE['latent_video_shape'],'audio_latents':CASE['latent_audio_shape']}),
                              ('condition_latents.safetensors',{'first':CASE['condition_shape']})):
        p=directory/filename;tensors=load_file(str(p),device='cpu')
        if set(tensors)!=set(expected) or any(list(t.shape)!=expected[k] or t.dtype!=torch.float32 or not bool(torch.isfinite(t).all()) for k,t in tensors.items()):
            raise RuntimeError('Public latent shape/dtype/finite values differ: '+filename)
        result[filename]=dict(bytes=p.stat().st_size,sha256=sha256(p),tensors={k:dict(shape=list(t.shape),dtype=str(t.dtype)) for k,t in tensors.items()})
    return dict(status='PUBLIC_CASE_MEDIA_AND_FINITE_LATENTS_VERIFIED',video=video,audio=audio,artifacts=result,
                similarity_to_reference_verified=False,preliminary_admission_passed=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    try:report=inspect(args.directory)
    except Exception:report=dict(status='FAILED_PUBLIC_MEDIA',traceback=traceback.format_exc())
    write_json(args.directory/'output_inspection.json',report);print(json.dumps(report),flush=True)
    return 0 if report['status']=='PUBLIC_CASE_MEDIA_AND_FINITE_LATENTS_VERIFIED' else 2

if __name__=='__main__':raise SystemExit(main())
