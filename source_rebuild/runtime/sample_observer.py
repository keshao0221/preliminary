"""Observe real image conditioning and each H3 call; no model math is replaced."""
import functools
import hashlib
import inspect
from pathlib import Path
from case_contract import CASE, validate_geometry, complete_calls, placement_devices
from observer import Counter, assess, mapped_core_paths, sha256, write_json
from model_counts import expected_counts
from modes import activate, zero_candidates

def tensor_metadata(value):
    import torch
    if not isinstance(value,torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise RuntimeError('Nonfinite model tensor')
    return dict(shape=list(value.shape),dtype=str(value.dtype),all_finite=True)

def install(roots,directory,config,mode,audit):
    import torch
    from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3ResizeStep
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep,MiniMaxH3KeyframeVaeEncoderStep
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep,MiniMaxH3AudioDecodeStep
    roots=tuple(Path(r).resolve() for r in roots);directory=Path(directory)
    if mapped_core_paths()!=[str(roots[-1]/'libinfinicore_cpp_api.so')]:
        raise RuntimeError('Unexpected native core mapping')
    if sha256(roots[-1]/'libinfinicore_cpp_api.so')!=config['core_sha256'] or sha256(Path(inspect.getfile(MiniMaxH3TransformerRunner)))!=config['model_source_sha256']:
        raise RuntimeError('Frozen core/runner changed')
    if sha256(Path(inspect.getfile(MiniMaxH3Transformer3DModel)))!=config['torch_h3_source_sha256']:
        raise RuntimeError('Frozen PyTorch reference source changed')
    for name,root in zip(('cast','rotary','mrope','silu','swiglu','rms'),roots):
        stem='rms_norm' if name=='rms' else name
        if sha256(root/('libminimax_nine_'+stem+'_runtime.so'))!=config[name+'_host_sha256']:
            raise RuntimeError('Frozen launch library changed: '+name)
    activate(mode,roots);counter=Counter(*roots)
    report=dict(status='PUBLIC_CASE_OBSERVER_INSTALLED',mode=mode,configuration=config,calls=[],
                actual_model_executed=False,training_executed=False,first_frame_conditioning_verified=False,
                resize=[],text_encoder=[],keyframe_encoder=[],layout=[],decoders=[],
                mapped_core_paths=mapped_core_paths(),source_sha256={})
    target=directory/'public_case_observer.json';restores=[]
    def save():write_json(target,report)
    def wrap_step(cls,label,inspect_result):
        original=cls.__call__;report['source_sha256'][cls.__name__]=sha256(Path(inspect.getfile(cls)))
        @functools.wraps(original)
        def wrapped(self,*args,**kwargs):
            with audit.phase(label):result=original(self,*args,**kwargs)
            inspect_result(*result);save();return result
        cls.__call__=wrapped;restores.append((cls,'__call__',original))
    def resized(components,state):
        frames=state.get('keyframes');anchors=state.get('keyframe_anchors')
        if anchors!=('first',) or len(frames)!=1 or frames[0].size!=(CASE['width'],CASE['height']):
            raise RuntimeError('Expected one resized first-frame anchor')
        report['resize'].append(dict(anchors=list(anchors),size=list(frames[0].size),rgb_sha256=hashlib.sha256(frames[0].convert('RGB').tobytes()).hexdigest()))
    def text(components,state):
        embeds=state.get('prompt_embeds');tags=state.get('text_token_tags')
        item=tensor_metadata(embeds)
        if embeds.ndim!=3 or embeds.shape[0]!=1 or embeds.shape[1]<=0 or tags.numel()!=embeds.shape[1]:
            raise RuntimeError('Missing actual multimodal condition embeddings')
        item['video_tagged_rows']=int((tags==components.video_tag).sum())
        if item['video_tagged_rows']<=0:raise RuntimeError('Keyframe vision tokens missing from text conditioner')
        item['device_map']={str(k):str(v) for k,v in getattr(components.text_encoder,'hf_device_map',{}).items()}
        if mode=='reference':item['placement_devices']=placement_devices(item['device_map'],range(4))
        report['text_encoder'].append(item)
    def keyframe(components,state):
        from safetensors.torch import save_file
        values=state.get('condition_latents')
        if len(values)!=1:raise RuntimeError('Expected one actually encoded keyframe')
        item=tensor_metadata(values[0])
        if item['shape']!=CASE['condition_shape'] or item['dtype']!='torch.float32':
            raise RuntimeError('Keyframe encoded latent shape/dtype differs')
        path=directory/'condition_latents.safetensors';save_file({'first':values[0].detach().cpu().contiguous()},str(path))
        item['sha256']=sha256(path);report['keyframe_encoder'].append(item)
        report['first_frame_conditioning_verified']=True
        print('H3_PUBLIC_FIRST_FRAME_ENCODED',mode,flush=True)
    def layout(components,state):
        item={k:int(state.get(k)) for k in ('height','width','num_frames','num_latent_frames','latent_height','latent_width','num_audio_latents','num_condition_video_rows','num_condition_audio_rows')}
        wanted=dict(height=384,width=576,num_frames=124,num_latent_frames=37,latent_height=24,latent_width=36,num_audio_latents=207,num_condition_video_rows=216,num_condition_audio_rows=0)
        if item!=wanted:raise RuntimeError('Actual first-frame layout differs')
        item.update(video_rows=int(state.get('video_indices').numel()),audio_rows=int(state.get('audio_indices').numel()),text_rows=int(state.get('text_indices').numel()))
        pos=state.get('position_ids');item['positions']=dict(shape=list(pos.shape),dtype=str(pos.dtype),device=str(pos.device),stride=list(pos.stride()))
        validate_geometry(item);report['layout'].append(item)
    wrap_step(MiniMaxH3ResizeStep,'image_resize',resized)
    wrap_step(MiniMaxH3TextEncoderStep,'condition_encoder',text)
    wrap_step(MiniMaxH3KeyframeVaeEncoderStep,'video_vae_encode',keyframe)
    wrap_step(MiniMaxH3PrepareLayoutStep,'layout',layout)
    for cls,label in ((MiniMaxH3VideoDecodeStep,'video_vae_decode'),(MiniMaxH3AudioDecodeStep,'audio_vae_decode')):
        wrap_step(cls,label,lambda components,state,label=label:report['decoders'].append(label))
    original=MiniMaxH3TransformerRunner.__call__
    @functools.wraps(original)
    def native(self,*args,**kwargs):
        if mode!='candidate':raise RuntimeError('Reference unexpectedly called native H3')
        inputs=args[0] if args else kwargs['inputs'];positions=inputs.position_ids
        meta=dict(video_rows=int(inputs.video_indices.numel()),audio_rows=int(inputs.audio_indices.numel()),text_rows=int(inputs.text_indices.numel()),
                  positions=dict(shape=list(positions.shape),dtype=str(positions.dtype),device=str(positions.device),stride=list(positions.stride())))
        validate_geometry(meta)
        if meta['positions']['device']!='cuda:0':raise RuntimeError('Native TP8 position device differs')
        counts=expected_counts(self.engine.hf_config)
        call=begin(meta);call.update(h3_config_geometry={k:self.engine.hf_config[k] for k in ('num_layers','num_refiner_layers','token_refiner_num_layers','num_attention_heads','attention_head_dim','hidden_size','ffn_dim','rope_freq_dim') if k in self.engine.hf_config},resolved_operation_counts=counts,frequency_dim=int(self.rope_freq_dim),theta=float(self.rope_theta))
        try:
            with audit.phase('native_h3_bridge'):result=original(self,*args,**kwargs)
            for gpu in range(8):torch.cuda.synchronize(gpu)
            after=counter.snapshot();delta=assess(call['before'],after,positions.device.index or 0,'f64',int(positions.shape[0]),int(self.rope_freq_dim),counts['silu'],counts['swiglu'],counts)
            if any(delta['cast'][str(i)]['launches']!=counts['num_layers']+counts['token_refiner_num_layers']+2 or delta['mrope'][str(i)]['launches']!=2*counts['num_layers'] for i in range(8)):
                raise RuntimeError('Model-derived Cast/MRoPE count differs')
            for i in range(8):
                geometry=after['mrope'][str(i)]['last_launch_geometry']
                if geometry['heads']!=self.engine.hf_config['num_attention_heads']//8 or geometry['head_dim']!=self.engine.hf_config['attention_head_dim']:
                    raise RuntimeError('Actual per-rank heads differ from model configuration')
            call.update(after=after,delta=delta,outputs={n:tensor_metadata(getattr(result,n)) for n in ('sample','audio_sample')})
            end(call);return result
        except BaseException as error:call.update(status='FAILED',error=str(error));raise
        finally:save()
    def begin(meta):
        if not report['first_frame_conditioning_verified'] or len(report['layout'])!=1:
            raise RuntimeError('H3 started without actual keyframe encoding/layout')
        for gpu in range(8):torch.cuda.synchronize(gpu)
        index=len(report['calls'])+1
        if index>CASE['expected_h3_calls']:raise RuntimeError('Extra actual H3 call')
        call=dict(status='RUNNING',index=index,before=counter.snapshot(),geometry=meta);report['calls'].append(call);save()
        print('H3_PUBLIC_REAL_H3_CALL_BEGIN',mode,index,flush=True);return call
    def end(call):
        call['status']='PASSED';report.update(status='ACTUAL_PUBLIC_FIRST_FRAME_H3_OBSERVED',actual_model_executed=True)
        print('H3_PUBLIC_REAL_H3_CALL_COMPLETED',mode,call['index'],flush=True)
    MiniMaxH3TransformerRunner.__call__=native;restores.append((MiniMaxH3TransformerRunner,'__call__',original))
    if mode=='reference':
        original_torch=MiniMaxH3Transformer3DModel.forward
        signature=inspect.signature(original_torch)
        @functools.wraps(original_torch)
        def legacy(self,*args,**kwargs):
            fields=signature.bind(self,*args,**kwargs).arguments
            pos=fields['position_ids'];video=fields['hidden_states'];audio=fields['audio_hidden_states'];text_embed=fields['encoder_hidden_states']
            meta=dict(video_rows=int(video.shape[1]),audio_rows=int(audio.shape[1]),text_rows=int(text_embed.shape[1]),
                      positions=dict(shape=list(pos.shape),dtype=str(pos.dtype),device=str(pos.device),stride=list(pos.stride())))
            validate_geometry(meta);call=begin(meta)
            call['device_map']={str(k):str(v) for k,v in getattr(self,'hf_device_map',{}).items()}
            call['placement_devices']=placement_devices(call['device_map'],range(4,8))
            try:
                with audit.phase('reference_torch_h3'):result=original_torch(self,*args,**kwargs)
                for gpu in range(8):torch.cuda.synchronize(gpu)
                after=counter.snapshot();zero_candidates(call['before'],after)
                if not isinstance(result,tuple) or len(result)!=2:raise RuntimeError('Expected two actual H3 outputs')
                call.update(after=after,delta={},outputs={n:tensor_metadata(v) for n,v in zip(('sample','audio_sample'),result)})
                end(call);return result
            except BaseException as error:call.update(status='FAILED',error=str(error));raise
            finally:save()
        MiniMaxH3Transformer3DModel.forward=legacy;restores.append((MiniMaxH3Transformer3DModel,'forward',original_torch))
    save()
    return report,restores

def assess_complete(report):
    complete_calls(report['calls'])
    if not report['actual_model_executed'] or not report['first_frame_conditioning_verified'] or any(len(report[k])!=1 for k in ('resize','text_encoder','keyframe_encoder','layout')) or report['decoders']!=['video_vae_decode','audio_vae_decode']:
        raise RuntimeError('Incomplete actual public-case pipeline evidence')
