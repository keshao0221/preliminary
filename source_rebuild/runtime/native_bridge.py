"""Checked native bridge, framework math VAE attention and real latent checkpoint."""
import copy
import functools
import hashlib
import importlib
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
import runpy
import sys

UTILS_SHA256 = "46bc63296d77ba4519f467b907cf19199e28e38a0487aa85301bfdd0066f878a"


def install_bridge_fix():
    import torch
    import infinicore as ic
    utils = importlib.import_module("infinicore.utils")
    tensor_module = importlib.import_module("infinicore.tensor")
    if hashlib.sha256(Path(utils.__file__).read_bytes()).hexdigest() != UTILS_SHA256:
        raise RuntimeError("Unexpected InfiniCore conversion source")
    original = utils.to_infinicore_dtype
    if getattr(original, "_minimax_fp64_bridge", False):
        return

    @functools.wraps(original)
    def convert(torch_dtype):
        if torch_dtype == torch.float64:
            return ic.float64
        return original(torch_dtype)

    convert._minimax_fp64_bridge = True
    utils.to_infinicore_dtype = convert
    tensor_module.to_infinicore_dtype = convert
    print("MINIMAX_NATIVE_BRIDGE_FIX", "float64_metadata_preserved", flush=True)


def gpu_preflight():
    import torch
    import infinicore as ic
    from infinilm.multimodal.minimax_h3 import (
        MiniMaxH3TransformerRunner, minimax_h3_rotary_cache, _torch_rotary_cache_reference,
    )
    torch.cuda.set_device(0)
    ic.set_device(ic.device("cuda", 0))
    positions = torch.tensor([[0., 0., 0.], [.125, 1.5, -3.25], [123.456, 1e-5, 500.]], dtype=torch.float64)
    gpu_positions = positions.to("cuda:0")
    torch.cuda.synchronize(0)
    native = MiniMaxH3TransformerRunner._to_infinicore(gpu_positions)
    assert native.dtype == ic.float64 and native.data_ptr() == gpu_positions.data_ptr()
    cache = minimax_h3_rotary_cache(gpu_positions, dtype=torch.float32)
    expected = _torch_rotary_cache_reference(positions, 16, 10000.0, torch.float32)
    torch.testing.assert_close(cache.cpu(), expected, rtol=2e-5, atol=2e-5)
    print("MINIMAX_NATIVE_BRIDGE_PREFLIGHT", "FP64 input and native H3 rotary cache PASSED", flush=True)


@contextmanager
def math_attention():
    from diffusers.models.attention_dispatch import attention_backend
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with attention_backend("_native_math"), sdpa_kernel(SDPBackend.MATH):
        yield


def install_vae_math_fix():
    from diffusers.modular_pipelines.minimax_h3.decoders import (
        MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep,
    )
    for cls in (MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep):
        original = cls.__call__
        if getattr(original, "_minimax_math_attention", False):
            continue

        @functools.wraps(original)
        def wrapped(self, *args, _original=original, **kwargs):
            with math_attention():
                return _original(self, *args, **kwargs)

        wrapped._minimax_math_attention = True
        cls.__call__ = wrapped
    print("MINIMAX_VAE_ATTENTION_FIX", "video_and_audio_native_math", flush=True)


@__import__("torch").no_grad()
def vae_attention_preflight(device="cuda:0"):
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoViTDecoder3d
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import MiniMaxH3AudioCausalAttention
    results = []
    with torch.random.fork_rng(devices=[]), math_attention():
        torch.manual_seed(971)
        video = MiniMaxH3VideoViTDecoder3d(in_channels=24, patch_size=2, patch_size_t=4,
            num_layers=1, num_attention_heads=2, attention_head_dim=16,
            num_register_tokens=2, ffn_mult=2).eval()
        audio = MiniMaxH3AudioCausalAttention(in_dim=32, out_dim=8, num_heads=2).eval()
        for name, module, value in [("video_decoder", video, torch.randn(1, 24, 2, 2, 2)),
                                     ("audio_causal_attention", audio, torch.randn(2, 13, 32))]:
            expected = module(value)
            target = copy.deepcopy(module).to(device)
            for half in ([False, True] if str(device).startswith("cuda") else [False]):
                autocast = torch.autocast("cuda", dtype=torch.float16) if half else nullcontext()
                with autocast:
                    result = target(value.to(device)).float().cpu()
                torch.testing.assert_close(result, expected, rtol=1e-2 if half else 5e-4,
                                           atol=1e-2 if half else 5e-4)
                assert torch.isfinite(result).all()
                results.append({"module": name, "autocast_fp16": half,
                                "max_error": (result - expected).abs().max().item()})
    print("MINIMAX_VAE_MATH_PREFLIGHT", json.dumps({"device": device, "checks": results,
                                                   "status": "PASSED"}), flush=True)
    return results


def save_latents(state, directory):
    import torch
    from safetensors.torch import save_file
    tensors = {name: state.get(name).detach().cpu().contiguous()
               for name in ("latents", "audio_latents")}
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Non-finite denoised " + name)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "denoised_latents.safetensors"
    save_file(tensors, str(path), metadata={"source": "actual_H3_AfterDenoiseStep", "format_version": "1"})
    path.chmod(0o660)
    report = {"schema_version": 1, "file": str(path), "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "all_finite": True,
              "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
              "geometry": {k: state.get(k) for k in
                           ("height", "width", "num_frames", "num_latent_frames", "latent_height",
                            "latent_width", "num_audio_latents")}, "fps": 24}
    info = directory / "denoised_latents.json"
    info.write_text(json.dumps(report, indent=2), encoding="utf-8")
    info.chmod(0o660)
    print("MINIMAX_DENOISED_LATENTS_SAVED", json.dumps(report), flush=True)
    return report


def install_checkpoint_hook(directory):
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep
    original = MiniMaxH3AfterDenoiseStep.__call__

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        components, state = original(self, *args, **kwargs)
        save_latents(state, directory)
        return components, state

    MiniMaxH3AfterDenoiseStep.__call__ = wrapped


if __name__ == "__main__":
    install_bridge_fix()
    install_vae_math_fix()
    gpu_preflight()
    vae_attention_preflight()
    install_checkpoint_hook(os.environ["MINIMAX_SMOKE_LOG_DIRECTORY"])
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
