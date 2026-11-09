"""Torch-owned views -> real InfiniCore SiLU -> frozen NineToothed objects.

This bridge contains no SiLU math. Cross-framework streams are synchronized
before and after the native operation. This conservative first integration
prioritizes correctness over latency; it never falls back to Torch SiLU.
"""
import ctypes
from pathlib import Path
import os
import torch

VARIABLE = 'INFINICORE_NINETOOTHED_SILU_ROOT'
PHASES = ('video_vae_encode', 'video_vae_decode')


class SiluBridge:
    def __init__(self, root):
        import infinicore
        from infinicore.lib import _infinicore
        self.core, self.api = infinicore, _infinicore
        self.root = Path(root).resolve()
        if Path(os.environ.get(VARIABLE, '/missing')).resolve() != self.root:
            raise RuntimeError('Explicit frozen NineToothed SiLU root required')
        self.host = ctypes.CDLL(str(self.root / 'libminimax_nine_silu_runtime.so'))
        self.host.minimax_nine_silu_device_statistics.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
        self.host.minimax_nine_silu_device_statistics.restype = None

    def statistics(self, device):
        values = (ctypes.c_uint64 * 10)()
        self.host.minimax_nine_silu_device_statistics(device, values)
        return dict(launches=int(values[0]), last_stream=int(values[2]), last_device=int(values[3]),
                    f32=int(values[7]), f16=int(values[8]), bf16=int(values[9]))

    def __call__(self, source, *, inplace=False, out=None):
        if (source.device.type != 'cuda' or source.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or source.requires_grad or source.ndim > 8 or source.layout != torch.strided):
            raise RuntimeError('VAE Nine SiLU requires inference F32/F16/BF16 strided DCU tensor, rank<=8')
        if source.numel() > 2147483647:
            raise RuntimeError('Frozen rank1 SiLU indexing limit exceeded; no numerical fallback')
        target = source if inplace else out if out is not None else torch.empty_like(source)
        if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device or target.requires_grad:
            raise RuntimeError('VAE SiLU output metadata mismatch')
        device = source.device.index
        before = self.statistics(device)
        previous = self.api.get_device()
        try:
            with torch.cuda.device(device):
                # Producer completion before InfiniCore consumes Torch-owned views.
                torch.cuda.current_stream(device).synchronize()
                self.core.set_device(self.core.device('cuda', device))
                self.api.silu_(self.core.from_torch(target)._underlying, self.core.from_torch(source)._underlying)
                # Native completion before Torch consumes or releases either view.
                self.core.sync_stream()
        finally:
            self.api.set_device(previous)
        after = self.statistics(device)
        launches = int(source.numel() != 0)
        kind = {torch.float32: 'f32', torch.float16: 'f16', torch.bfloat16: 'bf16'}[source.dtype]
        if (after['launches'] - before['launches'] != launches
                or any(after[k] - before[k] != launches * int(k == kind) for k in ('f32', 'f16', 'bf16'))
                or launches and (after['last_device'] != device or not after['last_stream'])):
            raise RuntimeError('Real NineToothed SiLU launch/device/type proof missing')
        return target, dict(device=device, dtype=str(source.dtype), shape=list(source.shape),
                            input_stride=list(source.stride()), output_stride=list(target.stride()),
                            input_offset=int(source.storage_offset()), output_offset=int(target.storage_offset()),
                            inplace=inplace, provided_out=out is not None, launch_delta=launches,
                            before=before, after=after)
