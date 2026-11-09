"""Replace only video-VAE SiLU dispatch; retain evidence of all remaining math."""
from collections import Counter
import torch
from inventory import PhaseInventory
from vae_silu_bridge import SiluBridge
from vae_silu_contract import PHASES,assess_vae


class VideoSiluInventory(PhaseInventory):
    def __init__(self, root=None, *, bridge=None):
        super().__init__()
        self.bridge = SiluBridge(root) if bridge is None else bridge
        self.routed = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.stack and self.stack[-1] in PHASES and func in (
                torch.ops.aten.silu.default, torch.ops.aten.silu_.default, torch.ops.aten.silu.out):
            source = args[0] if args else kwargs['self']
            output, proof = self.bridge(source, inplace=func is torch.ops.aten.silu_.default, out=kwargs.get('out'))
            self.routed.append(dict(phase=self.stack[-1], operation=str(func), **proof))
            return output
        return super().__torch_dispatch__(func, types, args, kwargs)

    def report(self):
        result = super().report()
        result.update(status='TORCH_REMAINDER_AND_NINE_VIDEO_VAE_SILU_OBSERVED',
                      ninetoothed_video_vae_silu=self.routed,
                      video_vae_silu_counts=dict(Counter(p['phase'] for p in self.routed)),
                      ninetoothed_math_changed=False, synchronization='Torch producer sync -> InfiniCore SiLU -> InfiniCore stream sync',
                      note='Routed SiLU is excluded from residual Torch operations; all other Torch model computation remains listed')
        return result


