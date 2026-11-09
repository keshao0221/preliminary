"""Observe Torch dispatch within named model phases without changing tensors."""
from collections import Counter
from contextlib import contextmanager
from torch.utils._python_dispatch import TorchDispatchMode

class PhaseInventory(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.stack=[]
        self.counts=Counter()
    @contextmanager
    def phase(self,name):
        self.stack.append(name)
        try:
            yield
        finally:
            self.stack.pop()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if self.stack:
            self.counts[(self.stack[-1],str(func))]+=1
        return func(*args,**(kwargs or {}))
    def report(self):
        return dict(status='OBSERVED_TORCH_MODEL_OPERATIONS', complete_model_migration=False,
                    native_core_operations_included=False,
                    operations=[dict(phase=p,operation=f,count=n) for (p,f),n in sorted(self.counts.items())],
                    note='Torch phases are actual dependency evidence; counts do not grant third-party permission')
