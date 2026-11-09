"""H3 three-axis RoPE cache, expressed entirely in the NineToothed DSL."""
import ninetoothed as nt
import ninetoothed.language as nl
from ninetoothed.language import libdevice

TILE=(16,1,16)

def application(cosine,sine,positions,theta,frequency_dim):
    # Match the established positions.float() before angle arithmetic, including
    # F64 and integer inputs. Frequency indices come from the tensor layout.
    frequency=nl.cast(cosine.offsets(2),nl.float32)
    exponent=nl.div_rn(frequency,frequency_dim)
    inverse=nl.div_rn(1.0,libdevice.pow(theta,exponent))
    angle=nl.cast(positions,nl.float32)*inverse
    cosine=libdevice.cos(angle)
    sine=libdevice.sin(angle)

application.__annotations__=dict(cosine=nt.Tensor(3).tile(TILE),sine=nt.Tensor(3).tile(TILE),
    positions=nt.Tensor(3).tile(TILE),theta=nt.Tensor(0,dtype=nt.float32),frequency_dim=nt.Tensor(0,dtype=nt.float32))
