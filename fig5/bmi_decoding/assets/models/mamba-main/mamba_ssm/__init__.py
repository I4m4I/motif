__version__ = "2.2.6.post3"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simplemotif import Mambamotif

# Keep sequence modules importable even when optional LM dependencies
# (e.g. transformers version constraints) are unavailable.
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except Exception:
    MambaLMHeadModel = None
