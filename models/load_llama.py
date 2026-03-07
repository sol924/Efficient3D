"""Legacy LLaMA loader shim.

This module is retained only for backward compatibility with historical imports.
The current Efficient3D pipelines instantiate LLaMA inside `chat3d_fast_*.py`.
"""

from typing import Any


def init_llama_model(config: Any):
    """Legacy API placeholder."""
    raise NotImplementedError(
        "models/load_llama.py is a legacy entrypoint and is not used in the current "
        "Efficient3D/DVTIE pipeline. Use models/chat3d_fast_*.py entry classes."
    )

