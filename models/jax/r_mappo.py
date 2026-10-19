from __future__ import annotations

from typing import Any

from .r_pci import JAXRPCIAlgorithm


class JAXRMAPPOAlgorithm(JAXRPCIAlgorithm):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("use_centralized_critic", True)
        kwargs.setdefault("causal_lambda", 0.0)
        kwargs["global_reward"] = True
        super().__init__(*args, **kwargs)
