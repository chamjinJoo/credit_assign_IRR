from __future__ import annotations

from typing import Any

from .r_pci import JAXRPCIAlgorithm


class JAXPPOAlgorithm(JAXRPCIAlgorithm):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("causal_lambda", 0.0)
        kwargs.setdefault("bridge_epoch", 0)
        kwargs["global_reward"] = True
        super().__init__(*args, **kwargs)
