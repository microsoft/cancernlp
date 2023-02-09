"""
See https://github.com/deep-spin/entmax
"""

import logging
from functools import partial

import torch
from overrides import overrides

from .normalization import Normalization

logger = logging.getLogger()

try:
    # Majority of custom autograd functions are not traceable. Unfortunately, this is
    # true for `entmax15`, `entmax_bisect` and `sparsemax`. This methods wrap their
    # implementation in torch.autograd.function.Function which are then invoked with
    # `apply` method. This code flow is what is causing the tracer issues. To get
    # around this during tracing or ONNX export, we import the underlying Function
    # implementations and call `forward` method while passing in a dummy context
    # object. This workaround works for inference but should not be used during
    # training as the back propagation will be affected.
    from entmax import sparsemax
    from entmax.activations import SparsemaxFunction
    from torch.autograd.function import _ContextMethodMixin
except ModuleNotFoundError:
    logger.warning("entmax not found, calling entmax Normalization will fail")
    sparsemax = None


def _entmax_helper(
    entmax_fn, vector: torch.Tensor, mask: torch.Tensor = None, dim: int = -1
):
    """
    Any slice along dim that are all paddings get NaN's. This is consistent with the
    behavior of torch.nn.functional.softmax.
    """
    masked_vector = (
        vector.masked_fill(~(mask.to(dtype=torch.bool)), -1e32)
        if mask is not None
        else vector
    )
    values = entmax_fn(masked_vector, dim=dim)

    if mask is not None:
        # find whole slice that is mask, and make that slice is all NaN's
        # torch.all doesn't support regular onnx conversion
        # equivalent to: torch.all(mask == 0, dim=dim, keepdim=True)
        all_padding = ~torch.sum(mask, dim=dim, keepdim=True).bool()
        values = values.masked_fill(all_padding, float("nan"))

    return values


@Normalization.register("sparsemax")
class Sparsemax(Normalization):
    """
    Sparsemax
    """

    def __init__(self) -> None:
        if sparsemax is None:
            raise ModuleNotFoundError(
                "sparsemax unavailable, please install entmax package"
            )
        super().__init__()

    @overrides
    def forward(
        self, vector: torch.Tensor, mask: torch.Tensor = None, dim: int = -1
    ) -> torch.Tensor:

        entmax_fn = (
            partial(SparsemaxFunction.forward, _ContextMethodMixin())
            if torch.jit.is_tracing()
            else sparsemax
        )
        return _entmax_helper(entmax_fn, vector, mask, dim)
