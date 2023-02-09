import torch
from overrides import overrides
from torch.nn.functional import softmax

from .normalization import Normalization


@Normalization.register("softmax")
class Softmax(Normalization):
    """
    softmax
    """

    def __init__(self) -> None:
        super().__init__()

    @overrides
    def forward(
        self, vector: torch.Tensor, mask: torch.Tensor = None, dim: int = -1
    ) -> torch.Tensor:
        """
        Any slice along dim that are all paddings get NaN's. This is the behavior of
        torch.nn.functional.softmax.
        """
        masked_vector = (
            vector.masked_fill(~(mask.to(dtype=torch.bool)), -float("inf"))
            if mask is not None
            else vector
        )
        return softmax(masked_vector, dim=dim)
