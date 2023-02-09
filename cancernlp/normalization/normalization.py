import torch
from allennlp.common.registrable import Registrable
from overrides import overrides


class Normalization(torch.nn.Module, Registrable):
    """
    An `Normalization` takes two inputs: a (batched) vector, plus an
    optional mask.
    """

    def __init__(self) -> None:
        super().__init__()

    @overrides
    def forward(
        self, vector: torch.Tensor, mask: torch.Tensor = None, dim: int = -1
    ) -> torch.Tensor:
        """
        :param vector: arbitrary shape
        :param mask: same shape as vector
        :param dim: dimension along which the normalization is performed

        Output: normalized vectors: same shape as vector
        """
        raise NotImplementedError
