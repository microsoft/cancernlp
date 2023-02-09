"""
An *attention* module that computes the similarity between
an input vector and the rows of a matrix.
"""

import torch
from allennlp.common.registrable import Registrable
from overrides import overrides


class FixedAttention(torch.nn.Module, Registrable):
    """
    An `FixedAttention` takes two inputs: a (batched) vector, plus an
    optional binary mask where a zero indicates padding and one otherwise.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_input_dim(self) -> int:
        raise NotImplementedError

    @overrides
    def forward(self, vector: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param vector: arbitrary dim
        :param mask: shape == vector.shape[:-1]

        Output: one attention weight for each vector. shape vector.shape[:-1].
                Masked location (paddings) have attention weights of zero
        """
        raise NotImplementedError
