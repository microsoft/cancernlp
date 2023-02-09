import math
from typing import Optional

import torch
from allennlp.modules import FeedForward
from overrides import overrides

from .fixed_attention import FixedAttention


@FixedAttention.register("dot_product")
class DotProductAttention(FixedAttention):
    """
    Dot product attention
    """

    def __init__(
        self, embedding_dim: int, key_transform: Optional[FeedForward] = None
    ) -> None:
        super().__init__()
        self.key_transform = key_transform
        self.embedding_dim = (
            embedding_dim if key_transform is None else key_transform.get_output_dim()
        )
        self.scaling_factor = 1.0 / math.sqrt(self.embedding_dim)
        self.attention_vector = torch.nn.Parameter(torch.Tensor(self.embedding_dim, 1))
        torch.nn.init.xavier_uniform_(self.attention_vector)

        self.attention_weights = None

    @overrides
    def get_input_dim(self) -> int:
        return self.embedding_dim

    @overrides
    def forward(self, vector: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param vector: arbitrary shape, but last dim is the embbeding_dim passed to
                       __init__

        See parent docstring
        """

        masked = vector * mask.unsqueeze(-1) if mask is not None else vector

        if self.key_transform:
            masked = self.key_transform(masked)

        weights = torch.matmul(masked, self.attention_vector).squeeze(-1)
        weights = weights * self.scaling_factor

        if mask is not None:
            weights.masked_fill_(~(mask.to(dtype=torch.bool)), 0)

        self.attention_weights = weights

        return weights
