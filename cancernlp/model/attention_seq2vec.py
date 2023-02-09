from typing import Optional

import torch
from allennlp.modules import FeedForward, Seq2VecEncoder
from overrides import overrides

from ..fixed_attention import FixedAttention
from ..normalization import Normalization


@Seq2VecEncoder.register("attention")
class AttentionEncoder(Seq2VecEncoder):
    """
    An attention seq2vec encoder with configurable attention and normalization
    modules.
    """

    def __init__(
        self,
        attention: FixedAttention,
        normalization: Optional[Normalization] = None,
        value_transform: Optional[FeedForward] = None,
    ) -> None:
        """
        :param attention: module to compute one attention weight per input embedding
        :param normalization: optional module to normalize attention weights
        :param value_transform: transform input before linear combination according to
        (normalized) attention weigths.
        """
        super().__init__()

        self._attention = attention
        self._intput_dim = attention.get_input_dim()
        self._normalization = normalization

        self._value_transform = value_transform

        self._output_dim = (
            value_transform.get_output_dim() if value_transform else self._intput_dim
        )
        self.attention_weights = None
        self.unnormalized_attention_weights = None

    @overrides
    def get_input_dim(self) -> int:
        return self._intput_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        """
        :param tokens: shape (B, N, input_dim)
        :param mask: shape (B, N)

        return:
        combined vecotr: shape (B, output_dim)
        """

        weights = self._attention(tokens, mask)

        # save for external use
        self.unnormalized_attention_weights = weights  # (B, N)

        if self._normalization:
            weights = self._normalization(weights, mask)  # (B, N)

            # nan can occur when all tokens in a sequence are paddings.
            # in this case, set attention weights to zero
            # Note: to be strict, one should only replace a whole row of NaN's with zero
            weights.data.masked_fill_(torch.isnan(weights), 0)

        # save for external use
        self.attention_weights = weights  # (B, N)

        if self._value_transform:
            # (B, N, output_dim)
            tokens = self._value_transform(tokens * mask.unsqueeze(-1))

        x = tokens.transpose(-1, -2)  # (B, output_dim, N)
        y = weights.unsqueeze(-1)  # (B, N, 1)
        combined = torch.bmm(x, y).squeeze(-1)  # (B, output_dim)
        return combined
