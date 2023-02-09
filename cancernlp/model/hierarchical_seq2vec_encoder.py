import logging
from typing import Optional

import torch
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from overrides import overrides

logger = logging.getLogger()


@Seq2VecEncoder.register("hierarchical")
class HierarchicalEncoder(Seq2VecEncoder):
    """
    Hierarchical Encoder with configurable token and sentence level seq2seq and seq2vec
    encoders.
    """

    def __init__(
        self,
        token_seq2vec_encoder: Seq2VecEncoder,
        sentence_seq2vec_encoder: Seq2VecEncoder,
        token_seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        sentence_seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        return_norm_sent_attns: bool = True,
        return_norm_token_attns: bool = True,
    ) -> None:
        super().__init__()

        self._token_seq2seq_encoder = token_seq2seq_encoder
        self._token_seq2vec_encoder = token_seq2vec_encoder
        self._sentence_seq2seq_encoder = sentence_seq2seq_encoder
        self._sentence_seq2vec_encoder = sentence_seq2vec_encoder

        if self._token_seq2seq_encoder:
            self._input_dim = self._token_seq2seq_encoder.get_input_dim()
        else:
            self._input_dim = self._token_seq2vec_encoder.get_input_dim()

        self._output_dim = self._sentence_seq2vec_encoder.get_output_dim()

        self.mask = None  # cache for add_attention_annotations()
        self.return_norm_sent_attns = return_norm_sent_attns
        self.return_norm_token_attns = return_norm_token_attns

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        hidden_state: torch.Tensor = None,
    ):
        """
        :param tokens: shape (B, S, T, E)
        :param mask: shape (B, S, T)  tensor where 0 indicates padding and 1 otherwise.

        Returns:
        tensor shape (B, sentence_seq2vec_output_dim)
        """

        self.mask = mask.clone()  # caching for add_attention_annotations()

        input_shape = tokens.shape
        input_mask_shape = mask.shape

        # collapse batch and sentence dim for token-level encoding
        tokens = tokens.view((-1,) + input_shape[-2:])  # (B x S, T, E)
        mask = mask.view((-1, input_mask_shape[-1]))  # (B x S, T)

        if self._token_seq2seq_encoder:
            # (B x S, T, token_seq2seq_output_dim)
            tokens = self._token_seq2seq_encoder(
                tokens, mask=mask, hidden_state=hidden_state
            )

        # (B x S, token_seq2vec_output_dim)
        tokens = self._token_seq2vec_encoder(tokens, mask=mask)

        # (B, S, token_seq2vec_output_dim)
        tokens = tokens.view(input_shape[:2] + (-1,))
        mask = mask.view(input_mask_shape)  # (B, S, T)
        # torch.any doesn't support regular onnx conversion
        # equivalent to: torch.any(mask.bool(), dim=2)
        mask = torch.sum(mask, dim=2).bool()  # (B, S)

        if self._sentence_seq2seq_encoder:
            # (B x S, T, sentence_seq2seq_output_dim)
            tokens = self._sentence_seq2seq_encoder(tokens, mask=mask)

        # (B x sentence_seq2vec_output_dim)
        tokens = self._sentence_seq2vec_encoder(tokens, mask=mask)

        return tokens

    def get_sentence_attention_weights(self) -> torch.Tensor:
        op_name = (
            "attention_weights"
            if self.return_norm_sent_attns
            else "unnormalized_attention_weights"
        )
        return getattr(self._sentence_seq2vec_encoder, op_name, None)

    def get_token_attention_weights(self) -> torch.Tensor:
        op_name = (
            "attention_weights"
            if self.return_norm_token_attns
            else "unnormalized_attention_weights"
        )
        return getattr(self._token_seq2vec_encoder, op_name, None)
