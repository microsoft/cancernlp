import inspect
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn
from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.models import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, Seq2VecEncoder,
                              TextFieldEmbedder, TokenEmbedder)
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from more_itertools import one
from overrides import overrides

from ..annotation import AnnotatedText


def _update_bert_mask(mask):
    # We want to remove the mask elements corresponding the the
    # CLS and SEP tokens. They are not PAD tokens, and so have a 1
    # value in the mask.  CLS will always be at the beginning, but
    # if there is padding SEP will not be at the end.  However,
    # assuming the mask for each entry is a string of zero or more
    # 1's followed by a string of zero or more 0's, removing the
    # last 1 is equivalent to removing the first (or second) 1.
    # See the corresponding unit test for an example.
    return mask[..., 2:]


def compute_mask_from_indexer_tokens(tokens: TextFieldTensors):
    """
    Logic for creating a mask from TextFieldEmbedder output.  This is petty nasty,
    we should figure out a more principled way to manage other tensors, such as
    masks, that may need to be reshaped when the inputs are reshaped.
    :param tokens: tokens from a TextFieldEmbedder
    :return: appropriately-shaped mask for downstream use
    """
    tokens_embeddings = tokens["tokens"]
    if len(tokens_embeddings) == 1:
        # for standard embeddings there's just a single entry here
        tokens_tensor = one(tokens_embeddings.values())
    else:
        # BERT has mask and type IDs as well ask token IDs, try to get the token IDs
        tokens_tensor = tokens_embeddings["token_ids"]
    # set num_wrapping_dims correctly for hierarchical data when generating mask
    text_field_ndims = tokens_tensor.dim()
    # if tokens contains a 'mask' element, this returns it, otherwise it looks at
    # the indexed tokens and assumes padding has value zero.
    mask = get_text_field_mask(tokens, num_wrapping_dims=text_field_ndims - 2)

    # This is a hack, but if we have type_ids we assume it's BERT and remove the
    # first and last elements of the mask.
    # TODO: really need something more elegant
    if "type_ids" in tokens["tokens"]:
        mask = _update_bert_mask(mask)

    return mask


class DocumentClassifierBase(Model):
    """
    General-purpose document classification based on allennlp BasicClassifier.  Meant
    as a base class for multi-label and multi-class document classification models.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        embedding_dropout: float = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        two_stage_prediction: bool = False,
        two_stage_topk: int = 10,
    ) -> None:
        # Notice that we have to pass the vocab to the base class constructor.
        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder

        if embedding_dropout:
            self._embedding_dropout = torch.nn.Dropout2d(embedding_dropout)
        else:
            self._embedding_dropout = None

        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward

        if self._feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            if self._seq2vec_encoder is not None:
                self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()
            else:
                # Feed embedded text directly to classifier. Suitable for models like
                # bag-of-words
                self._classifier_input_dim = self._text_field_embedder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self._num_labels
        )

        # a flag to enable two-stage prediction that uses the top-k sentences
        # ranked by attention weight
        self.two_stage_prediction = two_stage_prediction
        self.two_stage_topk = two_stage_topk
        initializer(self)

    def forward(
        self,
        tokens: TextFieldTensors,
        labels: torch.Tensor = None,
        example: AnnotatedText = None,
        weight: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        logits, score, *_ = self.run_forward(token_dict=tokens)
        output_dict = {"logits": logits, "score": score}

        if labels is not None:
            if weight is not None:
                weight = weight.squeeze()
            loss = self._compute_loss_and_metrics(logits, labels, weights=weight)
            output_dict["loss"] = loss

        return output_dict

    def run_forward(
        self,
        token_dict: TextFieldTensors,
        hidden_shape: torch.LongTensor = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor]:
        """
        Core forward pass logic that can be used by traceable wrappers
        during onnx conversion of this model
        :param token_dict: Dict[str, Dict[str, torch.Tensor]], the dictionary of
            dictionaries containing token tensor inputs. It is expected to have at
            least a "tokens" key mapped to a dictionary with at least a `token_ids`
            key or a singleton dictionary if the key is different.
        :param hidden_shape: torch.LongTensor, this specifies the size of the hidden
            initial state for the RNN-based sequence encoders. It is very important
            during ONNX conversion for infering dynamic sizes. For more details about
            how it is used during ONNX conversion, see `rwep_onnx_lib` README file.
        :param kwargs: Any, extra keyword arguments.

        :returns a Tuple[torch.Tensor] with at least these tensors:
            `(logits, score, sentence attentions, token attentions, token embeddings)`
            Additional tensors could be returned based on model configuration.
        """
        mask = compute_mask_from_indexer_tokens(token_dict)
        token_embeddings = self._text_field_embedder(token_dict)
        logits, score = self._compute_logits_and_score(
            embedded_text=token_embeddings, mask=mask, hidden_shape=hidden_shape
        )

        sentence_attentions = self._get_attention_weights(
            encoder=self._seq2vec_encoder, op_name="get_sentence_attention_weights"
        )
        token_attentions = self._get_attention_weights(
            encoder=self._seq2vec_encoder, op_name="get_token_attention_weights"
        )
        if token_attentions.numel() > 0:
            token_attentions = token_attentions.view(token_embeddings.shape[:2] + (-1,))

        if self.two_stage_prediction:
            topk = torch.tensor(self.two_stage_topk, dtype=torch.long)
            _, sorted_indices = sentence_attentions.sort(1, descending=True)
            # preserve original sentence ordering after selecting topk
            topk_indices, _ = torch.narrow(
                sorted_indices, dim=1, start=0, length=topk
            ).sort(1, descending=False)

            _, score_using_topk = self._compute_logits_and_score(
                self._gather_by_2d_indices(token_embeddings, topk_indices, 1),
                self._gather_by_2d_indices(mask, topk_indices, 1),
                hidden_shape,
                hidden_shape_index=-1,
            )

            return (
                logits,
                score,
                sentence_attentions,
                token_attentions,
                token_embeddings,
                score_using_topk,
            )

        return (logits, score, sentence_attentions, token_attentions, token_embeddings)

    def _compute_logits_and_score(
        self, embedded_text, mask, hidden_shape=None, hidden_shape_index=0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = embedded_text.shape
        # use 2D dropout to drop full token embeddings
        if self._embedding_dropout:
            embedded_text = self._embedding_dropout(
                embedded_text.view(-1, input_shape[-1])
            ).view(input_shape)

        if self._seq2seq_encoder:
            # Apply seq2seq encoder
            # If the input is hierarchical, we collapse the batch and sentence dim.
            # Doing so, we lose the sentence-level info within a seq2seq encoder.
            # However this allows us to apply seq2seq encoders allennlp provides, such
            # as lstm and and gru. We restore the sentence-level dim after the seq2seq
            # encoder is applied.
            flat_mask = mask
            if len(input_shape) > 3:
                # assume input_shape = BATCH_SIZE x N_SENT x N_TOKENS x EMBEDDING_DIM
                # collapse sent and tokens dimensions
                embedded_text = embedded_text.view(
                    (-1, input_shape[-2], input_shape[-1])
                )
                flat_mask = flat_mask.view((-1, flat_mask.shape[-1]))

            embedded_text = self._call_encoder_with_optional_hidden_state(
                encoder=self._seq2seq_encoder,
                embedded_text=embedded_text,
                mask=flat_mask,
                hidden_shape=hidden_shape,
            )
            if self._dropout:
                embedded_text = self._dropout(embedded_text)

            if len(input_shape) > 3:
                # allennlp seq2seq encoder says the output sequence should have the
                # same sequence length as the input. Therefore, we can restore the shape
                # as follows.
                embedded_text = embedded_text.view(input_shape[:-1] + (-1,))

        if self._seq2vec_encoder:
            embedded_text = self._call_encoder_with_optional_hidden_state(
                encoder=self._seq2vec_encoder,
                embedded_text=embedded_text,
                mask=mask,
                hidden_shape=hidden_shape,
                hidden_shape_index=hidden_shape_index,
            )

            if self._dropout:
                embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)

        score = self._score_from_logits(logits)

        return logits, score

    def _score_from_logits(self, logits):
        raise NotImplementedError(
            "_score_from_logits should be implemented by subclasses"
        )

    def _compute_loss_and_metrics(self, logits, labels, weights=None):
        raise NotImplementedError(
            "_compute_loss_and_metrics should be implemented by subclasses"
        )

    def _get_attention_weights(self, encoder, op_name) -> torch.Tensor:
        attention_op = getattr(encoder, op_name, None)
        attention_weight = attention_op() if callable(attention_op) else torch.empty(0)

        return attention_weight

    def _gather_by_2d_indices(
        self, in_tensor: torch.Tensor, indices: torch.LongTensor, dim: int = 1
    ) -> torch.Tensor:
        """
        Gathers values along an axis specified by dim. Unlike the `torch.gather`, this
        function allows passing in an input tensor with a different dimension than the
        indices.
        :param in_tensor, torch.Tensor: the input tensor with at least 3-dimensions
        :param indices, torch.LongTensor: the index tensor for gathering values
        :param dim, int: the axis for indexing the input tensor
        """
        t_indices = indices.unsqueeze(-1)
        x = in_tensor.view(in_tensor.shape[:2] + (-1,))
        t_indices = t_indices.expand(indices.shape + x.shape[-1:])
        x = x.gather(dim, t_indices)
        return x.view(indices.shape + in_tensor.shape[2:])

    def _call_encoder_with_optional_hidden_state(
        self, encoder, embedded_text, mask, hidden_shape=None, hidden_shape_index=0
    ):
        if hidden_shape is not None and hidden_shape.numel() > 0:
            # pass hidden state if encoder accepts it
            if self._can_accept_keyword_arg(
                obj=encoder, op_name="forward", keyword="hidden_state"
            ):
                return encoder(
                    embedded_text,
                    mask=mask,
                    # generate zeros tensor using hidden state shape hint
                    hidden_state=torch.zeros(
                        torch.Size(hidden_shape[hidden_shape_index, :]),
                        dtype=torch.float,
                    ),
                )
        return encoder(embedded_text, mask=mask)

    def _can_accept_keyword_arg(self, obj: object, op_name: str, keyword: str) -> bool:
        method_op = getattr(obj, op_name, None)
        return callable(method_op) and any(
            True
            for param_name, param in inspect.signature(method_op).parameters.items()
            if param.kind == param.VAR_KEYWORD
            or (
                param_name == keyword
                and param.kind in {param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD}
            )
        )


@Model.register("multi_class_document_classifier")
class MultiClassDocumentClassifier(DocumentClassifierBase):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        embedding_dropout: float = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        two_stage_prediction: bool = False,
        two_stage_topk: int = 10,
    ) -> None:
        super().__init__(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            seq2seq_encoder=seq2seq_encoder,
            seq2vec_encoder=seq2vec_encoder,
            feedforward=feedforward,
            embedding_dropout=embedding_dropout,
            dropout=dropout,
            num_labels=num_labels,
            label_namespace=label_namespace,
            initializer=initializer,
            regularizer=regularizer,
            two_stage_prediction=two_stage_prediction,
            two_stage_topk=two_stage_topk,
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss(reduction="none")

    def _score_from_logits(self, logits):
        return torch.nn.functional.softmax(logits, dim=-1)

    def _compute_loss_and_metrics(self, logits, labels, weights=None):
        self._accuracy(logits, labels)
        loss_per_example = self._loss(logits, labels.long().view(-1))
        if weights is not None:
            loss_per_example *= weights
        return torch.mean(loss_per_example)

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        predictions = output_dict["score"].cpu().detach().numpy()
        labels = [
            self.vocab.get_token_from_index(np.argmax(_pred), namespace="labels")
            for _pred in predictions
        ]
        output_dict["labels"] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}


@Model.register("multi_label_document_classifier")
class MultiLabelDocumentClassifier(DocumentClassifierBase):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        two_stage_prediction: bool = False,
        two_stage_topk: int = 10,
    ) -> None:
        super().__init__(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            seq2seq_encoder=seq2seq_encoder,
            seq2vec_encoder=seq2vec_encoder,
            feedforward=feedforward,
            dropout=dropout,
            num_labels=num_labels,
            label_namespace=label_namespace,
            initializer=initializer,
            regularizer=regularizer,
            two_stage_prediction=two_stage_prediction,
            two_stage_topk=two_stage_topk,
        )
        self._accuracy = BooleanAccuracy()
        self._loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def _score_from_logits(self, logits):
        return torch.sigmoid(logits)

    def _compute_loss_and_metrics(self, logits, labels, weights=None):
        self._accuracy(logits.round().long(), labels)
        loss_per_example = self._loss(logits, labels.float())
        if weights:
            loss_per_example *= weights
        return torch.mean(loss_per_example)

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        predictions = output_dict["score"].cpu().detach().round().numpy()
        labels = [
            [
                self.vocab.get_token_from_index(_idx, namespace="labels")
                for _idx, _flag in enumerate(_pred)
                if _flag
            ]
            for _pred in predictions
        ]
        output_dict["labels"] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}


@TokenEmbedder.register("batched_embedding")
class BatchedEmbedding(TokenEmbedder):
    """
    Embed tokens in fixed-size batches sent to the wrapped TokenEmbedder.  Expects
    batches to be of size:
    BATCH_SIZE x N_SENT x N_TOKENS
    This will run N_SENT passes through the underlying embedder, each of size
    BATCH_SIZE x N_TOKENS
    and reconstuct a final embedding of size
    BATCH_SIZE x N_SENT x N_TOKENS x EMBEDDING_DIM
    as output.  This can be much more memory efficient than running as a single large
    batch.
    """

    def __init__(self, bert_embedder: TokenEmbedder):
        # init superclass
        super().__init__()

        self.bert_embedder = bert_embedder

    # pylint: disable=arguments-differ
    def forward(self, token_ids, *, mask=None, type_ids=None):

        if token_ids.ndim != 3:
            raise ValueError("Expected batch_size x n_sent x n_token tensor")

        token_ids_shape = token_ids.shape
        extra_params = {}
        if mask is not None:
            extra_params["mask"] = mask.view((-1,) + mask.shape[2:])

        if type_ids is not None:
            extra_params["type_ids"] = type_ids.view((-1,) + type_ids.shape[2:])

        embedded_tokens = self.bert_embedder(
            token_ids.view((-1,) + token_ids.shape[2:]), **extra_params
        )
        return embedded_tokens.view(token_ids_shape[:2] + embedded_tokens.shape[-2:])

    def get_output_dim(self) -> int:
        return self.wrapped_embedder.get_output_dim()
