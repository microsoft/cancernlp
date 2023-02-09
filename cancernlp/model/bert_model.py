import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

from allennlp.data import DatasetReader, TokenIndexer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules import TokenEmbedder
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from overrides import overrides

from ..annotation import AnnotatedText, Annotation, Token
from ..transformer_utils import PretrainedTransformerModels
from .doc_class_reader import DocumentDatasetReaderBase, allen_Token
from .preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


@TokenIndexer.register("my-pretrained-indexer")
class MyTransformerIndexer(PretrainedTransformerIndexer):
    """
    Thin wrapper around AllenNLP's PretrainedTransformerIndexer that uses our
    PretrainedTransformerModels class to load custom pretrained models by name (e.g.
    scibert-uncased)
    """

    def __init__(
        self, model_name: str, namespace: str = "tags", max_length: int = None, **kwargs
    ) -> None:
        # do translation of our custom model names
        model_name = PretrainedTransformerModels.get_tokenizer_name(model_name)
        super().__init__(
            model_name=model_name, namespace=namespace, max_length=max_length, **kwargs
        )


@TokenEmbedder.register("my-pretrained-embedder")
class MyTransformerEmbedder(PretrainedTransformerEmbedder):
    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        train_n_layers: Optional[int] = None,
        train_pooler: bool = False,
        train_embedding: bool = False,
    ) -> None:
        # these are mostly copied from PretrainedTransformerEmbedder.__init__
        # FastBERT is not merged to AutoModel yet, so here we call our
        # PretrainedTransformerModels
        TokenEmbedder.__init__(self)

        self.transformer_model = PretrainedTransformerModels.get_pretrained_model(
            model_name
        )

        if gradient_checkpointing is not None:
            self.transformer_model.config.update(
                {"gradient_checkpointing": gradient_checkpointing}
            )

        self.config = self.transformer_model.config
        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length

        # I'm not sure if this works for all models; open an issue on github if you find
        # a case where it doesn't work.
        self.output_dim = self.config.hidden_size

        self._scalar_mix: Optional[ScalarMix] = None
        if not last_layer_only:
            self._scalar_mix = ScalarMix(self.config.num_hidden_layers)
            self.config.output_hidden_states = True

        # Wanted to use PretrainedTransformerModels.get_pretrained_tokenizer,
        # but we need to create PretrainedTransformerTokenizer here
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        tokenizer_kwargs[
            "do_lower_case"
        ] = PretrainedTransformerModels.should_do_lower_case(model_name)
        model_path = PretrainedTransformerModels.get_model_name(model_name)
        tokenizer = PretrainedTransformerTokenizer(
            model_path, tokenizer_kwargs=tokenizer_kwargs
        )
        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = (
            self._num_added_start_tokens + self._num_added_end_tokens
        )

        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        if train_n_layers is not None:
            PretrainedTransformerModels.set_bert_layers_to_train(
                bert_model=self.transformer_model,
                layers_to_train=train_n_layers,
                train_pooler=train_pooler,
                train_embedding=train_embedding,
            )


@TokenEmbedder.register("bert_wrapper")
class BertEmbeddingWrapper(TokenEmbedder):
    """
    Wraps BERT token embedder and removes CLS and SEP tokens from second-to-last
    dimension.
    """

    def __init__(self, wrapped_embedder: TokenEmbedder):
        # init superclass
        super().__init__()
        self.wrapped_embedder = wrapped_embedder

    # pylint: disable=arguments-differ
    def forward(self, *args, **kwargs):

        embedded_tokens = self.wrapped_embedder(*args, **kwargs)
        # chop off first (CLS) and last embeddings.  We don't know where SEP is,
        # it could be before the last position if there is padding.  However,
        # the end embedding should either be SEP or PAD, so chopping it is okay,
        # and if there was a padding mask calculated from the original tokens (before
        # CLS or SEP were added), then it should remove the SEP token wherever it is.
        embedded_tokens = embedded_tokens[..., 1:-1, :]

        return embedded_tokens

    def get_output_dim(self) -> int:
        return self.wrapped_embedder.get_output_dim()


@DatasetReader.register("doc_class_transformer_dataset_reader")
class TransformerDocumentDatasetReader(DocumentDatasetReaderBase):
    def __init__(
        self,
        token_indexer: Union[TokenIndexer, Dict[str, TokenIndexer]],
        token_ann_type: Type[Token] = "Annotation.Token.TransformerInputToken",
        sentence_ann_type: Optional[Type[Annotation]] = None,
        label_key_function: Optional[Callable[[AnnotatedText], List[str]]] = None,
        is_multilabel: bool = True,
        max_instances: Optional[int] = None,
        preprocessing: Optional[PreprocessingPipeline] = None,
        include_examples: bool = False,
        disable_caching: bool = False,
        weight_field: str = None,
        raise_on_empty_instance: bool = True,
    ) -> None:
        """
        :param token_indexer: either a TokenIndexer or a dict of TokenIndexer. If a
        TokenIndexer, it must have the attribute '_tokenizer'. If a dict of
        TokenIndexer, it must have an entry with the key 'tokens' whose value is a
        TokenIndexer that has a '_tokenizer' attribute.
        """

        if isinstance(token_indexer, dict):
            self.tokenizer = token_indexer["tokens"]._tokenizer
        else:
            self.tokenizer = token_indexer._tokenizer

        super().__init__(
            token_ann_type=token_ann_type,
            sentence_ann_type=sentence_ann_type,
            label_key_function=label_key_function,
            is_multilabel=is_multilabel,
            token_indexer=token_indexer,
            max_instances=max_instances,
            preprocessing=preprocessing,
            include_examples=include_examples,
            disable_caching=disable_caching,
            weight_field=weight_field,
            raise_on_empty_instance=raise_on_empty_instance,
        )

    @overrides
    def to_allen_tokens(self, tokens: List[Annotation]) -> List[allen_Token]:
        """Convert list of nlp_lib Tokens to an AllenNLP Tokens"""
        token_ids = [token.token_id for token in tokens]
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        tokens = [
            self.tokenizer._convert_id_to_token(token_id) for token_id in token_ids
        ]
        return [
            allen_Token(text=token_text, text_id=token_id, type_id=0)
            for token_text, token_id in zip(tokens, token_ids)
        ]
