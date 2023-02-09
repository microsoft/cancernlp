"""
AllenNLP DatasetReader that reads from an AnnotatedText Corpus
"""
import logging
import warnings
from functools import partial
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Type, Union)

import numpy as np
from allennlp.data import Instance
from allennlp.data import Token as allen_Token
from allennlp.data import TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import (ArrayField, LabelField, ListField,
                                  MetadataField, MultiLabelField, TextField)
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from ..annotation import AnnotatedText, Annotation, Corpus
from ..collection_utils import collection_utils
from .preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


class ExtendedToken(allen_Token):
    """
    Add to allennlp.data.Token the following fields:
        doc_length: the length of the document containing self
        meta_fields: any additional info to pass to a token
    """

    def __init__(
        self,
        doc_length: Optional[int] = 0,
        numeric_value: Optional[int] = None,
        is_numeric_continuation: Optional[bool] = False,
        meta_fields: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.doc_length = doc_length
        self.numeric_value = numeric_value
        self.is_numeric_continuation = is_numeric_continuation
        self.meta_fields = meta_fields


# allows specifying annotation via a type name ("Annotation.Token") or the class
# directly (Token)
AnnotationType = Union[Type[Annotation], str]
# Either a function taking an AnnotatedTextInterface and returning a list of labels,
# or a string giving the attribute on an AnnotatedTextInterface containing the labels
LabelKeyFunctionType = Callable[[AnnotatedText], List[str]]
LabelKeyInputType = Union[str, LabelKeyFunctionType]


class DocumentDatasetReaderBase(DatasetReader):
    def __init__(
        self,
        token_ann_type: AnnotationType,
        sentence_ann_type: Optional[AnnotationType] = None,
        label_key_function: Optional[LabelKeyInputType] = None,
        is_multilabel: bool = True,
        max_instances: Optional[int] = None,
        token_indexer: Optional[Union[TokenIndexer, Dict[str, TokenIndexer]]] = None,
        preprocessing: Optional[PreprocessingPipeline] = None,
        include_examples: bool = False,
        disable_caching: bool = False,
        weight_field: str = None,
        raise_on_empty_instance: bool = True,
    ) -> None:
        """
        DatasetReader to read a Dataset from an AnnotatedText Corpus
        :param token_ann_type: annotation type to use as tokens
        :param sentence_ann_type: if specified, create a hierarchical dataset using
        this sentence type
        :param label_key_function: function or string indicating how to turn an Example
        (AnnotatedText) into a label.
        :param is_multilabel: if True, each 'label' should be a list of labels
        :param max_instances: maximum number of instances to load (useful for debugging)
        :param token_indexer: if specified, can be a TokenIndexer or a dict of
        TokenIndexer. If a TokenIndexer, the indexer key is 'tokens'. If not specified,
        defaults to SingleIdTokenIndexer with the key 'tokens'.
        :param preprocessing: Optional preprocessing pipeline to run on input
        AnnotatedText objects.
        :param include_examples: If True, include the original AnnotatedText as the
        metadata field "example" in each generated Instance.  The include_examples
        attribute can be directly modified to change this behavior in the instantiated
        object.
        :param disable_caching: If True, do not use cache for loading / storing
        preprocessing datasets.  This option is primarily for testing.
        :param weight_field: if given, each AnnotatedText should have a field by this
        name which contains a float value to use as a weight for this instance
        :param raise_on_empty_instance: raise an exception if an empty instance (no
        sentences / tokens) is encountered.
        """
        super().__init__(
            max_instances=max_instances,
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
        )
        self._do_not_cache = False
        if max_instances:
            if not disable_caching:
                logger.info("Disabling caching because max_instances is set.")
                self._do_not_cache = True
        if token_indexer is None:
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        elif isinstance(token_indexer, dict):
            self._token_indexers = token_indexer
        else:
            self._token_indexers = {"tokens": token_indexer}

        self._sentence_ann_type = sentence_ann_type
        self._token_ann_type = token_ann_type
        self._is_multilabel = is_multilabel
        self._preprocessing = preprocessing
        self.include_examples = include_examples
        self._weight_field = weight_field
        self.raise_on_empty_instance = raise_on_empty_instance

        self._disable_caching = disable_caching

        if isinstance(label_key_function, str):
            label_key_function = partial(
                collection_utils.get_by_path, path=label_key_function
            )

        self._label_key_function = label_key_function

    def from_corpus(
        self, corpus: Union[Corpus, Iterable[AnnotatedText]]
    ) -> Generator[Instance, None, None]:
        """
        Convert Corpus to Instances.  Called internally by _read for final conversion to
        instances.  If passed an iterable, it will yield results lazily.
        """
        for instance in self._from_corpus(corpus):
            self.apply_token_indexers(instance)
            yield instance

    def _from_corpus(
        self, corpus: Union[Corpus, Iterable[AnnotatedText]]
    ) -> Generator[Instance, None, None]:
        """Convert Corpus to Instances.  Called internally by _read for final conversion
        to instances.  If passed an iterable, it will yield results lazily. This
        internal interface does not apply token indexers (important for distributed data
        loading)
        """
        for ann_text in self.shard_iterable(corpus):
            yield self.text_to_instance(ann_text)

    def to_allen_tokens(self, tokens: List[Annotation]) -> List[allen_Token]:
        """Convert the nlp_lib Token to an AllenNLP Token"""
        raise NotImplementedError("subclass should provide to_allen_tokens")

    def _corpus_from_preprocessing(self, dataset_name, max_records=None):
        return self._preprocessing.preprocess(
            dataset_name,
            force_preprocessing=self._disable_caching,
            do_not_cache=self._disable_caching or self._do_not_cache,
            max_records=max_records,
        )

    def preprocess_doc(self, doc: AnnotatedText) -> List[AnnotatedText]:
        """Run preprocessing pipeline on the given AnnotatedText.  May return None if
        the doc is filtered by one of the PipelineComponents"""
        if not self._preprocessing:
            logger.warning("No preprocessing configured for this reader, skipping.")
            return [doc]
        return self._preprocessing.preprocess_doc(doc)

    def preprocess_corpus(
        self, corpus: Corpus, quiet: bool = False
    ) -> Iterator[AnnotatedText]:
        """Helper function to preprocess the given corpus, returning an iterator of
        processed AnnotatedText objects."""
        if not self._preprocessing:
            logger.warning("No preprocessing configured for this reader, skipping.")
            return corpus
        return self._preprocessing.preprocess_corpus(corpus, quiet=quiet)

    @overrides
    def _read(self, file_path_or_dataset_name):
        if self._preprocessing:
            logger.info("Running Preprocessing")
            corpus = self._corpus_from_preprocessing(
                dataset_name=file_path_or_dataset_name, max_records=self.max_instances
            )
        else:
            corpus = Corpus.stream(file_path_or_dataset_name)

        return self._from_corpus(corpus)

    def apply_token_indexers(self, instance: Instance) -> None:
        tokens_field = instance.fields["tokens"]
        if isinstance(tokens_field, TextField):
            tokens_field._token_indexers = self._token_indexers
        elif isinstance(tokens_field, ListField):
            for field in tokens_field.field_list:
                field._token_indexers = self._token_indexers
        else:
            raise ValueError(
                f"Cannot assign token indexer to field type {type(tokens_field)}"
            )

    def doc_to_instance(self, ann_text: AnnotatedText) -> Instance:
        instance = self.text_to_instance(ann_text)
        self.apply_token_indexers(instance)
        return instance

    @overrides
    def text_to_instance(self, ann_text: AnnotatedText) -> Instance:
        """Convert AnnotatedText to allennlp Instance.
        Note this method is meant for distributed loading and DOES NOT add token
        indexers, use doc_to_instance if you want to use the generated instances
        directly.
        """
        fields = {}
        # set up "tokens" field
        if self._sentence_ann_type:
            sentences = [
                TextField(
                    self.to_allen_tokens(list(_sent.covering(self._token_ann_type))),
                )
                for _sent in ann_text.covering(self._sentence_ann_type)
            ]
            if not sentences:
                if self.raise_on_empty_instance:
                    raise ValueError(
                        "No sentences found in AnnotatedText, is it "
                        "annotated with sentences of type "
                        f"{self._sentence_ann_type}"
                    )
                else:
                    # a list of TextFields is expected, so give them an empty one
                    sentences = [TextField([])]
            tokens_field = ListField(sentences)
        else:
            tokens = self.to_allen_tokens(list(ann_text.covering(self._token_ann_type)))
            tokens_field = TextField(tokens, self._token_indexers)
        fields["tokens"] = tokens_field

        # if we have a 'weight' field on the ann_text object, create a 'weight'
        if self._weight_field:
            weight = collection_utils.get_by_path(ann_text, self._weight_field)
            # single-value ArrayField is the closest thing to a scalar
            fields["weight"] = ArrayField(np.array([weight], dtype=np.float32))

        # set up labels field
        if self._label_key_function:
            try:
                label = self._label_key_function(ann_text)
                fields["labels"] = self._label_to_label_field(label)
            except KeyError:
                warnings.warn("labels not found in instances!")
        else:
            warnings.warn(
                "no label_key_function, labels will not be added to instances"
            )

        if self.include_examples:
            fields["example"] = MetadataField(ann_text)
        return Instance(fields)

    def _label_to_label_field(self, label):
        if self._is_multilabel:
            field = MultiLabelField(label, skip_indexing=False)
        else:
            if not isinstance(label, str):
                # just take the first label if there are multiple labels and we're not
                # treating this as a multi-label problem.  If there are no labels,
                # convert this to the string "None"
                label = label[0] if len(label) else "None"
            field = LabelField(label, skip_indexing=False)
        return field


@DatasetReader.register("doc_class_dataset_reader")
class DocumentDatasetReader(DocumentDatasetReaderBase):
    def __init__(
        self,
        token_ann_type: AnnotationType,
        sentence_ann_type: Optional[AnnotationType],
        label_key_function: Optional[LabelKeyInputType] = None,
        text_key_function: Optional[Callable[[Annotation], str]] = None,
        is_multilabel: bool = True,
        max_instances: Optional[int] = None,
        token_indexer: Optional[Union[TokenIndexer, Dict[str, TokenIndexer]]] = None,
        preprocessing: Optional[PreprocessingPipeline] = None,
        include_examples: bool = False,
        disable_caching: bool = False,
        weight_field: str = None,
        raise_on_empty_instance: bool = True,
    ) -> None:

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

        if isinstance(text_key_function, str):
            text_key_function = partial(
                collection_utils.get_by_path, path=text_key_function
            )

        def default_text_key(token: Annotation) -> str:
            """Default to just returning the token text."""
            return token.text

        self._text_key_function = text_key_function or default_text_key

    @overrides
    def to_allen_tokens(self, tokens: List[Annotation]) -> List[ExtendedToken]:
        """Convert list of nlp_lib Tokens to an AllenNLP Tokens"""
        return [
            ExtendedToken(
                text=self._text_key_function(token),
                idx=token.begin,
                doc_length=len(token.annotated_text.text),
            )
            for token in tokens
        ]
