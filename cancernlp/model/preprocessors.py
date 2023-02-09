"""
Annotators wrapped as pipeline components
"""
from typing import List, Optional

from overrides import overrides

from ..annotation import AnnotatedText, AnnotatedTextInterface
from ..annotators import (FixedLengthSentencizer, MaxTokensAnnotator,
                          SimpleTokenizer)
from ..transformer import TransformerInputToken
from ..transformer_utils.transformer_utils import WordPieceTokenizer
from .preprocessing import (AnnotatorPipelineComponentBase, PipelineComponent,
                            PipelineProcessorComponent)


@PipelineComponent.register("simple-tokenizer")
class SimpleTokenizerPreprocessor(AnnotatorPipelineComponentBase):
    def __init__(self, add_tokens: bool = True, add_sentences: bool = True):
        annotator = SimpleTokenizer(add_tokens=add_tokens, add_sentences=add_sentences)
        super().__init__(annotator)


@PipelineComponent.register("fixed-length-sentencizer")
class FixedLengthSentencizerPreprocessor(AnnotatorPipelineComponentBase):
    def __init__(self, token_type: str, sentence_length: int):
        annotator = FixedLengthSentencizer(
            token_type=token_type, sentence_length=sentence_length
        )
        super().__init__(annotator)


@PipelineComponent.register("max-tokens-annotator")
class MaxTokensAnnotatorPreprocessor(PipelineProcessorComponent):
    """
    Limit the number of tokens in a document / sentence.

    This isn't implemented as an AnnotatorPipelineComponent because we want to run on
    the top-level object when we have MultiAnnotatedText documents in order to limit
    the total number of tokens; i.e. we want to have 6400 tokens max per
    MultiAnnotatedText, not per AnnotatedText within a MultiAnnotatedText.
    """

    def __init__(
        self, token_type: str, max_tokens: int, sentence_type: Optional[str] = None
    ):
        self.annotator = MaxTokensAnnotator(
            token_type=token_type, max_tokens=max_tokens, sentence_type=sentence_type
        )

    def __call__(self, ann_text: AnnotatedText) -> AnnotatedText:
        self.annotator(ann_text)
        return ann_text


class PipelineFilterBase(PipelineComponent):
    """Base class for simple per-instance filters"""

    def is_valid_instance(self, ann_text: AnnotatedTextInterface) -> bool:
        raise NotImplementedError

    @overrides
    def process(self, ann_text: AnnotatedTextInterface) -> List[AnnotatedTextInterface]:
        if self.is_valid_instance(ann_text):
            return [ann_text]
        return []


@PipelineComponent.register("wordpiece-tokenizer")
class WordPieceTokenizerPreprocessor(AnnotatorPipelineComponentBase):
    def __init__(
        self,
        model_name: str,
        do_lower_case: Optional[bool] = None,
        sentence_type: Optional[str] = None,
        token_type: Optional[str] = None,
        max_input_length: Optional[int] = None,
        output_token_type: Optional[str] = TransformerInputToken.get_type_name(),
    ):
        annotator = WordPieceTokenizer(
            tokenizer=model_name,
            do_lower_case=do_lower_case,
            sentence_type=sentence_type,
            token_type=token_type,
            max_input_length=max_input_length,
            output_token_type=output_token_type,
        )
        super().__init__(annotator)
