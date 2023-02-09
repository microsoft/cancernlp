"""
Utilities for working with huggingface transformers library
"""
import logging
from typing import Any, Optional, Type, Union

from transformers import PreTrainedTokenizer

from ..annotation import Annotation
from ..transformer import TransformerInputToken, WordPieceTokenizerBase
from .pretrained_transformer_models import PretrainedTransformerModels

logger = logging.getLogger(__name__)


def pad_sequence(sequence, max_length: int, pad_element: Any = 0):
    """
    Truncate or extend sequence to be max_length.  If sequence must be extended, pad
    with instances of pad_element
    """
    sequence = sequence[:max_length]
    sequence.extend([pad_element] * (max_length - len(sequence)))
    return sequence


class WordPieceTokenizer(WordPieceTokenizerBase):
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer],
        do_lower_case: Optional[bool] = None,
        sentence_type: Optional[Type[Annotation]] = None,
        token_type: Optional[Type[Annotation]] = None,
        max_input_length: Optional[int] = None,
        output_token_type: Union[
            str, Type[TransformerInputToken]
        ] = TransformerInputToken,
        raise_alignment_exceptions: bool = False,
    ):
        """
        Tokenizer adding WordPiece tokens from PreTrainedTokenizer (e.g.
        BertTokenizer instance) to AnnotatedText documents.  Either sentence_type or
        token_type must be specified.

        :param tokenizer: The WordPiece tokenizer to use, from huggingface transformers
        package.
        :param sentence_type: If specified, each sentence_type Annotation will be
        tokenized separately.  This is useful in reducing computation time for token
        alignment, and also defining the unit of text over witch max_input_length
        applies. If not specified, the full text of the AnnotatedText will be tokenized.
        :param token_type: If specified, the given tokenization will be respected
        when creating WordPiece tokens -- WordPiece tokens will only be created
        within existing token_type annotations.
        :param max_input_length: If specified, the maximum number of token
        annotations to create within one sentence_type annotation.
        :param output_token_type: Subclass of TransformerInputToken to create as
        WordPiece Annotations.  Defaults to TransformerInputToken.
        :param raise_alignment_exceptions: If True, raise an exception if wordpiece
        alignment fails.  If False (default), do not add WordPiece Annotations if
        alignment fails.
        """

        # if we passed in a name or a file, load that model
        if isinstance(tokenizer, str):
            tokenizer_name = tokenizer
            tokenizer = PretrainedTransformerModels.get_pretrained_tokenizer(
                tokenizer, do_lower_case=do_lower_case
            )
            if tokenizer is None:
                raise ValueError(f"Could not find tokenizer {tokenizer_name}")

        super().__init__(
            tokenizer=tokenizer,
            sentence_type=sentence_type,
            token_type=token_type,
            max_input_length=max_input_length,
            output_token_type=output_token_type,
            raise_alignment_exceptions=raise_alignment_exceptions,
        )

    @property
    def vocab(self):
        """Return the underlying tokenizer vocab"""
        return self.tokenizer.vocab

    @property
    def cls_token(self):
        """Return the underlying tokenizer special token"""
        return self.tokenizer.cls_token

    @property
    def sep_token(self):
        """Return the underlying tokenizer special token"""
        return self.tokenizer.sep_token

    @property
    def pad_token(self):
        """Return the underlying tokenizer special token"""
        return self.tokenizer.pad_token
