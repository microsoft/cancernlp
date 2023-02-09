"""
Utilities for working with huggingface transformers library
"""
import logging
from typing import Any, List, Optional, Type, Union

from transformers import BertTokenizer

from ..annotation import (AlignmentError, AnnotatedText, Annotation, Token,
                          regex_align_tokens)

logger = logging.getLogger(__name__)

BERT_SUBSTITUTIONS = {"[UNK]": ".+?"}


class TransformerInputToken(Token):
    """Class representing an input token to pytorch_transformer model (e.g. BERT)"""

    def __init__(
        self,
        begin: int,
        end: int,
        annotation_id: str = None,
        annotated_text: Optional[AnnotatedText] = None,
        text: str = None,
        token_text: str = None,
        token_id: int = None,
    ):
        """
        Adds token_text (original text of tokenization, including '##' prefix for
        WordPiece pieces), and token_id, the embedding ID of the token.
        """
        super().__init__(
            begin=begin,
            end=end,
            annotation_id=annotation_id,
            text=text,
            annotated_text=annotated_text,
        )
        self.token_text = token_text
        self.token_id = token_id

    def is_continuation_piece(self):
        """Is this token a wordpiece continuation token?"""
        return self.token_text.startswith("##")


class WordPieceTokenizerBase:
    def __init__(
        self,
        tokenizer: Any,
        sentence_type: Optional[Type[Annotation]] = None,
        token_type: Optional[Type[Annotation]] = None,
        max_input_length: Optional[int] = None,
        output_token_type: Union[
            str, Type[TransformerInputToken]
        ] = TransformerInputToken,
        raise_alignment_exceptions: bool = False,
    ):
        """
        Tokenizer adding WordPiece tokens from pre-trained BERT Tokenizer to
        AnnotatedText documents. Either sentence_type or token_type must be specified.

        :param tokenizer: The WordPiece tokenizer to use either from huggingface
            transformers package or custom trained.
        :param sentence_type: If specified, each sentence_type Annotation will be
            tokenized separately. This is useful in reducing computation time for
            token alignment, and also defining the unit of text over which
            max_input_length applies. If not specified, the full text of the
            AnnotatedText will be tokenized.
        :param token_type: If specified, the given tokenization will be respected
            when creating WordPiece tokens -- WordPiece tokens will only be created
            within existing token_type annotations.
        :param max_input_length: If specified, the maximum number of token
            annotations to create within one sentence_type annotation.
        :param output_token_type: Subclass of TransformerInputToken to create as
            WordPiece Annotations. Defaults to TransformerInputToken.
        :param raise_alignment_exceptions: If True, raises an exception when wordpiece
            alignment fails. Defaults to False.
        """

        if isinstance(output_token_type, str):
            output_token_type = Annotation.get_class_by_type_name(output_token_type)

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.token_type = token_type
        self.sentence_type = sentence_type
        self.output_token_type = output_token_type
        self.raise_alignment_exceptions = raise_alignment_exceptions

        if self.sentence_type is None and self.token_type is None:
            raise ValueError(
                "either sentence_type or token_type must be specified so "
                "that there are Annotations to iterate over during "
                "tokenization."
            )

    def get_token_annotations(self, covering_ann: Annotation):
        text = covering_ann.text
        offset = covering_ann.begin
        token_strings = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(token_strings)
        cleaned_token_strings = [
            _tok if not _tok.startswith("##") else _tok[2:] for _tok in token_strings
        ]

        try:
            alignment = regex_align_tokens(
                text.lower(),
                cleaned_token_strings,
                substitution_list=BERT_SUBSTITUTIONS,
            )
            per_tag_info = list(zip(alignment, token_strings, token_ids))

            token_anns = [
                self.output_token_type(
                    begin=span.begin + offset,
                    end=span.end + offset,
                    annotated_text=covering_ann.annotated_text,
                    token_text=tok_text,
                    token_id=tok_id,
                )
                for span, tok_text, tok_id in per_tag_info
            ]
        except AlignmentError:
            if self.raise_alignment_exceptions:
                raise
            logger.exception("Error aligning WordPiece tokenization")
            # give up on trying to add tokenization to this Annotation
            token_anns = []
        return token_anns

    def __call__(self, ann_text: AnnotatedText):
        """
        Add huggingface transformer model tokenization (e.g. BERT tokenization) to
        the given Corpus object.
        """
        if self.sentence_type is not None:
            segments_to_annotate = list(ann_text.covering(self.sentence_type))
        else:
            segments_to_annotate = [ann_text]

        wordpiece_tokens = []
        for segment in segments_to_annotate:
            # create annotations to add
            if self.token_type is not None:
                annotations = []
                for token in segment.covering(self.token_type):
                    annotations.extend(self.get_token_annotations(covering_ann=token))
            else:
                annotations = self.get_token_annotations(covering_ann=segment)

            # truncate to at most max_input_length
            if self.max_input_length is not None:
                if len(annotations) > self.max_input_length:
                    logger.warning(
                        f"Truncating tokenization, {len(annotations)} > "
                        f"{self.max_input_length} (max)"
                    )
                annotations = annotations[: self.max_input_length]
            wordpiece_tokens.extend(annotations)

        ann_text.add_annotations(wordpiece_tokens)


class TransformersHelper:
    # cache of loaded tokenizers
    _loaded_tokenizers = {}

    @classmethod
    def load_pretrained_tokenizer(
        cls, model_name: str, do_lower_case: Optional[bool] = None
    ):
        """
        Returns a singleton BertTokenizer object per model name
        """

        tokenizer = cls._loaded_tokenizers.get(model_name)
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(
                model_name, do_lower_case=do_lower_case
            )
            cls._loaded_tokenizers[model_name] = tokenizer

        return tokenizer

    @staticmethod
    def pad_sequence(sequence, max_length: int, pad_element: Any = 0):
        """
        Truncate or extend sequence to be max_length.  If sequence must be extended, pad
        with instances of pad_element
        """
        sequence = sequence[:max_length]
        sequence.extend([pad_element] * (max_length - len(sequence)))
        return sequence

    @staticmethod
    def sentence_to_bert_inputs(
        sentence: Annotation,
        max_input_length: int,
        cls_token_id: Any,
        sep_token_id: Any,
        pad_token_id: Any,
        output_token_type: Type[TransformerInputToken] = TransformerInputToken,
        pad_inputs: bool = True,
    ) -> List[int]:
        """
        Given a 'sentence' with wordpiece token annotations, return a list of
        WordPiece tokens ready for use as inputs to BERT.  Includes [CLS] and [SEP]
        tokens.

        Note: The sentence should already have wordpiece token annotations from being
        run through add_wordpiece_token_annotations (probably along with the rest of
        the associated Corpus)

        :param sentence: input Sentence annotation (could also be paragraph etc.)
        :return: list of BERT WordPiece IDs
        """

        tokens = list(sentence.covering(output_token_type))

        # skip sentences where tokens aren't present, probably because of an error
        # in creating tokenization
        if not tokens:
            # this will just result in a sentence of all PAD tokens (with CLS and SEP)
            logger.info("Found sentence with no WordPiece tokens.")

        token_ids = [int(_t.token_id) for _t in tokens]
        # truncate to max length
        if max_input_length:
            token_ids = token_ids[:max_input_length]
        # add [CLS] and [SEP] token IDs
        token_ids = [cls_token_id] + token_ids + [sep_token_id]
        if pad_inputs:
            token_ids = TransformersHelper.pad_sequence(
                token_ids,
                max_length=max_input_length + 2,  # account for [CLS] and [SEP]
                pad_element=pad_token_id,
            )

        return token_ids
