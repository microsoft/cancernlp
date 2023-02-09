"""
Core classes for Annotations and AnnotatedText

TODO: more here
"""
from .annotation import (Annotatable, AnnotatedSubText, AnnotatedText,
                         AnnotatedTextInterface, Annotation, AnnotationError,
                         Entity, Sentence, Token)
from .corpus import Corpus, CorpusLikeType, MultiAnnotatedText
from .token_alignment import (AlignmentError, CharSpan, align_tokens,
                              regex_align_tokens)
