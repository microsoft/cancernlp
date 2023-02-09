"""
Defines Corpus, a collection of AnnotatedText objects with some convenience functions
allowing users to iterate over all Annotations in the corpus.
"""
import itertools
import logging
import random
import warnings
from typing import (Callable, Iterable, Iterator, List, Optional, Sequence,
                    Union)
from uuid import uuid4

from tqdm import tqdm

from ..collection_utils import group_by
from .annotation import (ANN_LIST_TYPE, Annotatable, AnnotatedText,
                         AnnotatedTextInterface, Annotation, CoveringInterface)

logger = logging.getLogger(__name__)


def _add_annotations(
    annotations: Union[Annotation, List[Annotation]],
    mode=Annotatable.ADD_MODE_APPEND,
    show_progress: bool = False,
):
    """Add annotations to the AnnotatedText objects they reference.  This method
    groups Annotations by AnnotatedText and adds them all at once.

    Annotations must have had the annotated_text constructor param set.
    """
    if isinstance(annotations, Annotation):
        annotations = [annotations]

    for ann in annotations:
        if ann.annotated_text is None:
            raise ValueError(
                "Annotation must have annotated_text specified in "
                "constructor in order to use with "
                "Corpus.add_annotations, otherwise it is impossible "
                "to infer which AnnotatedText the Annotation should "
                "be added to."
            )
    # have to find these in each document
    anns_by_ann_text_id = group_by(annotations, key=lambda _ann: _ann.annotated_text.id)
    # Create lookup of annotated_texts by id.
    # Do a check to make sure there are not duplicate IDs.
    docs_by_id = {}
    for ann in annotations:
        doc_id = ann.annotated_text.id
        cur_doc = docs_by_id.get(doc_id)
        if cur_doc:
            if cur_doc is not ann.annotated_text:
                raise ValueError(f"Found duplicate documents with id {doc_id}")
        else:
            docs_by_id[doc_id] = ann.annotated_text
    # add the Annotations to the referenced AnnotatedText objects
    for ann_text_id, anns_to_add in tqdm(
        anns_by_ann_text_id.items(),
        desc="adding annotations",
        disable=not show_progress,
    ):
        docs_by_id[ann_text_id].add_annotations(anns_to_add, mode=mode)


def _remove_annotations(
    annotations: Union[Annotation, List[Annotation]], show_progress: bool = False
):
    """
    Remove the given annotations from their referenced AnnotatedTexts.
    """
    if isinstance(annotations, Annotation):
        annotations = [annotations]
    # have to find these in each document
    anns_by_ann_text_id = group_by(annotations, key=lambda ann: ann.annotated_text.id)
    # Create lookup of annotated_texts by id.  This doesn't acutally ensure that
    # the referenced AnnotatedText objects are part of this Corpus
    docs_by_id = {ann.annotated_text.id: ann.annotated_text for ann in annotations}
    # remove the annotations from the correct documents
    for ann_text_id, anns_to_remove in tqdm(
        anns_by_ann_text_id.items(),
        desc="removing annotations",
        disable=not show_progress,
    ):
        removed_anns = docs_by_id[ann_text_id].remove_annotations(anns_to_remove)
        not_removed_anns = set(removed_anns).difference(anns_to_remove)
        if not removed_anns:
            not_removed_ann_ids = [ann.id for ann in not_removed_anns]
            logger.warning(
                f"Unable to remove annotations:\n{not_removed_ann_ids}"
                f"\nfrom AnnotatedText {ann_text_id}"
            )


class MultiAnnotatedText(AnnotatedTextInterface):
    """
    Collection of AnnotatedText objects treated as a single object
    """

    def __init__(
        self,
        annotated_texts: List[AnnotatedTextInterface],
        doc_id: Optional[str] = None,
    ):
        self.id = doc_id if doc_id else str(uuid4())
        self.annotated_texts = list(annotated_texts)

    def get_annotations_by_type(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator[Annotation]:
        """Get iterator over annotations in the given type_allowlist."""
        return itertools.chain(
            *[
                _ann_text.get_annotations_by_type(type_allowlist)
                for _ann_text in self.annotated_texts
            ]
        )

    def flatten(self) -> Iterable[AnnotatedText]:
        return itertools.chain(
            *[ann_text.flatten() for ann_text in self.annotated_texts]
        )

    def annotate(self, annotator: Callable[[AnnotatedTextInterface], None]) -> None:
        for ann_text in self.annotated_texts:
            ann_text.annotate(annotator=annotator)

    def add_annotations(
        self,
        annotations: Union[Annotation, List[Annotation]],
        mode=Annotatable.ADD_MODE_APPEND,
    ):
        """Add annotations to AnnotatedText object in this Corpus.

        Annotations must have had the annotated_text constructor param set.
        """
        _add_annotations(annotations=annotations, mode=mode)

    # pylint: disable=duplicate-code
    def remove_annotations(
        self,
        annotations: Optional[Union[Annotation, List[Annotation]]] = None,
        types: Optional[ANN_LIST_TYPE] = None,
    ):
        """
        Remove annotations either by giving the exact annotation(s) to remove, or by
        type.
        """
        if annotations:
            _remove_annotations(annotations=annotations)

        if types:
            for ann_text in self.annotated_texts:
                ann_text.remove_annotations(types=types)

    def covering(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator[Annotation]:
        """Shorthand for get_annotations_by_type"""
        return self.get_annotations_by_type(type_allowlist)

    def __len__(self):
        return self.annotated_texts.__len__()

    def __iter__(self):
        return self.annotated_texts.__iter__()

    def __getitem__(self, key):
        result = self.annotated_texts.__getitem__(key)
        # return slice results as a new Corpus object
        if isinstance(key, slice):
            result = self.__class__(result)
        return result


# various types that can be cast to a Corpus via ensure_corpus()
CorpusLikeType = Union[
    "Corpus", AnnotatedTextInterface, Iterable[Union["Corpus", AnnotatedTextInterface]]
]


class Corpus(CoveringInterface):
    def __init__(self, annotated_texts: Optional[CorpusLikeType] = None):
        self.annotated_texts = []
        self.add_text(annotated_texts)

    @classmethod
    def ensure_corpus(cls, corpus: CorpusLikeType) -> "Corpus":
        """
        ensure that the given input is a corpus of the current corpus subclass, convert
        it if needed
        """
        if isinstance(corpus, AnnotatedTextInterface):
            corpus = [corpus]
        if not issubclass(type(corpus), cls):
            # assume we can treat it like a list of AnnotatedText
            corpus = cls(corpus)
        return corpus

    def get_annotations_by_type(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator[Annotation]:
        """Get iterator over annotations in the given type_allowlist."""
        return itertools.chain(
            *[
                _ann_text.get_annotations_by_type(type_allowlist)
                for _ann_text in self.annotated_texts
            ]
        )

    def covering(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator["Annotation"]:
        return self.get_annotations_by_type(type_allowlist=type_allowlist)

    def add_text(self, annotated_texts: CorpusLikeType):
        """Add AnnotatedText objects to this corpus"""
        if isinstance(annotated_texts, AnnotatedTextInterface):
            annotated_texts = [annotated_texts]
        for _text in annotated_texts:
            self.annotated_texts.append(_text)

    def split(self, split_portion: float):
        """split corpus into two sub-corpora

        :split_portion: value between 0-1 defining the split position.
        :return: 2-tuple of corpuses, first having 'split_portion' portion of the data
        and the second having '1.0-split_portion' portion of the data.
        """
        if split_portion > 1.0:
            raise ValueError("split_portion must be in the range [0.0, 1.0]")

        split_idx = int(len(self) * split_portion)
        return self[:split_idx], self[split_idx:]

    def __add__(self, other):
        if not isinstance(other, Corpus):
            raise TypeError(f"__add__ only supports Corpus objects, not {type(other)}")
        return Corpus(self.annotated_texts + other.annotated_texts)

    def shuffle(self):
        """Shuffle the document order in-place."""
        random.shuffle(self.annotated_texts)

    @staticmethod
    def add_annotations(
        annotations: Union[Annotation, Sequence[Annotation]],
        mode=Annotatable.ADD_MODE_APPEND,
        show_progress: bool = False,
    ):
        """Add annotations to AnnotatedText object in this Corpus.

        Annotations must have had the annotated_text constructor param set.
        """
        _add_annotations(
            annotations=annotations, mode=mode, show_progress=show_progress
        )

    # pylint: disable=duplicate-code
    def remove_annotations(
        self,
        annotations: Optional[Union[Annotation, List[Annotation]]] = None,
        types: Optional[ANN_LIST_TYPE] = None,
        show_progress: bool = False,
    ):
        """
        Remove annotations either by giving the exact annotation(s) to remove, or by
        type.
        """
        if annotations:
            _remove_annotations(annotations=annotations, show_progress=show_progress)

        if types:
            for ann_text in tqdm(
                self.annotated_texts,
                desc="removing ann types",
                disable=not show_progress,
            ):
                ann_text.remove_annotations(types=types)

    def annotate(
        self,
        annotator: Callable[[AnnotatedTextInterface], None],
        show_progress: Optional[bool] = None,
        desc: Optional[str] = None,
        use_dask: bool = False,
        n_dask_partitions: int = 128,
    ) -> None:
        """
        Annotate each AnnotatedText in the Corpus in-place with the given annotator.

        Note: use_dask is deprecated, call nlp_lib.distributed.annotate instead.
        """

        if use_dask:
            warnings.warn(
                "use_dask is deprecated: call nlp_lib.distributed.annotate instead.  "
                "Will use local computation.",
                DeprecationWarning,
            )
        # use of dask parameters deprecated
        del n_dask_partitions

        for _ in self.annotate_stream(
            self.annotated_texts,
            annotator=annotator,
            show_progress=show_progress,
            desc=desc,
        ):
            # just iterate over the annotated stream, forcing annotation to run on each
            # AnnotatedText
            pass

    def flatten(self) -> "Corpus":
        ann_texts = itertools.chain(*[ann_text.flatten() for ann_text in self])
        return Corpus(ann_texts)

    @classmethod
    def annotate_stream(
        cls,
        records: Iterable[AnnotatedTextInterface],
        annotator: Callable[[AnnotatedTextInterface], None],
        show_progress: Optional[bool] = None,
        desc: Optional[str] = None,
    ) -> Iterable[AnnotatedTextInterface]:
        """
        Annotate an iterator of AnnotatedText objects lazily, returning an iterator
        of annotated objects.
        """
        if show_progress is None:
            show_progress = False if desc is None else True

        for ann_text in tqdm(records, desc=desc, disable=not show_progress):
            ann_text.annotate(annotator)
            yield ann_text

    def __len__(self):
        return self.annotated_texts.__len__()

    def __iter__(self):
        return self.annotated_texts.__iter__()

    def __getitem__(self, key):
        result = self.annotated_texts.__getitem__(key)
        # return slice results as a new Corpus object
        if isinstance(key, slice):
            result = self.__class__(result)
        return result
