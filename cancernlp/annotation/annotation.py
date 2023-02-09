"""
The core classes for representing annotated text.

AnnotatedText represents a span of raw unicode text, and Annotation is the base class
for any  annotations (e.g. marking sentence or token spans, headers, entities,
relationships, etc.) on the text.

Because an Annotation represents some information about a particular span of text
(represented by character offsets), AnnotatedText and Annotation objects are tightly
coupled.  AnnotatedText holds a list of Annotations, and the Annotations each hold a
reference to the containing AnnotatedText.  While it is possible to create an Annotation
without a corresponding AnnotatedText object, most of the functionality (such as finding
covered/covering Annotations) is unavailable until the object has been added to an
AnnotatedText object via the add_annotations function.

The basic general-purpose Annotation subclasses are also set up here, such as Sentence,
Token, and Entity.  Applications should subclass these when creating annotations, and
use these general classes (when appropriate) for processing Annotations.
"""

import logging
from functools import total_ordering
from itertools import chain
from typing import (Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Set, Type, Union)
from uuid import uuid4

from more_itertools import one

from .span_list import SpanList

logger = logging.getLogger(__name__)

# we can specify Annotation types by name (e.g. Annotation.Token.MyToken) or by the
# actual class (e.g. MyToken)
ANN_TYPE = Union[str, Type["Annotation"]]
# lists of ANN_TYPE can either take a single type or a list of types
ANN_LIST_TYPE = Union[ANN_TYPE, List[ANN_TYPE]]


class AnnotationError(Exception):
    """
    This is a general-purpopse error arising from Annotation / AnnotatedText processing.
    """


class RegistrarMetaClass(type):
    """
    This meta-class allows automatic registration of Annotation subclasses.

    This allows lookup of Annotation classes by name, which is important for
    deserialization.
    """

    def __new__(cls, clsname, bases, attrs):
        newclass = super().__new__(cls, clsname, bases, attrs)
        cls.register_class(newclass)
        return newclass

    # mapping from canonical names to class instances
    _class_by_type_name: Dict[str, Type["Annotation"]] = {}

    @classmethod
    def register_class(cls, annotation_subclass: Type["Annotation"]) -> None:
        """Register this class with the global subclass registry.  Used on type
        instantiation."""
        class_name = annotation_subclass.get_type_name()
        registered_class = cls._class_by_type_name.get(class_name)
        if registered_class and registered_class is not cls:
            logger.warning(f"Re-registering class {class_name} with a new type!")
        cls._class_by_type_name[class_name] = annotation_subclass

    @classmethod
    def get_class_by_type_name(cls, annotation_type_name) -> Type["Annotation"]:
        """Return Annotation subclass corresponding to the given type name (e.g.
        "Annotation.Token.SpacyToken")"""
        ann_class = cls._class_by_type_name.get(annotation_type_name)
        if ann_class is None:
            raise AnnotationError(
                f"Specified class name {annotation_type_name} not found (has it been "
                "imported?)"
            )
        return ann_class


class CoveringInterface:
    """
    Common interface for convenience functions related to getting Annotations covered by
    the current object.  Represents shared functionality between Annotations and
    AnnotatedText objects.
    """

    def covering(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator["Annotation"]:
        """
        Get annotations within this object that match one or more of the types specified
        in type_allowlist
        """
        raise NotImplementedError("covering must be implemented by subclasses")

    @property
    def sentences(self) -> List["Sentence"]:
        """Convenience function returning sentences covered by this Annotation"""
        return list(self.covering(Sentence))

    @property
    def tokens(self) -> List["Token"]:
        """Convenience function returning tokens covered by this Annotation"""
        return list(self.covering(Token))

    @property
    def entities(self) -> List["Entity"]:
        """Convenience function returning entities covered by this Annotation"""
        return list(self.covering(Entity))

    def annotation_types(self) -> Set[str]:
        """Get set of Annotation type names covered by this object."""
        return set((_ann.get_type_name() for _ann in self.covering()))


class AnnotationBase:
    """Base class for all Annotations, required to allow references to the base class
    within the Annotation class (python doesn't allow direct references to Annotation
    within the Annotation class definition).  Do not directly subclass this class."""


@total_ordering
class Annotation(AnnotationBase, CoveringInterface, metaclass=RegistrarMetaClass):
    """
    Parent class for text annotations.  Specific Annotation types should subclass this
    class.

    An Annotation represents a character span on an AnnotatedText object.  Each
    Annotation should be added to the corresponding AnnotatedText object via the
    `add_annotations` method.  The Annotations in the AnnotatedText object can then be
    traversed using methods on individual Annotations.

    There are four basic methods for traversal and processing the document structure:
    covering, covered_by, overlapping, and matching.  For instance, to move from a Token
    to the containing Sentence, you would use `token.covered_by(Sentence)`, and to get
    the list of tokens within a sentence you would use `sentence.covering(Token)`.
    `overlapping` returns Annotations of the specified type(s) whose span overlaps the
    current Annotation's span by at least one character, and `matching` returns
    Annotations whose span exactly matches that of the current Annotation.
    """

    # cache of names by type to speed up get_type_name() calls
    # TODO: exclude this from serialization
    _name_by_type = {}

    def __init__(
        self,
        begin: int,
        end: int,
        annotation_id: Optional[str] = None,
        text: Optional[str] = None,
        annotated_text: Optional["AnnotatedText"] = None,
    ):
        """
        Create an Annotation object from begin/end span indices.

        Optionally a fixed annotation_id can be specified (otherwise a uuid is used).
        The text of the span can also be given, in which case the .text attribute can be
        used prior to adding this Annotation to an AnnotatedText object, and an
        additional check will be run when the Annotation is added to make sure the
        AnnotatedText span matches the given text.

        The AnnotatedText object to which this Annotation should be added can also be
        specified, however note that this **DOES NOT** add the annotation.  The
        Annotation must still be added via AnnotatedText.add_annotations().  The
        parameter is available to allow batching of Annotation creations across
        AnnotatedText objects while keeping track of which Annotations go with which
        AnnotatedTexts.
        """
        if end < begin:
            raise ValueError(
                "Annotation cannot have end before begin: end {end} < begin {begin}."
            )
        self.id = annotation_id if annotation_id else str(uuid4())
        self.begin = begin
        self.end = end
        self._text = text
        self.annotated_text = annotated_text
        self.predictions = []

    @classmethod
    def get_type_name(cls) -> str:
        """Get a canonical name for this class suitable for serialization, e.g.
        Annotation.Token.SpacyToken."""

        if cls in cls._name_by_type:
            return cls._name_by_type[cls]

        class_name_list = []
        cur_cls = cls
        while True:
            cur_class_name = cur_cls.__name__
            if cur_cls is AnnotationBase:
                break
            class_name_list.append(cur_class_name)
            cur_cls_bases = [
                _cls for _cls in cur_cls.__bases__ if issubclass(_cls, AnnotationBase)
            ]
            # We'll allow multiple inheritance, but only if exactly one parent is a
            # subclass of Annotation.
            cur_cls = one(
                cur_cls_bases,
                too_short=AnnotationError(
                    f"{cur_class_name} is not a subclass of Annotation!"
                ),
                too_long=AnnotationError(
                    f"{cur_class_name} has multiple superclasses that are subclasses "
                    "of Annotation!"
                ),
            )
        type_name = ".".join(reversed(class_name_list))

        # register name in cache
        cls._name_by_type[cls] = type_name

        return type_name

    @property
    def type(self) -> str:
        """Convenience function to get normalized type name as a property"""
        return self.get_type_name()

    @classmethod
    def get_class_by_type_name(cls, annotation_type_name) -> Type["Annotation"]:
        """Return Annotation subclass corresponding to the given type name (e.g.
        "Annotation.Token.SpacyToken")"""
        return RegistrarMetaClass.get_class_by_type_name(annotation_type_name)

    @property
    def text(self) -> str:
        """Return text span from containing document."""
        if self.annotated_text:
            # If we have the AnnotatedText object, directly return the substring
            # representing the span of this Annotation.
            return self.annotated_text.text[self.begin : self.end]
        if self._text:
            # If we haven't added this Annotation to an AnnotatedText, but a text string
            # was specified when the Annotation was created, return that string.
            return self._text
        raise AnnotationError(
            "Annotation has not yet been added to AnnotatedText, no text available!"
        )

    def __lt__(self, other):
        # Sort based on begin offset, then based on reversed end offset.
        # This allows long annotations (sentences) to come before short annotations
        # (tokens) that have the same begin index.
        return (self.begin, other.end) < (other.begin, self.end)

    # def __eq__(self, other):
    #     # We assume that ids are globally unique.  If two Annotations have the same ID
    #     # they represent the same Annotation, even if they are two distinct objects.
    #     return self.id == other.id

    def __repr__(self):
        return (
            f"{self.get_type_name()}(text={self.text}, "
            f"begin={self.begin}, end={self.end})"
        )

    def __len__(self):
        return self.end - self.begin

    def duplicate(self, offset=0, new_id=False):
        """Create a duplicate of this Annotation.

        The resulting Annotation will not maintain the annotated_text link, but will
        have the same ID unless new_id is True.  If offset is given, the begin and end
        of the annotation will be adjusted by this offset.
        """
        cls = self.__class__
        dupe = cls.__new__(cls)
        dupe.__dict__.update(vars(self))
        dupe.annotated_text = None
        dupe._text = self.text
        dupe.begin += offset
        dupe.end += offset
        if new_id:
            dupe.id = str(uuid4())
        return dupe

    def as_subtext(self, type_allowlist: Optional[ANN_LIST_TYPE] = None):
        """Create an AnnotatedSubText object from this Annotation.

        Duplicates of any Annotations covered by the span are automatically added,
        optionaly filtered by type_allowlist.
        """
        if not self.annotated_text:
            raise AnnotationError(
                "Can only call 'as_document' on Annotation that has "
                "been added to an AnnotatedText object."
            )
        return self.annotated_text.get_subtext(
            begin=self.begin, end=self.end, type_allowlist=type_allowlist
        )

    def covering(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator["Annotation"]:
        """Get annotations that this annotation is covering, i.e. the [start, end) span
        of the returned annotations fall within the [start, end) span of this
        annotation. Includes annotations with identical spans.
        """
        if not self.annotated_text:
            raise AnnotationError(
                "Can only call 'covering' on Annotation that has "
                "been added to an AnnotatedText object."
            )
        return self.annotated_text.get_annotations_covered_by(
            annotation=self, type_allowlist=type_allowlist
        )

    def covered_by(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator["Annotation"]:
        """Get annotations that this annotation is covered by, i.e. the [start, end)
        span of this annotation falls within the [start, end) span of the returned
        annotations. Includes annotations with identical spans.
        """
        if not self.annotated_text:
            raise AnnotationError(
                "Can only call 'covered_by' on Annotation that has "
                "been added to an AnnotatedText object."
            )
        return self.annotated_text.get_annotations_covering(
            annotation=self, type_allowlist=type_allowlist
        )

    def overlapping(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator["Annotation"]:
        """Get annotations whose span overlaps by at least one character with the span
        of this annotation.
        """

        if not self.annotated_text:
            raise AnnotationError(
                "Can only call 'overlapping' on Annotation that has "
                "been added to an AnnotatedText object."
            )
        return self.annotated_text.get_annotations_overlapping(
            annotation=self, type_allowlist=type_allowlist
        )

    def matching(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator["Annotation"]:
        """Get annotations whose span exactly matches the span of this annotation."""
        if not self.annotated_text:
            raise AnnotationError(
                "Can only call 'matching' on Annotation that has "
                "been added to an AnnotatedText object."
            )
        return self.annotated_text.get_annotations_matching(
            annotation=self, type_allowlist=type_allowlist
        )


class Sentence(Annotation):
    """Base type for other Sentence annotations to inherit from"""


class Token(Annotation):
    """Base type for other Token annotations to inherit from"""


class Entity(Annotation):
    """Base type for other Entity annotations to inherit from"""

    def __init__(
        self,
        begin: int,
        end: int,
        label: Optional[str] = None,
        annotation_id: Optional[str] = None,
        text: Optional[str] = None,
    ):
        super().__init__(begin=begin, end=end, annotation_id=annotation_id, text=text)
        self.label = label


def _normalize_type(ann_type: ANN_TYPE) -> Type[Annotation]:
    """Given a string representing an Annotation subclass, return the class.  Pass
    through objects that are already subclasses of Annotation, and raise errors on
    anything else."""
    if isinstance(ann_type, type):
        if not issubclass(ann_type, Annotation):
            raise AnnotationError(
                f"Non-Annotation type {ann_type.__name__} specified in annotation type "
                "list."
            )
    elif isinstance(ann_type, str):
        ann_type = Annotation.get_class_by_type_name(ann_type)
    else:
        raise AnnotationError(
            f"Unrecognized object {ann_type} specified in annotation type list."
        )
    return ann_type


def _normalize_type_list(
    ann_type_list: Optional[ANN_LIST_TYPE],
) -> List[Type[Annotation]]:
    """Given None or a list of strings or Annotation subclasses, return a list of
    Annotation subclasses."""
    ann_type_list = ann_type_list if ann_type_list else []
    if not isinstance(ann_type_list, list):
        ann_type_list = [ann_type_list]
    ann_type_list = [_normalize_type(_ann_type) for _ann_type in ann_type_list]
    return ann_type_list


def _type_filter(
    annotations: Iterable[Annotation], type_allowlist: Optional[ANN_LIST_TYPE]
) -> Iterator[Annotation]:

    type_allowlist = _normalize_type_list(type_allowlist)
    for _ann in annotations:
        for _type in type_allowlist:
            if isinstance(_ann, _type):
                yield _ann
                continue


class Annotatable:
    """Interface for objects that can have Annotations added / removed from them"""

    # Modes for add_annotations()
    # append -- no checks, just add the requested annotations
    ADD_MODE_APPEND = "append"
    # overwrite -- first remove any annotations of types that exist in the list to add
    ADD_MODE_OVERWRITE = "overwrite"
    # create -- raise an error if the AnnotatedText already contains Annotations of any
    # type that exists in the list to add
    ADD_MODE_CREATE = "create"
    ADD_MODES = [ADD_MODE_APPEND, ADD_MODE_OVERWRITE, ADD_MODE_CREATE]

    def add_annotations(
        self, annotations: Union[Annotation, List[Annotation]], mode=ADD_MODE_APPEND
    ):
        raise NotImplementedError()

    def remove_annotations(
        self,
        annotations: Optional[Union[Annotation, List[Annotation]]] = None,
        types: Optional[ANN_LIST_TYPE] = None,
    ):
        raise NotImplementedError()


class AnnotatedTextInterface(CoveringInterface, Annotatable):
    def optimize(self):
        """
        Compute optimization structures making covering/covered_by/etc. much more
        efficient.
        """
        raise NotImplementedError()

    def get_annotations_by_type(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator[Annotation]:
        raise NotImplementedError()

    def covering(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator["Annotation"]:
        """
        covering() returns all annotations in the AnnotatedText matching the
        type_allowlist.
        """
        return self.get_annotations_by_type(type_allowlist=type_allowlist)

    def annotate(self, annotator: Callable[["AnnotatedTextInterface"], None]) -> None:
        """Annotate the AnnotatedText with the given annotator"""
        annotator(self)

    def flatten(self) -> Iterable["AnnotatedText"]:
        """Return an iterable of AnnotatedText objects comprising this
        AnnotatedTextInterface"""
        raise NotImplementedError

    def add_annotations(
        self,
        annotations: Union[Annotation, List[Annotation]],
        mode: str = Annotatable.ADD_MODE_APPEND,
    ):
        raise NotImplementedError()

    def remove_annotations(
        self,
        annotations: Optional[Union[Annotation, List[Annotation]]] = None,
        types: Optional[ANN_LIST_TYPE] = None,
    ):
        pass


class AnnotatedText(AnnotatedTextInterface):
    """
    Class representing a text field supporting annotations.
    """

    def __init__(
        self,
        text: str,
        text_id: Optional[str] = None,
        annotations: Optional[Union[Annotation, List[Annotation]]] = None,
        auto_optimize: bool = False,
    ):
        """Create a new AnnotatedText object from raw text and optionally a list of
        Annotations.
        """
        self.text = text
        self.id = text_id if text_id else str(uuid4())
        self.annotations = SpanList(
            max_span_offset=len(text), auto_optimize=auto_optimize
        )
        if annotations:
            self.add_annotations(annotations)
        self.predictions = []

    def __repr__(self):
        text_to_show = self.text[:1000]
        if len(self.text) > len(text_to_show):
            text_to_show += "..."
        return f'AnnotatedText("{text_to_show}")'

    def flatten(self) -> Iterable["AnnotatedText"]:
        return [self]

    def add_annotations(
        self,
        annotations: Union[Annotation, Sequence[Annotation]],
        mode=Annotatable.ADD_MODE_APPEND,
    ):
        """Add annotations to this AnnotatedText.  These must not have been added to
        any other AnnotatedText spans.
        """
        if isinstance(annotations, Annotation):
            annotations = [annotations]
        for _ann in annotations:
            if _ann.annotated_text is not None and _ann.annotated_text is not self:
                raise ValueError(
                    "Attempting to add annotation to multiple AnnotatedText objects."
                )
            _ann.annotated_text = self

        if mode not in self.ADD_MODES:
            raise ValueError(f"Unknown mode {mode}")

        # check for existing annotations and handle duplicate types based on 'mode'
        if mode != self.ADD_MODE_APPEND:
            new_ann_types = {_ann.get_type_name() for _ann in annotations}
            existing_ann_types = self.annotation_types()
            duplicate_types = new_ann_types.intersection(existing_ann_types)
            if duplicate_types:
                if mode == self.ADD_MODE_OVERWRITE:
                    self.remove_annotations(types=list(duplicate_types))
                if mode == self.ADD_MODE_CREATE:
                    raise AnnotationError(
                        f"Attempting to add types {duplicate_types} to AnnotatedText "
                        f"that already contains them with mode {self.ADD_MODE_CREATE}"
                    )

        self.annotations.update(annotations)

        # now that we've associated the annotation with the target text, make sure any
        # specified text matches the text referenced by the begin/end offsets
        for _ann in annotations:
            if _ann._text is not None and _ann._text != _ann.text:
                raise AnnotationError(
                    f'Specified text "{_ann._text}" does not match offset text '
                    f'"{_ann.text}".'
                )

    def remove_annotations(
        self,
        annotations: Optional[Union[Annotation, List[Annotation]]] = None,
        types: Optional[ANN_LIST_TYPE] = None,
    ) -> List[Annotation]:
        """
        Remove annotations either by giving the exact annotation(s) to remove, or by
        type.  Returns the removed annotations.
        """
        if not annotations:
            annotations = []
        if isinstance(annotations, Annotation):
            annotations = [annotations]

        if types:
            annotations = chain(annotations, self.get_annotations_by_type(types))

        filtered_annotations = self.annotations.filter(annotations)
        for _ann in filtered_annotations:
            _ann._text = _ann.text
            _ann.annotated_text = None
        return filtered_annotations

    def get_annotations_by_type(
        self, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator[Annotation]:
        """Get iterator over annotations in the given type_allowlist."""
        if type_allowlist is None:
            return iter(self.annotations)
        return _type_filter(self.annotations, type_allowlist=type_allowlist)

    def get_annotations_covering(
        self, annotation: Annotation, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator[Annotation]:
        """Get annotations covering the given annotation, i.e. the [start, end)
        span of the given annotation falls within the [start, end) span of the returned
        annotations. Includes annotations with identical spans.
        """
        annotations = self.annotations.get_spans_covering(
            begin=annotation.begin, end=annotation.end
        )
        if type_allowlist is not None:
            annotations = _type_filter(annotations, type_allowlist=type_allowlist)
        # filter out current annotation
        annotations = (_ann for _ann in annotations if _ann is not annotation)
        return annotations

    def get_annotations_covered_by(
        self, annotation: Annotation, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator[Annotation]:
        """Get annotations covered by the span of the given annotation, i.e. the
        [start, end) span of the returned annotations falls within the [start, end) span
        of the given annotation. Includes annotations with identical spans.
        """
        annotations = self.annotations.get_spans_covered_by(
            begin=annotation.begin, end=annotation.end
        )
        if type_allowlist is not None:
            annotations = _type_filter(annotations, type_allowlist=type_allowlist)
        # filter out current annotation
        annotations = (_ann for _ann in annotations if _ann is not annotation)
        return annotations

    def get_annotations_overlapping(
        self, annotation: Annotation, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator[Annotation]:
        """Get annotations whose span overlaps by at least one character with the span
        of the given annotation.
        """
        annotations = self.annotations.get_spans_overlapping(
            begin=annotation.begin, end=annotation.end
        )
        if type_allowlist is not None:
            annotations = _type_filter(annotations, type_allowlist=type_allowlist)
        # filter out current annotation
        annotations = (_ann for _ann in annotations if _ann is not annotation)
        return annotations

    def get_annotations_matching(
        self, annotation: Annotation, type_allowlist: Optional[ANN_LIST_TYPE] = None
    ) -> Iterator[Annotation]:
        """Get annotations whose span exactly matches the span of the given
        annotation."""
        annotations = self.annotations.get_spans_matching(
            begin=annotation.begin, end=annotation.end
        )
        if type_allowlist is not None:
            annotations = _type_filter(annotations, type_allowlist=type_allowlist)
        # filter out current annotation
        annotations = (_ann for _ann in annotations if _ann is not annotation)
        return annotations

    def optimize(self):
        """
        Compute optimization structures making covering/covered_by/etc. much more
        efficient.
        """
        self.annotations.optimize()

    def get_subtext(
        self,
        begin: Optional[int] = None,
        end: Optional[int] = None,
        type_allowlist: Optional[ANN_LIST_TYPE] = None,
    ) -> "AnnotatedSubText":
        """Create an AnnotatedSubText object from the span and type_allowlist given."""
        if begin is None:
            begin = 0
        if end is None:
            end = len(self.text)
        annotations = self.annotations.get_spans_covered_by(begin=begin, end=end)
        if type_allowlist is not None:
            annotations = _type_filter(annotations, type_allowlist=type_allowlist)
        # pylint: disable=invalid-unary-operand-type
        annotations = [_ann.duplicate(offset=-begin) for _ann in annotations]
        text = self.text[begin:end]
        return AnnotatedSubText(
            text=text, parent_text_id=self.id, offset=begin, annotations=annotations
        )

    def update(self, subtext: Union["AnnotatedText", "AnnotatedSubText"]):
        """Update the current annotations with all new annotations from subtext."""
        my_ann_ids = {_ann.id for _ann in self.annotations}
        offset = subtext.offset if hasattr(subtext, "offset") else 0
        annotations = [
            _ann.duplicate(offset)
            for _ann in subtext.annotations
            if _ann.id not in my_ann_ids
        ]
        self.add_annotations(annotations)


class AnnotatedSubText(AnnotatedText):
    """
    Class representing a section of an AnnotatedText, or a subset of the original
    annotations.

    Not meant to be instantiated directly, use AnnotatedText.get_subtext or
    Annotation.as_subtext.
    """

    def __init__(
        self,
        text: str,
        parent_text_id: str,
        offset: Optional[int] = 0,
        text_id: Optional[str] = None,
        annotations: Optional[Union[Annotation, List[Annotation]]] = None,
    ):
        super().__init__(text=text, text_id=text_id, annotations=annotations)
        self.offset = offset
        self.parent_text_id = parent_text_id
