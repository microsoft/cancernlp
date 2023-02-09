"""
Utilities for efficiently searching for spans in a list of spans via covering,
corvered_by, matching, and overlapping
"""
import logging
from functools import partial
from typing import Callable, Iterator, List, TypeVar

logger = logging.getLogger(__name__)

Span = TypeVar("Span")


def _is_covering(a_begin: int, a_end: int, b_begin: int, b_end: int):
    """Test if span [a_begin, a_end) covers span [b_begin, b_end)."""
    return a_begin <= b_begin and a_end >= b_end


def _is_covered_by(a_begin: int, a_end: int, b_begin: int, b_end: int):
    """Test if span [a_begin, a_end) is covered by span [b_begin, b_end)."""
    return _is_covering(a_begin=b_begin, a_end=b_end, b_begin=a_begin, b_end=a_end)


def _is_overlapping(a_begin: int, a_end: int, b_begin: int, b_end: int):
    """Test if the spans [a_begin, a_end) and [b_begin, b_end) overlap."""
    return (
        a_begin <= b_begin < a_end
        or a_begin < b_end <= a_end
        or b_begin <= a_begin < b_end
        or b_begin < a_end < b_end
    )


def _is_matching(a_begin: int, a_end: int, b_begin: int, b_end: int):
    """Test if the spans [a_begin, a_end) and [b_begin, b_end) match."""
    return a_begin == b_begin and a_end == b_end


class SpanList:
    """
    A list of spans.  Each item in the list should have `begin` and `end` properties.
    Adds support for retrieving spans by covering/covered_by/overlapping/matching spans.
    """

    def __init__(self, max_span_offset=None, auto_optimize=False):
        self.spans = list()
        self._optimized = False
        self.max_span_offset = max_span_offset
        self.auto_optimize = auto_optimize

    #
    # delegate most work to underlying list, but monitor modifications
    #
    def _update(self):
        self._optimized = False
        # TODO: totally inefficient, we should maintain a sorted list
        self.spans.sort()
        if self.auto_optimize:
            self.optimize()

    def add(self, span: Span):
        """Add new span to list"""
        self.spans.append(span)
        self._update()

    def update(self, spans: List[Span]):
        """Add many spans to list"""
        self.spans.extend(spans)
        self._update()

    def remove(self, span: Span) -> Span:
        """Remove span from list"""
        try:
            idx = self.spans.index(span)
            removed_span = self.spans.pop(idx)
        finally:
            self._update()
        return removed_span

    def filter(self, spans_to_remove) -> List[Span]:
        """Remove many spans from list, returning the removed spans"""
        kept_spans = []
        removed_spans = []
        if not isinstance(spans_to_remove, set):
            spans_to_remove = set(spans_to_remove)

        for span in self.spans:
            if span in spans_to_remove:
                removed_spans.append(span)
            else:
                kept_spans.append(span)

        self.spans = kept_spans
        self._update()
        return removed_spans

    def __iter__(self):
        return self.spans.__iter__()

    def __len__(self):
        return self.spans.__len__()

    def __repr__(self):
        return self.spans.__repr__()

    def __str__(self):
        return self.spans.__str__()

    #
    # Logic for querying by span
    #

    def _filter_spans(self, span_test: Callable[[int, int], bool]) -> Iterator[Span]:
        filtered_anns = (_ann for _ann in self if span_test(_ann.begin, _ann.end))
        return filtered_anns

    def get_spans_covering(self, begin: int, end: int) -> Iterator[Span]:
        """Return all spans covering the span [begin, end)"""
        if self._optimized:
            return self._opt_get_spans_covering(begin, end)
        span_test = partial(_is_covering, b_begin=begin, b_end=end)
        return self._filter_spans(span_test)

    def get_spans_covered_by(self, begin: int, end: int) -> Iterator[Span]:
        """Return all spans covered by the span [begin, end)"""
        if self._optimized:
            return self._opt_get_spans_covered_by(begin, end)
        span_test = partial(_is_covered_by, b_begin=begin, b_end=end)
        return self._filter_spans(span_test)

    def get_spans_overlapping(self, begin: int, end: int) -> Iterator[Span]:
        """Return all spans overlapping the span [begin, end) by at least one
        character"""
        span_test = partial(_is_overlapping, b_begin=begin, b_end=end)
        return self._filter_spans(span_test)

    def get_spans_matching(self, begin: int, end: int) -> Iterator[Span]:
        """Return all spans exactly matching the span [begin, end)"""
        span_test = partial(_is_matching, b_begin=begin, b_end=end)
        return self._filter_spans(span_test)
